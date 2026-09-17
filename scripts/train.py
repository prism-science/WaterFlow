"""
Training pipeline for WaterFlow model.

This module provides the main training script for the WaterFlow water placement
model. It handles:
- Dataset loading and preprocessing with configurable quality filters
- Model construction with pluggable encoders (GVP, SLAE, ESM)
- Training loop with gradient accumulation and warmup scheduling
- Validation and evaluation with RK4 trajectory integration
- Checkpointing and W&B logging

Usage:
    python -m scripts.train \\
        --train_list /path/to/train.txt \\
        --val_list /path/to/val.txt \\
        --processed_dir /path/to/cache \\
        --base_pdb_dir /path/to/pdbs \\
        --epochs 200 \\
        --batch_size 4
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.constants import DEFAULT_EDGE_CUTOFF, DEFAULT_MIN_EDIA
from src.dataset import get_dataloader, ProteinWaterDataset
from src.distributed import (
    all_reduce_means,
    ddp_barrier,
    ddp_is_active,
    ddp_rank_and_world,
    is_main_process,
    run_once_on_main,
    setup_distributed,
    teardown_distributed,
)
from src.encoder_base import build_encoder
from src.flow import DYNAMIC_EDGE_POLICIES, FlowMatcher, FlowWaterGVP
from src.utils import (
    compute_placement_metrics,
    compute_rmsd,
    create_trajectory_gif,
    plot_3d_frame,
    setup_logging_for_tqdm,
)


# best.pt selection. blend = 0.85*F1 + 0.15*AUC-PR. Generative metrics are
# averaged over the last SEL_ROLLING_WINDOW eval epochs to smooth sampling noise.
BLEND_F1_WEIGHT = 0.85
BLEND_AUC_PR_WEIGHT = 1.0 - BLEND_F1_WEIGHT
SEL_ROLLING_WINDOW = 3


def generate_run_name(args: argparse.Namespace) -> str:
    """Generate a run name from timestamp and key parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers = f"L{args.flow_layers}"
    hidden = f"h{args.hidden_s}"
    name = f"{timestamp}_{args.encoder_type}_{layers}_{hidden}"
    return name


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace with all training hyperparameters and paths
    """
    # TODO: Add support for loading configuration from YAML/JSON config files.
    # This would allow users to save and share training configurations easily.
    # Example: --config config.yaml would load all arguments from the file,
    # with CLI args taking precedence for overrides.

    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help=(
            "Cache root. Geometry caches are expected in <processed_dir>/geometry, "
            "embeddings in <processed_dir>/<encoder_name>."
        ),
    )
    p.add_argument(
        "--base_pdb_dir",
        type=str,
        required=True,
        help="Base directory of PDB subdirectories used to build the geometry cache.",
    )
    p.add_argument(
        "--geometry_cache_name",
        type=str,
        default="geometry",
        help="Base name for geometry cache directory (e.g., 'geometry' -> geometry/ or geometry_unfiltered/)",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        help="Include symmetry mate atoms as protein nodes",
    )
    p.add_argument(
        "--include_ligands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include ligand, ion, cofactor and nucleic acid heavy atoms as protein "
            "nodes. Disabling appends '_noligands' to the geometry cache directory, "
            "so the two configs cache separately."
        ),
    )
    p.add_argument(
        "--duplicate_single_sample",
        type=int,
        default=1,
        help="If training on single sample, duplicate it N times for more gradient updates per epoch",
    )

    # dataset quality checks (always on)
    p.add_argument(
        "--max_com_dist",
        type=float,
        default=25.0,
        help="Quality: max allowed protein-water center-of-mass distance (Angstroms).",
    )
    p.add_argument(
        "--max_clash_fraction",
        type=float,
        default=0.05,
        help="Quality: max allowed fraction of waters clashing with protein.",
    )
    p.add_argument(
        "--clash_dist",
        type=float,
        default=2.0,
        help="Quality: distance threshold for defining a water-protein clash (Angstroms).",
    )
    p.add_argument(
        "--interface_dist_threshold",
        type=float,
        default=4.0,
        help="Quality: max inter-chain interface distance to treat chains as interacting (Angstroms).",
    )
    p.add_argument(
        "--min_water_residue_ratio",
        type=float,
        default=0.1,
        help=(
            "Quality: minimum waters/residue ratio required per structure. Applied "
            "at cache-write time, so it decides which structures the cache holds."
        ),
    )

    # per-water filtering (toggleable)
    p.add_argument(
        "--max_protein_dist",
        type=float,
        default=5.0,
        help="Water filter: remove waters farther than this from nearest protein atom (Angstroms).",
    )
    p.add_argument(
        "--min_edia",
        type=float,
        default=DEFAULT_MIN_EDIA,
        help="Water filter: remove waters with EDIA below this threshold.",
    )
    p.add_argument(
        "--max_bfactor_zscore",
        type=float,
        default=2.0,
        help=(
            "Water filter: remove waters with normalized B-factor above this "
            "threshold. Baked in at cache-write time, so a warm cache built at a "
            "different value is refused rather than extended."
        ),
    )
    p.add_argument(
        "--no_filter_by_distance",
        dest="filter_by_distance",
        action="store_false",
        help="Disable distance-from-protein water filtering (ignores --max_protein_dist).",
    )
    p.add_argument(
        "--no_filter_by_edia",
        dest="filter_by_edia",
        action="store_false",
        help="Disable EDIA-based water filtering (ignores --min_edia).",
    )
    p.add_argument(
        "--no_filter_by_bfactor",
        dest="filter_by_bfactor",
        action="store_false",
        help="Disable B-factor-based water filtering (ignores --max_bfactor_zscore).",
    )
    p.set_defaults(filter_by_distance=True, filter_by_edia=True, filter_by_bfactor=True)

    # model
    p.add_argument(
        "--encoder_type",
        type=str,
        default="esm",
        choices=["gvp", "slae", "esm"],
        help="Protein encoder. 'esm' (default) and 'slae' need embeddings "
        "precomputed under --processed_dir; 'gvp' learns from coordinates alone. "
        "'slae' is legacy and untested with the current pipeline.",
    )
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--hidden_s", type=int, default=256)
    p.add_argument("--hidden_v", type=int, default=64)
    p.add_argument("--flow_layers", type=int, default=3)
    p.add_argument(
        "--n_message_gvps",
        type=int,
        default=2,
        help="Number of GVPs in message function per edge type (default: 2)",
    )
    p.add_argument(
        "--n_update_gvps",
        type=int,
        default=2,
        help="Number of GVPs in node update function (default: 2)",
    )
    p.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for GVP layers (default: 0.1)",
    )
    # flow-matching prior
    p.add_argument(
        "--sampling_strategy",
        type=str,
        default="uniform_ball",
        choices=["uniform_ball", "scaled_gaussian"],
        help=(
            "Source distribution for the flow prior. Also resolves "
            "--dynamic_edge_policy auto (default: uniform_ball)"
        ),
    )

    # edge construction
    p.add_argument(
        "--dynamic_edge_policy",
        type=str,
        default="auto",
        choices=["auto", *DYNAMIC_EDGE_POLICIES],
        help=(
            "How water-touching edges are built: 'radius' connects everything "
            "within --cutoff, 'knn' takes a fixed neighbour count, "
            "'knn_if_isolated' is radius plus a rescue for stranded waters. "
            "'auto' picks radius under uniform_ball and knn_if_isolated under "
            "scaled_gaussian (default: auto)"
        ),
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_EDGE_CUTOFF,
        help=f"Distance cutoff in Angstroms for radius edges (default: {DEFAULT_EDGE_CUTOFF})",
    )
    p.add_argument(
        "--max_neighbors",
        type=int,
        default=256,
        help="Per-source cap on radius query results (default: 256)",
    )
    p.add_argument(
        "--knn_fallback_k",
        type=int,
        default=8,
        help=(
            "Nearest neighbours attached to waters the radius query stranded; "
            "0 disables the rescue. Ignored under --dynamic_edge_policy knn "
            "(default: 8)"
        ),
    )
    p.add_argument(
        "--disable_ww",
        action="store_true",
        help="Ablate water->water edges",
    )
    p.add_argument(
        "--disable_wp",
        action="store_true",
        help="Ablate water->protein edges",
    )
    p.add_argument(
        "--k_pw",
        type=int,
        default=12,
        help="Nearest neighbours for protein->water edges under 'knn' (default: 12)",
    )
    p.add_argument(
        "--k_ww",
        type=int,
        default=8,
        help="Nearest neighbours for water->water edges under 'knn' (default: 8)",
    )
    p.add_argument(
        "--k_wp",
        type=int,
        default=8,
        help="Nearest neighbours for water->protein edges under 'knn' (default: 8)",
    )

    # optional cached-embedding override
    p.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Optional cached embedding dimension override for SLAE/ESM encoders",
    )

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Number of batches to prefetch per worker",
    )
    p.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for faster CPU-GPU transfer",
    )
    p.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep workers alive between epochs",
    )
    p.add_argument(
        "--sample_cache_size",
        type=int,
        default=0,
        help="Per-worker in-process dataset sample LRU cache size (0 disables caching)",
    )
    p.add_argument(
        "--cache_load_mmap",
        action="store_true",
        default=False,
        help="Use mmap-backed torch.load for dataset cache files when supported",
    )

    # scheduler
    p.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )
    p.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps")
    p.add_argument(
        "--eta_min_factor",
        type=float,
        default=0.001,
        help="eta_min = lr * eta_min_factor",
    )
    p.add_argument(
        "--lr_decay_epochs",
        type=int,
        default=None,
        help="Cosine T_max in epochs. The LR reaches eta_min after this many epochs "
        "and holds there. Defaults to --epochs.",
    )
    p.add_argument(
        "--step_size", type=int, default=50, help="StepLR step size (epochs)"
    )
    p.add_argument("--step_gamma", type=float, default=0.5, help="StepLR gamma")

    # mixed precision / optimizer
    p.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the training forward pass under bfloat16 autocast (CUDA only). "
        "On by default; pass --no-use_amp to disable.",
    )
    p.add_argument(
        "--fused_adamw",
        action="store_true",
        help="Use the fused AdamW implementation (CUDA only).",
    )

    # checkpointing
    p.add_argument("--save_dir", type=str, default="flow_checkpoints")
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (auto-generated if not provided)",
    )
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_eval_samples", type=int, default=3)
    p.add_argument(
        "--eval_method",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="Integrator for the sampling eval.",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Integration steps for the sampling eval.",
    )
    p.add_argument(
        "--selection_metric",
        type=str,
        default="blend",
        choices=["val_loss", "f1", "auc_pr", "blend"],
        help="Metric that selects best.pt. 'val_loss' is checked every epoch. "
        f"'f1', 'auc_pr' and 'blend' ({BLEND_F1_WEIGHT}*F1 + {BLEND_AUC_PR_WEIGHT:g}"
        "*AUC-PR) come from the sampling eval, so they are checked on eval epochs "
        f"only and averaged over the last {SEL_ROLLING_WINDOW} of them.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest epoch checkpoint under save_dir/run_name.",
    )
    p.add_argument(
        "--save_gifs", action="store_true", help="Save trajectory GIFs during eval"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold in Angstroms for precision/recall (default: 1.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for weight init and data shuffling. Pass a negative "
        "value to leave them unseeded.",
    )
    p.add_argument(
        "--val_seed",
        type=int,
        default=1234,
        help="Seed for picking the eval samples and for their prior noise.",
    )

    # logging / wandb
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="If set, log to this wandb project. Omit to disable wandb "
        "(no login required).",
    )
    p.add_argument("--wandb_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    if args.encoder_type == "gvp" and args.embedding_dim is not None:
        p.error("--embedding_dim is only valid for cached encoders: slae or esm")
    if args.sample_cache_size < 0:
        p.error("--sample_cache_size must be >= 0")
    if args.resume and args.run_name is None:
        p.error("--resume requires --run_name (generated names are timestamped)")
    return args


def _extract_quality_config(args: argparse.Namespace) -> dict:
    """Extract dataset quality check parameters (always active in preprocessing)."""
    return {
        "max_com_dist": args.max_com_dist,
        "max_clash_fraction": args.max_clash_fraction,
        "clash_dist": args.clash_dist,
        "interface_dist_threshold": args.interface_dist_threshold,
        "min_water_residue_ratio": args.min_water_residue_ratio,
    }


def _extract_water_filter_config(args: argparse.Namespace) -> dict:
    """Extract per-water filtering parameters (toggleable)."""
    return {
        "max_protein_dist": args.max_protein_dist,
        "min_edia": args.min_edia,
        "max_bfactor_zscore": args.max_bfactor_zscore,
        "filter_by_distance": args.filter_by_distance,
        "filter_by_edia": args.filter_by_edia,
        "filter_by_bfactor": args.filter_by_bfactor,
    }


def _build_dataset_config(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """
    Build grouped dataset configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (dataset_kwargs, quality_kwargs, water_filter_kwargs):
            - dataset_kwargs: Merged dict for DataLoader creation
            - quality_kwargs: Structure-level quality check parameters
            - water_filter_kwargs: Per-water filtering parameters
    """
    quality_kwargs = _extract_quality_config(args)
    water_filter_kwargs = _extract_water_filter_config(args)
    dataset_kwargs = {
        "encoder_type": args.encoder_type,
        "base_pdb_dir": args.base_pdb_dir,
        "geometry_cache_name": args.geometry_cache_name,
        "include_mates": args.include_mates,
        "include_ligands": args.include_ligands,
        "sample_cache_size": args.sample_cache_size,
        "cache_load_mmap": args.cache_load_mmap,
        **quality_kwargs,
        **water_filter_kwargs,
    }
    return dataset_kwargs, quality_kwargs, water_filter_kwargs


def _ignored_water_filter_thresholds(args) -> list[str]:
    """
    Identify water filter thresholds that are disabled.

    Args:
        args: Parsed command-line arguments with filter_by_* flags

    Returns:
        List of threshold parameter names that are disabled (e.g., ['min_edia'])
    """
    ignored = []
    if not args.filter_by_distance:
        ignored.append("max_protein_dist")
    if not args.filter_by_edia:
        ignored.append("min_edia")
    if not args.filter_by_bfactor:
        ignored.append("max_bfactor_zscore")
    return ignored


def _log_dataset_filter_config(args, quality_kwargs: dict):
    """
    Log dataset quality check and water filter configuration.

    Args:
        args: Parsed command-line arguments with filter settings
        quality_kwargs: Structure-level quality check parameters to log
    """
    active_filters = {
        "distance": args.filter_by_distance,
        "edia": args.filter_by_edia,
        "bfactor": args.filter_by_bfactor,
    }
    logger.info(f"Dataset quality checks (always on): {quality_kwargs}")
    logger.info(f"Water filters (toggleable): {active_filters}")

    ignored = _ignored_water_filter_thresholds(args)
    if ignored:
        logger.info(f"Ignored water-filter thresholds (disabled): {ignored}")


def _required_embedding_field(encoder_type: str) -> str | None:
    """
    Get the required embedding field name for a given encoder type.

    Args:
        encoder_type: Encoder identifier ('gvp', 'slae', or 'esm')

    Returns:
        Field name string (e.g., 'embedding') or None if encoder doesn't need embeddings
    """
    if encoder_type in {"slae", "esm"}:
        return "embedding"
    return None


def _uses_cached_embeddings(encoder_type: str) -> bool:
    """Return whether the selected encoder consumes cached protein embeddings."""
    return _required_embedding_field(encoder_type) is not None


def _resolve_embedding_dim(
    sample_data,
    encoder_type: str,
    override_dim: int | None,
) -> int | None:
    """
    Infer or validate embedding dimension from sample data.

    Args:
        sample_data: HeteroData sample from the dataset
        encoder_type: Encoder identifier ('gvp', 'slae', or 'esm')
        override_dim: User-specified dimension override, or None to infer

    Returns:
        Embedding dimension, or None if encoder doesn't use embeddings

    Raises:
        ValueError: If required embedding field is missing or dimension mismatch
    """
    field = _required_embedding_field(encoder_type)
    if field is None:
        return None
    if field not in sample_data["protein"]:
        raise ValueError(
            f"Selected encoder '{encoder_type}' requires protein.{field}, "
            f"but it is missing from dataset samples. "
            f"Expected cached embeddings in data['protein'].embedding from "
            f"--processed_dir/{encoder_type}/<cache_key>.pt."
        )

    embedding_type = sample_data["protein"].get("embedding_type")
    if embedding_type is not None and embedding_type != encoder_type:
        raise ValueError(
            f"Selected encoder '{encoder_type}' requires protein.embedding_type="
            f"'{encoder_type}', but sample data has '{embedding_type}'."
        )

    inferred_dim = int(sample_data["protein"][field].shape[-1])
    if override_dim is not None and int(override_dim) != inferred_dim:
        raise ValueError(
            f"{encoder_type} dim override mismatch: override={override_dim}, "
            f"inferred={inferred_dim} from sample data"
        )
    return inferred_dim if override_dim is None else int(override_dim)


def resolve_encoder_config(args, sample_data, node_scalar_in: int):
    """
    Build a registry-friendly encoder config with inferred dimensions.

    Args:
        args: Parsed command-line arguments containing encoder settings
        sample_data: HeteroData sample used to infer embedding dimensions
        node_scalar_in: Number of input scalar features per node

    Returns:
        dict: Encoder configuration ready for build_encoder(), e.g.:
            - GVP: {"encoder_type": "gvp", "hidden_s": 256, "hidden_v": 64, ...}
            - SLAE: {"encoder_type": "slae", "embedding_key": "embedding", "embedding_dim": 128, ...}
            - ESM: {"encoder_type": "esm", "embedding_key": "embedding", "embedding_dim": 1536, ...}
    """
    encoder_config = {
        "encoder_type": args.encoder_type,
        "hidden_s": args.hidden_s,
        "hidden_v": args.hidden_v,
        "node_scalar_in": node_scalar_in,
        "freeze_encoder": args.freeze_encoder,
        "encoder_ckpt": args.encoder_ckpt,
    }

    if _uses_cached_embeddings(args.encoder_type):
        encoder_config["embedding_key"] = "embedding"
        encoder_config["embedding_dim"] = _resolve_embedding_dim(
            sample_data, args.encoder_type, args.embedding_dim
        )

    return encoder_config


def log_encoder_sample_stats(sample_data: HeteroData, encoder_type: str) -> None:
    """Log summary statistics for the selected encoder input features."""
    field = _required_embedding_field(encoder_type)
    if field is None:
        return
    emb = sample_data["protein"][field]
    embedding_type = sample_data["protein"].get("embedding_type", "unknown")
    logger.info(
        f"{field} type={embedding_type} shape={tuple(emb.shape)} "
        f"mean={emb.mean():.4f} std={emb.std():.4f} min={emb.min():.4f} max={emb.max():.4f}"
    )


def build_model(
    args: argparse.Namespace, device: torch.device, encoder_config: dict
) -> FlowWaterGVP:
    """
    Build encoder and flow model using registry-based encoder construction.

    Args:
        args: Parsed command-line arguments with model hyperparameters
        device: Torch device to place the model on
        encoder_config: Registry-friendly config from resolve_encoder_config()

    Returns:
        FlowWaterGVP: Initialized model with the specified encoder
    """
    logger.info(f"Building model with {args.encoder_type.upper()} encoder")
    logger.info(f"Resolved encoder config: {encoder_config}")

    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(args.hidden_s, args.hidden_v),
        layers=args.flow_layers,
        n_message_gvps=args.n_message_gvps,
        n_update_gvps=args.n_update_gvps,
        drop_rate=args.drop_rate,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        dynamic_edge_policy=args.dynamic_edge_policy,
        # "auto" depends on which prior the run uses, so pass that through.
        sampling_strategy=args.sampling_strategy,
        knn_fallback_k=args.knn_fallback_k,
        disable_ww=args.disable_ww,
        disable_wp=args.disable_wp,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
        k_wp=args.k_wp,
    ).to(device)

    return model


def run_eval_sampling(
    flow_matcher, val_loader, args, epoch, device, eval_indices, run_dir
):
    """Sample the fixed eval set with the configured integrator and return metrics.

    Args:
        eval_indices: Fixed list of dataset indices to evaluate (sampled once at start)
        run_dir: Path to run directory for saving outputs
    """
    flow_matcher.model.eval()

    # Rank r handles samples r, r + world_size, ... The sums are all-reduced below.
    rank, world_size = ddp_rank_and_world()
    results = []

    integrate = (
        flow_matcher.euler_integrate
        if args.eval_method == "euler"
        else flow_matcher.rk4_integrate
    )
    # The prior noise is the only randomness in eval. Seeding a fresh generator
    # here gives every eval epoch the same noise and leaves the training RNG alone.
    eval_rng = torch.Generator(device=device).manual_seed(args.val_seed)
    for i, idx in enumerate(eval_indices):
        # Shard by global position i, so plot/GIF filenames never collide.
        if i % world_size != rank:
            continue
        graph = val_loader.dataset[idx]
        if graph["water"].num_nodes == 0:
            continue

        out = integrate(
            graph,
            num_steps=args.eval_steps,
            device=device,
            return_trajectory=args.save_gifs,  # frames for the GIFs
            generator=eval_rng,
        )[0]  # integrators return a list; take the single result

        # compute metrics
        final_metrics = compute_placement_metrics(
            pred=out["water_pred"], true=out["water_true"], threshold=args.threshold
        )

        final_rmsd = compute_rmsd(out["water_pred"], out["water_true"])

        results.append(
            {
                "rmsd": final_rmsd,
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1": final_metrics["f1"],
                "auc_pr": final_metrics["auc_pr"],
            }
        )

        # plot final frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax,
            out["protein_pos"],
            None,
            out["water_pred"],
            out["water_true"],
            title=f"Epoch {epoch} Sample {i} | RMSD={final_rmsd:.2f}A | F1={final_metrics['f1']:.3f}",
        )

        plot_path = run_dir / "plots" / f"epoch{epoch}_sample{i}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if args.save_gifs:
            gif_path = run_dir / "gifs" / f"epoch{epoch}_sample{i}.gif"
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            create_trajectory_gif(
                trajectory=out["trajectory"],
                protein_pos=out["protein_pos"],
                water_true=out["water_true"],
                save_path=str(gif_path),
                title=f"Epoch {epoch} Sample {i}",
                fps=10,
                pdb_id=graph.pdb_id,
            )

    # Every rank must reach this collective, even one with no results.
    avg_metrics, _ = all_reduce_means(
        {
            f"eval/avg_{key}": sum(r[key] for r in results)
            for key in ("rmsd", "precision", "recall", "f1", "auc_pr")
        },
        len(results),
        device,
    )
    return avg_metrics


def _needs_grad_sync(step: int, n_batches: int, accum_steps: int) -> bool:
    """
    Whether this micro-step's backward must all-reduce gradients under DDP.

    True at every accumulation boundary and for the leftover steps at the end of
    the epoch, which also end in an optimizer.step(). Stepping on gradients that
    were never all-reduced leaves the ranks out of sync for good.
    """
    if (step + 1) % accum_steps == 0:
        return True
    return step >= n_batches - (n_batches % accum_steps)


def train_epoch(
    flow_matcher: FlowMatcher,
    train_loader: DataLoader,
    optimizer: AdamW,
    warmup_scheduler,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    optimizer_step_count: int,
) -> tuple[dict[str, float], int, int]:
    """Single training epoch with gradient accumulation and warmup support."""
    flow_matcher.model.train()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue

        # Only all-reduce gradients on micro-steps followed by an optimizer.step().
        no_sync = ddp_is_active() and not _needs_grad_sync(
            step, len(train_loader), args.grad_accum_steps
        )
        with flow_matcher.model.no_sync() if no_sync else contextlib.nullcontext():
            metrics = flow_matcher.training_step(
                batch,
                accumulation_steps=args.grad_accum_steps,
            )

        if metrics["per_sample_info"] is not None:
            per_sample_losses = metrics["per_sample_info"]["losses"].cpu()
            num_graphs = metrics["per_sample_info"]["num_graphs"]

            if hasattr(batch, "pdb_id"):
                pdb_ids = (
                    batch.pdb_id if isinstance(batch.pdb_id, list) else [batch.pdb_id]
                )
                logger.warning("=" * 60)
                logger.warning(f"Batch loss {metrics['loss']:.2f} exceeded 100.0!")
                logger.warning(f"Per-sample losses ({num_graphs} samples):")
                for i in range(num_graphs):
                    pdb_id = pdb_ids[i] if i < len(pdb_ids) else "unknown"
                    sample_loss = per_sample_losses[i].item()
                    logger.warning(f"[{i}] {pdb_id}: {sample_loss:.2f}")
                logger.warning("=" * 60)

        processed_batches += 1
        total_loss += metrics["loss"]
        total_rmsd += metrics["rmsd"]

        # Step optimizer every grad_accum_steps
        if (step + 1) % args.grad_accum_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in flow_matcher.model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

            # Step warmup scheduler per optimizer step
            if (
                warmup_scheduler is not None
                and optimizer_step_count <= args.warmup_steps
            ):
                warmup_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            rmsd=f"{metrics['rmsd']:.2f}",
            lr=f"{current_lr:.2e}",
        )

        global_step = (epoch - 1) * len(train_loader) + step
        wandb.log(
            {
                "train/iter_loss": metrics["loss"],
                "train/iter_rmsd": metrics["rmsd"],
                "lr": current_lr,
            },
            step=global_step,
        )

    # Handle remaining gradients at end of epoch
    if (step + 1) % args.grad_accum_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in flow_matcher.model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_count += 1
        if warmup_scheduler is not None and optimizer_step_count <= args.warmup_steps:
            warmup_scheduler.step()

    final_global_step = (epoch - 1) * len(train_loader) + len(train_loader) - 1

    # All-reduce before the zero-batch return below. A rank that skipped every
    # batch would otherwise never join the collective and hang the others.
    train_metrics, processed_batches = all_reduce_means(
        {"train/epoch_loss": total_loss, "train/epoch_rmsd": total_rmsd},
        processed_batches,
        device,
    )

    if processed_batches == 0:
        logger.warning(
            f"Epoch {epoch}: skipped all {skipped_batches} train batches (no waters)."
        )
        return (
            {"train/epoch_loss": float("inf"), "train/epoch_rmsd": float("inf")},
            final_global_step,
            optimizer_step_count,
        )

    logger.info(
        f"Epoch {epoch} [Train] processed_batches={processed_batches}, skipped_batches={skipped_batches}"
    )
    return train_metrics, final_global_step, optimizer_step_count


@torch.no_grad()
def val_epoch(
    flow_matcher: FlowMatcher,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Single validation epoch."""
    flow_matcher.model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        batch = batch.to(device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue
        metrics = flow_matcher.validation_step(batch)
        processed_batches += 1
        total_loss += metrics["loss"]
        total_rmsd += metrics["rmsd"]

    # Selection reads val/loss, so every rank needs the same value.
    val_metrics, processed_batches = all_reduce_means(
        {"val/loss": total_loss, "val/rmsd": total_rmsd},
        processed_batches,
        device,
    )

    if processed_batches == 0:
        logger.warning(
            f"Epoch {epoch}: skipped all {skipped_batches} val batches (no waters)."
        )
        return {"val/loss": float("inf"), "val/rmsd": float("inf")}

    logger.info(
        f"Epoch {epoch} [Val] processed_batches={processed_batches}, skipped_batches={skipped_batches}"
    )
    return val_metrics


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _latest_epoch_checkpoint(ckpt_dir: Path) -> Path | None:
    """Highest-numbered epoch_<n>.pt in ckpt_dir, or None if there is none."""
    ckpts = list(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def save_checkpoint(
    model,
    optimizer,
    warmup_scheduler,
    main_scheduler,
    epoch,
    optimizer_step_count,
    path,
    best=False,
    best_val_loss=None,
    best_sel_score=None,
    selection_metric=None,
    sel_history=None,
):
    """
    Save model checkpoint with optimizer and scheduler states.

    Args:
        model: FlowWaterGVP model instance
        optimizer: AdamW optimizer instance
        warmup_scheduler: LinearLR warmup scheduler, or None
        main_scheduler: Main LR scheduler (CosineAnnealingLR or StepLR), or None
        epoch: Current epoch number
        optimizer_step_count: Total number of optimizer steps taken
        path: Path object for checkpoint file destination
        best: If True, log as best checkpoint
        best_val_loss: Best val loss so far (resume metadata)
        best_sel_score: Best selection score so far, on selection_metric's scale
        selection_metric: Metric behind best_sel_score. A resume resets the
            score if this changes.
        sel_history: Per-eval-epoch selection scores, so a resume continues the
            rolling window
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_step_count": optimizer_step_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict()
            if warmup_scheduler
            else None,
            "main_scheduler_state_dict": main_scheduler.state_dict()
            if main_scheduler
            else None,
            "best_val_loss": best_val_loss,
            "best_sel_score": best_sel_score,
            "selection_metric": selection_metric,
            "sel_history": sel_history,
        },
        path,
    )
    logger.info(f"{'Best ' if best else ''}Checkpoint saved: {path}")


def build_scheduler(optimizer, args):
    """
    Build warmup and main learning rate schedulers.

    Supports hybrid stepping: warmup scheduler steps per optimizer step,
    main scheduler steps per epoch after warmup completes.

    Args:
        optimizer: AdamW optimizer instance
        args: Parsed arguments with scheduler configuration

    Returns:
        Tuple of (warmup_scheduler, main_scheduler), either may be None
    """
    # Warmup scheduler (stepped per optimizer step)
    warmup_scheduler = None
    if args.warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup_steps
        )

    # Main scheduler (stepped per epoch, after warmup)
    main_scheduler = None
    if args.scheduler == "cosine":
        t_max = (
            args.lr_decay_epochs if args.lr_decay_epochs is not None else args.epochs
        )
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.lr * args.eta_min_factor
        )
    elif args.scheduler == "step":
        main_scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    return warmup_scheduler, main_scheduler


def _build_cache_shard(
    list_file: str, processed_dir: str, dataset_kwargs: dict
) -> None:
    """
    Pool worker: build the geometry cache for one shard's list.

    Already-cached entries are skipped, and shards hold disjoint keys, so workers
    never write the same file.
    """
    ProteinWaterDataset(
        pdb_list_file=list_file,
        processed_dir=processed_dir,
        preprocess=True,
        **dataset_kwargs,
    )


def build_cache(args: argparse.Namespace) -> None:
    """
    Build the geometry cache for the train+val lists as the sole writer.

    Preprocessing is CPU/PyMOL only, so this runs before the DDP group exists --
    hence race-free. A warm cache is a fast no-op; a cold build is parallelized
    across CPU cores over disjoint key shards.
    """
    dataset_kwargs, _, _ = _build_dataset_config(args)
    ids = set()
    for lst in (args.train_list, args.val_list):
        with open(lst) as f:
            ids.update(line.strip() for line in f if line.strip())
    if not ids:
        return
    sorted_ids = sorted(ids)

    tmpdir = Path(tempfile.mkdtemp(prefix="wf_build_"))
    try:
        union = tmpdir / "union.txt"
        union.write_text("\n".join(sorted_ids) + "\n")
        # Parse-only probe to find which entries still need building.
        probe = ProteinWaterDataset(
            pdb_list_file=str(union),
            processed_dir=args.processed_dir,
            preprocess=False,
            **dataset_kwargs,
        )
        # Entries can share a cache_key. Dedup so each file is checked once.
        keys = list(dict.fromkeys(entry["cache_key"] for entry in probe.entries))
        missing = [k for k in keys if not (probe.geometry_dir / f"{k}.pt").is_file()]
        if not missing:
            return  # warm cache: nothing to build

        logger.info(f"build_cache: preprocessing {len(missing)} missing entries")
        # One worker per CPU, each on its own shard of keys. Always use the pool,
        # even for one shard, so PyMOL never runs in the parent process.
        n_shards = max(1, min(len(missing), os.cpu_count() or 1))
        shard_files = []
        for i in range(n_shards):
            shard = tmpdir / f"shard_{i}.txt"
            shard.write_text("\n".join(missing[i::n_shards]) + "\n")
            shard_files.append(str(shard))
        # spawn (not fork): safe alongside PyMOL's C extension and any threads.
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_shards) as pool:
            pool.starmap(
                _build_cache_shard,
                [(shard, args.processed_dir, dataset_kwargs) for shard in shard_files],
            )
    except Exception:
        logger.exception("build_cache failed; other ranks will block until timeout")
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    """Run the full training pipeline."""
    args = parse_args()

    # Seed weight init and shuffling. A negative --seed leaves them unseeded.
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Build the cache on rank 0 before the NCCL group exists. A cold build can
    # take longer than NCCL's timeout. Other ranks wait on a CPU store, which is
    # then reused to set up NCCL.
    store = run_once_on_main(lambda: build_cache(args), key="wf_cache_ready")

    # Under torchrun each rank gets its own GPU. A plain launch yields (0, 0, 1).
    rank, local_rank, world_size = setup_distributed(store=store)
    main_proc = is_main_process(rank)
    # Under DDP use the GPU set in setup_distributed, otherwise --device.
    if ddp_is_active():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # AMP needs CUDA with bf16. Write the result back to args so config.json
    # shows what ran.
    if args.use_amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        reason = (
            "device is not CUDA"
            if device.type != "cuda"
            else "the GPU lacks bfloat16 support"
        )
        logger.warning(f"--use_amp set but {reason}; training without AMP.")
        args.use_amp = False

    if args.run_name is None:
        args.run_name = generate_run_name(args)

    run_dir = Path(args.save_dir) / args.run_name
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "gifs").mkdir(exist_ok=True)
    ddp_barrier()  # other ranks wait until run_dir exists

    # Only rank 0 writes the log file. Other ranks log to the console.
    log_file = Path(args.log_file) if args.log_file else run_dir / "train.log"
    setup_logging_for_tqdm(
        level=args.log_level, log_file=str(log_file) if main_proc else None
    )

    logger.info("=" * 60)
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    if ddp_is_active():
        logger.info(
            f"DDP active: rank={rank} local_rank={local_rank} world_size={world_size}"
        )
    logger.info("=" * 60)

    # data loaders
    dataset_kwargs, quality_kwargs, _ = _build_dataset_config(args)
    _log_dataset_filter_config(args, quality_kwargs)

    train_loader = get_dataloader(
        pdb_list_file=args.train_list,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        duplicate_single_sample=args.duplicate_single_sample,
        distributed=ddp_is_active(),
        **dataset_kwargs,
    )

    val_loader = get_dataloader(
        pdb_list_file=args.val_list,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        duplicate_single_sample=args.duplicate_single_sample,
        distributed=ddp_is_active(),
        **dataset_kwargs,
    )

    # Fixed eval indices. Use a local generator so --val_seed does not reseed
    # numpy's global RNG.
    rng = np.random.default_rng(args.val_seed)
    eval_indices = rng.choice(
        len(val_loader.dataset),
        min(args.n_eval_samples, len(val_loader.dataset)),
        replace=False,
    ).tolist()

    eval_indices_file = run_dir / "eval_indices.txt"
    if main_proc:
        with open(eval_indices_file, "w") as f:
            f.write("# Fixed evaluation sample indices\n")
            for idx in eval_indices:
                graph = val_loader.dataset[idx]
                pdb_id = getattr(graph, "pdb_id", "unknown")
                f.write(f"{idx}\t{pdb_id}\n")
        logger.info(f"Fixed eval indices saved to: {eval_indices_file}")
    logger.info(f"Evaluating on {len(eval_indices)} proteins at each eval epoch")

    # detect input dimension and resolve encoder configuration from sample data
    sample_data = train_loader.dataset[0]
    node_scalar_in = int(sample_data["protein"].x.shape[-1])
    logger.info(f"Detected protein input dimension: {node_scalar_in}")

    log_encoder_sample_stats(sample_data, args.encoder_type)
    encoder_config = resolve_encoder_config(
        args, sample_data, node_scalar_in=node_scalar_in
    )

    config_dict = vars(args).copy()
    config_dict["active_water_filters"] = {
        "distance": args.filter_by_distance,
        "edia": args.filter_by_edia,
        "bfactor": args.filter_by_bfactor,
    }
    config_dict["ignored_water_filter_thresholds"] = _ignored_water_filter_thresholds(
        args
    )
    config_dict["node_scalar_in"] = node_scalar_in
    config_dict["resolved_encoder_config"] = encoder_config
    config_file = run_dir / "config.json"
    if main_proc:
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to: {config_file}")

    # wandb logs only when --wandb_project is set, on rank 0; disabled mode
    # makes wandb.log/finish no-ops and needs no login. mode=None leaves the
    # WANDB_MODE environment variable in charge, so offline runs keep working.
    wandb.init(
        project=args.wandb_project,
        dir=args.wandb_dir,
        name=args.run_name,
        config=config_dict,
        mode=None if (main_proc and args.wandb_project) else "disabled",
    )

    model = build_model(args, device, encoder_config=encoder_config)
    if ddp_is_active():
        # Ablated edge types can leave parameters unused in a backward.
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    # Use the unwrapped module for parameters, sampling and state_dicts. Saving
    # the DDP wrapper would prefix every key with "module.".
    raw_model = getattr(model, "module", model)

    trainable_params, total_params = count_parameters(raw_model)
    logger.info("Model statistics:")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # quick forward pass sanity check for cached embedding encoders
    if _uses_cached_embeddings(args.encoder_type):
        logger.info(f"Testing forward pass with {args.encoder_type.upper()}...")
        raw_model.eval()
        batch = next(iter(train_loader)).to(device)
        with torch.no_grad():
            num_graphs = int(batch["protein"].batch.max().item()) + 1
            t = torch.zeros(num_graphs, device=device)
            v_out = raw_model(batch, t)
            logger.info(f"Forward pass successful! Output shape: {v_out.shape}")
            logger.info(f"Output stats: mean={v_out.mean():.4f}, std={v_out.std():.4f}")
            if v_out.std() < 1e-6:
                logger.warning("Model output is constant! This indicates a problem.")
        raw_model.train()

    flow_matcher = FlowMatcher(
        model=model,
        sampling_strategy=args.sampling_strategy,
        use_amp=args.use_amp,
    )

    # Fused AdamW is a CUDA-only kernel.
    optimizer = AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=args.fused_adamw and device.type == "cuda",
    )
    warmup_scheduler, main_scheduler = build_scheduler(optimizer, args)

    best_val_loss = float("inf")
    best_sel_score = float("-inf")
    sel_history: list[float] = []
    optimizer_step_count = 0
    start_epoch = 0
    ckpt_metric = None

    if args.resume:
        ckpt_path = _latest_epoch_checkpoint(run_dir / "checkpoints")
        if ckpt_path is None:
            raise FileNotFoundError(
                f"--resume set but no epoch_*.pt found under {run_dir / 'checkpoints'}."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if warmup_scheduler is not None and ckpt.get("warmup_scheduler_state_dict"):
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
        if main_scheduler is not None and ckpt.get("main_scheduler_state_dict"):
            main_scheduler.load_state_dict(ckpt["main_scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        optimizer_step_count = ckpt["optimizer_step_count"]
        best_val_loss = ckpt.get("best_val_loss")
        best_val_loss = float("inf") if best_val_loss is None else best_val_loss
        best_sel_score = ckpt.get("best_sel_score")
        best_sel_score = float("-inf") if best_sel_score is None else best_sel_score
        sel_history = ckpt.get("sel_history") or []
        ckpt_metric = ckpt.get("selection_metric")
        logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch}.")

    # Generative metrics only exist on eval epochs. If no eval epoch is left,
    # select on val_loss so best.pt is still written.
    selection_metric = args.selection_metric
    remaining_epochs = range(start_epoch + 1, args.epochs + 1)
    if selection_metric != "val_loss" and not any(
        e % args.eval_every == 0 for e in remaining_epochs
    ):
        logger.warning(
            f"--selection_metric {selection_metric} needs an eval epoch, but none "
            f"fall in epochs {start_epoch + 1}..{args.epochs} at --eval_every "
            f"{args.eval_every}. Selecting best.pt on val_loss instead."
        )
        selection_metric = "val_loss"

    # best_sel_score and sel_history are on the selection metric's scale. Reset
    # them if the metric differs from the checkpoint's.
    if ckpt_metric is not None and ckpt_metric != selection_metric:
        logger.warning(
            f"Selection metric changed ({ckpt_metric} -> {selection_metric}); "
            "resetting best_sel_score."
        )
        best_sel_score = float("-inf")
        sel_history = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Without this every epoch replays the same shard order on every rank.
        if ddp_is_active():
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_metrics, global_step, optimizer_step_count = train_epoch(
            flow_matcher,
            train_loader,
            optimizer,
            warmup_scheduler,
            args,
            device,
            epoch,
            optimizer_step_count,
        )
        # Tag epoch metrics with the epoch number.
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics, step=global_step)

        val_metrics = val_epoch(flow_matcher, val_loader, device, epoch)
        val_metrics["epoch"] = epoch
        wandb.log(val_metrics, step=global_step)

        # Step the main scheduler once per epoch after warmup. Cosine reaches
        # eta_min at T_max and would rise again past it, so stop stepping there.
        if main_scheduler is not None and optimizer_step_count >= args.warmup_steps:
            past_cosine_horizon = (
                isinstance(main_scheduler, CosineAnnealingLR)
                and main_scheduler.last_epoch >= main_scheduler.T_max
            )
            if not past_cosine_horizon:
                main_scheduler.step()

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['train/epoch_loss']:.4f}, "
            f"val_loss={val_metrics['val/loss']:.4f}, val_rmsd={val_metrics['val/rmsd']:.2f}"
        )

        # Eval runs before selection so generative metrics are available for it.
        # All ranks enter (it ends in a collective). Use the unwrapped module so
        # DDP hooks do not run during integration.
        eval_metrics = {}
        if epoch % args.eval_every == 0:
            wrapped = flow_matcher.model
            flow_matcher.model = raw_model
            try:
                eval_metrics = run_eval_sampling(
                    flow_matcher,
                    val_loader,
                    args,
                    epoch,
                    device,
                    eval_indices,
                    run_dir,
                )
            finally:
                flow_matcher.model = wrapped
            if eval_metrics:
                wandb.log(eval_metrics, step=global_step)
                logger.info(
                    f"Eval: RMSD={eval_metrics['eval/avg_rmsd']:.2f}A, "
                    f"Precision={eval_metrics['eval/avg_precision']:.2%}, "
                    f"Recall={eval_metrics['eval/avg_recall']:.2%}, "
                    f"F1={eval_metrics['eval/avg_f1']:.3f}, "
                    f"AUC-PR={eval_metrics['eval/avg_auc_pr']:.3f}"
                )

        # All values below are all-reduced, so every rank picks the same epoch.
        # Only rank 0 writes. best_val_loss is always tracked for the checkpoint.
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]

        improved = False
        if selection_metric == "val_loss":
            sel = -val_metrics["val/loss"]
            if sel > best_sel_score:
                best_sel_score = sel
                improved = True
        elif eval_metrics:  # generative metric, defined only on eval epochs
            if selection_metric == "blend":
                raw = (
                    BLEND_F1_WEIGHT * eval_metrics["eval/avg_f1"]
                    + BLEND_AUC_PR_WEIGHT * eval_metrics["eval/avg_auc_pr"]
                )
            else:
                raw = eval_metrics[f"eval/avg_{selection_metric}"]
            sel_history.append(raw)
            window = sel_history[-SEL_ROLLING_WINDOW:]
            sel = sum(window) / len(window)
            if sel > best_sel_score:
                best_sel_score = sel
                improved = True

        if improved and main_proc:
            save_checkpoint(
                raw_model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / "best.pt",
                best=True,
                best_val_loss=best_val_loss,
                best_sel_score=best_sel_score,
                selection_metric=selection_metric,
                sel_history=sel_history,
            )

        if epoch % args.save_every == 0 and main_proc:
            save_checkpoint(
                raw_model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / f"epoch_{epoch}.pt",
                best_val_loss=best_val_loss,
                best_sel_score=best_sel_score,
                selection_metric=selection_metric,
                sel_history=sel_history,
            )

        # Realign ranks: rank 0 may have spent extra time writing checkpoints.
        ddp_barrier()

    wandb.finish()
    teardown_distributed()
    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed with an unhandled exception.")
        raise
