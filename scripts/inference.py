# inference.py

"""
Inference script for WaterFlow model.

Takes a text file of PDB paths, runs trajectory integration (euler/rk4),
and outputs plots, gifs, and metrics for each PDB.

Usage:
    python -m scripts.inference \
        --flow_run_dir /path/to/run_directory \
        --pdb_list /path/to/pdb_list.txt \
        --processed_dir /path/to/cache \
        --base_pdb_dir /path/to/pdbs \
        --out_dir /path/to/output \
        --method rk4 \
        --num_steps 100 \
        --save_gifs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from src.constants import DEFAULT_EDGE_CUTOFF, DEFAULT_MIN_EDIA, NUM_RBF
from src.dataset import ProteinWaterDataset
from src.encoder_base import build_encoder, resolve_encoder_config
from src.flow import FlowMatcher, FlowWaterGVP
from src.utils import (
    compute_placement_metrics,
    compute_rmsd,
    create_trajectory_gif,
    plot_3d_frame,
    setup_logging_for_tqdm,
)


# Configure logging to work with tqdm progress bars
setup_logging_for_tqdm()


def parse_args():
    """
    Parse command-line arguments for inference configuration.

    Returns:
        argparse.Namespace with inference parameters and paths
    """
    # TODO: Add support for loading configuration from YAML/JSON config files.
    # This would simplify inference invocation and ensure reproducibility.
    # Example: --config config.yaml would load inference settings.

    p = argparse.ArgumentParser(description="Run WaterFlow inference on PDB files")

    p.add_argument(
        "--flow_run_dir",
        type=str,
        required=True,
        help="Path to training run directory (contains config.json and checkpoints/)",
    )
    p.add_argument(
        "--pdb_list",
        type=str,
        required=True,
        help="Text file with PDB entries (one per line, format: pdb_id_final or pdb_id_final_chainID)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save inference outputs",
    )

    # data arguments
    p.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Parent directory containing all cached preprocessed data. This directory holds "
        "protein graphs, embeddings, and geometry subdirectories (e.g., processed_dir/geometry/). "
        "Each PDB's preprocessed data is stored in subdirectories organized by cache type.",
    )
    p.add_argument(
        "--base_pdb_dir",
        type=str,
        required=True,
        help="Base directory containing PDB subdirectories for dataset creation",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        help="Include symmetry mate atoms as protein nodes",
    )
    p.add_argument(
        "--geometry_cache_name",
        type=str,
        default=None,
        help="Subdirectory name within processed_dir specifying which water coordinate set to use. "
        "Options include 'geometry' (filtered waters meeting quality criteria) or "
        "'geometry_unfiltered' (all crystallographic waters). Overrides the model's config if specified.",
    )

    # checkpoint arguments
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename within flow_run_dir/checkpoints/ (default: best.pt)",
    )

    # integration arguments
    p.add_argument(
        "--method",
        type=str,
        default="rk4",
        choices=["euler", "rk4"],
        help="Integration method (default: rk4)",
    )
    p.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of integration steps (default: 100)",
    )

    p.add_argument(
        "--save_gifs",
        action="store_true",
        help="Save trajectory GIFs (slower but useful for visualization)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold in Angstroms for precision/recall (default: 1.0)",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of proteins to process in parallel (default: 8)",
    )

    p.add_argument(
        "--water_ratio",
        type=float,
        default=None,
        help="Sample num_residues * water_ratio waters instead of using ground truth "
        "count. num_residues counts ASU and symmetry-mate residues, so with "
        "--include_mates the same ratio yields ~1.7x more waters than without: two "
        "runs share a sampling budget only if their mate settings match.",
    )

    p.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics computation (precision, recall, RMSD) that require ground truth. "
        "Use when running inference on structures without ground truth waters. "
        "Automatically enabled when --water_ratio is specified.",
    )

    args = p.parse_args()

    return args


def load_config(run_dir: Path) -> dict:
    """
    Load training configuration from run directory.

    Args:
        run_dir: Path to training run directory containing config.json

    Returns:
        Dict with training configuration parameters

    Raises:
        FileNotFoundError: If config.json doesn't exist in run_dir
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def _extract_dataset_filter_config(config: dict) -> dict:
    """Extract dataset filter params from training config with fallback to defaults."""
    return {
        "max_com_dist": config.get("max_com_dist", 25.0),
        "max_clash_fraction": config.get("max_clash_fraction", 0.05),
        "clash_dist": config.get("clash_dist", 2.0),
        "interface_dist_threshold": config.get("interface_dist_threshold", 4.0),
        "min_water_residue_ratio": config.get("min_water_residue_ratio", 0.1),
        "max_protein_dist": config.get("max_protein_dist", 5.0),
        "min_edia": config.get("min_edia", DEFAULT_MIN_EDIA),
        "max_bfactor_zscore": config.get("max_bfactor_zscore", 2.0),
        "filter_by_distance": config.get("filter_by_distance", True),
        "filter_by_edia": config.get("filter_by_edia", True),
        "filter_by_bfactor": config.get("filter_by_bfactor", True),
    }


def build_model_from_config(config: dict, device: torch.device) -> nn.Module:
    """
    Build model architecture from training configuration.

    Uses registry-based encoder construction to instantiate the correct
    encoder type (GVP, SLAE, or ESM) based on config.

    Args:
        config: Training configuration dict with model hyperparameters.
            Expected keys include:
            - encoder_type: "gvp", "slae", or "esm"
            - hidden_s, hidden_v: Hidden dimensions for scalars/vectors
            - flow_layers: Number of flow layers
            - For cached encoders: embedding_dim and embedding_key="embedding"
        device: Device to place model on

    Returns:
        FlowWaterGVP model instance
    """
    encoder_config = resolve_encoder_config(config)
    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(config.get("hidden_s") or 256, config.get("hidden_v") or 64),
        edge_scalar_dim=config.get("edge_scalar_dim") or NUM_RBF,
        layers=config.get("flow_layers") or 3,
        drop_rate=config.get("drop_rate", 0.1),
        n_message_gvps=config.get("n_message_gvps", 2),
        n_update_gvps=config.get("n_update_gvps", 2),
        cutoff=config.get("cutoff", DEFAULT_EDGE_CUTOFF),
        max_neighbors=config.get("max_neighbors", 256),
        dynamic_edge_policy=config.get("dynamic_edge_policy", "radius"),
        # "auto" depends on which prior the run uses, so pass that through.
        sampling_strategy=config.get("sampling_strategy", "uniform_ball"),
        knn_fallback_k=config.get("knn_fallback_k", 8),
        disable_ww=config.get("disable_ww", False),
        disable_wp=config.get("disable_wp", False),
        k_pw=config.get("k_pw", 12),
        k_ww=config.get("k_ww", 8),
        k_wp=config.get("k_wp", 8),
    ).to(device)

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    """
    Load model weights from checkpoint file.

    Args:
        model: FlowWaterGVP model instance to load weights into
        checkpoint_path: Path to checkpoint .pt file
        device: Device to map checkpoint tensors to

    Returns:
        Epoch number from checkpoint, or None if not stored

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint.get("epoch", None)


def run_inference_batch(
    flow_matcher: FlowMatcher,
    graphs: list,
    method: str,
    num_steps: int,
    device: str,
    water_ratio: float = None,
) -> list:
    """
    Run inference on a batch of graphs.

    Args:
        flow_matcher: FlowMatcher instance
        graphs: List of HeteroData graphs
        method: Integration method ('euler' or 'rk4')
        num_steps: Number of integration steps
        device: Device to run on
        water_ratio: If provided, sample num_residues * water_ratio waters

    Returns:
        List of result dicts, each with:
            - protein_pos, water_true, water_pred
            - trajectory (if rk4)
            - pdb_id
    """
    if method == "rk4":
        results = flow_matcher.rk4_integrate(
            graphs,
            num_steps=num_steps,
            device=device,
            return_trajectory=True,
            water_ratio=water_ratio,
        )
    else:  # euler
        results = flow_matcher.euler_integrate(
            graphs,
            num_steps=num_steps,
            device=device,
            water_ratio=water_ratio,
        )

    return results


def save_plot(
    result: dict,
    pdb_id: str,
    output_path: Path,
    metrics: dict | None,
):
    """
    Save 3D visualization plot of water prediction results.

    Args:
        result: Dict with 'protein_pos', 'water_pred', 'water_true' arrays
        pdb_id: PDB identifier for title
        output_path: Path to save PNG image
        metrics: Dict with 'rmsd', 'precision', 'recall', 'f1' for title, or None if no ground truth
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if metrics is not None:
        title = (
            f"{pdb_id} | RMSD={metrics['rmsd']:.2f}Å | "
            f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} F1={metrics['f1']:.3f}"
        )
    else:
        n_pred = result["water_pred"].shape[0]
        title = f"{pdb_id} | {n_pred} waters predicted (no ground truth)"

    # water_true may be None or empty when no ground truth is available
    water_true = result.get("water_true")
    if water_true is not None and water_true.shape[0] == 0:
        water_true = None

    plot_3d_frame(
        ax,
        result["protein_pos"],
        None,  # no separate mate positions
        result["water_pred"],
        water_true,
        title=title,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run inference pipeline on a list of PDB structures."""
    setup_logging_for_tqdm()
    args = parse_args()

    # setup paths
    run_dir = Path(args.flow_run_dir)
    output_root = Path(args.out_dir)
    output_dir = output_root / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    if args.save_gifs:
        (output_dir / "gifs").mkdir(exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # load config and build model
    logger.info(f"Loading model from: {run_dir}")
    config = load_config(run_dir)
    model = build_model_from_config(config, device)

    # load checkpoint
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    epoch = load_checkpoint(model, checkpoint_path, device)
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")

    # Create FlowMatcher
    flow_matcher = FlowMatcher(
        model=model,
        sampling_strategy=config.get("sampling_strategy", "uniform_ball"),
    )

    # Load dataset
    logger.info(f"Loading PDBs from: {args.pdb_list}")

    # Determine include_mates from args or config
    include_mates = args.include_mates or config.get("include_mates", False)
    encoder_type = config.get("encoder_type", "gvp")

    # Use --geometry_cache_name if provided, otherwise the config value
    geometry_cache_name = args.geometry_cache_name or config.get(
        "geometry_cache_name", "geometry"
    )

    # Extract dataset filter config from training config for consistency
    filter_config = _extract_dataset_filter_config(config)

    dataset = ProteinWaterDataset(
        pdb_list_file=args.pdb_list,
        processed_dir=args.processed_dir,
        base_pdb_dir=args.base_pdb_dir,
        encoder_type=encoder_type,
        include_mates=include_mates,
        geometry_cache_name=geometry_cache_name,
        preprocess=True,
        **filter_config,
    )

    logger.info(f"Found {len(dataset)} PDB entries")
    logger.info(f"Using geometry cache: {geometry_cache_name}")

    # run inference
    logger.info(f"Running inference with method={args.method}, steps={args.num_steps}")
    logger.info(f"Threshold for metrics: {args.threshold}Å")
    logger.info(f"Batch size: {args.batch_size}")

    # Determine if metrics should be skipped
    # Skip metrics when explicitly requested or when using water_ratio (no ground truth count)
    skip_metrics = args.skip_metrics or args.water_ratio is not None

    if args.water_ratio is not None:
        logger.info(
            f"Water ratio: {args.water_ratio} (sampling num_residues × {args.water_ratio} waters)"
        )
    else:
        logger.info("Water ratio: None (using ground truth water count)")
    if skip_metrics:
        logger.info("Metrics computation: DISABLED (no ground truth comparison)")
    else:
        logger.info("Metrics computation: ENABLED")
    logger.info("-" * 60)

    all_metrics = []

    # collect graphs for inference
    valid_graphs = []
    skipped_pdbs = []
    for idx in range(len(dataset)):
        graph = dataset[idx]
        # Only skip zero-water PDBs when we need ground truth for metrics
        if graph["water"].num_nodes == 0 and not skip_metrics:
            skipped_pdbs.append(graph.pdb_id)
        else:
            valid_graphs.append(graph)

    if skipped_pdbs:
        logger.info(f"Skipping {len(skipped_pdbs)} PDBs with no water molecules")

    # process in batches
    num_batches = (len(valid_graphs) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(valid_graphs))
        batch_graphs = valid_graphs[start_idx:end_idx]

        # run batched inference
        batch_results = run_inference_batch(
            flow_matcher,
            batch_graphs,
            method=args.method,
            num_steps=args.num_steps,
            device=args.device,
            water_ratio=args.water_ratio,
        )

        # process each result in the batch
        for result in batch_results:
            pdb_id = result.get("pdb_id", f"unknown_{len(all_metrics)}")
            water_pred = result["water_pred"]
            water_true = result["water_true"]
            has_ground_truth = water_true is not None and water_true.shape[0] > 0

            # compute metrics only when ground truth is available and metrics are enabled
            if not skip_metrics and has_ground_truth:
                metrics = compute_placement_metrics(
                    pred=water_pred,
                    true=water_true,
                    threshold=args.threshold,
                )
                metrics["rmsd"] = compute_rmsd(water_pred, water_true)
                metrics["pdb_id"] = pdb_id
                metrics["n_waters_true"] = water_true.shape[0]
                metrics["n_waters_pred"] = water_pred.shape[0]
                all_metrics.append(metrics)
            else:
                # no ground truth comparison - just store prediction info
                metrics = None

            plot_path = output_dir / "plots" / f"{pdb_id}.png"
            save_plot(result, pdb_id, plot_path, metrics)

            # save GIF if requested and trajectory available
            if args.save_gifs and result.get("trajectory") is not None:
                gif_path = output_dir / "gifs" / f"{pdb_id}.gif"
                # Use water_true only if available
                gif_water_true = water_true if has_ground_truth else None
                create_trajectory_gif(
                    trajectory=result["trajectory"],
                    protein_pos=result["protein_pos"],
                    water_true=gif_water_true,
                    save_path=str(gif_path),
                    title="",
                    fps=10,
                    pdb_id=pdb_id,
                )

            # print per-sample info
            if metrics is not None:
                tqdm.write(
                    f"  {pdb_id}: RMSD={metrics['rmsd']:.2f}Å | "
                    f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} "
                    f"F1={metrics['f1']:.3f} AUC-PR={metrics['auc_pr']:.3f}"
                )
            else:
                tqdm.write(f"  {pdb_id}: {water_pred.shape[0]} waters predicted")

    # compute and save summary metrics
    if all_metrics:
        summary = {
            "n_samples": len(all_metrics),
            "avg_rmsd": float(np.mean([m["rmsd"] for m in all_metrics])),
            "std_rmsd": float(np.std([m["rmsd"] for m in all_metrics])),
            "avg_precision": float(np.mean([m["precision"] for m in all_metrics])),
            "avg_recall": float(np.mean([m["recall"] for m in all_metrics])),
            "avg_f1": float(np.mean([m["f1"] for m in all_metrics])),
            "avg_auc_pr": float(np.mean([m["auc_pr"] for m in all_metrics])),
            "avg_n_waters_true": float(
                np.mean([m["n_waters_true"] for m in all_metrics])
            ),
            "avg_n_waters_pred": float(
                np.mean([m["n_waters_pred"] for m in all_metrics])
            ),
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY METRICS")
        logger.info("=" * 60)
        logger.info(f"  Samples processed: {summary['n_samples']}")
        logger.info(f"  Avg waters (true):  {summary['avg_n_waters_true']:.1f}")
        logger.info(f"  Avg waters (pred):  {summary['avg_n_waters_pred']:.1f}")
        logger.info(
            f"  Avg RMSD:      {summary['avg_rmsd']:.3f} ± {summary['std_rmsd']:.3f} Å"
        )
        logger.info(f"  Avg Precision: {summary['avg_precision']:.3%}")
        logger.info(f"  Avg Recall:    {summary['avg_recall']:.3%}")
        logger.info(f"  Avg F1:        {summary['avg_f1']:.4f}")
        logger.info(f"  Avg AUC-PR:    {summary['avg_auc_pr']:.4f}")
        logger.info("=" * 60)

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "per_sample": all_metrics,
                    "config": {
                        "flow_run_dir": str(run_dir),
                        "checkpoint": args.checkpoint,
                        "method": args.method,
                        "num_steps": args.num_steps,
                        "threshold": args.threshold,
                        "include_mates": include_mates,
                        "water_ratio": args.water_ratio,
                        "geometry_cache_name": geometry_cache_name,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Metrics saved to: {metrics_path}")

    else:
        if skip_metrics:
            logger.info(f"Processed {len(valid_graphs)} samples (metrics disabled)")
        else:
            logger.warning("No valid samples processed.")

    logger.info(f"Plots saved to: {output_dir / 'plots'}")
    if args.save_gifs:
        logger.info(f"GIFs saved to: {output_dir / 'gifs'}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
