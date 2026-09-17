# predict_waters.py

"""Predict waters for raw PDB/CIF files.

Per structure: drop existing waters, build the flow graph from protein + hets,
sample candidate waters with a flow checkpoint, score them with a confidence
checkpoint, cluster and threshold, then write the input structure with the
predicted waters added (`<name>_pred.pdb|cif`) and a `<name>_waters.txt` of
`x y z confidence` rows. No ground truth is involved.

NOTE: scripts/inference.py on the other hand, evaluates the flow model alone,
on cached training-format graphs, against the ground-truth waters.

For esm/slae encoders the protein embeddings must already be in
--processed_dir (see generate_esm_embeddings.py / generate_slae_embeddings.py).

Models are loaded from a --ckpt_dir holding flow.pt, confidence.pt,
flow_config.json and confidence_config.json. It defaults to the mates models
shipped in the repo (checkpoints/mates), or to checkpoints/mates_off when no
input has usable crystal symmetry (predicted structures); pass a directory to
choose one yourself.

Usage (the default models use esm, so embeddings come first):
    python -m scripts.generate_esm_embeddings --struc protein.cif --processed_dir cache/
    python -m scripts.predict_waters --struc protein.cif --processed_dir cache/ --out_dir out/

    Run without symmetry mates:
        ... --ckpt_dir checkpoints/mates_off

    Density mode keeps a fixed count per residue instead of a cutoff:
        ... --selection density --density_ratio 0.6
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import biotite.structure as bts
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from scripts.inference import build_model_from_config, run_inference_batch
from src.confidence import build_confidence_model, cluster_waters_vdw, ConfidenceGVP
from src.confidence_dataset import _oxygen_features
from src.constants import DEFAULT_EDGE_CUTOFF
from src.dataset import crystal_symmetry_check, parse_asu_with_biotite
from src.flow import FlowMatcher
from src.inference_graph import build_inference_graph
from src.structure_io import merge_waters, read_space_group, write_structure
from src.utils import setup_logging_for_tqdm


DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # confidence mode
DEFAULT_DENSITY_RATIO = 0.6  # density mode, waters per ASU residue


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Load weights: a model param with no weight is fatal (it would run at
    init); checkpoint keys with no matching module are dropped.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"{checkpoint_path.name}: {len(result.missing_keys)} model params "
            f"missing from the checkpoint (e.g. {result.missing_keys[:3]}); "
            "refusing to run with layers left at init."
        )
    if result.unexpected_keys:
        logger.info(
            f"{checkpoint_path.name}: dropped {len(result.unexpected_keys)} "
            f"checkpoint keys with no matching module (e.g. "
            f"{result.unexpected_keys[:3]})"
        )
    model.eval()


# ---------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------


def score_candidates(
    conf_model: ConfidenceGVP,
    graph,
    candidate_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Confidence score in [0, 1] for each candidate, given the protein graph."""
    n = candidate_pos.size(0)
    if n == 0:
        return candidate_pos.new_zeros(0)
    scored = graph.clone()
    scored["water"].pos = candidate_pos.to(device)
    scored["water"].x = _oxygen_features(n, device=device)
    scored["water"].num_nodes = n
    scored = scored.to(device)
    with torch.inference_mode():
        return conf_model(scored).detach().cpu()


def select_waters(
    candidate_pos: torch.Tensor,
    confidences: torch.Tensor,
    *,
    mode: str,
    threshold: float | None = None,
    density_ratio: float | None = None,
    num_asu_residues: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster candidates and cull to the final set.

    confidence: candidates with confidence >= threshold are clustered (the rest
    cannot pull a centroid) and every resulting centroid is kept.
    density: cluster with no cutoff, then keep the top
    floor(density_ratio * num_asu_residues) centroids by confidence, or all of
    them if there are fewer.
    """
    if mode == "confidence":
        return cluster_waters_vdw(candidate_pos, confidences, threshold=threshold)
    if mode == "density":
        if density_ratio is None or num_asu_residues is None:
            raise ValueError(
                "density selection needs density_ratio and num_asu_residues"
            )
        if not (math.isfinite(density_ratio) and density_ratio > 0):
            raise ValueError(
                f"density_ratio must be finite and > 0, got {density_ratio}"
            )
        pos, conf = cluster_waters_vdw(candidate_pos, confidences)  # sorted desc
        n_keep = int(density_ratio * num_asu_residues)
        if n_keep > len(pos):
            logger.warning(
                f"density: asked for {n_keep} waters but only {len(pos)} centroids"
            )
        return pos[:n_keep], conf[:n_keep]
    raise ValueError(f"Unknown selection mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-structure prediction
# ---------------------------------------------------------------------------


def _input_frame(struc_path: str) -> tuple[bts.AtomArray, str | None]:
    """Atoms to write out and the input space group.

    Hets are always written, whether or not the flow model saw them: the output
    is the input structure plus waters.
    """
    protein_atoms, _waters, ligand_atoms = parse_asu_with_biotite(struc_path)
    kept = protein_atoms + ligand_atoms if len(ligand_atoms) else protein_atoms
    return kept, read_space_group(struc_path)


def _candidate_path(cache: Path, name: str) -> Path:
    """Cached sampled waters for one structure inside the predict cache."""
    return cache / "candidates" / f"{name}.pt"


def predict_structures(
    struc_paths: list[str],
    flow_matcher: FlowMatcher | None,
    conf_model: ConfidenceGVP,
    flow_config: dict,
    args: argparse.Namespace,
    device: torch.device,
    cache: Path | None = None,
) -> None:
    """Predict + write final waters for a batch of structures.

    With cache set, flow input graphs are reused from <cache>/<name>.pt and
    candidate waters from _candidate_path(cache, name); missing entries are
    computed and written. The caller derives the directory from the run
    settings, so the structure name alone identifies a file here.
    """
    graphs, frames, out_names = [], [], []
    for path in struc_paths:
        graphs.append(
            build_inference_graph(
                path,
                encoder_type=flow_config.get("encoder_type", "gvp"),
                processed_dir=args.processed_dir,
                include_mates=args.include_mates,
                include_ligands=flow_config.get("include_ligands", True),
                cutoff=flow_config.get("cutoff", DEFAULT_EDGE_CUTOFF),
                max_neighbors=flow_config.get("max_neighbors", 256),
                cache_dir=cache,
            )
        )
        frames.append(_input_frame(path))
        out_names.append(Path(path).stem)

    # Candidate waters (centred frame): cached ones are reused, the rest are
    # sampled in one batch.
    candidates: list[torch.Tensor | None] = [None] * len(graphs)
    todo_idx, todo_graphs = [], []
    for i, name in enumerate(out_names):
        cand_pt = _candidate_path(cache, name) if cache else None
        if cand_pt is not None and cand_pt.exists():
            candidates[i] = torch.load(cand_pt, weights_only=True)["candidate_pos"]
        else:
            todo_idx.append(i)
            todo_graphs.append(graphs[i])

    if todo_graphs:
        if flow_matcher is None:
            raise RuntimeError("flow model needed for sampling but not loaded")
        results = run_inference_batch(
            flow_matcher,
            todo_graphs,
            method=args.method,
            num_steps=args.num_steps,
            device=str(device),
            water_ratio=args.water_ratio,
        )
        for i, result in zip(todo_idx, results):
            cand = torch.as_tensor(result["water_pred"], dtype=torch.float32)
            candidates[i] = cand
            if cache is not None:
                cand_pt = _candidate_path(cache, out_names[i])
                cand_pt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"candidate_pos": cand}, cand_pt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for graph, candidate_pos, (kept, space_group), name in zip(
        graphs, candidates, frames, out_names
    ):
        conf = score_candidates(conf_model, graph, candidate_pos, device)
        sel_pos, sel_conf = select_waters(
            candidate_pos,
            conf,
            mode=args.selection,
            threshold=args.confidence_threshold,
            density_ratio=args.density_ratio,
            num_asu_residues=int(graph["protein"].num_protein_residues),
        )
        if len(sel_pos) == 0:
            logger.warning(f"{name}: no waters selected")
        # Back to the input frame, then write structure + scored coordinates.
        water_xyz = sel_pos.numpy() + graph.center.numpy()
        structure = merge_waters(kept, water_xyz)
        write_structure(
            structure,
            str(out_dir / f"{name}_pred{args.out_format}"),
            space_group=space_group,
        )
        xyz_conf = np.column_stack([water_xyz, sel_conf.numpy()])
        np.savetxt(
            out_dir / f"{name}_waters.txt",
            xyz_conf,
            fmt=["%.3f", "%.3f", "%.3f", "%.4f"],
            header="x y z confidence",
        )
        logger.info(f"{name}: {len(water_xyz)} waters -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _collect_struc_paths(args: argparse.Namespace) -> list[str]:
    """--struc files, or --pdb_list entries resolved under --base_pdb_dir
    (with or without a .pdb/.cif extension).

    Outputs, embeddings and cache entries are all keyed by file stem, so
    duplicate stems are rejected.
    """
    if args.struc:
        paths = [str(p) for p in args.struc]
    else:
        base = Path(args.base_pdb_dir)
        names = [
            ln.strip()
            for ln in Path(args.pdb_list).read_text().splitlines()
            if ln.strip()
        ]
        paths = []
        for name in names:
            if (base / name).suffix.lower() in (".cif", ".pdb"):
                candidates = [base / name]
            else:
                candidates = [base / f"{name}{ext}" for ext in (".cif", ".pdb")]
            match = next((c for c in candidates if c.exists()), None)
            if match is not None:
                paths.append(str(match))
            else:
                logger.warning(f"No structure file found for {name!r} under {base}")
    counts = Counter(Path(p).stem for p in paths)
    duplicates = sorted(s for s, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate structure names {duplicates}: outputs and cache entries "
            "are keyed by file name, so every input needs a distinct one."
        )
    return paths


def resolve_ckpt_dir(explicit: str | None, paths: list[str]) -> Path:
    """Pick the checkpoint directory when the user did not name one.

    The shipped mates models expect crystal symmetry in the input. When no
    input has any, fall back to the shipped mates_off models so predicted
    structures work out of the box. A mix of inputs with and without symmetry
    is refused: one model cannot serve both, so such runs must be split.
    """
    if explicit is not None:
        return Path(explicit)
    errors = {p: crystal_symmetry_check(p) for p in paths}
    bad = [p for p, why in errors.items() if why is not None]
    if not bad:
        return Path("checkpoints/mates")
    if len(bad) < len(paths):
        names = [Path(p).name for p in bad]
        raise ValueError(
            f"Inputs mix structures with and without usable crystal symmetry "
            f"({names} lack it); split them into separate runs."
        )
    off_dir = Path("checkpoints/mates_off")
    needed = ("flow.pt", "confidence.pt", "flow_config.json", "confidence_config.json")
    missing = [f for f in needed if not (off_dir / f).exists()]
    if missing:
        raise ValueError(
            f"No input has usable crystal symmetry and the mates_off fallback "
            f"is missing {missing} under {off_dir}; pass --ckpt_dir with "
            "models trained without mates."
        )
    example = bad[0]
    logger.warning(
        f"No input has usable crystal symmetry ({Path(example).name}: "
        f"{errors[example]}); switching to the mates_off models"
    )
    return off_dir


def _check_embeddings(
    paths: list[str], encoder_type: str, processed_dir: str | None
) -> None:
    """Fail before any model is loaded if an esm/slae embedding
    (<processed_dir>/<encoder_type>/<stem>.pt) is missing. gvp needs none.
    """
    if encoder_type == "gvp":
        return
    if processed_dir is None:
        raise ValueError(f"--processed_dir is required for encoder_type={encoder_type}")
    emb_dir = Path(processed_dir) / encoder_type
    missing = [p for p in paths if not (emb_dir / f"{Path(p).stem}.pt").exists()]
    if missing:
        raise ValueError(
            f"Missing {encoder_type} embeddings under {emb_dir} for "
            f"{[Path(p).name for p in missing]}. Generate them first with "
            f"scripts/generate_{encoder_type}_embeddings.py"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt_dir",
        default=None,
        help="Directory holding flow.pt, confidence.pt, flow_config.json and "
        "confidence_config.json. Default: the mates models shipped in the repo, "
        "or the mates_off models when no input has usable crystal symmetry.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--struc", nargs="+", help="One or more PDB/CIF files.")
    src.add_argument(
        "--pdb_list",
        help="Text file of structure names under --base_pdb_dir, one per line; "
        "each may include or omit a .pdb/.cif extension.",
    )
    p.add_argument("--base_pdb_dir", help="Directory --pdb_list names resolve against.")
    p.add_argument(
        "--processed_dir",
        default=None,
        help="Embedding cache root for esm/slae encoders (unused for gvp). "
        "Embeddings are loaded, not generated: run generate_esm_embeddings.py or "
        "generate_slae_embeddings.py first. Looked up by file stem under "
        "processed_dir/<encoder_type>.",
    )
    p.add_argument(
        "--predict_cache",
        default=None,
        help="Root directory to reuse flow input graphs and sampled candidate "
        "waters across runs. Each checkpoint + sampling configuration gets its "
        "own subdirectory, so runs never mix files. Off by default.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_format", default=".pdb", choices=[".pdb", ".cif"])

    p.add_argument(
        "--selection",
        default="confidence",
        choices=["confidence", "density"],
        help="Final-water selection rule.",
    )
    p.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="confidence mode: keep candidates with confidence >= this, in [0, 1] "
        f"(default {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    p.add_argument(
        "--density_ratio",
        type=float,
        default=None,
        help="density mode: keep the top floor(ratio * ASU residues) waters by "
        "confidence, or all if fewer; ratio > 0 "
        f"(default {DEFAULT_DENSITY_RATIO}).",
    )

    p.add_argument(
        "--include_mates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add symmetry mates. Default: the flow run's setting.",
    )
    p.add_argument(
        "--water_ratio",
        type=float,
        default=8.0,
        help="Candidates = ratio * num_residues.",
    )
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--method", default="euler", choices=["euler", "rk4"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_level", default="INFO")

    args = p.parse_args(argv)
    if args.pdb_list and not args.base_pdb_dir:
        p.error("--pdb_list requires --base_pdb_dir")
    # Each mode has one knob. Reject the other mode's and fill the default.
    if args.selection == "confidence":
        if args.density_ratio is not None:
            p.error("--density_ratio only applies to --selection density")
        if args.confidence_threshold is None:
            args.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        if not 0.0 <= args.confidence_threshold <= 1.0:
            p.error("--confidence_threshold must be in [0, 1]")
    else:  # density (argparse choices rejects anything else)
        if args.confidence_threshold is not None:
            p.error("--confidence_threshold only applies to --selection confidence")
        if args.density_ratio is None:
            args.density_ratio = DEFAULT_DENSITY_RATIO
        if not (math.isfinite(args.density_ratio) and args.density_ratio > 0):
            p.error("--density_ratio must be finite and > 0")
    return args


def main() -> None:
    args = parse_args()
    setup_logging_for_tqdm(level=args.log_level)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    paths = _collect_struc_paths(args)
    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir, paths)
    flow_config = json.loads((ckpt_dir / "flow_config.json").read_text())
    conf_config = json.loads((ckpt_dir / "confidence_config.json").read_text())
    conf_config = conf_config.get("flow_config", conf_config)  # confidence runs nest it
    ckpt_mates = flow_config.get("include_mates", False)
    if args.include_mates is None:
        args.include_mates = ckpt_mates
    elif args.include_mates != ckpt_mates:
        logger.warning(
            f"--include_mates={args.include_mates} but the checkpoint was trained "
            f"with include_mates={ckpt_mates}; graphs will not match training"
        )

    # Graphs are built with ESM or with plain GVP, based on the checkpoint
    flow_encoder = flow_config.get("encoder_type", "gvp")
    conf_encoder = conf_config.get("encoder_type", "gvp")
    if conf_encoder not in ("gvp", flow_encoder):
        raise ValueError(
            f"Confidence checkpoint uses encoder {conf_encoder!r} but graphs are "
            f"built for {flow_encoder!r}; use checkpoints whose encoders match."
        )

    _check_embeddings(paths, flow_encoder, args.processed_dir)

    # Cache subdir named by every setting that shapes its files, so runs with
    # different settings never share one (the cache_candidates scheme). The
    # checkpoint is identified by its directory name, so retraining in place
    # makes the cache stale.
    cache = None
    if args.predict_cache:
        mates = "mates" if args.include_mates else "nomates"
        cache = Path(args.predict_cache) / (
            f"{ckpt_dir.resolve().name}_ckpt_{mates}"
            f"_{args.method}{args.num_steps}_r{args.water_ratio:g}"
        )

    conf_model = build_confidence_model(conf_config, device)
    load_checkpoint(conf_model, ckpt_dir / "confidence.pt", device)

    # The flow checkpoint only samples candidates; skip it when all are cached.
    if cache is not None and all(
        _candidate_path(cache, Path(p).stem).exists() for p in paths
    ):
        flow_matcher = None
        logger.info("All candidate waters cached; flow checkpoint not loaded")
    else:
        flow_model = build_model_from_config(flow_config, device)
        load_checkpoint(flow_model, ckpt_dir / "flow.pt", device)
        flow_matcher = FlowMatcher(
            model=flow_model,
            sampling_strategy=flow_config.get("sampling_strategy", "uniform_ball"),
        )

    logger.info(f"Predicting waters for {len(paths)} structure(s) on {device}")
    for start in tqdm(range(0, len(paths), args.batch_size), desc="predict"):
        predict_structures(
            paths[start : start + args.batch_size],
            flow_matcher,
            conf_model,
            flow_config,
            args,
            device,
            cache=cache,
        )


if __name__ == "__main__":
    main()
