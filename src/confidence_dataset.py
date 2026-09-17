"""
Dataset for confidence-model training.

`ConfidenceDataset` rides on the flow dataset/cache layout: it composes a
`ProteinWaterDataset` (protein graph + embeddings + GT waters from
`{processed_dir}/geometry[_mates]` + `esm/`) with per-structure candidate
files (`<candidate_dir>/<pdb_id>.pt = {"candidate_pos": (Nc, 3)}`). Each item
swaps the GT water nodes for the sampled candidates and computes the target on
the fly, so no bespoke confidence-cache format is needed.

Targets are computed per item rather than cached: benchmarking showed this is
~80x cheaper than the disk load that already happens, and caching would bake
the sharpness hyperparameters into the files.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from src.confidence import nearest_gt_distance, smootherstep_confidence
from src.constants import NODE_FEATURE_DIM, OXYGEN_INDEX
from src.dataset import ProteinWaterDataset


def _oxygen_features(n: int, device: torch.device | None = None) -> Tensor:
    """Oxygen one-hot feature tensor, equal to `element_onehot(['O'] * n)`.

    Candidates are all oxygen by construction, so this skips the per-atom
    string lookups of the general encoder (~6x faster in `__getitem__`, which
    runs per structure per epoch); equivalence is pinned by a unit test.
    """
    idx = torch.full((n,), OXYGEN_INDEX, dtype=torch.long, device=device)
    return F.one_hot(idx, num_classes=NODE_FEATURE_DIM).float()


class ConfidenceDataset(Dataset):
    """Confidence dataset over the flow dataset/cache layout + candidate files.

    Composes a `ProteinWaterDataset` (the flow model's own dataset -- protein
    graph, embeddings, GT waters, PP edges) with a directory of per-structure
    candidate files. For each structure it swaps the GT water nodes for the
    flow-sampled candidates and computes the target on the fly from each
    candidate's distance to its nearest GT water.

    PW edges are NOT cached, even though the candidate positions are fixed:
    at scoring time candidates come straight from the flow sampler with no
    cache in sight, so `ConfidenceGVP` builds its PW edges dynamically
    (`dynamic_edge_policy="knn_if_isolated"`) and training must exercise that
    same path to avoid a train/inference skew. `max_candidates` also redraws
    the candidate subset each epoch, which would invalidate cached edge
    indices anyway -- and rebuilding is cheap next to storing edge indices +
    RBF features per structure. The candidate file therefore stores only
    `candidate_pos`.

    Args:
        flow_dataset: A constructed `ProteinWaterDataset` (or compatible) whose
            items are flow `HeteroData` with `data["water"].pos` = GT waters and
            `data.pdb_id` = the cache key. Exposes `.entries[i]["cache_key"]`.
        candidate_dir: Directory holding one candidate file per structure,
            `<pdb_id>.pt = {"candidate_pos": (Nc, 3)}`.
        r_in, r_out: smootherstep plateau / floor radii (Å) -- target is 1
            within `r_in`, decays C2-smoothly, and is 0 past `r_out`.
        hard_label: Train on `1[d <= accept_radius]` instead of the soft target.
        accept_radius: Radius (Å) defining the binary AUC-PR label.
        max_candidates: Optional per-structure cap on the candidate cloud.
        strict: If True, every structure must have a candidate file. If False,
            structures without one are dropped (logged).
    """

    def __init__(
        self,
        flow_dataset: ProteinWaterDataset,
        candidate_dir: str | Path,
        *,
        r_in: float = 0.5,
        r_out: float = 1.5,
        hard_label: bool = False,
        accept_radius: float = 1.0,
        max_candidates: int | None = None,
        strict: bool = True,
    ):
        self.flow_dataset = flow_dataset
        self.candidate_dir = Path(candidate_dir)
        if not self.candidate_dir.exists():
            raise FileNotFoundError(
                f"Candidate directory not found: {self.candidate_dir}. "
                "Build it with scripts/cache_candidates.py first."
            )
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.hard_label = bool(hard_label)
        self.accept_radius = float(accept_radius)
        self.max_candidates = max_candidates
        if self.r_out <= self.r_in:
            raise ValueError("r_out must exceed r_in.")
        if self.accept_radius < 0:
            raise ValueError("accept_radius must be non-negative.")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be positive.")

        # Map each flow-dataset index to its candidate file (cheap existence
        # checks via the entries list).
        entries = getattr(flow_dataset, "entries", None)
        if entries is None:
            raise TypeError(
                "flow_dataset must expose `.entries` (a ProteinWaterDataset)."
            )
        self._indices: list[int] = []
        self._paths: list[Path] = []  # per kept index: its candidate file
        missing: list[str] = []
        for i, entry in enumerate(entries):
            key = entry["cache_key"]
            path = self.candidate_dir / f"{key}.pt"
            if path.exists():
                self._indices.append(i)
                self._paths.append(path)
            else:
                missing.append(key)

        if missing and strict:
            raise FileNotFoundError(
                f"{len(missing)} structures lack a candidate file under "
                f"{self.candidate_dir} (first: {missing[0]}). Generate them "
                "with scripts/cache_candidates.py or pass strict=False."
            )
        if missing:
            logger.warning(
                f"ConfidenceDataset: skipping {len(missing)} structures with no "
                f"candidate file (first: {missing[0]})."
            )
        if not self._indices:
            raise RuntimeError(
                f"ConfidenceDataset: no candidate files matched under "
                f"{self.candidate_dir} for the {len(entries)} requested entries."
            )
        logger.info(
            f"ConfidenceDataset: {len(self._indices)} structures; smootherstep "
            f"target (r_in={self.r_in}, r_out={self.r_out})."
        )

    def __len__(self) -> int:
        return len(self._indices)

    def _compute_targets(
        self, candidate_pos: Tensor, gt_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (training target, within-`accept_radius` label).

        The nearest-GT distance (shared with `confidence.nearest_gt_distance`, so
        both paths agree) drives the regression target (soft smootherstep, or a
        hard 1[d<=accept_radius] when `hard_label`) and the AUC-PR label
        (1[d<=accept_radius]).
        """
        if candidate_pos.numel() == 0 or gt_pos.numel() == 0:
            empty = candidate_pos.new_empty(0)
            return empty, empty
        d = nearest_gt_distance(candidate_pos, gt_pos)  # (Nc,)
        label = (d <= self.accept_radius).float()
        target = (
            label
            if self.hard_label
            else smootherstep_confidence(d, r_in=self.r_in, r_out=self.r_out)
        )
        return target, label

    def __getitem__(self, idx: int) -> HeteroData:
        flow_idx = self._indices[idx]
        data = self.flow_dataset[flow_idx]

        gt_pos: Tensor = data["water"].pos.float().clone()
        candidate_pos: Tensor = torch.load(
            self._paths[idx], map_location="cpu", weights_only=True
        )["candidate_pos"].float()
        # Optional per-structure cap: random subsample of the candidate cloud.
        # Free for quality (candidates are scored independently -- no
        # water-water edges) but bounds per-step memory. A fresh draw each
        # epoch covers all candidates over training.
        if (
            self.max_candidates is not None
            and candidate_pos.size(0) > self.max_candidates
        ):
            sel = torch.randperm(candidate_pos.size(0))[: self.max_candidates]
            candidate_pos = candidate_pos[sel]
        n_cand = candidate_pos.size(0)

        target, within_accept_radius = self._compute_targets(candidate_pos, gt_pos)

        # Swap GT water nodes for the candidates to be scored.
        data["water"].pos = candidate_pos
        data["water"].x = _oxygen_features(n_cand, device=candidate_pos.device)
        data["water"].num_nodes = n_cand
        data["water"].target_confidence = target
        data["water"].within_accept_radius = within_accept_radius

        return data
