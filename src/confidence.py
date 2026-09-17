"""
Confidence model and post-processing for candidate waters.

Mirrors the SuperWater (Nature Comm. Chem. s42004-025-01789-4) confidence +
clustering stage that sits after the generator in the DiffDock-style two-stage
pipeline.

This module provides:
- smootherstep_target / smootherstep_confidence: per-candidate supervision from
  a soft cutoff of the nearest-GT-water distance
- cluster_waters_vdw: vdW-radius clustering with confidence-weighted centroids
  and NMS over those centroids
- ConfidenceGVP: scores candidate waters through the same GVP backbone as
  FlowWaterGVP, minus time conditioning
- build_confidence_model: ConfidenceGVP from a flow run's recorded config
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import HeteroData

from src.constants import (
    DEFAULT_EDGE_CUTOFF,
    EDGE_PP,
    EDGE_PW,
    NUM_RBF,
    OXYGEN_VDW_RADIUS,
)
from src.encoder_base import BaseProteinEncoder, build_encoder, resolve_encoder_config
from src.flow import ProteinWaterUpdate
from src.gvp import GVP


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def nearest_gt_distance(candidate_pos: Tensor, gt_pos: Tensor) -> Tensor:
    """
    Distance from each candidate to its nearest ground-truth water.

    Single-protein (no batch dim); the caller guards the empty cases.

    Args:
        candidate_pos: (Nc, 3) candidate positions.
        gt_pos: (Ng, 3) ground-truth positions.

    Returns:
        (Nc,) Euclidean distances.
    """
    diffs = candidate_pos.unsqueeze(1) - gt_pos.unsqueeze(0)  # (Nc, Ng, 3)
    return diffs.norm(dim=-1).min(dim=1).values


def smootherstep(x: Tensor) -> Tensor:
    """
    Perlin smootherstep S(x) = 6x^5 - 15x^4 + 10x^3 on x in [0, 1].

    C2-continuous: S, S' and S'' all vanish at both ends, so it joins flat
    plateaus with no kink in value, slope, or curvature.

    Args:
        x: Input, already clamped to [0, 1] by the caller.

    Returns:
        S(x), same shape as `x`.
    """
    # Horner form: degree-5 polynomial, no transcendentals.
    return x * x * x * (10.0 + x * (x * 6.0 - 15.0))


def smootherstep_confidence(
    d: Tensor,
    r_in: float = 0.5,
    r_out: float = 1.5,
) -> Tensor:
    """
    Soft-cutoff confidence from a nearest-GT distance.

        conf = 1                    for d <= r_in
        conf = 1 - smootherstep(u)  for r_in < d < r_out, u = (d-r_in)/(r_out-r_in)
        conf = 0                    for d >= r_out

    A candidate within `r_in` of a GT water is on the site within experimental
    error; past `r_out` it is background. The 0.5-crossing sits at the midpoint
    and the band width sets the steepness, so location and sharpness are
    decoupled.

    Args:
        d: (N,) nearest-GT distances in Angstroms.
        r_in: Plateau radius -- at or under this, confidence is 1.
        r_out: Floor radius -- at or over this, confidence is 0.

    Returns:
        (N,) confidences in [0, 1].

    Raises:
        ValueError: If `r_out` does not exceed `r_in`.
    """
    if r_out <= r_in:
        raise ValueError(f"r_out ({r_out}) must exceed r_in ({r_in}).")
    u = ((d - r_in) / (r_out - r_in)).clamp(0.0, 1.0)
    return 1.0 - smootherstep(u)


def smootherstep_target(
    candidate_pos: Tensor,
    gt_pos: Tensor,
    r_in: float = 0.5,
    r_out: float = 1.5,
) -> Tensor:
    """
    Confidence target for each candidate: higher means closer to a GT water.

    Single-protein (no batch dim).

    Args:
        candidate_pos: (Nc, 3) candidate water positions.
        gt_pos: (Ng, 3) ground-truth water positions, Ng >= 1.
        r_in: Plateau radius (A). See `smootherstep_confidence`.
        r_out: Floor radius (A). See `smootherstep_confidence`.

    Returns:
        (Nc,) target confidences in [0, 1]; empty when there are no candidates.

    Raises:
        ValueError: If `gt_pos` is empty.
    """
    if gt_pos.numel() == 0:
        raise ValueError("smootherstep_target requires at least one GT water.")
    if candidate_pos.numel() == 0:
        return candidate_pos.new_empty(0)
    return smootherstep_confidence(
        nearest_gt_distance(candidate_pos, gt_pos), r_in=r_in, r_out=r_out
    )


# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------


def cluster_waters_vdw(
    positions: Tensor,
    confidences: Tensor,
    radius: float = OXYGEN_VDW_RADIUS,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Two-pass vdW clustering of scored candidates (SuperWater Fig. 4 / Methods).

    Round 1 absorbs: seed a cluster with the highest-confidence unassigned
    water, absorb every unassigned water within `radius`, and emit a
    confidence-weighted centroid carrying the cluster's max confidence.
    Round 2 runs NMS over those centroids, dropping the lower-confidence member
    of any pair still within `radius`.

    Args:
        positions: (N, 3) candidate water positions.
        confidences: (N,) scalar confidences, higher is better.
        radius: Absorption and NMS radius in Angstroms. Defaults to the vdW
            radius of oxygen.
        threshold: Drop candidates scoring below this before clustering. None
            keeps all.

    Returns:
        ((M, 3) positions, (M,) confidences), in descending confidence order.

    Raises:
        ValueError: If the input shapes disagree or are not (N, 3) / (N,).
    """
    if positions.dim() != 2 or positions.size(-1) != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(positions.shape)}")
    if confidences.dim() != 1 or confidences.size(0) != positions.size(0):
        raise ValueError(
            f"confidences must be (N,), got {tuple(confidences.shape)} "
            f"for positions {tuple(positions.shape)}"
        )

    if threshold is not None:
        keep = confidences >= threshold
        positions = positions[keep]
        confidences = confidences[keep]

    if positions.numel() == 0:
        return positions, confidences

    # --- Round 1: absorb into confidence-weighted centroids ---
    # stable so ties keep input order, which makes the whole routine deterministic
    # and lets both later stages rely on the descending order instead of re-sorting.
    order = torch.argsort(confidences, descending=True, stable=True)
    pos_sorted = positions[order]
    conf_sorted = confidences[order]

    n = pos_sorted.size(0)
    assigned = torch.zeros(n, dtype=torch.bool, device=pos_sorted.device)
    r2 = float(radius) ** 2

    centroid_positions: list[Tensor] = []
    centroid_confidences: list[Tensor] = []

    for i in range(n):
        if assigned[i]:
            continue
        seed_pos = pos_sorted[i]
        d2 = ((pos_sorted - seed_pos) ** 2).sum(dim=-1)
        # Always includes i: d2[i] == 0 and assigned[i] is False.
        within = (d2 <= r2) & (~assigned)
        cluster_pos = pos_sorted[within]
        cluster_conf = conf_sorted[within]

        # Confidences are sigmoid outputs (>= 0), so the weights sum to 0 only
        # when every member scored 0; fall back to an unweighted mean there to
        # yield a position rather than NaN.
        w_sum = cluster_conf.sum()
        if w_sum > 0:
            centroid = (cluster_pos * cluster_conf.unsqueeze(-1)).sum(dim=0) / w_sum
        else:
            centroid = cluster_pos.mean(dim=0)

        centroid_positions.append(centroid)
        centroid_confidences.append(cluster_conf.max())
        assigned = assigned | within

    if not centroid_positions:
        return pos_sorted.new_empty(0, 3), conf_sorted.new_empty(0)

    cent_pos = torch.stack(centroid_positions, dim=0)
    cent_conf = torch.stack(centroid_confidences, dim=0)

    # --- Round 2: NMS between centroids ---
    keep_mask = torch.ones(cent_pos.size(0), dtype=torch.bool, device=cent_pos.device)
    # Centroids are already in descending confidence order: seeds are visited
    # high-to-low and each centroid inherits its seed's (max) confidence.
    for idx in range(cent_pos.size(0)):
        if not keep_mask[idx]:
            continue
        d2 = ((cent_pos - cent_pos[idx]) ** 2).sum(dim=-1)
        collide = (d2 <= r2) & keep_mask
        collide[idx] = False  # never drop self
        keep_mask[collide] = False

    # Masking preserves the descending order downstream thresholding relies on.
    return cent_pos[keep_mask], cent_conf[keep_mask]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConfidenceGVP(nn.Module):
    """
    DiffDock-style confidence model scoring candidate waters.

    Mirrors the flow model's structure but drops time (inputs are clean
    samples), emits one scalar per candidate water, and runs on PW and PP edges
    only -- candidates come from inference and are not refined here, so WW and
    WP carry nothing.

    PW edges come from the dataset when it supplies them, otherwise from a
    radius query plus a nearest-neighbour pass for candidates it left with no
    edges -- candidates can land in sparse regions the generator never visits.

    Keeping the backbone (encoder, encoder_to_flow, updater) identical to the
    generator is what lets those weights warm-start from a flow checkpoint. The
    scalar encoders differ -- the generator concatenates a time channel these do
    not -- and the score head is new, so those two are trained from scratch.
    """

    def __init__(
        self,
        encoder: BaseProteinEncoder,
        hidden_dims: tuple[int, int] = (256, 32),
        edge_scalar_dim: int = NUM_RBF,
        layers: int = 4,
        drop_rate: float = 0.1,
        n_message_gvps: int = 2,
        n_update_gvps: int = 2,
        vector_gate: bool = True,
        cutoff: float = DEFAULT_EDGE_CUTOFF,
        max_neighbors: int = 256,
        dynamic_edge_policy: str = "knn_if_isolated",
        knn_fallback_k: int = 8,
        water_input_dim: int = 16,
    ):
        """
        Initialize the confidence scorer.

        Args mirror `FlowWaterGVP`; see that class for the shared ones. The
        differences: no time dimension is added to the scalar encoders, and the
        head projects hidden -> scalar logit (vo=0) instead of a vector field.

        Args:
            encoder: Protein encoder implementing BaseProteinEncoder.
            hidden_dims: (scalar_dim, vector_dim) hidden dimensions.
            edge_scalar_dim: Dimension of edge scalar features.
            layers: Number of heterogeneous GVP message passing layers.
            drop_rate: Dropout rate.
            n_message_gvps: GVP modules in each edge type's message function.
            n_update_gvps: GVP modules in the node update function.
            vector_gate: Whether to use vector gating in GVP layers.
            cutoff: Distance cutoff in Angstroms for radius PW edges.
            max_neighbors: Per-source cap on radius results.
            dynamic_edge_policy: How PW edges are built when not cached.
            knn_fallback_k: Neighbours attached to candidates the radius query
                left with no edges; 0 disables that pass.
            water_input_dim: Input dimension for water node features.
        """
        super().__init__()
        # "auto" resolves against a sampling strategy the confidence model does not
        # have, so it would silently fall through to "radius". Reject it instead.
        if dynamic_edge_policy == "auto":
            raise ValueError(
                "ConfidenceGVP has no sampling strategy to resolve 'auto'; pass a "
                "concrete dynamic_edge_policy such as 'radius' or 'knn_if_isolated'."
            )
        self.encoder = encoder
        self.hidden_dims = hidden_dims
        self.edge_scalar_dim = edge_scalar_dim
        self.layers = layers
        self.drop_rate = drop_rate
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.vector_gate = vector_gate

        s_h, v_h = hidden_dims

        self.encoder_to_flow = GVP(
            in_dims=encoder.output_dims,
            out_dims=hidden_dims,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
        )

        # No time concat, unlike FlowWaterGVP: these are clean samples.
        self.protein_scalar_encoder = nn.Sequential(
            nn.Linear(s_h, s_h),
            nn.GELU(),
            nn.LayerNorm(s_h),
        )
        self.water_scalar_encoder = nn.Sequential(
            nn.Linear(water_input_dim, s_h),
            nn.GELU(),
            nn.LayerNorm(s_h),
        )

        self.updater = ProteinWaterUpdate(
            hidden_dims=hidden_dims,
            rbf_dim=edge_scalar_dim,
            layers=layers,
            drop_rate=drop_rate,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            vector_gate=vector_gate,
            aggr_edges="sum",
            use_dst_feats=True,
            etypes=[EDGE_PW, EDGE_PP],
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            dynamic_edge_policy=dynamic_edge_policy,
            knn_fallback_k=knn_fallback_k,
        )

        # vo=0 makes GVP return a tensor rather than a (scalar, vector) tuple.
        self.score_head = GVP(
            in_dims=hidden_dims,
            out_dims=(1, 0),
            activations=(None, None),
            vector_gate=False,
        )

    def forward(
        self,
        data: HeteroData,
        return_logits: bool = False,
    ) -> Tensor:
        """
        Score every water node in `data`.

        Args:
            data: HeteroData with 'protein' and 'water' node types, where
                `water.pos` holds the candidate positions to score.
            return_logits: Return raw pre-sigmoid logits, for callers using
                BCEWithLogits instead of MSE.

        Returns:
            (N_w,) scores, in [0, 1] unless `return_logits` is set.
        """
        device = data["protein"].pos.device

        if "water" not in data.node_types or data["water"].num_nodes == 0:
            # No candidates. Return an empty (0,) result that keeps a grad path to
            # the score head: under DDP the backward must reach the reducer, or
            # this rank skips the all-reduce and hangs ranks that had candidates.
            # Route through the water head only -- 0 nodes cannot build edges.
            in_features = self.water_scalar_encoder[0].in_features
            water_x = (
                data["water"].x
                if "water" in data.node_types
                else torch.zeros(0, in_features, device=device)
            )
            s_w = self.water_scalar_encoder(water_x)  # (0, s_h)
            v_w = torch.zeros(0, self.hidden_dims[1], 3, device=device)
            logits = self.score_head((s_w, v_w)).squeeze(-1)  # (0,)
            return logits if return_logits else torch.sigmoid(logits)

        s_all, v_all, pp_edge_attr = self.encoder(data)
        encoder_input = (s_all, v_all) if self.encoder.output_dims[1] > 0 else s_all
        s_p_latent, v_p_latent = self.encoder_to_flow(encoder_input)

        s_p = self.protein_scalar_encoder(s_p_latent)
        s_w = self.water_scalar_encoder(data["water"].x)

        v_w = torch.zeros(
            data["water"].num_nodes,
            self.hidden_dims[1],
            3,
            device=device,
        )

        x_dict = {
            "protein": (s_p, v_p_latent),
            "water": (s_w, v_w),
        }
        x_dict = self.updater(
            x_dict,
            data,
            pp_edge_attr=pp_edge_attr,
        )

        logits = self.score_head(x_dict["water"]).squeeze(-1)  # (N_w,)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def build_confidence_model(config: dict, device: torch.device) -> ConfidenceGVP:
    """
    Instantiate `ConfidenceGVP` from the flow run's hyperparameters.

    Mirroring the flow's encoder and hidden dims is what lets the head
    warm-start from a checkpoint that shares the same backbone shape.

    Args:
        config: The flow run's recorded config.
        device: Device to build on.

    Returns:
        The model, on `device`.
    """
    encoder = build_encoder(resolve_encoder_config(config), device)
    return ConfidenceGVP(
        encoder=encoder,
        hidden_dims=(config.get("hidden_s") or 256, config.get("hidden_v") or 64),
        edge_scalar_dim=config.get("edge_scalar_dim") or NUM_RBF,
        layers=config.get("flow_layers") or 3,
        drop_rate=config.get("drop_rate", 0.1),
        n_message_gvps=config.get("n_message_gvps", 2),
        n_update_gvps=config.get("n_update_gvps", 2),
        cutoff=config.get("cutoff", DEFAULT_EDGE_CUTOFF),
        max_neighbors=config.get("max_neighbors", 256),
        knn_fallback_k=config.get("knn_fallback_k", 8),
        # Fixed, not from flow config: candidates are scored with no cached PW
        # edges and can land where the radius query leaves them isolated, so the
        # knn fallback is always needed (see ConfidenceGVP docstring).
        dynamic_edge_policy="knn_if_isolated",
    ).to(device)
