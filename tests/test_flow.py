"""Unit tests for flow.py

All test cases created with assistance from Claude Code and refined.
"""

import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import knn, radius

from src.constants import (
    ALL_EDGE_TYPES,
    EDGE_PP,
    EDGE_PW,
    EDGE_WP,
    EDGE_WW,
    get_active_edge_types,
)
from src.flow import (
    _batch_from_counts,
    build_dynamic_edges,
    FlowMatcher,
    FlowWaterGVP,
    ProteinWaterUpdate,
    resolve_edge_policy,
    sample_waters_scaled_gaussian,
    sample_waters_uniform_ball,
)
from src.gvp_encoder import GVPEncoder, make_gvp_encoder_data, ProteinGVPEncoder


@pytest.fixture
def warning_log():
    """Collect loguru warning messages; loguru does not reach pytest's caplog."""
    from loguru import logger

    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def simple_hetero_data(device):
    """Minimal HeteroData with protein and water nodes."""
    data = HeteroData()

    # Protein: 10 atoms
    data["protein"].pos = torch.randn(10, 3, device=device)
    data["protein"].x = torch.randn(10, 16, device=device)
    data["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)

    # Water: 5 molecules
    data["water"].pos = torch.randn(5, 3, device=device)
    data["water"].x = torch.randn(5, 16, device=device)
    data["water"].batch = torch.zeros(5, dtype=torch.long, device=device)

    # Protein-protein edges
    data["protein", "pp", "protein"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long, device=device
    )

    return data


@pytest.fixture
def batched_hetero_data(device):
    """HeteroData with 2 graphs batched."""
    data = HeteroData()

    # Protein: 20 atoms (10 per graph)
    data["protein"].pos = torch.randn(20, 3, device=device)
    # One-hot encoded element types (16 classes: 15 elements + 1 "other")
    protein_elem_indices = torch.randint(0, 16, (20,), device=device)
    data["protein"].x = F.one_hot(protein_elem_indices, num_classes=16).float()
    data["protein"].batch = torch.cat(
        [torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)]
    ).to(device)

    # Water: 8 molecules (4 per graph)
    data["water"].pos = torch.randn(8, 3, device=device)
    # One-hot encoded element types (water is oxygen, index 2 in ELEMENT_VOCAB)
    water_elem_indices = torch.full((8,), 2, dtype=torch.long, device=device)
    data["water"].x = F.one_hot(water_elem_indices, num_classes=16).float()
    data["water"].batch = torch.cat(
        [torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)]
    ).to(device)

    data["protein", "pp", "protein"].edge_index = torch.tensor(
        [[0, 1, 10, 11], [1, 2, 11, 12]], dtype=torch.long, device=device
    )

    return data


@pytest.fixture
def mock_encoder(device):
    """Mock BaseProteinEncoder."""
    encoder = Mock()
    encoder.output_dims = (256, 32)  # Required by FlowWaterGVP
    encoder.encoder_type = "mock"
    encoder.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
    encoder.eval = Mock()

    def mock_forward(data):
        n = data["protein"].pos.size(0)
        s = torch.randn(n, 256, device=device)
        v = torch.randn(n, 32, 3, device=device)
        # Return 3 values: (s, V, pp_edge_attr)
        # Mock encoder returns None for edge features (like SLAE/ESM)
        return s, v, None

    encoder.side_effect = mock_forward
    encoder.__call__ = mock_forward
    return encoder


# base_encoder and gvp_encoder fixtures are defined in conftest.py


@pytest.mark.unit
class TestBuildDynamicEdgesKnn:
    def test_basic_knn(self, device):
        src = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device
        )
        dst = torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], device=device)

        edges = build_dynamic_edges(src, dst, k=2, policy="knn", r=8.0)

        assert edges.shape[0] == 2
        assert edges.shape[1] >= 4  # At least 2 dst points × 2 neighbors each
        assert edges.dtype == torch.long

        # dst[0] at 0.5 should connect to src[0] (dist=0.5) and src[1] (dist=0.5)
        # dst[1] at 1.5 should connect to src[1] (dist=0.5) and src[2] (dist=0.5)
        edge_set = set(zip(edges[0].tolist(), edges[1].tolist()))
        assert (0, 0) in edge_set, f"Missing edge src[0]->dst[0], got {edge_set}"
        assert (1, 0) in edge_set, f"Missing edge src[1]->dst[0], got {edge_set}"
        assert (1, 1) in edge_set, f"Missing edge src[1]->dst[1], got {edge_set}"
        assert (2, 1) in edge_set, f"Missing edge src[2]->dst[1], got {edge_set}"

    def test_empty_src(self, device):
        src = torch.empty(0, 3, device=device)
        dst = torch.randn(5, 3, device=device)

        edges = build_dynamic_edges(src, dst, k=3, policy="knn", r=8.0)

        assert edges.shape == (2, 0)

    def test_empty_dst(self, device):
        src = torch.randn(5, 3, device=device)
        dst = torch.empty(0, 3, device=device)

        edges = build_dynamic_edges(src, dst, k=3, policy="knn", r=8.0)

        assert edges.shape == (2, 0)

    def test_self_edges_removed(self, device):
        pos = torch.randn(10, 3, device=device)

        edges = build_dynamic_edges(pos, pos, k=5, policy="knn", r=8.0)

        # No self-loops
        assert (edges[0] != edges[1]).all()

    def test_with_batch(self, device):
        src = torch.randn(10, 3, device=device)
        dst = torch.randn(8, 3, device=device)
        batch_src = torch.cat([torch.zeros(5), torch.ones(5)]).long().to(device)
        batch_dst = torch.cat([torch.zeros(4), torch.ones(4)]).long().to(device)

        edges = build_dynamic_edges(
            src, dst, k=3, batch_src=batch_src, batch_dst=batch_dst, policy="knn", r=8.0
        )

        assert edges.shape[0] == 2
        assert edges.shape[1] > 0


@pytest.mark.unit
class TestBuildDynamicEdgesDirection:
    """Row-convention tests for both policies, on asymmetric geometry.

    The KNN cases spread srcs out and put both dsts near src[0], so "k nearest
    srcs per dst" (correct) and "k nearest dsts per src" (the x/y swap) give
    different edge sets -- no distance ties to mask a mix-up. The radius cases
    keep the two index ranges different sizes for the same reason.

    Both policies feed row 0 = src, row 1 = dst, but reach it differently: KNN
    swaps the rows PyG returns, radius inverts the arguments instead. Each needs
    its own pin.
    """

    SRC = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    DST = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    def test_exact_edge_set_is_per_destination(self, device):
        """Both dsts are nearest src[0], so src[1]/src[2] must not appear. An x/y
        swap gives {(0,0), (1,1), (2,1)} instead."""
        src = torch.tensor(self.SRC, device=device)
        dst = torch.tensor(self.DST, device=device)

        edges = build_dynamic_edges(src, dst, k=1, policy="knn", r=8.0)

        edge_set = set(zip(edges[0].tolist(), edges[1].tolist()))
        assert edge_set == {(0, 0), (0, 1)}, f"got {sorted(edge_set)}"

    def test_every_destination_is_covered(self, device):
        """Coverage is per-destination: every dst gets k in-edges; an unneeded src
        may be absent."""
        src = torch.tensor(self.SRC, device=device)
        dst = torch.tensor(self.DST, device=device)
        k = 2

        edges = build_dynamic_edges(src, dst, k=k, policy="knn", r=8.0)

        dst_row = edges[1]
        for d in range(len(self.DST)):
            assert (dst_row == d).sum().item() == k, f"dst {d} lacks {k} in-edges"
        # src[2] (x=20) is not among the 2 nearest srcs of either dst
        assert 2 not in set(edges[0].tolist())

    def test_output_rows_are_src_then_dst(self, device):
        """Row 0 = src, row 1 = dst, pinned by index range.

        Own geometry: both dsts are nearest src[2], so row 0 must reach index 2
        while row 1 only reaches 1. Swapping the rows puts 2 in row 1, which is
        out of range for two dsts. (The class fixture can't pin this -- its row 0
        is all zeros, so both ranges hold either way round.)
        """
        src = torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], device=device
        )
        dst = torch.tensor([[19.0, 0.0, 0.0], [21.0, 0.0, 0.0]], device=device)

        edges = build_dynamic_edges(src, dst, k=1, policy="knn", r=8.0)

        assert edges[0].max().item() == 2  # src[2]; >= len(dst), so a swap breaks
        assert edges[1].max().item() < len(dst)

    def test_torch_geometric_knn_row_convention_unchanged(self, device):
        """Pin knn's undocumented rows: row 0 = y (query), row 1 = x (neighbor).
        build_dynamic_edges swaps them, so a flip here would reverse every edge."""
        x = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], device=device)  # N=3
        y = torch.tensor([[0.1, 0.0], [19.9, 0.0]], device=device)  # M=2

        out = knn(x, y, k=1)

        assert out[0].tolist() == [0, 1]  # queries (y), in order
        assert out[1].tolist() == [0, 2]  # nearest x: y[0]->x[0], y[1]->x[2]

    def test_torch_geometric_radius_row_convention_unchanged(self, device):
        """Pin radius's rows: row 0 = y (query), row 1 = x (neighbor) -- the same
        convention as knn. build_dynamic_edges relies on this by passing src as
        `y`, so it lands in row 0 with no swap. Index ranges are kept distinct
        (3 x's vs 2 y's) so a flip cannot pass silently."""
        x = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [10.0, 0.0], [10.1, 0.0]],
            device=device,
        )  # N=5
        y = torch.tensor([[0.05, 0.0], [10.05, 0.0]], device=device)  # M=2

        out = radius(x, y, r=0.5)

        assert out[0].max().item() <= 1  # queries (y), max index 1
        assert out[1].max().item() == 4  # neighbours (x), reaches index 4

    def test_radius_rows_are_src_then_dst(self, device):
        """Row 0 = src, row 1 = dst under the radius policy too, pinned by index
        range: 2 srcs and 5 dsts, so a swap puts an out-of-range index in row 0."""
        src = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
        dst = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [10.2, 0.0, 0.0],
            ],
            device=device,
        )

        edges = build_dynamic_edges(src, dst, policy="radius", k=1, r=1.0)

        assert edges[0].max().item() < len(src)
        assert edges[1].max().item() == 4  # dst[4]; >= len(src), so a swap breaks

    def test_radius_self_graph_has_no_self_loops(self, device):
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [99.0, 0.0, 0.0]], device=device
        )

        edges = build_dynamic_edges(pos, pos, policy="radius", k=1, r=1.0)

        assert (edges[0] != edges[1]).all()
        assert set(zip(edges[0].tolist(), edges[1].tolist())) == {(0, 1), (1, 0)}


@pytest.mark.unit
class TestResolveEdgePolicy:
    """Replaying a recorded config must reproduce the original run. Every run on
    record carries "auto", so rejecting it would strand all of them."""

    @pytest.mark.parametrize(
        "policy,strategy,expected",
        [
            # "auto" reads off the prior: the uniform ball never strands a water,
            # the Gaussian can, so only the latter earns the rescue.
            ("auto", "uniform_ball", "radius"),
            ("auto", "scaled_gaussian", "knn_if_isolated"),
            # An explicit policy ignores the prior.
            ("radius", "scaled_gaussian", "radius"),
            ("knn", "uniform_ball", "knn"),
            ("knn_if_isolated", "uniform_ball", "knn_if_isolated"),
        ],
    )
    def test_resolves(self, policy, strategy, expected):
        assert resolve_edge_policy(policy, strategy) == expected

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="dynamic_edge_policy"):
            resolve_edge_policy("knn_if_lonely")


@pytest.mark.unit
class TestMakeEncoderData:
    def test_basic_output(self, simple_hetero_data):
        enc_data = make_gvp_encoder_data(simple_hetero_data)

        assert isinstance(enc_data, Data)
        assert hasattr(enc_data, "x")
        assert hasattr(enc_data, "pos")
        assert hasattr(enc_data, "edge_index")

    def test_shapes(self, simple_hetero_data):
        enc_data = make_gvp_encoder_data(simple_hetero_data)

        n_nodes = simple_hetero_data["protein"].pos.size(0)
        n_edges = simple_hetero_data["protein", "pp", "protein"].edge_index.size(1)

        assert enc_data.x.shape[0] == n_nodes
        assert enc_data.pos.shape == (n_nodes, 3)
        assert enc_data.edge_index.shape == (2, n_edges)

    def test_batch_preserved(self, batched_hetero_data):
        enc_data = make_gvp_encoder_data(batched_hetero_data)

        assert hasattr(enc_data, "batch")
        assert enc_data.batch.shape[0] == batched_hetero_data["protein"].pos.size(0)

    def test_no_edges(self, device):
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        # No edges defined

        enc_data = make_gvp_encoder_data(data)

        assert enc_data.edge_index.shape == (2, 0)


@pytest.mark.unit
class TestProteinWaterUpdate:
    def test_init(self):
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            rbf_dim=16,
            layers=2,
        )

        assert len(updater.blocks) == 2
        assert ("protein", "pw", "water") in updater.etypes
        assert ("water", "ww", "water") in updater.etypes

    def test_init_defaults_to_all_edge_types(self):
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            rbf_dim=16,
            layers=2,
        )

        assert set(updater.etypes) == set(ALL_EDGE_TYPES)

    def test_init_rejects_unknown_etype(self):
        with pytest.raises(ValueError, match="subset"):
            ProteinWaterUpdate(
                hidden_dims=(128, 16),
                layers=1,
                etypes=[EDGE_PW, ("water", "xx", "water")],
            )

    def test_knn_if_isolated_builds_a_radius_graph_and_rescues(self):
        """It differs from plain radius only by the rescue; build_edges is handed
        "radius" either way."""
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16), layers=1, dynamic_edge_policy="knn_if_isolated"
        )

        assert updater.dynamic_edge_policy == "radius"
        assert updater.rescue_isolated

    def test_plain_radius_does_not_rescue(self):
        """The rescue belongs to knn_if_isolated, so a positive knn_fallback_k
        must not switch it on by itself."""
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            layers=1,
            dynamic_edge_policy="radius",
            knn_fallback_k=8,
        )

        assert not updater.rescue_isolated

    def test_init_rejects_negative_fallback_k(self):
        with pytest.raises(ValueError, match="knn_fallback_k"):
            ProteinWaterUpdate(hidden_dims=(128, 16), layers=1, knn_fallback_k=-1)

    def test_ablated_etypes_are_absent_from_edge_dict(self, simple_hetero_data):
        updater = ProteinWaterUpdate(
            hidden_dims=(128, 16),
            layers=1,
            etypes=get_active_edge_types(disable_ww=True, disable_wp=True),
        )

        edge_dict = updater.build_edges(simple_hetero_data)

        assert set(edge_dict) == {EDGE_PW, EDGE_PP}

    def test_radius_strands_far_water_and_fallback_rescues_it(self, device):
        """A water parked far outside `cutoff` gets no radius PW edges. With the
        rescue enabled it is reconnected to its nearest protein atoms anyway."""
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["water"].pos = torch.cat(
            [torch.randn(4, 3, device=device), torch.full((1, 3), 500.0, device=device)]
        )

        def stranded(knn_fallback_k):
            updater = ProteinWaterUpdate(
                hidden_dims=(32, 4),
                layers=1,
                cutoff=8.0,
                dynamic_edge_policy="knn_if_isolated",
                knn_fallback_k=knn_fallback_k,
            )
            edge_index = updater.build_edges(data)[EDGE_PW]
            connected = torch.zeros(5, dtype=torch.bool, device=device)
            connected[edge_index[1].unique()] = True
            return int((~connected).sum())

        assert stranded(knn_fallback_k=0) == 1
        assert stranded(knn_fallback_k=3) == 0

    def test_radius_strands_far_water_on_wp_and_fallback_rescues_it(self, device):
        """The rescue is symmetric across edge direction. On water->protein the
        water is the source, so a water parked outside `cutoff` loses its WP edges
        too; the fallback reconnects it to its nearest protein atoms."""
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["water"].pos = torch.cat(
            [torch.randn(4, 3, device=device), torch.full((1, 3), 500.0, device=device)]
        )

        def stranded(knn_fallback_k):
            updater = ProteinWaterUpdate(
                hidden_dims=(32, 4),
                layers=1,
                cutoff=8.0,
                dynamic_edge_policy="knn_if_isolated",
                knn_fallback_k=knn_fallback_k,
            )
            # water is the source of WP, so it lives on row 0.
            edge_index = updater.build_edges(data)[EDGE_WP]
            connected = torch.zeros(5, dtype=torch.bool, device=device)
            connected[edge_index[0].unique()] = True
            return int((~connected).sum())

        assert stranded(knn_fallback_k=0) == 1
        assert stranded(knn_fallback_k=3) == 0

    def test_knn_policy_never_strands_a_water(self, device):
        """KNN budgets per destination, so distance cannot isolate a node and the
        rescue is unnecessary."""
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["water"].pos = torch.cat(
            [torch.randn(4, 3, device=device), torch.full((1, 3), 500.0, device=device)]
        )
        updater = ProteinWaterUpdate(
            hidden_dims=(32, 4), layers=1, dynamic_edge_policy="knn", k_pw=3
        )

        edge_index = updater.build_edges(data)[EDGE_PW]

        assert set(edge_index[1].tolist()) == {0, 1, 2, 3, 4}

    def test_cached_pw_edges_are_used_verbatim(self, simple_hetero_data, device):
        """A dataset that precomputes PW edges (the confidence pipeline) must not
        have them rebuilt underneath it."""
        cached = torch.tensor([[0, 1], [2, 3]], device=device)
        simple_hetero_data[EDGE_PW].edge_index = cached
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data)

        assert torch.equal(edge_dict[EDGE_PW], cached)

    def test_build_edges(self, simple_hetero_data):
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data)

        assert ("protein", "pw", "water") in edge_dict
        assert ("water", "ww", "water") in edge_dict
        assert edge_dict[("protein", "pw", "water")].shape[0] == 2

    def test_build_edges_empty_water(self, device):
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        data["water"].pos = torch.empty(0, 3, device=device)
        data["water"].x = torch.empty(0, 16, device=device)

        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)
        edge_dict = updater.build_edges(data)

        assert edge_dict[("protein", "pw", "water")].shape == (2, 0)
        assert edge_dict[("water", "ww", "water")].shape == (2, 0)

    def test_forward_shapes(self, simple_hetero_data, device):
        s_h, v_h = 128, 16
        updater = ProteinWaterUpdate(hidden_dims=(s_h, v_h), layers=1).to(device)

        n_p = simple_hetero_data["protein"].pos.size(0)
        n_w = simple_hetero_data["water"].pos.size(0)

        x_dict = {
            "protein": (
                torch.randn(n_p, s_h, device=device),
                torch.randn(n_p, v_h, 3, device=device),
            ),
            "water": (
                torch.randn(n_w, s_h, device=device),
                torch.randn(n_w, v_h, 3, device=device),
            ),
        }

        out = updater(x_dict, simple_hetero_data)

        assert out["water"][0].shape == (n_w, s_h)
        assert out["water"][1].shape == (n_w, v_h, 3)


# ============== Tests for FlowWaterGVP ==============


@pytest.mark.unit
class TestFlowWaterGVP:
    def test_init(self, mock_encoder, device):
        model = FlowWaterGVP(
            encoder=mock_encoder,
            hidden_dims=(128, 16),
            layers=2,
        ).to(device)

        assert model.hidden_dims == (128, 16)
        assert model.layers == 2

    def test_forward_output_shape(self, simple_hetero_data, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        t = torch.tensor([0.5], device=device)
        v_pred = model(simple_hetero_data, t)

        n_water = simple_hetero_data["water"].num_nodes
        assert v_pred.shape == (n_water, 3)

    def test_forward_no_water(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        data["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)
        data["protein", "pp", "protein"].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )
        # No water nodes

        t = torch.tensor([0.5], device=device)
        v_pred = model(data, t)

        assert v_pred.shape == (0, 3)


# ============== Tests for FlowMatcher ==============


class _ConstantVelocity(torch.nn.Module):
    """Velocity field v(x, t) = c, independent of state and time.

    Integrating dx/dt = c from t=0 to t=1 gives x(1) = x(0) + c exactly, so it
    lets the integrators be checked against a closed-form trajectory.
    """

    def __init__(self, velocity):
        super().__init__()
        self.velocity = torch.as_tensor(velocity, dtype=torch.float32)

    def forward(self, g, t):
        pos = g["water"].pos
        return self.velocity.to(pos).view(1, 3).expand(pos.shape[0], 3)


class _LinearDecayVelocity(torch.nn.Module):
    """Velocity field v(x, t) = -k x. Solves to x(1) = x(0) * exp(-k).

    A genuine ODE where the step count matters, used to check that RK4 with few
    steps converges to Euler with many.
    """

    def __init__(self, k):
        super().__init__()
        self.k = float(k)

    def forward(self, g, t):
        return -self.k * g["water"].pos


class _TimeRampVelocity(torch.nn.Module):
    """Velocity field v(x, t) = a * t: independent of state, linear in time.

    The fields above ignore t, so nothing about them notices a stage evaluated at
    the wrong time. Here the quadrature is known in closed form for both methods
    (RK4's Simpson weights are exact on a linear integrand, Euler's left
    rectangles fall short by a known amount), which pins the stage times down.
    """

    def __init__(self, a):
        super().__init__()
        self.a = torch.as_tensor(a, dtype=torch.float32)

    def forward(self, g, t):
        pos = g["water"].pos
        t_w = t.to(pos)[g["water"].batch].view(-1, 1)
        return self.a.to(pos).view(1, 3) * t_w


class _PerGraphVelocity(torch.nn.Module):
    """Velocity field v = (graph_index + 1) * c, constant in state and time.

    Each graph in a batch moves by a different amount, so results split back to
    the wrong graph -- or waters mixed across graphs -- cannot pass.
    """

    def __init__(self, c):
        super().__init__()
        self.c = torch.as_tensor(c, dtype=torch.float32)

    def forward(self, g, t):
        pos = g["water"].pos
        scale = (g["water"].batch.to(pos) + 1.0).view(-1, 1)
        return self.c.to(pos).view(1, 3) * scale


@pytest.mark.unit
class TestFlowMatcher:
    @pytest.fixture
    def flow_matcher(self, device, gvp_encoder):
        model = FlowWaterGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        return FlowMatcher(model)

    def test_compute_sigma(self, simple_hetero_data):
        sigma = FlowMatcher.compute_sigma(simple_hetero_data)

        assert isinstance(sigma, float)
        assert sigma > 0

    def test_compute_sigma_per_graph_zero_protein_raises(self, device):
        """A graph with no protein atoms has no meaningful sigma."""
        g0 = HeteroData()
        g0["protein"].pos = torch.randn(4, 3, device=device)
        g1 = HeteroData()
        g1["protein"].pos = torch.empty(0, 3, device=device)

        with pytest.raises(ValueError, match="zero protein atoms"):
            FlowMatcher.compute_sigma_per_graph(Batch.from_data_list([g0, g1]), device)

    def test_training_step(self, flow_matcher, simple_hetero_data, device):
        optimizer = torch.optim.Adam(flow_matcher.model.parameters(), lr=1e-4)

        optimizer.zero_grad()
        result = flow_matcher.training_step(simple_hetero_data)
        optimizer.step()

        assert "loss" in result
        assert "rmsd" in result
        assert "sigma" in result
        assert result["loss"] >= 0

    def test_validation_step(self, flow_matcher, simple_hetero_data):
        result = flow_matcher.validation_step(simple_hetero_data)

        assert "loss" in result
        assert "rmsd" in result
        assert result["loss"] >= 0

    def test_edge_config_propagates_to_updater(self, device, gvp_encoder):
        """Edge construction is configured once on the model and used for both
        training and integration, so it must reach the updater that builds the
        edges. An earlier design resolved it per batch and let the two diverge."""
        model = FlowWaterGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
            dynamic_edge_policy="knn",
            cutoff=6.0,
            knn_fallback_k=0,
            disable_ww=True,
        ).to(device)

        assert model.updater.dynamic_edge_policy == "knn"
        assert model.updater.cutoff == 6.0
        assert model.updater.knn_fallback_k == 0
        assert EDGE_WW not in model.updater.etypes

    @pytest.mark.slow
    def test_euler_integrate(self, flow_matcher, simple_hetero_data, device):
        results = flow_matcher.euler_integrate(
            simple_hetero_data, num_steps=5, device=str(device)
        )
        # euler_integrate returns List[Dict], one per input graph
        result = results[0]
        assert "water_pred" in result
        assert "protein_pos" in result
        assert "water_true" in result
        assert "pdb_id" in result
        assert "trajectory" not in result

        water_pred = result["water_pred"]
        n_water = simple_hetero_data["water"].num_nodes
        assert water_pred.shape == (n_water, 3)
        assert isinstance(water_pred, np.ndarray)

    @pytest.mark.slow
    def test_euler_trajectory(self, flow_matcher, simple_hetero_data, device):
        result = flow_matcher.euler_integrate(
            simple_hetero_data, num_steps=5, device=str(device), return_trajectory=True
        )[0]
        assert len(result["trajectory"]) == 5
        n_water = simple_hetero_data["water"].num_nodes
        assert all(frame.shape == (n_water, 3) for frame in result["trajectory"])
        np.testing.assert_array_equal(result["trajectory"][-1], result["water_pred"])

    @pytest.mark.slow
    def test_rk4_integrate(self, flow_matcher, simple_hetero_data, device):
        results = flow_matcher.rk4_integrate(
            simple_hetero_data,
            num_steps=5,
            device=str(device),
            return_trajectory=True,
        )
        # rk4_integrate returns List[Dict], one per input graph
        result = results[0]

        assert "water_pred" in result
        assert "water_true" in result
        assert "protein_pos" in result
        assert "trajectory" in result
        assert len(result["trajectory"]) == 5

    def test_sample_euler(self, flow_matcher, simple_hetero_data, device):
        water_pred = flow_matcher.sample(
            simple_hetero_data, num_steps=3, method="euler", device=str(device)
        )

        n_water = simple_hetero_data["water"].num_nodes
        assert water_pred.shape == (n_water, 3)

    def test_sample_rk4(self, flow_matcher, simple_hetero_data, device):
        water_pred = flow_matcher.sample(
            simple_hetero_data, num_steps=3, method="rk4", device=str(device)
        )

        n_water = simple_hetero_data["water"].num_nodes
        assert water_pred.shape == (n_water, 3)

    # ---- Integration correctness (issue #58) ----
    # The tests above check output shapes/keys; these check the numerical
    # behavior of the Euler and RK4 steppers against closed-form solutions,
    # driven by analytic velocity fields (mocked model) rather than the encoder.

    @pytest.mark.parametrize("method", ["euler", "rk4"])
    def test_constant_field_gives_linear_trajectory(
        self, flow_matcher, simple_hetero_data, device, method
    ):
        """A constant velocity field c must produce a straight-line trajectory
        with total displacement exactly c, for both Euler and RK4 and regardless
        of step count (dx/dt = c over t in [0, 1] => x(1) = x(0) + c)."""
        c = torch.tensor([1.0, -2.0, 0.5], device=device)
        flow_matcher.model = _ConstantVelocity(c).to(device)
        integrate = getattr(flow_matcher, f"{method}_integrate")

        num_steps = 11
        result = integrate(
            simple_hetero_data,
            num_steps=num_steps,
            device=str(device),
            return_trajectory=True,
        )[0]
        trajectory = result["trajectory"]
        assert len(trajectory) == num_steps

        x0 = trajectory[0]
        c_np = c.cpu().numpy()

        # Net displacement equals the (constant) velocity, independent of steps.
        np.testing.assert_allclose(
            result["water_pred"] - x0,
            np.broadcast_to(c_np, x0.shape),
            atol=1e-3,
        )
        # Every intermediate frame lies on the line x0 + (k / (num_steps-1)) * c.
        for k, frame in enumerate(trajectory):
            expected = x0 + (k / (num_steps - 1)) * c_np
            np.testing.assert_allclose(frame, expected, atol=1e-3)

    def test_rk4_matches_euler_with_more_steps(
        self, flow_matcher, simple_hetero_data, device
    ):
        """On the decay field dx/dt = -x, coarse RK4 should reach the same
        solution as fine-step Euler, and be far more accurate than Euler at the
        same (small) step count -- i.e. RK4's higher order converges faster."""
        flow_matcher.model = _LinearDecayVelocity(k=1.0).to(device)

        def run(method: str, num_steps: int, seed: int, trajectory: bool = False):
            # Same seed => same prior noise (the integration's only randomness),
            # so the runs are compared from identical initial positions.
            gen = torch.Generator(device=device).manual_seed(seed)
            integrate = getattr(flow_matcher, f"{method}_integrate")
            return integrate(
                simple_hetero_data,
                num_steps=num_steps,
                device=str(device),
                return_trajectory=trajectory,
                generator=gen,
            )[0]

        seed = 1234
        # The prior is drawn before the first step, so a 2-step run exposes the
        # same initial noise the long runs start from, without keeping 2000
        # frames alive (each one copies to CPU and converts to NumPy).
        x0_euler = run("euler", num_steps=2, seed=seed, trajectory=True)["trajectory"][
            0
        ]
        x0_rk4 = run("rk4", num_steps=2, seed=seed, trajectory=True)["trajectory"][0]
        np.testing.assert_allclose(x0_rk4, x0_euler, atol=1e-5)

        euler_fine = run("euler", num_steps=2000, seed=seed)
        rk4_coarse = run("rk4", num_steps=21, seed=seed)
        euler_coarse = run("euler", num_steps=21, seed=seed)

        reference = euler_fine["water_pred"]
        # Coarse RK4 converges to the fine-step Euler solution.
        np.testing.assert_allclose(
            rk4_coarse["water_pred"], reference, rtol=2e-2, atol=2e-2
        )
        # RK4 is closer to the reference than Euler at the same step count.
        rk4_err = np.linalg.norm(rk4_coarse["water_pred"] - reference)
        euler_err = np.linalg.norm(euler_coarse["water_pred"] - reference)
        assert rk4_err < euler_err

    def test_time_dependent_field_uses_correct_stage_times(
        self, flow_matcher, simple_hetero_data, device
    ):
        """On v(x, t) = a*t both integrators have a closed-form answer that
        depends only on the times they evaluate the field at: RK4 integrates a
        linear integrand exactly, Euler's left rectangles miss by a known
        deficit. The constant/decay fields above are time-independent, so this
        is what catches a stage fed the wrong t."""
        a = torch.tensor([2.0, -1.0, 4.0], device=device)
        flow_matcher.model = _TimeRampVelocity(a).to(device)
        a_np = a.cpu().numpy()

        num_steps = 11
        kwargs = {
            "num_steps": num_steps,
            "device": str(device),
            "return_trajectory": True,
        }
        rk4 = flow_matcher.rk4_integrate(simple_hetero_data, **kwargs)[0]
        euler = flow_matcher.euler_integrate(simple_hetero_data, **kwargs)[0]

        # RK4: integral of a*t over [0, 1] = a/2, exactly.
        rk4_shift = rk4["water_pred"] - rk4["trajectory"][0]
        np.testing.assert_allclose(
            rk4_shift, np.broadcast_to(a_np / 2.0, rk4_shift.shape), atol=1e-4
        )

        # Euler: dt * sum(t_i) over the num_steps-1 left endpoints, which is
        # a * (num_steps - 2) / (2 * (num_steps - 1)).
        euler_shift = euler["water_pred"] - euler["trajectory"][0]
        expected = a_np * (num_steps - 2) / (2 * (num_steps - 1))
        np.testing.assert_allclose(
            euler_shift, np.broadcast_to(expected, euler_shift.shape), atol=1e-4
        )

    @pytest.mark.parametrize("method", ["euler", "rk4"])
    def test_batched_integration_splits_results_per_graph(
        self, flow_matcher, simple_hetero_data, device, method
    ):
        """Inference integrates a list of graphs in one batch, so the per-graph
        split of the final positions and trajectories has to be right. The two
        graphs differ in water count and in velocity, so a wrong mask shows up."""
        g0 = simple_hetero_data
        g1 = copy.deepcopy(simple_hetero_data)
        # Second graph: 3 waters instead of 5, so a mis-split changes shapes too.
        g1["water"].pos = torch.randn(3, 3, device=device)
        g1["water"].x = torch.randn(3, 16, device=device)
        g1["water"].batch = torch.zeros(3, dtype=torch.long, device=device)

        c = torch.tensor([1.0, -2.0, 0.5], device=device)
        flow_matcher.model = _PerGraphVelocity(c).to(device)
        integrate = getattr(flow_matcher, f"{method}_integrate")

        num_steps = 7
        results = integrate(
            [g0, g1], num_steps=num_steps, device=str(device), return_trajectory=True
        )

        assert len(results) == 2
        c_np = c.cpu().numpy()
        for i, (result, n_water) in enumerate(zip(results, (5, 3))):
            assert result["water_pred"].shape == (n_water, 3)
            assert len(result["trajectory"]) == num_steps
            assert all(f.shape == (n_water, 3) for f in result["trajectory"])
            # Graph i moves by (i + 1) * c over t in [0, 1].
            shift = result["water_pred"] - result["trajectory"][0]
            np.testing.assert_allclose(
                shift, np.broadcast_to((i + 1) * c_np, shift.shape), atol=1e-3
            )


# ============== Tests for water sampling strategies ==============


@pytest.mark.unit
class TestUniformBallSampling:
    def test_shapes_and_counts(self, device):
        torch.manual_seed(0)
        protein_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            device=device,
        )
        batch_p = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
        batch_w = _batch_from_counts(
            torch.tensor([4, 3], dtype=torch.long, device=device), device
        )

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=2.0,
            device=device,
        )

        assert pos.shape == (7, 3)

    def test_all_within_cutoff(self, device):
        torch.manual_seed(42)
        protein_pos = torch.randn(20, 3, device=device) * 50
        batch_p = torch.cat([torch.zeros(10), torch.ones(10)]).long().to(device)
        batch_w = _batch_from_counts(
            torch.tensor([50, 50], dtype=torch.long, device=device), device
        )
        cutoff = 8.0

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=cutoff,
            device=device,
        )

        for g in range(2):
            g_waters = pos[batch_w == g]
            g_protein = protein_pos[batch_p == g]
            dists = torch.cdist(g_waters, g_protein)
            assert dists.min(dim=1).values.max().item() <= cutoff + 1e-5

    def test_empty_waters(self, device):
        protein_pos = torch.randn(5, 3, device=device)
        batch_p = torch.zeros(5, dtype=torch.long, device=device)
        batch_w = torch.empty(0, dtype=torch.long, device=device)

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=8.0,
            device=device,
        )

        assert pos.shape == (0, 3)

    def test_zero_protein_graph_raises(self, device):
        """Requesting waters for a graph with no protein atoms fails fast."""
        # graph 0 has protein atoms, graph 1 has none but requests waters
        protein_pos = torch.randn(5, 3, device=device)
        batch_p = torch.zeros(5, dtype=torch.long, device=device)
        batch_w = _batch_from_counts(
            torch.tensor([3, 4], dtype=torch.long, device=device), device
        )

        with pytest.raises(ValueError, match="zero protein atoms"):
            sample_waters_uniform_ball(
                protein_pos=protein_pos,
                batch_p=batch_p,
                batch_w=batch_w,
                cutoff=8.0,
                device=device,
            )

    def test_anchor_mask_is_per_graph(self, device):
        """Masked-out atoms are never ball centres, and each graph stays within
        its own eligible atoms rather than its neighbour's."""
        torch.manual_seed(0)
        protein_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [100.0, 0.0, 0.0], [900.0, 0.0, 0.0]],
            device=device,
        )
        batch_p = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
        batch_w = _batch_from_counts(
            torch.tensor([50, 50], dtype=torch.long, device=device), device
        )
        anchor_mask = torch.tensor([True, False, True, False], device=device)

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=2.0,
            device=device,
            anchor_mask=anchor_mask,
        )

        assert pos[batch_w == 0][:, 0].abs().max().item() < 5.0
        assert (pos[batch_w == 1][:, 0] - 100.0).abs().max().item() < 5.0

    def test_anchor_mask_skipped_for_batch_when_a_graph_starves(
        self, device, warning_log
    ):
        """If the mask would leave a water-requesting graph with no anchor, the
        whole batch anchors on all atoms (as local_flow does) and warns."""
        torch.manual_seed(0)
        protein_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # graph 0, masked out
                [10.0, 0.0, 0.0],  # graph 0, masked out -> graph 0 is starved
                [1000.0, 0.0, 0.0],  # graph 1, eligible
                [2000.0, 0.0, 0.0],  # graph 1, masked out
            ],
            device=device,
        )
        batch_p = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
        batch_w = _batch_from_counts(
            torch.tensor([100, 100], dtype=torch.long, device=device), device
        )
        anchor_mask = torch.tensor([False, False, True, False], device=device)
        cutoff = 2.0

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=cutoff,
            device=device,
            anchor_mask=anchor_mask,
        )

        # each water still sits within cutoff of one of its own graph's atoms
        d0 = torch.cdist(pos[batch_w == 0], protein_pos[:2]).min(dim=1).values
        assert d0.max().item() <= cutoff + 1e-5
        # the mask was dropped for the whole batch, so graph 1 uses its masked-out
        # atom at x=2000 too, not only the eligible one at x=1000
        g1_x = pos[batch_w == 1][:, 0]
        assert (g1_x > 1500.0).any()

        assert any("all protein atoms" in message for message in warning_log)

    def test_anchor_mask_none_matches_all_true_mask(self, device):
        """An all-True mask changes neither the draws nor their order."""
        protein_pos = torch.randn(12, 3, device=device) * 10
        batch_p = torch.cat([torch.zeros(6), torch.ones(6)]).long().to(device)
        batch_w = _batch_from_counts(
            torch.tensor([20, 15], dtype=torch.long, device=device), device
        )

        torch.manual_seed(7)
        without = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=8.0,
            device=device,
        )
        torch.manual_seed(7)
        with_mask = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=8.0,
            device=device,
            anchor_mask=torch.ones(12, dtype=torch.bool, device=device),
        )

        assert torch.equal(without, with_mask)

    def test_large_spread_protein_succeeds(self, device):
        """The scenario that crashes truncated Gaussian (sigma~50) works here."""
        torch.manual_seed(0)
        protein_pos = torch.randn(500, 3, device=device) * 50
        batch_p = torch.zeros(500, dtype=torch.long, device=device)
        batch_w = _batch_from_counts(
            torch.tensor([301], dtype=torch.long, device=device), device
        )

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=8.0,
            device=device,
        )

        assert pos.shape == (301, 3)

    @pytest.mark.slow
    def test_real_structure_cutoff_and_batch(self, device, pdb_6eey):
        """Cutoff guarantee holds on real protein geometry; batch indexing is correct
        when two structures with different water counts are packed into one call."""
        import biotite.structure as bts
        from biotite.structure.io.pdb import get_structure, PDBFile

        torch.manual_seed(0)

        pdb_file = PDBFile.read(pdb_6eey)
        atoms = get_structure(pdb_file, model=1, altloc="occupancy")
        atoms = atoms[atoms.element != "H"]
        protein_atoms = atoms[bts.filter_amino_acids(atoms)]

        protein_pos_np = protein_atoms.coord  # (N, 3) float64
        n_atoms = len(protein_pos_np)

        # batch two copies: graph 0 gets 50 waters, graph 1 gets 30
        protein_pos = torch.tensor(protein_pos_np, dtype=torch.float32, device=device)
        protein_pos_both = torch.cat([protein_pos, protein_pos], dim=0)
        batch_p = torch.cat(
            [
                torch.zeros(n_atoms, dtype=torch.long, device=device),
                torch.ones(n_atoms, dtype=torch.long, device=device),
            ]
        )
        num_waters = torch.tensor([50, 30], dtype=torch.long, device=device)
        batch_w = _batch_from_counts(num_waters, device)
        cutoff = 8.0

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos_both,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=cutoff,
            device=device,
        )

        # correct total count and per-graph split
        assert pos.shape == (80, 3)
        assert (batch_w == 0).sum().item() == 50
        assert (batch_w == 1).sum().item() == 30

        # every water must be within cutoff of at least one protein atom in its graph
        for g, n_w in enumerate(num_waters.tolist()):
            g_waters = pos[batch_w == g]  # (n_w, 3)
            g_protein = protein_pos_both[batch_p == g]  # (n_atoms, 3)
            dists = torch.cdist(g_waters, g_protein)  # (n_w, n_atoms)
            min_dists = dists.min(dim=1).values  # (n_w,)
            assert min_dists.max().item() <= cutoff + 1e-4, (
                f"Graph {g}: water too far from protein "
                f"(max dist {min_dists.max().item():.4f} > {cutoff})"
            )


@pytest.mark.unit
class TestScaledGaussianSampling:
    def test_shapes_and_counts(self, device):
        torch.manual_seed(0)
        batch_w = _batch_from_counts(
            torch.tensor([4, 3], dtype=torch.long, device=device), device
        )
        sigma = torch.tensor([1.0, 2.0], device=device)

        pos = sample_waters_scaled_gaussian(
            batch_w=batch_w,
            sigma_per_graph=sigma,
            device=device,
            dtype=torch.float32,
        )

        assert pos.shape == (7, 3)

    def test_sigma_broadcasts_per_graph(self, device):
        """Each graph's waters must be scaled by that graph's own sigma."""
        batch_w = _batch_from_counts(
            torch.tensor([4, 3], dtype=torch.long, device=device), device
        )
        sigma = torch.tensor([1.0, 2.0], device=device)

        torch.manual_seed(0)
        pos = sample_waters_scaled_gaussian(
            batch_w=batch_w,
            sigma_per_graph=sigma,
            device=device,
            dtype=torch.float32,
        )

        # randn is the sampler's only RNG draw, so the same seed reproduces it
        torch.manual_seed(0)
        expected = torch.randn(7, 3, device=device, dtype=torch.float32) * sigma[
            batch_w
        ].unsqueeze(-1)

        assert torch.allclose(pos, expected)

    def test_empty_waters(self, device):
        batch_w = torch.empty(0, dtype=torch.long, device=device)
        sigma = torch.tensor([1.0], device=device)

        pos = sample_waters_scaled_gaussian(
            batch_w=batch_w,
            sigma_per_graph=sigma,
            device=device,
            dtype=torch.float32,
        )

        assert pos.shape == (0, 3)


@pytest.mark.unit
class TestCrystalMateAwareSampling:
    """Both samplers work from ASU atoms only when the batch carries is_mate."""

    @staticmethod
    def _matcher(strategy):
        return FlowMatcher(model=Mock(cutoff=8.0), sampling_strategy=strategy)

    @staticmethod
    def _graph(pos, batch, is_mate=None):
        data = HeteroData()
        data["protein"].pos = pos
        data["protein"].batch = batch
        if is_mate is not None:
            data["protein"].is_mate = is_mate
        return data

    def test_uniform_ball_anchors_on_asu_only(self, device):
        """Waters spawn around ASU atoms, not around the distant symmetry mates."""
        torch.manual_seed(0)
        data = self._graph(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [500.0, 0.0, 0.0],
                    [900.0, 0.0, 0.0],
                ],
                device=device,
            ),
            torch.zeros(4, dtype=torch.long, device=device),
            torch.tensor([False, False, True, True], device=device),
        )
        batch_w = torch.zeros(200, dtype=torch.long, device=device)

        matcher = self._matcher("uniform_ball")
        pos = matcher._sample_waters(data, batch_w, device)

        # ASU spans x in [0, 1]; a mate anchor would land a water near 500 or 900
        assert pos[:, 0].max().item() < 1.0 + matcher.graph_cutoff + 1e-5
        assert pos[:, 0].min().item() > -matcher.graph_cutoff - 1e-5

    def test_uniform_ball_skips_mask_for_all_mate_graph(self, device, warning_log):
        """An all-mate graph has no ASU anchor, so the batch anchors on all atoms
        and warns; every graph still draws from its own atoms."""
        torch.manual_seed(0)
        data = self._graph(
            torch.tensor([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]], device=device),
            torch.tensor([0, 1], dtype=torch.long, device=device),
            torch.tensor([False, True], device=device),
        )
        batch_w = _batch_from_counts(
            torch.tensor([20, 20], dtype=torch.long, device=device), device
        )

        pos = self._matcher("uniform_ball")._sample_waters(data, batch_w, device)

        assert pos[batch_w == 0][:, 0].abs().max().item() < 10.0
        assert (pos[batch_w == 1][:, 0] - 500.0).abs().max().item() < 10.0
        assert any("all protein atoms" in message for message in warning_log)

    def test_all_false_is_mate_matches_no_attribute(self, device):
        """Dropping the .any() guard is safe: an all-True mask draws exactly what
        no mask draws, bit for bit."""
        protein_pos = torch.randn(12, 3, device=device) * 10
        batch_p = torch.cat([torch.zeros(6), torch.ones(6)]).long().to(device)
        batch_w = _batch_from_counts(
            torch.tensor([20, 15], dtype=torch.long, device=device), device
        )
        matcher = self._matcher("uniform_ball")

        plain = self._graph(protein_pos, batch_p)
        flagged = self._graph(
            protein_pos, batch_p, torch.zeros(12, dtype=torch.bool, device=device)
        )

        torch.manual_seed(7)
        without = matcher._sample_waters(plain, batch_w, device)
        torch.manual_seed(7)
        with_mask = matcher._sample_waters(flagged, batch_w, device)

        assert torch.equal(without, with_mask)

    def test_sigma_ignores_distant_mates(self, device):
        """Counting distant mates would inflate sigma and push the prior out past
        the targets."""
        torch.manual_seed(0)
        asu = torch.randn(30, 3, device=device)
        mates = torch.randn(30, 3, device=device) + 500.0

        asu_only = self._graph(asu, torch.zeros(30, dtype=torch.long, device=device))
        with_mates = self._graph(
            torch.cat([asu, mates], dim=0),
            torch.zeros(60, dtype=torch.long, device=device),
            torch.cat(
                [torch.zeros(30, dtype=torch.bool), torch.ones(30, dtype=torch.bool)]
            ).to(device),
        )

        reference = FlowMatcher.compute_sigma_per_graph(asu_only, device)
        masked = FlowMatcher.compute_sigma_per_graph(
            with_mates, device, node_mask=FlowMatcher._asu_mask(with_mates)
        )
        unmasked = FlowMatcher.compute_sigma_per_graph(with_mates, device)

        # adding the mates leaves sigma untouched once they are masked out
        assert torch.allclose(masked, reference, atol=1e-4)
        # ...and would otherwise blow it up by two orders of magnitude
        assert unmasked.item() > 50 * reference.item()

    def test_sigma_skips_mask_for_all_mate_graph(self, device, warning_log):
        """If any graph has no ASU atom, sigma is computed over all atoms for the
        whole batch (as local_flow does) with a warning, not a degenerate value."""
        torch.manual_seed(0)
        g0 = HeteroData()
        g0["protein"].pos = torch.cat(
            [torch.randn(20, 3), torch.randn(20, 3) + 500.0]
        ).to(device)
        g0["protein"].is_mate = torch.cat(
            [torch.zeros(20, dtype=torch.bool), torch.ones(20, dtype=torch.bool)]
        ).to(device)
        g1 = HeteroData()
        g1["protein"].pos = torch.randn(20, 3).to(device)
        g1["protein"].is_mate = torch.ones(20, dtype=torch.bool, device=device)

        batch = Batch.from_data_list([g0, g1])
        masked = FlowMatcher.compute_sigma_per_graph(
            batch, device, node_mask=FlowMatcher._asu_mask(batch)
        )
        full = FlowMatcher.compute_sigma_per_graph(batch, device)

        # graph 1 has no ASU atom, so the mask is skipped for the whole batch:
        # sigma matches the unmasked computation everywhere, non-degenerate
        assert torch.allclose(masked, full)
        assert masked[1].item() > 0.5
        assert any("all protein atoms" in message for message in warning_log)


@pytest.mark.unit
class TestSamplingHonoursNodeOrder:
    """Samplers return one water per batch_w entry, in batch_w's own order."""

    @staticmethod
    def _two_graphs_far_apart(device):
        """Graph 0's protein at the origin, graph 1's 100A away."""
        protein_pos = torch.cat(
            [torch.zeros(4, 3, device=device), torch.full((4, 3), 100.0, device=device)]
        )
        batch_p = torch.tensor([0] * 4 + [1] * 4, dtype=torch.long, device=device)
        # water nodes interleaved across graphs rather than grouped
        batch_w = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

        return protein_pos, batch_p, batch_w

    def test_uniform_ball_follows_interleaved_batch(self, device):
        """Each water anchors on its own graph even when nodes are interleaved."""
        torch.manual_seed(0)
        protein_pos, batch_p, batch_w = self._two_graphs_far_apart(device)

        pos = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=2.0,
            device=device,
        )

        assert pos[batch_w == 0].abs().max().item() < 10.0
        assert (pos[batch_w == 1] - 100.0).abs().max().item() < 10.0

    def test_scaled_gaussian_follows_interleaved_batch(self, device):
        """Sigma follows each water's own graph when nodes are interleaved."""
        batch_w = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
        sigma = torch.tensor([1.0, 2.0], device=device)

        torch.manual_seed(0)
        pos = sample_waters_scaled_gaussian(
            batch_w=batch_w, sigma_per_graph=sigma, device=device, dtype=torch.float32
        )

        torch.manual_seed(0)
        expected = torch.randn(4, 3, device=device) * sigma[batch_w].unsqueeze(-1)

        assert torch.allclose(pos, expected)

    def test_ot_coupling_pairs_within_graph_when_interleaved(self, device):
        """A graph's waters pair with its own prior, not a neighbour's."""
        from src.utils import ot_coupling

        torch.manual_seed(0)
        protein_pos, batch_p, batch_w = self._two_graphs_far_apart(device)
        x1 = torch.tensor(
            [[0.0] * 3, [100.0] * 3, [0.0] * 3, [100.0] * 3], device=device
        )

        x0 = sample_waters_uniform_ball(
            protein_pos=protein_pos,
            batch_p=batch_p,
            batch_w=batch_w,
            cutoff=2.0,
            device=device,
        )
        x0_star, x1_star = ot_coupling(x1=x1, batch=batch_w, x0=x0)

        # pairings stay inside their graph, so nothing is dragged 100A across
        assert (x1_star - x0_star).norm(dim=-1).max().item() < 10.0


@pytest.mark.unit
class TestWaterCountValidation:
    def test_negative_water_count_raises(self, device):
        """A negative water_count is rejected before any sampling work."""
        fm = FlowMatcher(model=Mock(cutoff=8.0))
        g = HeteroData()  # guard fires before touching graph contents

        with pytest.raises(ValueError, match="water_count must be >= 0"):
            fm._setup_water_nodes_from_count(g, -1, device)


# ============== Edge case tests ==============


@pytest.mark.unit
class TestEdgeCases:
    def test_single_water_molecule(self, device):
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        data["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)
        data["water"].pos = torch.randn(1, 3, device=device)  # Single water
        data["water"].x = torch.randn(1, 16, device=device)
        data["water"].batch = torch.zeros(1, dtype=torch.long, device=device)
        data["protein", "pp", "protein"].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )

        t = torch.tensor([0.5], device=device)
        v_pred = model(data, t)

        assert v_pred.shape == (1, 3)

    def test_frozen_gvp_encoder(self, device):
        """Freezing is handled by the encoder itself, not FlowWaterGVP."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 8),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)
        encoder = GVPEncoder(encoder=base_encoder, freeze=True)

        model = FlowWaterGVP(
            encoder=encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        # Verify encoder params are frozen
        for p in model.encoder.encoder.parameters():
            assert p.requires_grad is False


# ============== Tests for edge connectivity ==============


@pytest.mark.unit
class TestWaterEdgeConnectivity:
    """Tests to ensure all waters have edges (both protein-water and water-water)."""

    def test_all_waters_have_protein_edges(self, simple_hetero_data):
        """Ensure every water has at least one protein-water edge."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data)
        pw_edges = edge_dict[("protein", "pw", "water")]

        n_water = simple_hetero_data["water"].num_nodes

        # Check that all water nodes appear in the protein-water edges
        water_nodes_with_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_edges) == n_water, (
            f"Only {len(water_nodes_with_edges)}/{n_water} waters have protein edges"
        )

    def test_all_waters_have_water_edges(self, simple_hetero_data):
        """Ensure every water has at least one water-water edge (if multiple waters exist)."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(simple_hetero_data)
        ww_edges = edge_dict[("water", "ww", "water")]

        n_water = simple_hetero_data["water"].num_nodes

        if n_water > 1:
            # WW edges are built per destination (knn query per water), so every
            # water is guaranteed to appear as a destination (row 1); a water that
            # is no other water's nearest neighbor would be missing from the source
            # row (row 0). Assert coverage on the destination/query row.
            water_nodes_with_edges = torch.unique(ww_edges[1])
            assert len(water_nodes_with_edges) == n_water, (
                f"Only {len(water_nodes_with_edges)}/{n_water} waters have water-water edges"
            )

    def test_batched_waters_have_edges(self, batched_hetero_data):
        """Ensure all waters in a batched graph have edges."""
        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)

        edge_dict = updater.build_edges(batched_hetero_data)
        pw_edges = edge_dict[("protein", "pw", "water")]
        ww_edges = edge_dict[("water", "ww", "water")]

        n_water = batched_hetero_data["water"].num_nodes

        # Check protein-water edges
        water_nodes_with_pw_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_pw_edges) == n_water, (
            f"Only {len(water_nodes_with_pw_edges)}/{n_water} waters have protein edges in batched data"
        )

        # Check water-water edges. WW edges are built per destination, so every
        # water appears as a destination (row 1); assert coverage on the
        # destination/query row rather than the source row.
        if n_water > 1:
            water_nodes_with_ww_edges = torch.unique(ww_edges[1])
            assert len(water_nodes_with_ww_edges) == n_water, (
                f"Only {len(water_nodes_with_ww_edges)}/{n_water} waters have water-water edges in batched data"
            )

    def test_single_water_has_protein_edges_no_water_edges(self, device):
        """A single water should have protein edges but no water-water edges."""
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        data["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)
        data["water"].pos = torch.randn(1, 3, device=device)  # Single water
        data["water"].x = torch.randn(1, 16, device=device)
        data["water"].batch = torch.zeros(1, dtype=torch.long, device=device)
        data["protein", "pp", "protein"].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )

        updater = ProteinWaterUpdate(hidden_dims=(128, 16), layers=1)
        edge_dict = updater.build_edges(data)

        pw_edges = edge_dict[("protein", "pw", "water")]
        ww_edges = edge_dict[("water", "ww", "water")]

        # Single water should have protein edges
        assert pw_edges.shape[1] > 0, (
            "Single water should have at least one protein edge"
        )
        water_nodes_with_edges = torch.unique(pw_edges[1])
        assert len(water_nodes_with_edges) == 1, "Single water must have protein edges"

        # Single water should have no water-water edges (since k_ww excludes self-loops)
        assert ww_edges.shape[1] == 0, "Single water should have no water-water edges"
