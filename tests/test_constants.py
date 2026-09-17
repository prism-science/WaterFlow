"""Tests for shared constants consolidated into src.constants (issue #104).

These guard against values drifting back out of ``constants.py`` into
re-declared literals scattered across the source modules.
"""

import inspect

import pytest

from src.constants import (
    DEFAULT_EDGE_CUTOFF,
    DEFAULT_MIN_EDIA,
    ELEM_IDX,
    NUM_RBF,
    OXYGEN_INDEX,
    OXYGEN_VDW_RADIUS,
    RBF_CUTOFF,
)


@pytest.mark.unit
class TestSharedConstants:
    def test_oxygen_vdw_radius_value(self):
        assert OXYGEN_VDW_RADIUS == pytest.approx(1.52)

    def test_default_min_edia_value(self):
        assert DEFAULT_MIN_EDIA == pytest.approx(0.4)

    def test_oxygen_index_matches_element_vocab(self):
        # OXYGEN_INDEX must stay in sync with the element vocabulary it derives from.
        assert OXYGEN_INDEX == ELEM_IDX["O"]


@pytest.mark.unit
class TestDefaultsMatchSharedConstants:
    """Downstream defaults should *match* the shared constants. These compare each
    recorded default value against the constant, catching drift where a constant is
    changed but a call site keeps a stale literal. (They assert value equality, not
    that the source literally imports the name.)"""

    def test_cluster_waters_vdw_uses_oxygen_vdw_radius(self):
        from src.confidence import cluster_waters_vdw

        default = inspect.signature(cluster_waters_vdw).parameters["radius"].default
        assert default == OXYGEN_VDW_RADIUS

    def test_filter_waters_by_quality_uses_default_min_edia(self):
        from src.dataset import filter_waters_by_quality

        default = (
            inspect.signature(filter_waters_by_quality).parameters["min_edia"].default
        )
        assert default == DEFAULT_MIN_EDIA

    def test_dataset_init_uses_default_min_edia(self):
        from src.dataset import ProteinWaterDataset

        default = (
            inspect.signature(ProteinWaterDataset.__init__)
            .parameters["min_edia"]
            .default
        )
        assert default == DEFAULT_MIN_EDIA

    def test_compute_edge_features_uses_rbf_constants(self):
        from src.utils import compute_edge_features

        params = inspect.signature(compute_edge_features).parameters
        assert params["cutoff"].default == RBF_CUTOFF
        assert params["num_gaussians"].default == NUM_RBF

    def test_edge_cutoff_defaults_use_default_edge_cutoff(self):
        # Every graph-construction entry point should default its edge cutoff to
        # the shared DEFAULT_EDGE_CUTOFF rather than a bare 8.0.
        from src.confidence import ConfidenceGVP
        from src.dataset import ProteinWaterDataset
        from src.inference_graph import build_inference_graph

        for func in (
            ConfidenceGVP.__init__,
            ProteinWaterDataset.__init__,
            build_inference_graph,
        ):
            default = inspect.signature(func).parameters["cutoff"].default
            assert default == DEFAULT_EDGE_CUTOFF, f"{func.__qualname__} cutoff default"
