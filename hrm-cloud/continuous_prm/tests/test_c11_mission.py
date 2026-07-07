import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import continuous_prm_common as C
import continuous_prm_c11_headroom as C11P
import continuous_prm_c11_mission as M


# ---------------------------------------------------------------------------
# Test 1: cell grid shape.
# ---------------------------------------------------------------------------

def test_cell_grid_shape():
    cells = M.build_cell_grid()
    assert len(cells) == 11

    # No ("C", 0) cell -- config C is dropped at K=0 (degenerates to A).
    c_k0 = [c for c in cells if c["config_label"] == "C" and c["K"] == 0]
    assert c_k0 == []

    expected_keys = {"config_label", "spec_name", "config_idx", "K", "doors"}
    for cell in cells:
        assert set(cell.keys()) == expected_keys

    config_idx_by_label = {c["config_label"]: c["config_idx"] for c in cells}
    assert config_idx_by_label["A"] == 0
    assert config_idx_by_label["B"] == 1
    assert config_idx_by_label["C"] == 2

    # A and B present at all four K values; C only at K in (2, 4, 8).
    a_ks = sorted(c["K"] for c in cells if c["config_label"] == "A")
    b_ks = sorted(c["K"] for c in cells if c["config_label"] == "B")
    c_ks = sorted(c["K"] for c in cells if c["config_label"] == "C")
    assert a_ks == [0, 2, 4, 8]
    assert b_ks == [0, 2, 4, 8]
    assert c_ks == [2, 4, 8]

    # doors flag matches config label.
    for cell in cells:
        expected_doors = cell["config_label"] == "C"
        assert cell["doors"] == expected_doors

    # spec_name mapping.
    for cell in cells:
        if cell["config_label"] in ("A", "C"):
            assert cell["spec_name"] == "C_hard_maze"
        else:
            assert cell["spec_name"] == "C_hard_rooms_large"


# ---------------------------------------------------------------------------
# Test 2: TRAIN/TEST seed disjointness, pooled across all 11 cells.
# ---------------------------------------------------------------------------

def test_train_test_seed_disjointness():
    cells = M.build_cell_grid()
    train_seeds = set()
    test_seeds = set()
    for cell in cells:
        config_idx = cell["config_idx"]
        K = cell["K"]
        for w in range(60):
            train_seeds.add(M.train_seed(w, config_idx, K))
            test_seeds.add(M.test_seed(w, config_idx, K))

    assert train_seeds.isdisjoint(test_seeds)
    # Sanity: both sets are actually populated (not accidentally identical
    # formulas collapsing to one giant coincidental overlap check).
    assert len(train_seeds) > 0
    assert len(test_seeds) > 0


# ---------------------------------------------------------------------------
# Test 3: collect_world_bundle at K=2, config A (no doors).
# ---------------------------------------------------------------------------

def _find_first_valid_test_bundle(cell, max_attempts=60):
    """Loop TEST seeds like collect_cell_dataset does, returning the first
    valid bundle. Mirrors the probe's own eval_cell skip-and-retry loop."""
    for w in range(max_attempts):
        seed = M.test_seed(w, cell["config_idx"], cell["K"])
        try:
            return M.collect_world_bundle(cell, seed)
        except RuntimeError:
            continue
    raise RuntimeError(f"no valid TEST bundle found for cell {cell} in {max_attempts} attempts")


def test_collect_world_bundle_k2():
    cells = M.build_cell_grid()
    cell_a_k2 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 2)

    bundle = _find_first_valid_test_bundle(cell_a_k2)

    assert len(bundle.wp) == 2
    assert bundle.oracle.shape == (3, 192)
    assert bundle.adj_valid is None

    targets = bundle.targets
    assert targets.ndim == 2
    assert targets.shape[1] == 3
    assert targets.shape[0] > 0
    y = targets[:, 2]
    assert np.all(y >= 0.0)
    assert np.all(y <= 4.0)
    assert bundle.preclip_min >= -1e-9

    # Determinism: two calls on the same (cell, seed) give identical targets.
    bundle2 = M.collect_world_bundle(cell_a_k2, bundle.seed)
    assert np.array_equal(bundle.targets, bundle2.targets)


# ---------------------------------------------------------------------------
# Test 4: collect_world_bundle at K=0 -- formulation-equivalence guard.
# ---------------------------------------------------------------------------

def test_collect_world_bundle_k0():
    cells = M.build_cell_grid()
    cell_a_k0 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 0)

    bundle = _find_first_valid_test_bundle(cell_a_k0)

    assert bundle.wp == ()
    assert bundle.oracle.shape == (1, 192)

    # K=0 legsum must be exactly euclid-to-goal (node 1) -- the C7-continuity
    # formulation guard. Sample up to 5 nodes with finite oracle at stage 0.
    N = bundle.rm.points.shape[0]
    finite_nodes = [i for i in range(N) if bundle.oracle[0, i] < C.INF / 10.0]
    assert len(finite_nodes) >= 1
    sample = finite_nodes[:5]
    for i in sample:
        expected = float(np.linalg.norm(bundle.rm.points[i] - bundle.rm.points[1]))
        assert bundle.hl(i, 0) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 5: collect_cell_dataset counts + seed-formula provenance.
# ---------------------------------------------------------------------------

def test_collect_cell_dataset_counts():
    cells = M.build_cell_grid()
    cell_a_k2 = next(c for c in cells if c["config_label"] == "A" and c["K"] == 2)

    train_bundles = M.collect_cell_dataset(cell_a_k2, split="train", n_worlds=3)
    assert len(train_bundles) == 3
    for b in train_bundles:
        # seed = 900001 + 7919*w + 104729*config_idx + 15485863*K for SOME w.
        assert (b.seed - 900001 - 104729 * 0 - 15485863 * 2) % 7919 == 0

    test_bundles = M.collect_cell_dataset(cell_a_k2, split="test", n_worlds=2)
    assert len(test_bundles) == 2
    for b in test_bundles:
        # seed = 1234 + 7919*w + 104729*config_idx + 15485863*K for SOME w.
        assert (b.seed - 1234 - 104729 * 0 - 15485863 * 2) % 7919 == 0

    # Cross-check: no test bundle's seed satisfies the train formula and
    # vice versa (disjoint residues given the formulas' distinct constants).
    train_seed_set = {b.seed for b in train_bundles}
    test_seed_set = {b.seed for b in test_bundles}
    assert train_seed_set.isdisjoint(test_seed_set)


# ---------------------------------------------------------------------------
# Test 6: doors cell (config C) -- adj_valid present, oracle >= plain oracle.
# ---------------------------------------------------------------------------

def test_doors_cell_bundle():
    cells = M.build_cell_grid()
    cell_c_k2 = next(c for c in cells if c["config_label"] == "C" and c["K"] == 2)

    bundle = _find_first_valid_test_bundle(cell_c_k2, max_attempts=80)

    assert bundle.adj_valid is not None

    oracle_plain = C11P.product_oracle(bundle.rm, bundle.wp)
    finite_mask = (oracle_plain < C.INF / 10.0) & (bundle.oracle < C.INF / 10.0)
    assert finite_mask.any()
    assert np.all(bundle.oracle[finite_mask] >= oracle_plain[finite_mask] - 1e-9)
