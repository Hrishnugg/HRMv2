import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import numpy as np
import continuous_prm_common as C
import continuous_prm_providers as P
import continuous_prm_c7_hard_maps as M


def _world_prm():
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    for seed in range(60):
        w = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
        if w is None:
            continue
        rm = C.build_prm(w, C.RoadmapConfig(n_nodes=128, k_neighbors=7), seed=seed)
        if rm is not None and rm.connected_to_goal[0]:
            return w, rm
    raise RuntimeError("no world")


def test_run_arm_records_shape_and_suboptimality():
    world, rm = _world_prm()
    providers = {"euclid": P.EuclidProvider(), "oracle": P.OracleProvider()}
    recs = P.run_world_arms(world, rm, providers, budgets=[200], w_values=[1.0, 1.5], goal_idx=1)
    names = {(r["provider"], r["mode"], r.get("w")) for r in recs}
    assert ("euclid", "astar", None) in names
    assert ("oracle", "astar", None) in names
    assert ("euclid", "focal", 1.0) in names
    assert ("oracle", "focal", 1.5) in names
    for r in recs:
        assert set(["provider", "mode", "budget", "found", "expansions", "closed", "cost", "suboptimality"]).issubset(r)
        if r["found"]:
            assert r["suboptimality"] >= 1.0 - 1e-6
            if r["mode"] == "focal":
                assert r["suboptimality"] <= r["w"] + 1e-6
    by_key = {(r["provider"], r["mode"], r.get("w"), r["budget"]): r for r in recs}
    ea = by_key[("euclid", "astar", None, 200)]
    ef = by_key[("euclid", "focal", 1.0, 200)]
    if ea["found"] and ef["found"]:
        assert np.isclose(ea["cost"], ef["cost"], rtol=1e-9)  # focal w=1 == A* for admissible euclid
    for r in recs:
        if r["found"]:
            assert np.isfinite(r["optimal"]) and r["optimal"] > 0


def test_matched_worlds_identical_across_seeds():
    M.install_c7_hard_maps()
    spec = C.build_anchor_specs()["C_hard_spiral"]
    cfg = C.RoadmapConfig(n_nodes=128, k_neighbors=7)
    seed = None
    for s in range(60):
        w = C.build_world(spec, seed=s, min_start_goal_dist_frac=0.5)
        if w is None:
            continue
        rm = C.build_prm(w, cfg, seed=s)
        if rm is not None and rm.connected_to_goal[0]:
            seed = s
            break
    assert seed is not None
    w1 = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
    w2 = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
    rm1 = C.build_prm(w1, cfg, seed=seed)
    rm2 = C.build_prm(w2, cfg, seed=seed)
    assert rm1 is not None and rm2 is not None
    assert np.array_equal(rm1.points, rm2.points)
    assert rm1.adj == rm2.adj
