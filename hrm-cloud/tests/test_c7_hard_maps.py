import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_common as C  # noqa: E402
import continuous_prm_c7_hard_maps as M  # noqa: E402


NEW_SUITES = ["C_hard_spiral", "C_hard_bugtrap", "C_hard_rooms_large"]


def test_install_registers_new_suites():
    M.install_c7_hard_maps()
    specs = C.build_anchor_specs()
    for s in NEW_SUITES:
        assert s in specs
    assert "C_hard_maze" in specs  # C5 hard suites still present (composition preserved)


def test_new_suites_build_valid_connected_worlds():
    M.install_c7_hard_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in NEW_SUITES:
        spec = C.build_anchor_specs()[suite]
        built = 0
        for seed in range(40):
            world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if world is None:
                continue
            assert C.is_point_free(world.start, world.side_len, world.obstacles)
            assert C.is_point_free(world.goal, world.side_len, world.obstacles)
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is not None and rm.connected_to_goal[0]:
                built += 1
            if built >= 5:
                break
        assert built >= 5, f"{suite}: only built {built} connected worlds in 40 seeds"


def test_new_suites_force_detours():
    M.install_c7_hard_maps()
    cfg = C.RoadmapConfig(n_nodes=192, k_neighbors=7)
    for suite in NEW_SUITES:
        spec = C.build_anchor_specs()[suite]
        ratios = []
        for seed in range(60):
            world = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if world is None:
                continue
            rm = C.build_prm(world, cfg, seed=seed)
            if rm is None or not rm.connected_to_goal[0]:
                continue
            straight = float(np.linalg.norm(world.start - world.goal))
            ratios.append(rm.dist_to_goal[0] / max(1e-6, straight))
            if len(ratios) >= 8:
                break
        assert len(ratios) >= 8
        assert float(np.median(ratios)) >= 1.15, f"{suite}: detour ratio {np.median(ratios):.2f} too low"
