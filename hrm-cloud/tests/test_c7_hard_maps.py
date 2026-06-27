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


def test_install_idempotent_and_preserves_c5_routing():
    M.install_c7_hard_maps()
    spec_maze = C.build_anchor_specs()["C_hard_maze"]
    w1 = C.build_world(spec_maze, seed=3, min_start_goal_dist_frac=0.5)
    M.install_c7_hard_maps()  # second install must not double-wrap or lose C5 routing
    w2 = C.build_world(spec_maze, seed=3, min_start_goal_dist_frac=0.5)
    assert (w1 is None) == (w2 is None)
    if w1 is not None:
        assert len(w1.obstacles) == len(w2.obstacles)  # identical C5 geometry, no stacking
    # C7 suite still builds after the double install
    spec_spiral = C.build_anchor_specs()["C_hard_spiral"]
    assert any(C.build_world(spec_spiral, seed=s, min_start_goal_dist_frac=0.5) is not None for s in range(10))


def test_c7_world_feature_parity_with_c5_hard():
    import continuous_prm_c5_hard_obstacle_encoder as C5
    M.install_c7_hard_maps()
    cfg = C.RoadmapConfig(n_nodes=96, k_neighbors=7)
    fcfg = C.FeatureConfig()

    def first_feat(suite):
        spec = C.build_anchor_specs()[suite]
        for seed in range(60):
            w = C.build_world(spec, seed=seed, min_start_goal_dist_frac=0.5)
            if w is None:
                continue
            rm = C.build_prm(w, cfg, seed=seed)
            if rm is not None and rm.connected_to_goal[0]:
                return w, C5.make_hard_features_for_roadmap(w, rm, fcfg)
        raise RuntimeError(suite)

    w_c5, f_c5 = first_feat("C_hard_maze")
    w_c7, f_c7 = first_feat("C_hard_spiral")
    assert f_c5.shape[1:] == f_c7.shape[1:]  # same per-node feature dims
    # Encoder mode parity: C7 worlds must read as the C5 hard mode so the encoder
    # emits the same hard-mode indicator token and the same (hard) task descriptor.
    assert w_c7.meta.get("mode") == C5.HARD_MODE
    assert w_c5.meta.get("mode") == C5.HARD_MODE
    # The hard-mode indicator lives at seq[0, 15]; both families must set it to 1.0.
    assert float(f_c5[0, 0, 15]) == 1.0
    assert float(f_c7[0, 0, 15]) == 1.0
    # Descriptor index 5 is the C5 hard-mode flag (1.0) rather than the base
    # "narrow" flag (0.0); C7 worlds must carry the hard flag for descriptor parity.
    assert float(w_c7.descriptor[5]) == 1.0
    assert float(w_c5.descriptor[5]) == 1.0


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
