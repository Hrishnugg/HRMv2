"""Planner selection + env-forwarding wiring for focal search."""
import residual_tasklora_v2 as R


def test_forward_env_includes_planner(monkeypatch):
    monkeypatch.setenv("PLANNER", "focal")
    monkeypatch.setenv("FOCAL_W", "2.5")
    d = R._eval_forward_env()
    assert d.get("PLANNER") == "focal"
    assert d.get("FOCAL_W") == "2.5"


def test_run_policy_episode_routes_to_focal(monkeypatch):
    suite = [s for s in R.build_eval_suites(False, 10) if s.suite_id == "ID_A32_static"][0]
    calls = {"focal": 0, "astar": 0}
    real_focal = R.space_time_focal_astar
    real_astar = R.space_time_astar

    def spy_focal(*a, **k):
        calls["focal"] += 1
        return real_focal(*a, **k)

    def spy_astar(*a, **k):
        calls["astar"] += 1
        return real_astar(*a, **k)

    monkeypatch.setattr(R, "space_time_focal_astar", spy_focal)
    monkeypatch.setattr(R, "space_time_astar", spy_astar)

    monkeypatch.setattr(R, "PLANNER", "focal")
    monkeypatch.setattr(R, "FOCAL_W", 2.0)
    R.run_policy_episode(suite, seed=0, model=None, alpha=1.0, max_expansions=80, device="cpu")
    assert calls["focal"] > 0 and calls["astar"] == 0

    calls["focal"] = calls["astar"] = 0
    monkeypatch.setattr(R, "PLANNER", "astar")
    R.run_policy_episode(suite, seed=0, model=None, alpha=1.0, max_expansions=80, device="cpu")
    assert calls["astar"] > 0 and calls["focal"] == 0
