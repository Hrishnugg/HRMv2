"""Unit tests for space_time_focal_astar (CPU, dummy heuristics, no model)."""
import residual_tasklora_v2 as R
from residual_tasklora_v2 import PlanResult


def _small_static():
    # 16x16 static map, no dynamics -> deterministic, solvable, fast.
    ep = R.make_episode(7, "A", 16, 60, 0, 0, 0)
    occ = R.simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    return ep, occ


def _zero_delta(states):
    return [0.0 for _ in states]


def _optimal(ep, occ, horizon=40, budget=20000):
    # space_time_astar with zero residual == optimal A* with admissible Manhattan.
    return R.space_time_astar(ep.start, ep.goal, 0, horizon, budget, occ, _zero_delta, alpha=1.0)


def test_w1_matches_optimal_cost():
    ep, occ = _small_static()
    opt = _optimal(ep, occ)
    assert opt.found
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=1.0)
    assert foc.found
    assert len(foc.actions) == len(opt.actions)  # w=1 is optimal


def test_w2_respects_suboptimality_bound():
    ep, occ = _small_static()
    opt = _optimal(ep, occ)
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert foc.found
    assert len(foc.actions) <= 2 * len(opt.actions)  # bounded suboptimal


def test_completeness_and_interface():
    ep, occ = _small_static()
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert isinstance(foc, PlanResult)
    assert foc.found
    assert len(foc.actions) >= 1
    assert foc.expansions > 0


def test_determinism():
    ep, occ = _small_static()
    a = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    b = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert a.actions == b.actions
    assert a.expansions == b.expansions


def test_perfect_signal_beats_uninformative_at_same_w():
    # The intended benefit: at a fixed w, a perfect focal ordering expands no more nodes
    # than an uninformative (zero) one. (vs-plain-A* only holds at small w, since a wide
    # band admits non-optimal nodes regardless of ordering — see test below.)
    ep, occ = _small_static()
    dist = R.compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
    gx, gy = ep.goal
    maxt = occ["blocked"].shape[0] - 1

    def perfect_delta(states):
        out = []
        for x, y, t_rel in states:
            d = int(dist[min(t_rel, maxt), x, y])
            out.append(1e9 if d >= R.INF else float(max(0, d - R.manhattan(x, y, gx, gy))))
        return out

    for w in (1.5, 2.0):
        fp = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, perfect_delta, w=w)
        fz = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=w)
        assert fp.found
        assert fp.expansions <= fz.expansions


def test_perfect_signal_beats_astar_at_small_w():
    ep, occ = _small_static()
    dist = R.compute_true_cost_to_goal(occ["blocked"], ep.goal, ep.max_steps)
    gx, gy = ep.goal
    maxt = occ["blocked"].shape[0] - 1

    def perfect_delta(states):
        out = []
        for x, y, t_rel in states:
            d = int(dist[min(t_rel, maxt), x, y])
            out.append(1e9 if d >= R.INF else float(max(0, d - R.manhattan(x, y, gx, gy))))
        return out

    astar = R.space_time_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, alpha=1.0)
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, perfect_delta, w=1.25)
    assert foc.found
    assert foc.expansions <= astar.expansions


def test_no_double_expansion():
    ep, occ = _small_static()
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert len(set(foc.closed)) == len(foc.closed)


def test_bound_holds_with_dynamics():
    ep = R.make_episode(11, "A", 24, 120, 1, 1, 1)  # gates/pats/drifts present
    occ = R.simulate_occupancy(ep.walls, ep.gates, ep.pats, ep.drifts, ep.max_steps)
    opt = R.space_time_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, alpha=1.0)
    foc = R.space_time_focal_astar(ep.start, ep.goal, 0, 40, 20000, occ, _zero_delta, w=2.0)
    assert isinstance(foc, R.PlanResult)
    if opt.found and foc.found:
        assert len(foc.actions) <= 2 * len(opt.actions)
