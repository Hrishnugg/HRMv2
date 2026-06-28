import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "continuous_prm"))

import continuous_prm_dynamics as D


def test_moving_circle_triangle_wave():
    # endpoints A=(0,0), B=(1,0), period=4 -> at t=0 at A, t=2 at B, t=4 back at A.
    mc = D.MovingCircle(ax=0.0, ay=0.0, bx=1.0, by=0.0, period=4.0, radius=0.1)
    assert np.allclose(mc.center_at(0.0), [0.0, 0.0], atol=1e-9)
    assert np.allclose(mc.center_at(2.0), [1.0, 0.0], atol=1e-9)
    assert np.allclose(mc.center_at(4.0), [0.0, 0.0], atol=1e-9)


def test_node_wait_blocked_when_circle_overlaps():
    mc = D.MovingCircle(ax=0.0, ay=0.0, bx=2.0, by=0.0, period=4.0, radius=0.3)
    dyn = D.Dynamics([mc])
    node = np.array([1.0, 0.0])  # circle passes through here at t=1
    assert dyn.node_free(node, 1.0, 2.0, samples=8) is False
    assert dyn.node_free(node, 0.0, 0.2, samples=8) is True


def test_edge_sweep_blocked_when_circle_crosses():
    mc = D.MovingCircle(ax=1.0, ay=-1.0, bx=1.0, by=1.0, period=4.0, radius=0.25)
    dyn = D.Dynamics([mc])
    a = np.array([0.0, 0.0]); b = np.array([2.0, 0.0])  # horizontal edge through x=1
    assert dyn.edge_free(a, b, t0=0.0, t1=2.0, samples=16) is False
    assert dyn.edge_free(a, b, t0=2.0, t1=2.2, samples=16) is True


# --- scalar reference implementations (pre-vectorization semantics) ---------

def _node_free_scalar(dyn, p, t0, t1, samples=8):
    """Reference: per-sample Python loop using ``center_at`` (closed boundary)."""
    p = np.asarray(p, dtype=np.float64)
    ts = np.linspace(t0, t1, max(2, samples))
    for t in ts:
        for c in dyn.circles:
            if np.linalg.norm(p - c.center_at(float(t))) <= c.radius:
                return False
    return True


def _edge_free_scalar(dyn, a, b, t0, t1, samples=16):
    """Reference: per-sample Python loop using ``center_at`` (closed boundary)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ts = np.linspace(t0, t1, max(2, samples))
    for t in ts:
        frac = 0.0 if t1 <= t0 else (float(t) - t0) / (t1 - t0)
        pos = a + (b - a) * frac
        for c in dyn.circles:
            if np.linalg.norm(pos - c.center_at(float(t))) <= c.radius:
                return False
    return True


def test_centers_at_matches_center_at():
    rng = np.random.default_rng(7)
    for _ in range(200):
        period = float(rng.choice([0.0, -1.0, rng.uniform(0.5, 8.0)]))
        c = D.MovingCircle(
            ax=float(rng.uniform(-3, 3)), ay=float(rng.uniform(-3, 3)),
            bx=float(rng.uniform(-3, 3)), by=float(rng.uniform(-3, 3)),
            period=period, radius=float(rng.uniform(0.05, 0.5)),
        )
        ts = np.linspace(rng.uniform(-5, 0), rng.uniform(0, 5), int(rng.integers(2, 20)))
        vec = c.centers_at(ts)
        for i, t in enumerate(ts):
            assert np.array_equal(vec[i], c.center_at(float(t)))


def test_vectorized_matches_scalar_reference():
    """Vectorized edge_free/node_free must be byte-identical (boolean) to the
    scalar per-sample reference across many random circle configs and queries."""
    rng = np.random.default_rng(1234)
    n_cases = 0
    for _ in range(400):
        n_circ = int(rng.integers(0, 4))  # include the empty-Dynamics case
        circles = []
        for _ in range(n_circ):
            period = float(rng.choice([0.0, rng.uniform(0.5, 6.0)]))
            circles.append(D.MovingCircle(
                ax=float(rng.uniform(-2, 2)), ay=float(rng.uniform(-2, 2)),
                bx=float(rng.uniform(-2, 2)), by=float(rng.uniform(-2, 2)),
                period=period, radius=float(rng.uniform(0.05, 0.8)),
            ))
        dyn = D.Dynamics(circles)

        a = rng.uniform(-2, 2, size=2)
        b = rng.uniform(-2, 2, size=2)
        p = rng.uniform(-2, 2, size=2)
        t0 = float(rng.uniform(-3, 3))
        # mix of t1>t0, t1==t0, t1<t0 to exercise the frac rule
        dt = float(rng.choice([0.0, -rng.uniform(0, 2), rng.uniform(0, 4)]))
        t1 = t0 + dt
        e_samples = int(rng.integers(1, 24))
        n_samples = int(rng.integers(1, 24))

        assert dyn.edge_free(a, b, t0, t1, samples=e_samples) == \
            _edge_free_scalar(dyn, a, b, t0, t1, samples=e_samples)
        assert dyn.node_free(p, t0, t1, samples=n_samples) == \
            _node_free_scalar(dyn, p, t0, t1, samples=n_samples)
        n_cases += 1
    assert n_cases == 400
