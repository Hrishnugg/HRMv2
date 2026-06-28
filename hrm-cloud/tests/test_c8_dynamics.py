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
