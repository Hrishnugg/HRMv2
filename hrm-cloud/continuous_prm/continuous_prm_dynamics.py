"""Deterministic moving circular obstacles + time-feasibility checks (C8 dynamics).

Pure geometry over numpy; agent is a point (MVP)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def _tri(x: float) -> float:
    """Triangle wave with period 1, range [0,1]: 0->1 over [0,0.5], 1->0 over [0.5,1]."""
    x = x - np.floor(x)
    return float(2.0 * x if x < 0.5 else 2.0 * (1.0 - x))


def _tri_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized triangle wave (period 1, range [0,1]); matches ``_tri`` elementwise."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.floor(x)
    return np.where(x < 0.5, 2.0 * x, 2.0 * (1.0 - x))


@dataclass
class MovingCircle:
    ax: float
    ay: float
    bx: float
    by: float
    period: float
    radius: float

    def center_at(self, t: float) -> np.ndarray:
        frac = _tri(t / self.period) if self.period > 0 else 0.0
        return np.array([self.ax + (self.bx - self.ax) * frac,
                         self.ay + (self.by - self.ay) * frac], dtype=np.float64)

    def centers_at(self, ts: np.ndarray) -> np.ndarray:
        """Vectorized ``center_at`` over an array of times ``ts`` (S,) -> (S, 2).

        Triangle-wave interpolation between endpoints A and B; ``period <= 0``
        yields ``frac == 0`` (stationary at A), matching ``center_at``.
        """
        ts = np.asarray(ts, dtype=np.float64)
        if self.period > 0:
            fracs = _tri_vec(ts / self.period)
        else:
            fracs = np.zeros(ts.shape, dtype=np.float64)
        out = np.empty(ts.shape + (2,), dtype=np.float64)
        out[..., 0] = self.ax + (self.bx - self.ax) * fracs
        out[..., 1] = self.ay + (self.by - self.ay) * fracs
        return out


class Dynamics:
    def __init__(self, circles: Sequence[MovingCircle]):
        self.circles: List[MovingCircle] = list(circles)

    def point_free(self, p: np.ndarray, t: float) -> bool:
        for c in self.circles:
            if np.linalg.norm(np.asarray(p, dtype=np.float64) - c.center_at(t)) <= c.radius:
                return False
        return True

    def node_free(self, p: np.ndarray, t0: float, t1: float, samples: int = 8) -> bool:
        if not self.circles:
            return True
        p = np.asarray(p, dtype=np.float64)
        ts = np.linspace(t0, t1, max(2, samples))
        for c in self.circles:
            centers = c.centers_at(ts)  # (S, 2)
            d = np.linalg.norm(p[None, :] - centers, axis=1)  # (S,)
            if (d <= c.radius).any():
                return False
        return True

    def edge_free(self, a: np.ndarray, b: np.ndarray, t0: float, t1: float, samples: int = 16) -> bool:
        if not self.circles:
            return True
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        ts = np.linspace(t0, t1, max(2, samples))
        if t1 <= t0:
            fracs = np.zeros(ts.shape, dtype=np.float64)
        else:
            fracs = (ts - t0) / (t1 - t0)
        pos = a[None, :] + (b - a)[None, :] * fracs[:, None]  # (S, 2)
        for c in self.circles:
            centers = c.centers_at(ts)  # (S, 2)
            d = np.linalg.norm(pos - centers, axis=1)  # (S,)
            if (d <= c.radius).any():
                return False
        return True
