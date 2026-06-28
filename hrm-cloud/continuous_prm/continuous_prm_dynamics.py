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


class Dynamics:
    def __init__(self, circles: Sequence[MovingCircle]):
        self.circles: List[MovingCircle] = list(circles)

    def point_free(self, p: np.ndarray, t: float) -> bool:
        for c in self.circles:
            if np.linalg.norm(np.asarray(p, dtype=np.float64) - c.center_at(t)) <= c.radius:
                return False
        return True

    def node_free(self, p: np.ndarray, t0: float, t1: float, samples: int = 8) -> bool:
        ts = np.linspace(t0, t1, max(2, samples))
        return all(self.point_free(p, float(t)) for t in ts)

    def edge_free(self, a: np.ndarray, b: np.ndarray, t0: float, t1: float, samples: int = 16) -> bool:
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        ts = np.linspace(t0, t1, max(2, samples))
        for t in ts:
            frac = 0.0 if t1 <= t0 else (float(t) - t0) / (t1 - t0)
            pos = a + (b - a) * frac
            if not self.point_free(pos, float(t)):
                return False
        return True
