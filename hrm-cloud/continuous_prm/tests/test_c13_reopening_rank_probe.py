from __future__ import annotations

import numpy as np

import continuous_prm_c13_lhbl_c7_comparison as X
import continuous_prm_c13_reopening_rank_probe as P
import continuous_prm_c13_shared_queue as Q


def test_reopening_rank_repairs_better_g_after_expansion() -> None:
    graph = [[] for _ in range(4)]
    for a, b, cost in (
        (0, 2, 1.05),
        (0, 3, 1.0),
        (3, 2, 0.01),
        (2, 1, 1.0),
    ):
        graph[a].append((b, cost))
        graph[b].append((a, cost))
    rank = np.array([0.0, 100.0, 0.0, 10.0])
    no_reopen = X.astar_with_path(graph, rank, 8)
    reopened = P.reopening_rank_astar(graph, rank, 8)
    assert no_reopen["cost"] == 2.05
    assert reopened["cost"] == 2.01
    assert reopened["reexpansions"] == 1
    assert reopened["max_expansions_per_state"] == 2
    assert Q.validate_path(graph, reopened["path"], reopened["cost"])["valid"]
