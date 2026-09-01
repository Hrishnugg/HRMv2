# Weighted-A* baseline on the confirmation cohort

| Suite | $w_h$ | WA* succ | Anchor succ | Blind U-Net succ | Blind$-$WA* dsucc [CI] | WA*/anchor ratio (n) | Blind/WA* ratio (n) | WA* subopt | Blind subopt |
|---|---|---|---|---|---|---|---|---|---|
| Crossing | 1.5 | 1.00 | 0.12 | 0.92 | -0.08 [-0.16,-0.02] | 0.139 (6) | 0.825 (46) | 1.008 | 1.164 |
| Maze | 5 | 0.96 | 0.12 | 0.96 | +0.00 [-0.06,+0.06] | 0.120 (6) | 0.601 (47) | 1.017 | 1.012 |
| Dense maze | 5 | 0.70 | 0.06 | 0.70 | +0.00 [-0.06,+0.06] | 0.106 (3) | 0.982 (34) | 1.006 | 1.008 |
| Rooms | 3 | 0.98 | 0.42 | 1.00 | +0.02 [+0.00,+0.06] | 0.158 (21) | 0.554 (49) | 1.015 | 1.019 |
| Large rooms | 1.5 | 0.98 | 0.82 | 1.00 | +0.02 [+0.00,+0.06] | 0.335 (41) | 0.728 (49) | 1.003 | 1.082 |
| Spiral | 5 | 0.70 | 0.16 | 1.00 | +0.30 [+0.18,+0.42] | 0.567 (8) | 0.217 (35) | 1.013 | 1.011 |

Blind/WA* ratio < 1 means the learned heuristic expands fewer nodes than tuned weighted A* on jointly solved maps. Suboptimality = arrival / optimal arrival.