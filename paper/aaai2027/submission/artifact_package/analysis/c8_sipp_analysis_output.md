# SIPP baseline on the confirmation cohort

All 300 instances pass the correctness gate (SIPP arrival ==
space-time optimal arrival; unsolved iff optimum infinite).

| Suite | SIPP succ | SIPP succ@binding | Interval-exp med | t interval build (s) | t search (s) | Learned succ |
|---|---|---|---|---|---|---|
| Crossing | 1.00 | 1.00 | 80 | 1.20 | 0.128 | 0.92 |
| Maze | 0.98 | 0.98 | 254 | 0.96 | 0.210 | 0.96 |
| Dense maze | 0.94 | 0.94 | 290 | 1.44 | 0.234 | 0.70 |
| Rooms | 1.00 | 1.00 | 195 | 0.76 | 0.150 | 1.00 |
| Large rooms | 1.00 | 1.00 | 105 | 1.66 | 0.144 | 1.00 |
| Spiral | 1.00 | 1.00 | 282 | 0.73 | 0.162 | 1.00 |

Interval-state expansions are a different unit from (v,t) expansions and are never merged into the space-time columns. SIPP is optimal for earliest arrival on this substrate; its success at the binding thresholds is at or above every other arm on every suite.
