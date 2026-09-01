# Reachable supervised-label recount (C8 dynamic training pipelines)

Slots = usable worlds x 192 nodes x 111 time steps (all worlds verified at exactly 192 nodes and t_max=110). Reachable = finite backward space-time Dijkstra value (the training loss mask).

| Pipeline | Worlds (mz/rm/sp) | Slots | Reachable | % | Per-world reachable (min/mean/max) |
|---|---|---|---|---|---|
| canonical_1234 | 53 (17/19/17) | 1,129,536 | 821,850 | 72.8% | 13,028 / 15,507 / 17,322 |
| seed2001 | 139 (41/50/48) | 2,962,368 | 2,115,331 | 71.4% | 12,282 / 15,218 / 17,025 |
| seed2002 | 149 (40/54/55) | 3,175,488 | 2,257,512 | 71.1% | 10,932 / 15,151 / 17,705 |

Gate: per-suite usable-world counts reproduce each run's train_manifest.json exactly (asserted at runtime).
