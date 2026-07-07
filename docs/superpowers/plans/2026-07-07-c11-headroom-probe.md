# C11 G0-H Headroom Probe — Spec + Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Probe-scale: 4 tasks. Design user-approved 2026-07-07 (inline); derived from `hrm-cloud/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md` §7.

**Goal:** MEASURE whether compositional missions on the existing hard maps create real heuristic headroom — the go/no-go gate before building the C11 phase. No learning anywhere; CPU-only; ~a day.

**Gate (pre-registered):** the exact oracle must cut **≥40–50% of A\* expansions** vs the strong admissible `leg-sum` baseline at the binding budget on ≥1 config — ideally with the gap **growing** in `n_stages` (the dose-response signature). PASS → build C11. FAIL → reject the substrate for the cost of a probe.

**Branch:** `c11-headroom`. New files ONLY: `hrm-cloud/continuous_prm/continuous_prm_c11_headroom.py` + `hrm-cloud/continuous_prm/tests/test_c11_headroom.py`. NEVER modify `continuous_prm_common.py` / `transfer_astar_*` / any existing module. Reuse by import: `continuous_prm_common as C` (worlds, PRM, INF), `continuous_prm_c7_hard_maps as H7` (`install_c7_hard_maps()` → specs incl `C_hard_maze`, `C_hard_rooms_large`).

---

## Core definitions (shared by all tasks — implement EXACTLY)

**Mission.** Given a world + roadmap (`C.build_world(spec, seed, 0.45)`, `C.build_prm(world, C.RoadmapConfig(n_nodes=192, k_neighbors=7), seed+17)`; start=node 0, goal=node 1): sample K waypoints as **distinct roadmap node indices** (exclude 0 and 1), drawn seeded from nodes connected to the goal (`rm.connected_to_goal`), min pairwise euclidean separation ≥ 0.25·side (retry ≤ 200, relax to 0.15·side if unfillable, else skip world). Mission = visit `wp[0], wp[1], …, wp[K-1]` in order, then reach the goal (node 1). Waypoints are node indices → no roadmap changes needed.

**Product graph.** State `(i, s)`, `s ∈ 0..K` = number of waypoints completed. Moving along roadmap edge `i→j` (cost = euclidean edge length from `rm.adj`) yields `s' = s+1 if (s < K and j == wp[s]) else s`. Goal state: `(1, K)` — reaching node 1 only terminates when `s == K` (passing through node 1 earlier is allowed, no transition). Waypoints are distinct so no cascaded transitions.

**Heuristics (both admissible by triangle inequality):**
- `h_next(i, s) = ‖p_i − p_{tgt(s)}‖` where `tgt(s) = wp[s]` if `s < K` else node 1.
- `h_legsum(i, s) = ‖p_i − p_{tgt(s)}‖ + Σ_{t=s}^{K-1} ‖p_{next_t} − p_{next_{t+1}}‖` with the chain `wp[s] → wp[s+1] → … → wp[K-1] → node 1` (straight-line remaining legs). `h_legsum ≥ h_next`, both ≤ h\*.
- `h_oracle(i, s) = h*(i, s)`: exact cost-to-go via **backward Dijkstra on the product graph** from `(1, K)`: predecessors of `(j, s')` are `(i, s)` such that edge `i↔j` exists (stage-valid — Task 3 adds stage-dependent edges) and the forward transition rule maps `(i,s)→(j,s')`. Implement with `heapq` over `(K+1)·N ≤ 1728` states; unreachable = `C.INF`.

**Product A\*:** mirror `C.astar_search`'s contract (budget-limited expansions, returns `{found, cost, expansions, closed}`) but keyed on `(node, stage)` with an `h(i, s)` callable and a stage-aware adjacency accessor. Tie-break by g (or insertion) — match `C.astar_search`'s conventions where sensible (read it first).

**Keys→doors (config C).** `D = 2` doors: axis-aligned rectangles placed on the straight line between `wp[d]` and the NEXT mission target (midpoint-centered, half-width ≈ 0.02·side across the corridor axis, half-height ≈ 0.10·side perpendicular), for d ∈ {0, 1}. Door `d` is OPEN once `s > d` (its "key" is waypoint d). Stage-dependent adjacency: precompute per-door blocked edge set (segment-rectangle intersection on roadmap edges — implement a local `_segment_intersects_rect`; do NOT modify the world/obstacles); `adj_valid(s)` masks edges blocked by any door `d ≥ s`... precisely: door d blocked for stages `s ≤ d`. **Validation per world:** each door must block ≥ 3 roadmap edges AND the product goal must remain reachable from (0,0) (oracle finite at start) — else resample doors (≤ 20 tries) then skip world. Doors must NOT overlap waypoint or start/goal node positions.

**Measurement protocol (mirrors the C-series convention):** per (config, K): 25 valid worlds (seeded, `seed = 1234 + 7919·world_idx + 104729·config_idx + 15485863·K`); budgets grid `[100, 200, 400, 800, 1600, 3200]`; **binding budget** = lowest with `h_legsum` success ≥ 0.05 across the cell's worlds (if none qualifies, use the largest and flag DEGENERATE). At the binding budget report per arm (`h_next`, `h_legsum`, `h_oracle`): success rate, median expansions, and the **matched ratio** `oracle_expansions / legsum_expansions` on both-solved worlds (median + IQR) plus the success gap (oracle_succ − legsum_succ).

---

## Task 1: Mission layer + product graph + exact oracle (+ tests) — the ground truth

**Files:** create module + test file (headers, config dataclass `C11ProbeConfig`, `sample_mission`, `product_oracle`, `h_next`, `h_legsum`).
TDD tests (all CPU, small): (1) oracle at goal state = 0; (2) forward-consistency: greedy descent on h\* from (0,0) reaches (1,K) with total cost ≈ h\*(0,0) (±1e-6); (3) **admissibility on 200 sampled states: `h_next ≤ h_legsum ≤ h_oracle + 1e-9`** (the load-bearing invariant — if legsum ever exceeds the oracle, the mission/oracle wiring is wrong); (4) monotone in K: h\*(0,0) for K=4 ≥ h\*(0,0) for K=2 on the same world/waypoint-prefix; (5) waypoint transition rule: oracle at (wp[0], 0) equals oracle at (wp[0], 1) + 0 (arrival transition consistency — define precisely per your implementation and assert it).
Commit: `feat(c11-probe): mission layer + product graph + exact backward-Dijkstra oracle`.

## Task 2: Product A\* + calibration + matched eval (+ tests)

`astar_product(adj_accessor, h_fn, budget, K, wp, start=(0,0))`; `calibrate_binding_budget(cells)`; `eval_cell(config, K, n_worlds, budgets)` returning per-arm records. Tests: A\* with h_oracle finds cost == h\*(0,0) with the FEWEST expansions of the three arms on 3 worlds; A\* with h_next/h_legsum also optimal (admissible ⇒ optimal) when found within budget; budget exhaustion returns found=False.
Commit: `feat(c11-probe): product A* + binding-budget calibration + matched three-arm eval`.

## Task 3: Keys→doors config (+ tests)

Stage-dependent adjacency + door placement/validation per the spec above. Tests: door blocks ≥3 edges; oracle finite at start after placement; **oracle with doors ≥ oracle without doors** on the same world/mission (doors only remove edges); an edge blocked at s=0 becomes valid at s=2; h_legsum still ≤ h_oracle with doors (admissibility preserved — legsum ignores doors, doors only increase h\*).
Commit: `feat(c11-probe): keys->doors stage-dependent adjacency + placement validation`.

## Task 4: Probe runner + report + gate verdict

Run configs A (`C_hard_maze` waypoints), B (`C_hard_rooms_large` waypoints), C (`C_hard_maze` + doors) × K ∈ {2,4,8} × 25 worlds (CPU; parallelize worlds with `concurrent.futures` if trivial, else serial — estimate first, keep the full probe < 2h wall). Write `hrm-cloud/continuous_prm/C11_HEADROOM.md`: per-cell table (binding budget, per-arm success/median expansions, matched oracle/legsum ratio + IQR, success gap), the **dose-response read** (ratio & gap vs K per config), the pre-registered gate verdict (PASS/FAIL: oracle ratio ≤ 0.5–0.6 on ≥1 config; note trend), honest caveats (probe scale, waypoint sampling choices), and the decision implication (build C11 / reject). Also drop the raw records CSV under `runs/c11_probe/`. Commits: runner (`feat`), then results doc + CSV (`docs(c11-probe): G0-H headroom verdict`).

---

**Self-review:** transition rule, heuristic chain, oracle direction, and door-stage semantics are each pinned by a dedicated test; the admissibility chain test (T1.3) catches most wiring errors; measurement mirrors the validated C-series matched-eval convention; no placeholder steps; nothing touches frozen modules.
