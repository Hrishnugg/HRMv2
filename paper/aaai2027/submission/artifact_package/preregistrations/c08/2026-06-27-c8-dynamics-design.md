# C8 — Dynamics (time-varying obstacles): design spec

**Date:** 2026-06-27
**Status:** design approved; ready for implementation plan
**Scope:** Phase 2 of the continuous-PRM program. Brings deterministic moving obstacles into the continuous-PRM learned-heuristic line, extending the C7 integration comparison into the time domain and spotlighting temporal/hierarchical heuristics. Cluster-scale confirmation and richer dynamics (drifters, stochastic/predicted motion) are explicit follow-ons.

Related docs:
- `docs/experiments/continuous/c07/results/C7_RESULTS.md` + `docs/experiments/continuous/c07/design/2026-06-27-c7-integration-comparison-design.md` (the static integration comparison this extends).
- `docs/experiments/continuous/c06/results/C6_RESULTS.md` (value-field stage).
- `docs/experiments/discrete/learned-heuristic/results/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md` (discrete space-time focal A\*ε — the dynamics reference).

---

## 1. Motivation

The continuous-PRM line is validated **statically** (C7: additive learned heuristics cut A\* expansions 15–48% over Euclidean and generalize). The natural next frontier is **dynamics** — time-varying obstacles — which is (a) more realistic, (b) where the discrete line operated, and (c) where a **recurrent/hierarchical** heuristic should pull ahead, because cost-to-go now depends on *temporal phase* (when you arrive at a node determines whether a route is open). C8 extends the C7 comparison into the time domain and makes dynamics the regime that tests the hierarchical-model thesis on its home turf.

**Two goals (both):** (1) extend C7's rigorous integration comparison (additive vs focal vs value-field × HRM/ON-LSTM/U-Net, matched, McNemar/BH + Wilcoxon/CI) to moving obstacles; (2) spotlight whether **time-aware** learned heuristics beat **time-blind** ones (and the geometric baseline) when timing matters.

## 2. Goals / non-goals

**Goals**
- A matched space-time comparison on roadmaps with deterministically moving circular obstacles, time-aware learned heuristics, the same arms + a new time-blind ablation.
- Primary metric expansions on matched-solved `(node,t)` states; secondary success; first-class makespan suboptimality.
- A graded OOD story (density / structural / scale) under dynamics.
- Runs locally on the RTX 5090 at validation scale; cluster-ready via config (no code change), as in C7.
- A publication-grade writeup (`C8_RESULTS.md`).

**Non-goals (this phase)**
- Stochastic / observed-only motion requiring a learned **motion predictor** (motion is known/deterministic to the planner here; the only learned component is the heuristic). Deferred follow-on.
- SIPP / continuous-time search (we use time-discretized space-time A\* for harness reuse; SIPP is a future efficiency swap).
- New backbones beyond HRM / ON-LSTM / U-Net.
- Cluster headline run (conditional, after local validation).

## 3. Planner & occupancy model

**Moving obstacles.** Each is a circle with a deterministic, known trajectory. MVP: linear back-and-forth "patrollers" — `center(t) = A + (B−A)·tri(t/period)` (triangle wave), fixed radius. Position at any `t` is closed-form, so the planner has exact future occupancy; **no prediction**.

**Roadmap stays static.** The PRM is built on the **static** obstacles only (reuse `continuous_prm_common.build_prm` unchanged) — it captures static free-space connectivity. Moving circles are not baked into edges; they make nodes/edges **time-dependently infeasible** at plan time. The roadmap must be dense enough to detour around a blocked region (192 nodes / k=7, as in C7).

**Search = time-discretized space-time A\* over `(node, t)`** (integer steps of size `Δt`):
- **Move** `(u,t) → (v, t+τ)`, `τ = ⌈len(u,v)/(v_agent·Δt)⌉` steps, admitted iff sweeping the agent (a point, MVP) along `u→v` over `[t, t+τ]` stays clear of every moving circle (sampled at sub-steps; static clearance already guaranteed by the PRM edge).
- **Wait** `(u,t) → (u,t+1)`, admitted iff node `u` is clear of all moving circles over `[t,t+1]`.
- **Cost = arrival time (makespan):** `g(node,t)=t`. Goal = reach the goal node at any `t ≤ T_max`, within an expansion budget. Metric = expansions over `(node,t)` states (the C7 metric, time-expanded).

**Oracle & training labels.** `h*(node,t)` = true minimum time-to-go, computed once per world by a **backward space-time Dijkstra** from the goal over the time-expanded graph (bounded by `T_max`). Serves as the **oracle ceiling arm** and the **supervised target** for the learned heuristics. (This is the dynamic analog of the static `dijkstra_to_goal`.)

**Admissible baseline.** `euclid_time(node) = euclid(node,goal)/v_agent` — a true lower bound on time-to-go (can't beat straight-line at max speed, ignoring obstacles). It is the Euclid arm and the admissible ordering for focal; it is **consistent** on this graph (including across wait edges: `euclid_time(u) ≤ 1 + euclid_time(u)`), so the A\*ε bound holds.

## 4. Time-aware heuristic representation

**Interface.** C7 providers returned `h[node]`. C8 providers return a precomputed **table `h[node, t]`** (nodes × steps `0…T_max`), built once per world, so the A\* loop does O(1) lookups — **no model calls inside the search**.

**Per-arm computation:**
- **Euclid-time** (baseline): `h[node,t] = euclid_time(node)`, t-independent, admissible. No model.
- **Oracle** (ceiling): `h[node,t] = h*(node,t)` from the backward space-time Dijkstra.
**Time normalization.** All learned heuristics are additive on `euclid_time` with a normalized residual converted to time units by the map-crossing time `T_scale = side_len / v_agent` (the time analog of C7's `side_len` length-normalization). The supervised target is the normalized extra time beyond the admissible baseline: `residual_target(node,t) = clip(h*(node,t) − euclid_time(node), 0, ·) / T_scale`.

- **Scalar / sequence** (HRM, ON-LSTM): for each node, **one recurrent forward pass over the dynamics rollout** (the moving circles' local occupancy across future frames from the query node) emits a **time-to-go estimate per timestep** — the whole `h[node, ·]` row in one pass. `h[node,t] = euclid_time(node) + T_scale·clip(residual[node,t], 0, B)`. N forward passes per world (N=192), cached. *This is the temporal showcase*: the recurrent model consumes the future-occupancy sequence and reasons about phase.
- **Value field** (U-Net + field-HRM/ON-LSTM): precompute a **field per time step** — a forward pass on the occupancy grid rendered at `t`, with a short future window of frames as extra channels (so the field "sees" where obstacles head) → cost-to-go field; `h[node,t] = euclid_time(node) + T_scale·sample(field_t, node)`. `T_max+1` passes per world, cached.
- **Time-blind ablation** (for HRM, and optionally each learned arm): the *same* model fed only the **current snapshot** at `t` (no future-rollout window / a single frame) instead of the rollout. Tests whether temporal-phase reasoning — not just per-frame occupancy — is what helps.

**Focal.** OPEN ordered by `f = g + euclid_time` (g = arrival time); band `{f ≤ w·f_min}` expanded by minimum learned `h[node,t]`. The proven `≤ w` makespan bound carries over (as in the discrete space-time focal). Collapsed/uninformative `h` degrades to plain space-time A\* on `euclid_time`.

**Admissibility/honesty.** Additive scalar/field `h` may exceed true time-to-go (mildly inadmissible) → makespan suboptimality is recorded per instance (`arrival/optimal`); focal carries the `≤ w` bound. Nonfinite `h` fails loudly with a count (the discrete-residual lesson).

## 5. Dynamic suites & calibration

**Suites** = static C7 layouts + patroller circles placed to force timing decisions (a patroller periodically sweeps a corridor/gate → wait-or-detour). 
- **Train (in-dist):** `C_dyn_maze`, `C_dyn_rooms`, `C_dyn_spiral`.
- **Held-out (graded OOD):** `C_dyn_maze_dense` (more/faster patrollers — density-OOD), `C_dyn_crossing` (open arena, crossing patrollers — pure-timing structural-OOD), `C_dyn_rooms_large` (scale-OOD).
- Difficulty knobs: patroller count, speed/period, radius, placement.

**World validity:** keep a world only if **solvable in space-time** — the backward space-time Dijkstra reaches the start within `T_max`. Reject permanent traps. (Built by adding a patroller layer to the C7 world generator.)

**Calibration (Gate 1):** sweep `(budget, T_max, v_agent, patroller density)` running only Euclid-time + Oracle so Euclid-time sits in a binding band (waiting/detours make it expand heavily or miss within budget) while the oracle solves efficiently — headroom = the Euclid→oracle expansion gap in space-time. Extends C7's `calibrate` mode; writes `calibration.json` of the same shape.

## 6. Eval, metrics, arms, statistics

**Arms** (matched — same worlds + same seeded patroller trajectories): Euclid-time / additive-scalar{hrm,onlstm} / value-field{unet,onlstm,hrm} / focal × those + oracle, **plus the time-blind ablation** arm(s).

**Metrics:** primary = expansions on matched-solved `(node,t)`; secondary = success within budget; honesty = makespan suboptimality `arrival/optimal` (focal ≤ w; additive reports actual); gap-to-ceiling vs oracle; the time-aware-vs-time-blind delta.

**Statistics:** reuse C7's `mcnemar_exact_p` + `bh_q_values` (success, learned arms only) + paired Wilcoxon (ratio-space) + seeded bootstrap CI (expansions). Multiplicity disclosed as in C7.

**Pre-registered comparisons:**
1. time-aware learned vs Euclid-time (expansions + success), per suite.
2. **time-aware vs time-blind** (same backbone) — the spotlight: does temporal-phase reasoning help?
3. additive vs focal under dynamics — does C7's "additive wins on a weak baseline" hold when the baseline is `euclid_time`?
4. recurrent (HRM / ON-LSTM) vs field-U-Net — do temporal models win when timing matters?
5. learned vs oracle — gap-to-ceiling.
6. in-distribution vs each held-out OOD axis.

## 7. Scope, MVP, module structure

**New code** under `hrm-cloud/continuous_prm/`, mirroring the C7 layout:
- `continuous_prm_dynamics.py` *(new)* — moving-obstacle model (`MovingCircle` trajectory + `occupancy_at(t)`), rollout, and time-feasibility checks (edge sweep, node wait).
- `continuous_prm_spacetime.py` *(new)* — `space_time_astar_prm(...)` + `space_time_focal_prm(...)` (pure search over the time-expanded graph; mirror `continuous_prm_focal.py` shape) + `backward_spacetime_dijkstra(...)` (oracle + labels).
- `continuous_prm_c8_dynamic_maps.py` *(new)* — the dynamic suites via the runtime-install pattern (compose on the C7 static suites + add patrollers).
- `continuous_prm_dynamic_providers.py` *(new)* — time-aware providers returning `h[node,t]` tables (Euclid-time, Oracle, ScalarResidual, ValueField, time-blind ablation) + `run_world_arms_spacetime`.
- `continuous_prm_c8_dynamics_compare.py` *(new)* — orchestrator (collect/train/eval/calibrate/analyze/full + scale presets), reusing C7's stats and analyze structure.

**Reuse:** static `build_prm`; the C5/C6 model architectures (extended to ingest the rollout sequence / occupancy stack); C7's stats (`mcnemar_exact_p`/`bh_q_values`/Wilcoxon/bootstrap) and analyze/preregistered structure; the provider/arm/matched-eval pattern; scale presets.

**MVP-first, staged gates (as C7):**
- **Gate 0** — units: space-time A\* + focal (bound/optimality/wait/no-double-expansion tests), `backward_spacetime_dijkstra` == brute-force on tiny graphs, time-feasibility checks, providers (`h[node,t]` shapes + euclid-time admissibility + oracle==backward-Dijkstra), dynamic-suite validity + matched-integrity.
- **Gate 1** — calibrate the binding band.
- **Gate 2** — local validation run (all suites + all arms incl. time-blind ablation) + directional sanity gate.
- **Gate 3** *(conditional)* — cluster scale-up.
- Then `C8_RESULTS.md` writeup. **Subagent-driven** execution.

## 8. Testing strategy

- **Space-time search:** `w=1` optimality (min makespan), `w>1` bound never violated, completeness (finds a path iff space-time-reachable within T_max as budget→∞), determinism, no double-expansion of a closed `(node,t)`, wait-action correctness, collapsed-`h` degrades to A\*.
- **Oracle:** `backward_spacetime_dijkstra` matches a brute-force space-time BFS/Dijkstra on small hand-built graphs; oracle heuristic makes space-time A\* expand minimally.
- **Feasibility:** edge-sweep and node-wait checks correctly admit/reject against a known moving circle (analytic cases).
- **Providers:** `h[node,t]` table shapes; Euclid-time admissibility (`≤ h*`); Oracle == backward-Dijkstra; nonfinite guard; time-blind vs time-aware produce different tables.
- **Suites:** dynamic worlds are solvable-in-space-time (validity), patrollers actually block (timing pressure exists), matched worlds identical across seeds.
- **Stats:** reuse C7's stats tests; add space-time matched-set alignment by `(suite, world_index, budget)`.

## 9. Threats to validity (and mitigations)

- **Δt discretization** approximates continuous motion → choose Δt small enough that sub-step collision sampling is faithful; report the chosen Δt. (SIPP is the exact future swap.)
- **Search-space blow-up** in `(node,t)` → bound by `T_max` and the expansion budget; calibrate so the band is reachable.
- **Field-per-timestep precompute cost** (`T_max+1` passes/world) → cached once per world, out of the A\* loop; acceptable at local scale, parallelizable at cluster scale.
- **Unsolvable / trap worlds** → space-time-solvability validity filter (Section 5).
- **Hidden inadmissibility of additive `h`** → makespan suboptimality first-class (Section 4).
- **Matched comparison integrity** → shared worlds + seeded trajectories + matched-integrity test.
- **p-hacking** → pre-registered comparisons + BH (learned arms) as in C7.
- **Local scale** → conditional cluster Gate 3; design lifts via config.

## 10. Open decisions (deferred, not blocking)

- Exact `Δt`, `T_max`, `v_agent`, patroller density per suite — set by Gate 1 calibration.
- Future-window length `W` for the rollout/field input — small default (e.g. a few frames), tuned at Gate 2.
- Whether the time-blind ablation runs for all learned backbones or just HRM — decided at Gate 2 by wall-clock.
- Cluster Gate 3 — user decides after Gate 2.
- Richer dynamics (drifters, stochastic/predicted motion, agent-radius) — future phase.

## 11. Key file references

- Static PRM / A\* / Dijkstra: `hrm-cloud/continuous_prm/continuous_prm_common.py` (`build_prm` :666, `astar_search` :729, `dijkstra_to_goal` :712).
- C7 provider/arm/focal pattern to mirror: `continuous_prm_providers.py`, `continuous_prm_focal.py`.
- C7 orchestrator/stats to reuse: `continuous_prm_c7_integration_compare.py` (calibrate/eval/analyze, McNemar/BH/Wilcoxon/bootstrap).
- Discrete space-time A\*/focal reference: `hrm-cloud/residual_tasklora_v2.py` (`space_time_astar`, `space_time_focal_astar`).
- Field model architectures: `continuous_prm_c6_heatmap_value_field.py`; scalar/sequence: `continuous_prm_common.py` (HRM/ON-LSTM backbones) + `continuous_prm_c5_hard_obstacle_encoder.py`.
