# C7 — Integration Comparison (publication-grade): design spec

**Date:** 2026-06-27
**Status:** design approved; ready for implementation plan
**Scope:** Phase 1 of a two-phase effort. Phase 1 (this spec) hardens the continuous-PRM learned-heuristic result into a publication-grade, matched, multi-arm comparison. Phase 2 (dynamics — out of scope here) gets its own spec→plan→build cycle.

Related docs:
- `../../../hrm-cloud/continuous_prm/C6_RESULTS.md` — value-field stage (the win this builds on).
- `../../../hrm-cloud/EXPERIMENT_RESULTS_FOCAL_REDESIGN.md` — discrete focal search (ranker integration).
- `../../../hrm-cloud/EXPERIMENT_RESULTS_COMPENDIUM.md` — full program history + methodology.
- `2026-06-23-learned-focal-search-design.md` — the discrete focal design being ported.

---

## 1. Motivation

The continuous-PRM program has one validated win (C6: a goal-conditioned **value field** beats Euclidean A* on hard maps and rescues HRM, which had failed as a scalar residual in C5) and one validated win on the discrete side (focal search: use the learned signal as a **ranker** inside an admissible band, not an additive magnitude). The program's throughline is that **how the learned signal is integrated / represented — not the model — has been the recurring bottleneck.**

This phase makes that thesis publication-grade by running a single **matched** comparison of all three integration strategies, across backbones, on a difficulty band wide enough to measure the effect, with honest accounting of the optimality trade-off.

### The scientific hook

C5 showed scalar-HRM fails because its predicted magnitude saturated (correction ≈ 0, constant at the residual cap). Focal needs only *ranking*, not magnitude. So focal-on-scalar-HRM is a clean test:
- If focal **rescues** scalar-HRM → **two independent demonstrations** that integration was the bottleneck (focal-ranker *and* field-representation).
- If it **cannot** (the scalar head truly collapsed to a constant, leaving no ranking signal) → the **value-field representation is uniquely necessary** — a sharper claim.

Either outcome is publishable, and the harness yields it without special-casing.

## 2. Goals / non-goals

**Goals**
- A matched, trustworthy comparison of `{Euclidean, additive-scalar, value-field, focal-ranker}` × `{HRM, ON-LSTM, U-Net}` + oracle ceiling, on six hard suites, over a binding-budget sweep.
- Primary metric **expansions on matched-solved instances** (never saturates); secondary **success** (McNemar + BH); first-class **path-suboptimality** so the additive-vs-focal trade-off is visible, not hidden.
- A graded generalization story across **near / structural / scale** OOD axes.
- Runs end-to-end **locally on the RTX 5090** at a validation scale, and lifts to a cluster later via config only (no code change).
- A publication-grade writeup (`C7_RESULTS.md`).

**Non-goals (this phase)**
- Dynamics / time-varying obstacles (Phase 2).
- New backbones beyond HRM / ON-LSTM / U-Net.
- Re-opening the framing (north-star is fixed: a hierarchical learned heuristic beats the algorithmic planner and generalizes).
- A Modal/cluster headline run is **conditional** (Gate 3) and decided after local validation; the design must not depend on it.

## 3. The integration matrix

| Integration ↓ \ Source → | Euclid | HRM | ON-LSTM | U-Net | Oracle |
|---|:--:|:--:|:--:|:--:|:--:|
| none (baseline) | ✓ | | | | |
| additive scalar residual (C5) | | ✓ | ✓ | n/a | |
| value field (C6) | | ✓ | ✓ | ✓ | ✓ |
| focal ranker (A\*ε, new in PRM) | | ✓ | ✓ | ✓ | |

U-Net is inherently a field, so it has no per-node-scalar arm. Every cell is one arm = `(provider, planner-mode)`.

## 4. Architecture (Approach B — heuristic-provider interface)

The planner already consumes a precomputed per-node heuristic array (`astar_search(adj, heuristic, budget, …)`, `continuous_prm_common.py:729`). We formalize "how the learned signal enters the planner" as a strategy boundary.

### 4.1 Provider interface

```
HeuristicProvider:
    name: str
    node_h(world, prm, goal_idx) -> np.ndarray   # per-node cost-to-go estimate; finite, >= 0
```

Providers (each independently unit-testable):
- `EuclidProvider` — straight-line distance to goal (admissible reference).
- `ScalarResidualProvider(model)` — `h = euclid + side_len * clip(yhat, 0, B)`; wraps a C5-style sequence model (HRM / ON-LSTM).
- `ValueFieldProvider(model)` — bilinearly samples the predicted **cost-to-go field** at node positions and returns it directly as `node_h` (mirrors C6's integration); wraps a C6 model (HRM / U-Net / ON-LSTM). Admissibility is not guaranteed — handled by the path-suboptimality metric (additive/direct arms) or the focal bound (focal arm).
- `OracleProvider` — exact Dijkstra cost-to-go (field ceiling).

All learned providers wrap models **trained by C7's `train` mode on the C7 train split** (Section 6.5). The C5/C6 *architectures and code* are reused; their *weights are not* — they were trained on different suites, so reusing them would invalidate the matched comparison.

### 4.2 Planner modes

- `astar_search(adj, h, budget, start, goal)` — existing `f = g + h`.
- `focal_astar_search(adj, euclid_h, rank_h, budget, w, start, goal)` — **new** (Section 5). OPEN ordered by admissible `f = g + euclid_h`; expand the band node minimizing `rank_h`.

Every arm is `(provider, mode)`. The harness always holds `euclid_h`, so focal can wrap any provider's `node_h` as its ranker — this is what lets us test "does focal rescue scalar-HRM."

### 4.3 Module layout (refactor, not greenfield)

- `continuous_prm_common.py` — add `focal_astar_search()` beside `astar_search()`; both pure functions over `adj`.
- `continuous_prm_providers.py` *(new)* — the `HeuristicProvider` classes + unified construction and checkpoint loading for both model families (architectures reused from C5/C6; weights trained fresh by C7's `train` mode).
- `continuous_prm_c7_integration_compare.py` *(new)* — orchestrator: world/PRM generation (extracted from C6), arm enumeration, eval loop, stats, output writers. Modes `collect/train/eval/analyze/full` mirroring C6.

### 4.4 Data flow (matched by construction)

1. Generate eval worlds (seeded) → build one PRM per world with exact Dijkstra labels. **Shared across all arms** — guarantees the comparison is matched.
2. Per world: compute `euclid_h` once; compute each provider's `node_h` once.
3. Per `(arm, budget[, w])`: run the search; record `success`, `expansions`, `path_cost`.
4. Aggregate (Section 6).

### 4.5 Error handling / honesty guards

- **Path-suboptimality is recorded, never hidden.** Additive scalar/field `h` can over-predict → inadmissible → costlier paths. Record `path_cost / dijkstra_optimal` per instance for every arm. Focal carries a proven `<= w` bound; additive does not — the table shows this.
- **Nonfinite `h` fails loudly with a count** (the discrete-residual lesson), not silent zero-fill.
- Missing / non-finite checkpoints, disconnected worlds → explicit errors / world rejection (reuse C5/C6 thresholds).

## 5. Focal search in the PRM

### 5.1 Why it ports cleanly

The PRM is a static weighted graph (`adj: node -> [(neighbor, euclid_weight)]`) — simpler than the discrete space-time graph (no wait action, no time-varying occupancy). Euclidean distance is **admissible and consistent** here: edge weights are straight-line segment lengths, so `euclid(u) <= weight(u,v) + euclid(v)` by the triangle inequality. Consistency is the precondition the A\*ε bound needs.

### 5.2 Algorithm — `focal_astar_search(adj, euclid_h, rank_h, budget, w, start, goal)`

- OPEN ordered by admissible `f = g + euclid_h`; `f_min = min f over OPEN`.
- FOCAL = `{ n in OPEN : f(n) <= w * f_min }`, for `w >= 1`.
- Expand the FOCAL node minimizing `rank_h` (the provider's estimate). Tie-break `(rank_h, f, insertion_counter)` — the discrete fix that stopped oscillation on flat signals.
- Otherwise standard: pop → close → relax neighbors → push with updated `g`. One expansion per pop (identical accounting to `astar_search`, so expansion counts are matched across arms).
- Terminate when goal is popped, budget hit, or OPEN empty.

### 5.3 Guarantee

With consistent `euclid_h` ordering OPEN, A\*ε returns a path of cost `<= w * C*` (optimal). `w = 1` → optimal A\* with learned tie-breaking (zero-risk floor); `w > 1` → bounded-suboptimal, fewer expansions. `w` *replaces* the additive arm's unbounded `alpha`.

### 5.4 Two behaviors that yield the science for free

- Informative `rank_h` → focal greedily follows it inside the safe band → fewer expansions.
- Collapsed `rank_h` (≈ constant, the C5 scalar-HRM failure) → FOCAL selection falls through `rank_h -> f -> counter` and degrades to plain A\* on Euclid → ≈ Euclid, no harm. So focal-on-collapsed-scalar-HRM *automatically* produces the informative null.

### 5.5 Implementation choice

Correctness-first: maintain FOCAL by rescanning OPEN for the band each step — O(|OPEN|) per pop, trivial at 192 nodes, easy to verify against spec. Produces *identical expansion counts* to the optimized two-heap version (same selection rule); expansions, not wall-clock, is the metric. The two-heap optimization remains available if a cluster run needs it.

### 5.6 `w` dial

A swept parameter alongside budget. Sweep `{1.0, 1.05, 1.1, 1.25}` for focal arms; report the expansion / suboptimality trade-off curve. Empirically verify `path_cost / optimal <= w` on every completed instance (test + sanity check).

## 6. Map families and eval design

### 6.1 Suites

Keep the existing three (continuity with C5/C6): `C_hard_maze`, `C_hard_maze_dense`, `C_hard_rooms`.

Add three heuristic-hostile families (large Euclid-vs-true-cost gap or deceptive geometry):
- `C_hard_spiral` — serpentine/spiral corridor; optimal route winds far from the start→goal line. Purest test of capturing global detour structure.
- `C_hard_bugtrap` — concave pockets straddling the straight line; Euclid-guided A\* is lured into dead-ends. Tests steering around deceptive local minima.
- `C_hard_rooms_large` — larger multi-room layout (≈3.0 side, ~8–12 rooms + doorways, longer horizon, several routes). Tests scale + multi-modal routing. **Droppable first** if local wall-clock is tight; spiral and bugtrap are must-haves.

### 6.2 Difficulty calibration (required pre-step — Gate 1)

For each new family, before any model eval, run a quick **Euclid-only + oracle probe** and tune the budget so:
- Euclid sits in the **~40–60%** band (binding), and
- oracle success `< ~0.95` at that budget (headroom for a measurable gap-to-ceiling).

Roadmap stays at **192 / k=7**, grid **64** (avoids the U-Net decoder skip-size crash). Per-family node/k tweaks only if a family will not enter the band.

### 6.3 Implementation

New obstacle-generation modes extending the C5 wall/gate/clutter machinery in `continuous_prm_common.py`: arc/segment walls (spiral), concave pocket assembly (bugtrap), room-grid + doorways (rooms_large). Each parameterized (wall count, gap width, clutter density, side length).

### 6.4 Validity guards (reused)

Start/goal forced to opposite sides (mandatory detours); reject worlds with disconnected start/goal or fewer than `n_nodes/3` connected PRM nodes; seeded generation so every arm sees identical worlds.

### 6.5 Train / held-out split (graded transfer)

- **Train on 3 diverse layouts:** `C_hard_maze`, `C_hard_rooms`, `C_hard_spiral`.
- **Hold out 3 OOD suites on distinct axes:** `C_hard_maze_dense` (*near*-OOD: variant of a trained map), `C_hard_bugtrap` (*structural*-OOD: unseen geometry), `C_hard_rooms_large` (*scale*-OOD: bigger version of a trained map).

Multi-suite training is used (the C6 finding: it closes the OOD gap).

### 6.6 Arms

backbone × integration × mode:
- baseline `Euclid·astar`; ceiling `Oracle·astar`.
- `astar` mode: `{HRM, ON-LSTM}·scalar·astar` (additive residual), `{HRM, U-Net, ON-LSTM}·field·astar` (direct cost-to-go).
- `focal` mode (w-sweep): `{HRM, ON-LSTM}·scalar·focal`, `{HRM, U-Net, ON-LSTM}·field·focal`.

### 6.7 Metrics

1. **Primary — expansions on matched-solved instances** (never saturates). Matched set = instances solved by Euclid; robustness set = solved-by-all-arms. Report median expansion ratio vs Euclid.
2. **Secondary — success rate** within budget.
3. **Honesty — path suboptimality** `= path_cost / dijkstra_optimal` per instance. Focal must be `<= w` (verified); additive arms report actual inflation + fraction of instances returned suboptimal.
4. **Gap-to-ceiling** — learned vs oracle on each metric.
5. **Diagnostic — Spearman(`node_h`, true cost-to-go)** per arm; explains *why* an arm wins/fails and predicts whether focal can help.

### 6.8 Sweeps

Budget = per-suite binding band from 6.2 (~3 in-band budgets at Euclid ≈ 40/55/70%) + one saturating budget as reference. `w ∈ {1.0, 1.05, 1.1, 1.25}` for focal arms.

### 6.9 Statistics

- Success: **McNemar paired** (each arm vs Euclid), **BH-corrected** q across the grid (existing C6 infra).
- Expansions: **paired Wilcoxon signed-rank** on the matched set + **bootstrap CI** on the median ratio.
- **Pre-registered primary comparisons** (everything else is exploratory; guards against p-hacking across the grid):
  1. `field-HRM` vs `Euclid` (expansions + success), per suite.
  2. `scalar-HRM-additive` vs `field-HRM` — the representation lever (C5 fail → C6 win, one table).
  3. `scalar-HRM-focal` vs `scalar-HRM-additive` — the integration lever (does focal rescue the collapsed scalar?).
  4. `field-focal` vs `field-additive` on the strong models — integration comparison.
  5. learned vs `oracle` — gap-to-ceiling.
  6. in-dist vs each held-out axis — generalization.

### 6.10 Outputs (reuse C6 writers)

Per-instance raw CSV; per-`(arm, suite, budget, w)` summary CSV; significance MD (McNemar/BH + Wilcoxon/CI); figures (expansion-ratio bars, gap-to-ceiling, suboptimality-vs-`w` curve).

## 7. Scale parameterization (local-validate → cluster)

One script, one set of code paths; scale lives in config only.

| Knob | `local` (validate on 5090) | `cluster` (headline) |
|---|---|---|
| eval worlds / suite | ~20–30 | 100–200 |
| train worlds / suite | 96 (C6-proven) | 160+ |
| epochs | 16 (C6-proven) | 24+ |
| seeds | 1 train / fixed eval | 3–5 (variance CI) |
| budget × w grid | reduced (2 budgets, w∈{1.0,1.1}) | full (3 budgets, w∈{1.0,1.05,1.1,1.25}) |
| suites / backbones | all 6 / all | all 6 / all |

**Lift-without-rework guarantees:**
- **Backend-agnostic core** — no top-level Modal import (like C6); plain Python + GPU. "Bigger cluster" = same script, `cluster` preset, more shards in flight.
- **Sharded, incremental outputs** — eval is embarrassingly parallel over `(suite, world-range)`; per-shard CSVs (C6 `_shards/` pattern) + a join step; survives interruption.
- **Deterministic seeding** — local and cluster reproducible; same seed → same worlds.

## 8. Staged gates (subagent-driven)

- **Gate 0** — provider + focal unit tests green; matched-integrity test green.
- **Gate 1** — calibration probe: confirm each suite's binding band (Euclid 40–60%, oracle headroom); adjust any out-of-band suite.
- **Gate 2** — local validation run (`local` preset): end-to-end; directional results match expectations (field-HRM beats Euclid; scalar-HRM-additive fails); matched-integrity holds.
- **Gate 3** *(conditional, user decides)* — cluster scale-up for publication numbers.

## 9. Testing strategy

- **Providers** — shape/finiteness/non-negativity; `Euclid` admissibility (`<= dijkstra` per node); `Oracle == dijkstra` cost-to-go.
- **Focal** (ported from discrete `test_focal.py`) — `w=1` optimality; `w>1` bound never violated; completeness (finds a path iff reachable as budget→∞); determinism; no double-expansion; perfect-ranker-vs-uninformative at the same `w`; collapsed-`rank_h` degrades to Euclid A\*.
- **Matched-integrity** — all arms see byte-identical node coords + adjacency per seed.
- **Numerical guard** — nonfinite `h` raises/counts rather than silently zero-fills.

## 10. Threats to validity (and mitigations)

- **Saturation** → binding-budget band + expansions-primary (Sections 6.2/6.7).
- **Unmatched comparison** → shared worlds/PRMs by construction + matched-integrity test (Sections 4.4/9).
- **Hidden suboptimality from inadmissible additive `h`** → path-suboptimality is first-class (Section 6.7).
- **p-hacking across a large grid** → pre-registered primary comparisons (Section 6.9).
- **Local scale too small for a headline** → conditional cluster Gate 3; design lifts via config only (Section 7).
- **Numerical instability** (the discrete nonfinite incident) → loud nonfinite accounting (Section 4.5).

## 11. Open decisions (deferred, not blocking)

- Exact in-band budgets per suite — determined by Gate 1 calibration, not pre-set.
- Whether `rooms_large` survives the local wall-clock budget — decided at Gate 2.
- Whether to run Gate 3 (cluster) at all — user decides after Gate 2.
- Phase 2 (dynamics) — separate spec.

## 12. Key file references

- A\* search / priority: `hrm-cloud/continuous_prm/continuous_prm_common.py:729` (`f = g + h` at `:752`).
- Additive heuristic integration: `hrm-cloud/continuous_prm/continuous_prm_stage_runner.py:~300` (`h = euclid + side_len * residual`).
- PRM build: `hrm-cloud/continuous_prm/continuous_prm_common.py:666–709`.
- Backbones + residual head: `continuous_prm_common.py` HRM `:1146–1186`, ON-LSTM `:1086–1143`, head `:1200–1223`.
- C6 value-field harness (infra to extract): `hrm-cloud/continuous_prm/continuous_prm_c6_heatmap_value_field.py`.
- Discrete focal reference: `hrm-cloud/residual_tasklora_v2.py` (`space_time_focal_astar`) + `hrm-cloud/tests/test_focal.py`.
</content>
</invoke>
