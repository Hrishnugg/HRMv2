# C11 — Compositional-Mission PRM Heuristics: Design Spec

**Date:** 2026-07-07 (user-approved design, this session)
**Derived from:** `docs/experiments/cross-space/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md` §7 (arms + gates) and the G0-H headroom probe (`docs/experiments/continuous/c11/results/C11_HEADROOM.md`: **PASSED** — matched oracle/legsum ratios 0.082–0.225 across all 9 cells, monotone dose-response in K).
**Scope decision (user):** architecture-only — NO transfer/adaptation protocol in C11 (C9-style transfer on this substrate is a follow-up only if an architecture signal appears). Iterative-compute arm = **full HRM-v2 (ACT + H/L)**, not a weight-tied refiner (user choice).
**Hardware:** local RTX 5090; everything CPU-friendly except training.

---

## 1. Question and pre-registered gates

**Question.** On a substrate with measured 5–12× oracle headroom and compositional structure by construction, do structured architectures (GNN over the product graph, sequence models on real mission traces, iterative-compute HRM-v2) beat an explicit MLP control at learned heuristic regression — and does the gap grow with mission depth K?

**Gates (pre-registered; verdicts reported against these verbatim):**
- **G1 — dose-response in structure.** Primary read: per (config, K), each structured arm vs the MLP control on matched per-world expansion-ratio differences (both-solved worlds) + success McNemar. *Continuity control:* at **K=0** the task reduces exactly to C7 (legsum ≡ euclid, no mission tokens beyond the goal leg) and all arms must statistically tie — if they don't, it is a formulation/implementation bug and the phase halts for diagnosis, not a result. *Positive verdict:* ≥1 structured arm beats the MLP with BH q < 0.05 at ≥2 of the 3 K∈{2,4,8} values on ≥1 config, with the gap monotone non-decreasing in K. Anything less is a negative for that arm.
- **G2 — depth-of-compute (HRM-v2 mechanism).** (a) Forced-segment curves: eval the trained ACT arm at forced k∈{1,2,4,8} segments; positive iff quality (state-level MAE and/or expansion-ratio) improves monotonically with bootstrap-CI separation between k=1 and k=8 on ≥1 config at K=8. (b) Learned halting: Spearman correlation of mean halt-steps vs K positive with q < 0.05 ("thinks longer on deeper missions").
- **G3 — honest closure.** If G0-H passed (it did), the I/O exposes real structure, and the MLP control still ties everything: "learned planning heuristics are architecture-agnostic" graduates to a strong publishable claim; the program pivots to the transfer+integration paper with the architecture chapter closed.

**Decision rule after C11:** dose-response positive → hierarchy thesis lives, invest in the high-DOF substrate (S1) next. Flat → write the paper (strategy memo thrust #5).

---

## 2. Substrate, datasets, splits

All mission/product-graph/oracle/A\* machinery **imports from `continuous_prm_c11_headroom.py`** (probe module — never modified): `sample_mission`, `transition_stage`, `product_oracle`, `h_legsum`, `astar_product`, `place_doors`/`door_adj_valid_factory`, `mission_reachable`, `calibrate_binding_budget`.

**Grid:** configs **A** (`C_hard_maze` waypoints, config_idx 0), **B** (`C_hard_rooms_large` waypoints, config_idx 1), **C** (`C_hard_maze` + keys→doors, config_idx 2) × **K ∈ {0, 2, 4, 8}**.

**TEST worlds:** for K∈{2,4,8}, the probe's exact 25 seeded worlds per cell (seed formula `1234 + 7919·w + 104729·config_idx + 15485863·K`), with the probe's calibrated binding budgets (A: 200/400/1600, B: 100/200/800, C: 200/400/1600 for K=2/4/8). For K=0: 25 fresh worlds from the same formula at K=0 (missions are empty; config C at K=0 has no doors — it degenerates to config A's distribution and is **dropped**, leaving A and B at K=0), binding budget calibrated by the probe's rule (lowest budget in {100,…,3200} with legsum success ≥ 0.05).

**TRAIN worlds:** 40 per (config, K) cell from a disjoint seed stream: `seed = 900001 + 7919·w + 104729·config_idx + 15485863·K`. One mission per world; skip-world semantics identical to the probe. Disjointness TRAIN∩TEST = ∅ enforced by a unit test on the generated seed sets (not just arithmetic).

**Labels:** the full exact `product_oracle` field per world — every (node, stage) with finite h\*. K=8: ≤1,728 states/world → ≈69k supervised states per cell. Data-rich by construction (C9b lesson); no subsampling.

**Residual formulation (all arms identical):** target `y = (h*(i,s) − h_legsum(i,s)) / side_len ≥ 0` (admissibility guarantees non-negativity). Eval-time heuristic `ĥ(i,s) = h_legsum(i,s) + side_len · clip(ŷ, 0, B)` with **B = the C7 scalar-arm softcap constant** (read from `continuous_prm_common`'s default at implementation; the same clip both in training targets and at eval). At K=0, h_legsum ≡ euclid-to-goal, so this is exactly C7's residual formulation — the basis of the G1 continuity control. States with infinite h\* are excluded from training (unreachable).

---

## 3. The I/O contract (structure-exposing, identical information per arm)

**Shared primitives** (one encoder module used by every arm):
- **Query-node token** (dim 12): `[x/side, y/side, 8 coarse ray distances (existing extractor, every other direction: indices 0,2,…,14 of the 16, /side), s/K_max, K_remaining/K_max]`.
- **Leg tokens** (dim 12, one per remaining leg t = s…K): `[Δx/side, Δy/side to leg target, leg length/side, (t−s)/K_max, is_door_key, door_open_at_s, remaining_frac, is_goal_leg, 0, 0, 0, 0]` (zero-padded to 12). Targets: `wp[t]` for t<K, node 1 for the goal leg.
- **Sequence** = [query token] + leg tokens, length s-dependent (≤ K_max+2 = 10), padded + masked.

**Per-arm consumption:**
1. **MLP control** — `flatten(pad(sequence))` = 120-dim input → GELU MLP, 3 hidden layers × width 768 (~1.3M params). No sequence/graph structure available. This is the arm the whole audit demanded.
2. **U-Net field** — the existing C6/C8 field U-Net class; input grids: occupancy, start/goal, current-target heatmap (stage s target), closed-door-rect mask at stage s; stage conditioning via FiLM on a 32-dim stage embedding; one forward per stage, per-node cell gather (C6/C8 convention). ~1–2M params.
3. **GNN (product graph)** — hand-rolled message passing (NO torch-geometric; Blackwell/Windows wheel risk): nodes = all (i, s) product states; edges = probe adjacency incl. stage-crossing arrival edges, door-masked edges removed per stage; node init = query-node-token features of i + `[dist(i, tgt(s))/side, s/K_max]`; edge features `[length/side, is_arrival, 1]`; hidden 128, **m = 8 rounds** (pre-registered; the residual-over-legsum formulation carries global structure, so m is local refinement, noted as a limitation), residual updates, mean aggregation, per-node scalar head. One forward predicts the whole field. ~1M params.
4. **HRM-trace / ON-LSTM-trace** — the existing `DeepSapientHRMBackbone` / ON-LSTM scalar backbones from `continuous_prm_common`, with the token sequence = the real mission trace above (NOT the 24-token feature bag). Same scalar head convention as C7/C8. ~1–3M params each.
5. **HRM-v2-ACT** (isolated module, §5) — same trace tokens, paper-faithful mechanism.
6. **Scaled addendum** — after the main read: best + worst arm re-instantiated at ~10–20M params, config A only, K∈{2,8}, 3 seeds, once. Kills/confirms the audit §4.4 scale confound.

---

## 4. Training and matching (native arms 1–4)

Matched recipe, identical across arms 1–4 (deviations = bugs): smooth-L1 (β=1.0) on ŷ, AdamW lr 2e-4, weight_decay 1e-4, grad-clip 1.0, batch 1024 states, **40 epochs** over the cell's pooled TRAIN states, 3 training seeds per (arm, config, K), fixed world sets. Param counts reported per arm and matched within ~1–3M. Run count: 4 arms × (3 configs × 3 K + 2 configs × 1 K) × 3 seeds = **132** + HRM-v2 33 + scaled 12 = **≈177 training runs**, each minutes on the 5090.

---

## 5. HRM-v2-ACT arm (isolated in `continuous_prm_c11_hrmv2_arm.py`)

Builds on the **fixed** HRM-v2 stack (post port-fix branch: verbatim ACTLossHead, per-segment optimizer steps, adam-atan2, sdpa contract, parity-tested).

- **Input:** trace tokens linear-projected to d_model=256; learned positionals; a dedicated readout token (remediation lesson: state-token readout, not mean-pool).
- **Architecture:** H/L = 2/2 blocks (hrm_v1.yaml family scaled to ~2M params), halt_max_steps M = 8, ACT exploration 0.1.
- **Loss head:** ACTLossHead ported with lm-loss → smooth-L1 on ŷ; **halting correctness binarized as |ŷ − y| ≤ 0.1** (side-length units; pre-registered); q_halt/q_continue BCE at 0.5 weights per the paper; per-segment optimizer steps via the faithful streaming loop (one segment per step, carry across segments).
- **Recipe deviation, accepted and flagged:** data/epochs/loss-on-ŷ matched to §4, but optimizer (adam-atan2 + constant LR + warmup) and the segmented training loop are paper-faithful — mechanism-faithfulness IS the arm; this confound is pre-registered as accepted.
- **Eval hooks:** (a) standard provider (ACT halting live) for G1; (b) forced-k providers k∈{1,2,4,8} (halting overridden) for G2a; (c) per-query halt-step logging for G2b.
- **Isolation:** lazy `import hrm` inside functions; the core module's arms, eval, and tests run with `hrm` absent. Registered into the same provider registry by name (`hrmv2_act`, `hrmv2_act_k{1,2,4,8}`).

---

## 6. Eval and stats

Probe machinery end-to-end: for each TEST world, product A\* (`astar_product`) with each arm's ĥ at the cell's binding budget (learned ĥ is inadmissible — C-series convention, costs recorded but optimality not asserted for learned arms; oracle/legsum/next remain the reference arms). Records CSV schema = probe schema + `arm` names for learned providers + seed column.

**Primary (G1):** per (config, K): arm-vs-MLP-control per-world paired expansion-ratio (both-solved set, median + bootstrap CI over worlds×seeds pooled) and success McNemar+BH across the arm family. **Secondary:** each arm vs legsum (absolute quality), oracle as ceiling; state-level MAE on TEST fields; G2 curves (§5). Analyzer is a pure function over the raw CSV (probe convention); outputs `C11_RESULTS.md` + curves/comparisons/significance CSVs under `runs/c11_local/`.

---

## 7. Modules, tests, conventions

- **`hrm-cloud/continuous_prm/continuous_prm_c11_mission.py`** — `C11MissionConfig` (all constants above as fields); TRAIN/TEST world+mission+label builders; the shared token/grid/graph encoders; arms 1–4 (models + matched trainer); provider registry `name → h(i,s)` callable; modes `collect / train / eval / analyze / full` + CLI (`--mode`, `--out-dir runs/c11_local`, `--arms`, `--configs`, `--k-values`, `--seeds`); checkpoint manifest like C9b's adapt manifest.
- **`hrm-cloud/continuous_prm/continuous_prm_c11_hrmv2_arm.py`** — §5 only; registers providers into the core registry.
- **`tests/test_c11_mission.py`** — TRAIN/TEST seed disjointness; encoder shapes/masks (incl. K=0 = query+goal-leg only; door flags only in config C); residual target non-negativity + clip round-trip; each arm forward-shape + one tiny overfit smoke (loss decreases); provider registry returns finite ĥ ≥ h_legsum; eval-records schema; analyzer on synthetic records (paired-ratio both-solved semantics, McNemar wiring); K=0 formulation-equivalence check (mission tokens reduce to goal leg; legsum == euclid on a fixture).
- **`tests/test_c11_hrmv2_arm.py`** — guarded by `pytest.importorskip("hrm")`: token projection shapes; correctness binarization; per-segment step count == segments run; forced-k override respected; halt-step logging; provider parity (forced-k=M equals ACT-off path).
- **Frozen, never modified/staged:** `continuous_prm_common.py`, `transfer_astar_heuristic_clean_parallel_fixed.py`, `continuous_prm_c7_hard_maps.py`, `continuous_prm_c11_headroom.py`, repo-root `models/`. HRM-v2 sources are consumed as an installed package (`pip install -e ./HRM-v2 --no-deps`), not edited from this phase.

---

## 8. Risks and honesty

- **HRM-v2 adaptation is the heavy item.** Mitigation: parity-tested components, its own test file, isolation behind the registry. If the arm underperforms, that is a result (the mechanism doesn't transfer to heuristic regression), not a reason to tune it beyond the pre-registered recipe.
- **GNN is a new model class** — hand-rolled and small; m=8 pre-registered (no post-hoc m tuning; a single pre-registered m-sensitivity appendix run at m∈{4,16} on one cell is allowed for interpretation only).
- **The task remains per-state scalar regression.** What changed vs C5–C10 is input structure (real sequence/graph) and compositional targets with measured headroom. If everything still ties, G3's claim is the deliverable — the phase cannot fail to produce a decision.
- **Formulation guard:** the K=0 continuity control is the canary for accidental formulation bugs; it must pass before any G1/G2 read is trusted.
