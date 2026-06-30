# C9b — Few-shot Transfer under Dynamics: Design

**Date:** 2026-06-30
**Status:** design (approved in brainstorming); spec for implementation.
**Branch:** new branch off `main` (e.g. `c9b-dynamics-transfer`).
**Builds on:** C8 (space-time A\*, dynamic suites, temporal providers, space-time oracle, calibration) + C9/C9h (adapt/eval/analyze harness, ADAPT⊥TEST split, bounded LoRA + conv-LoRA, matched-compute recipe, stats). All reused FROZEN. Related: `hrm-cloud/continuous_prm/C8_RESULTS.md`, `C9_RESULTS.md`, `C9H_RESULTS.md`, `CONTINUOUS_PRM_STORY.md`.

---

## 1. Goal & the two questions

C9/C9h established a transfer story on the **static** substrate: bounded LoRA is the robust/sample-efficient/capacity-limited adapter; full fine-tune is the high-variance/high-ceiling one; both ≫ from-scratch at low K. C8 established that under **dynamics**, time-awareness helps the optimal *plan* but **not** the learned *heuristic* (a time-blind present-frame model is an equal-or-better search heuristic).

C9b runs the C9 transfer protocol on the C8 space-time substrate to answer two questions:

1. **Replication:** does the C9/C9h crossover (LoRA robust / full-FT high-ceiling / both ≫ scratch) reproduce when the target is a held-out **dynamic** family?
2. **Probe (dynamics-specific):** does few-shot adaptation change C8's time-aware-vs-blind verdict? In particular, at high K does **full fine-tune of a time-aware source overtake the time-blind source** on a new dynamic family (the future window finally pays off once you can specialize to that family's timing), or does C8's negative persist across all adaptation arms?

"Transfer" = adapt a heuristic trained on one set of dynamic families to an *unseen* dynamic family from K labeled worlds. The aware/blind distinction is a property of the model input — **aware** = future-occupancy window (W>0 channels for field; length-W+1 rollout sequence for scalar); **blind** = present frame only (W=0) — carried identically through every arm.

## 2. Scope

Local (RTX 5090), **new-file-only**: `continuous_prm_c9b_dynamics_transfer.py`. Reuses C8/C9/C9h (imported, not modified). No edits to `continuous_prm_common.py` / `transfer_astar_*` (user WIP). Three backbones (scalar HRM, scalar ON-LSTM, field U-Net); both awareness variants; four arms; no new dynamics physics (C8's substrate as-is).

## 3. Source bases (reuse C8 heavy, frozen)

Six frozen pooled checkpoints from the C8 heavy run, trained on the C8 **train** suites (`C_dyn_maze`, `C_dyn_rooms`, `C_dyn_spiral`):

| backbone | aware | blind |
|---|---|---|
| scalar HRM | `runs/c8_local_heavy/checkpoints/c8_scalar__hrm.pt` | `…/c8_scalar__hrm_blind.pt` |
| scalar ON-LSTM | `…/c8_scalar__onlstm.pt` | `…/c8_scalar__onlstm_blind.pt` |
| field U-Net | `…/c8_field__unet.pt` | `…/c8_field__unet_blind.pt` |

These are the right transfer sources: pooled over the train families, never trained on the C9b targets. The C8 source recipe (12 epochs) differs from the C9b adapt recipe (§5) — that is fine: the source is the frozen init shared by all adapt arms, so it does not break matched-compute *across arms*. A `--retrain-sources` flag regenerates the six checkpoints (via C8's temporal trainer on the train suites) if the `.pt` files are absent, keeping C9b reproducible from a clean checkout.

## 4. Targets, disjointness, binding budgets

Three held-out dynamic families (disjoint from the train suites the sources saw), reusing C8's OOD held-out set:
- `C_dyn_maze_dense`, `C_dyn_crossing`, `C_dyn_rooms_large`.

`C_dyn_crossing` (open arena, no chokepoint) is the **control**: C8 showed aware ties blind there because there is no timing to exploit; under C9b it must keep tying even after adaptation (a guardrail against a spurious aware-win).

ADAPT and TEST worlds are disjoint per target (reuse C9's `world_fingerprint` + the rng-mirroring disjointness check, adapted to the space-time world generator). Binding budgets reuse C8 heavy's non-degenerate calibration — **maze_dense=2500, crossing=150, rooms_large=600** — recomputed via C8's calibrate path if a target's euclid-time success drifts out of the fair-fight band (lowest budget with euclid-time success ≥ 0.05).

## 5. Arms, recipe, grid

**Arms** (per target × backbone × awareness): `zero_shot` (the frozen C8 source) / `lora` (bounded; conv-LoRA for the field U-Net, `SingleAdapterLoRA` for scalar) / `full_ft` (all params from the source) / `scratch` (random init, same architecture/awareness, no source). Integration = additive residual on Euclidean-time, evaluated by C8's space-time A\* providers.

**Matched compute** for all trained arms (the C9h recipe): **epochs 10, lr 2e-4**, smooth-L1, grad-clip. Only low-rank/full-rank, source/random-init, and aware/blind differ. Bounded residual only (C9h showed the bound is irrelevant — no separate unbounded arm).

**Grid:** K ∈ {1, 4, 16}; **3 seeds**; 3 backbones × 2 awareness. Trained adapters = 3 bb × 2 aware × 3 arms × 3 targets × 3 K × 3 seeds = **486**, plus the 6 frozen zero-shot sources. Labels for ADAPT/TEST worlds come from C8's backward space-time Dijkstra oracle (the same cost-to-go labels C8 trained on).

## 6. Metrics, stats, gates

Primary: matched **space-time** A\* expansion-ratio vs Euclidean-time on solved TEST worlds at the binding budget (median + seeded bootstrap CI, pooled over seed × world). Success rate is first-class (McNemar exact + BH on the success grid). Stats identical to C8/C9.

**Success-aware reporting is mandatory (C8's lesson):** the matched expansion-ratio is computed only on worlds *both* arms solve, which *excludes the worlds where an aware arm's advantage is largest* (those only it solves). So the aware-vs-blind probe (§1 Q2) is judged on a **success composite** (success delta + ratio on the shared-solved set), never expansion-ratio alone. The analyze step reports both and flags any cell where the two disagree.

**Gates:**
- **G0 (integrity):** unit tests — ADAPT⊥TEST disjointness on the space-time generator; matched comparison (all arms on the same TEST worlds); LoRA/conv-LoRA round-trip on a C8 source; single binding budget per target; aware vs blind input shapes correct (W>0 vs W=0).
- **G1 (base sanity):** every frozen zero-shot source beats Euclidean-time on the held-out dynamic families (reproduces the C8 OOD direction).
- **G2 (replication):** the crossover — full_ft worst at K=1, best by K≥4–16; LoRA ≈ zero-shot (robust, flat); both ≫ scratch at low K — reproduces under dynamics (reported with CIs, per backbone/awareness; or a clear, honest miss).
- **G3 (the probe):** aware-vs-blind per arm/K on the success composite. The pre-registered question: does `full_ft` (high-ceiling) at K=16 flip C8's negative (aware > blind on a new dynamic family), while `crossing` (control) stays a tie? Report the result either way.

## 7. Architecture

New `continuous_prm_c9b_dynamics_transfer.py`: `C9bConfig`, modes `adapt` / `eval` / `analyze` / `full` + CLI. Reuses:
- **C8** (`continuous_prm_c8_dynamics_compare` + `continuous_prm_c8_dynamic_maps`): dynamic-suite installer, the space-time world/label generator, temporal feature construction (aware rollout-sequence / future-occupancy channels and the W=0 blind variant), `run_world_arms_spacetime`, the temporal providers, the space-time oracle, and the calibrate/binding-budget path.
- **C9** (`continuous_prm_c9_transfer`): ADAPT/TEST split + `world_fingerprint` disjointness, `train_scalar_model` (full-FT/scratch) pattern, `_binding_budget_for`, `RAW_COLS`/`_parse_*`, the analyze helpers + stats imports.
- **C9h** (`continuous_prm_c9h_transfer`): `train_scalar_lora` + `apply_conv_lora` + the matched recipe + the bounded provider loader.

**The one piece of real integration glue:** the adapt loop must drive C8's **temporal** feature pipeline (not C9's static one) through C9h's LoRA/full-FT trainers — i.e. collect K dynamic worlds for a target, build the aware/blind temporal dataset via C8, apply_lora/apply_conv_lora (or full-FT/scratch) on the C8 source architecture, train at matched compute, save, and eval via C8's space-time providers. Awareness selects W>0 vs W=0 inputs end-to-end.

## 8. Scale & risks

Local; a long single-GPU run (486 adapters + a large space-time eval — space-time A\* over the budget grid is the expensive part; expect an overnight run, larger than C9h's ~3 h static run). Risks: (a) **runtime** — if too long, the first trim is K∈{1,16} or 2 seeds (not chosen now; the user opted for the full 3-seed grid). (b) **maze_dense degeneracy** — handled by C8 heavy's 2500 binding budget; re-calibrate if it drifts. (c) **matched-ratio hides aware-wins** — mitigated by the mandatory success composite (§6). (d) **source-recipe asymmetry** — the frozen C8 sources used 12 epochs vs the 10-epoch adapt recipe; documented, and irrelevant to across-arm matching since the source is shared.

## 9. Acceptance criteria

- `continuous_prm_c9b_dynamics_transfer.py` runs `--mode full --scale local` end-to-end: (optional source check/retrain), adapt 486 arms, space-time eval, analyze → curves + comparisons + significance + an **aware-vs-blind probe report** (success composite) + manifest.
- G0 unit tests green; G1/G2/G3 reported (replication verdict + the probe verdict, honest either way).
- `C9B_RESULTS.md` written and cross-linked from `C8_RESULTS.md`, `C9H_RESULTS.md`, and `CONTINUOUS_PRM_STORY.md`.
- WIP files never staged; C8/C9/C9h/common unchanged.
