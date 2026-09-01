# C8-R preregistration: multi-seed replication of fixed-provider dynamic transfer

**Frozen design date:** 2026-07-23
**Status:** approved design; fresh-cohort evaluation of the canonical seed is running; no new-seed training has started
**Purpose:** replicate the fixed-provider dynamic zero-shot transfer result (field U-Net blind) across independent training seeds and an enlarged, disjoint evaluation cohort, answering the seed-variance and selection objections ahead of the AAAI-27 camera-ready.

## 1. Fixed choices (frozen before any new target evaluation)

- **Primary provider family:** field U-Net, blind (window $W{=}0$) and aware ($W{=}8$) twins. The blind twin is the primary arm; aware exists only for the twin contrast. This selection was made from source-side reasoning (global-input field models train most stably in the C8 harness) and from the already-published canonical run; it is frozen here before any new-seed or new-cohort result is observed.
- **Integration:** additive, at the canonical per-suite binding budgets from `runs/c8_local_heavy/calibration.json` (copied verbatim; never recalibrated).
- **Evaluation cohort:** the 50-maps-per-suite fresh cohort generated with `--seed 999999` (run `c8r_fresh_eval`), which the eval-seed formula guarantees is world-disjoint from the canonical 20-map cohort within every suite (within-suite seed offset bound 397,450 < seed delta 998,765). All seeds evaluate on this one common cohort so comparisons are map-paired.
- **Training seeds:** canonical 1234 (already trained) plus fresh seeds 2001 and 2002. New-seed runs redo collection and training from scratch (`--seed {2001,2002}` in dedicated out-dirs), so training worlds also resample — this is full-pipeline replication, not weight-perturbation.

## 2. Protocol

1. For each new seed $s \in \{2001, 2002\}$: `--mode collect` then `--mode train` with `--scalar-backbones "" --field-backbones unet` in `runs/c8r_seed{s}/` (64 train worlds, 12 epochs, local scale — identical to canonical).
2. Copy each seed's `c8_field__unet.pt` / `c8_field__unet_blind.pt` into `runs/c8r_seed{s}_eval/checkpoints/` with the canonical `calibration.json`, and run `--mode eval --seed 999999 --eval-worlds 50` — byte-identical eval configuration to `c8r_fresh_eval` except the checkpoints.
3. Analysis (map-level, 10k bootstraps, seed 20260723): per suite and per training seed, paired success delta vs Euclid and matched-solved median ratio for the blind arm; between-seed spread of both; aware-minus-blind twin contrast per seed.

## 3. Preregistered readout (not gates that block the submission — replication descriptors)

- **R1 (success replication):** for each new seed, the blind arm's paired success-delta CI excludes zero in at least five of six suites.
- **R2 (effect stability):** the across-seed range of per-suite success deltas stays within ±0.15 of the canonical seed's estimate in at least five suites.
- **R3 (twin stability):** no suite in any seed shows a significant aware-over-blind success advantage.

Failures are reported as-is; no retraining, reselection, or recalibration is permitted in response to R1–R3.

## 4. Compute and exclusions

Local RTX 5090, sequential with the running fresh-cohort eval; each seed is one collect+train (~single-backbone fraction of the canonical heavy run) plus one 50-map eval. No Modal port, no new suites, no architecture additions, no recalibration, no per-suite provider selection anywhere in the primary analysis.
