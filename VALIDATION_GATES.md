# Cross-Space Validation Gates — 2026-07-06

**Branch:** `hrm-v2-fixes` (contains: C6→C10 + C9b experiment stacks, the HRM-v2 port fixes, all audits).
**Gate:** every suite we built, in both spaces, must pass **by default** on a clean run. Secondary: measure `pytest-xdist -n auto` speedups (user asked whether cloud GPUs would speed testing — answer below).

## Results — ALL GREEN (194 passed, 0 failed)

| Suite | Scope | Result | Serial | xdist `-n auto` | Verdict |
|---|---|---|---:|---:|---|
| `HRM-v2/tests` | port fidelity: sdpa contract, ACTLossHead parity-vs-original, sparse-emb, AdamATan2, streaming loop, **bit-exact original-vs-port parity**, full-state checkpointing | **59 passed**, 4 skipped (flash-attn absent) | 3.3s | 8.9s | ✅ GREEN — run serial (worker startup dominates) |
| `hrm-cloud/tests` | discrete space: focal search, focal wiring, eval-speedup, C6/C7/C8 unit tests | **84 passed** | 6m03s | 4m58s | ✅ GREEN — xdist mildly helpful |
| `hrm-cloud/continuous_prm/tests` | continuous space: C8 binding budget, C9/C9h transfer, C10 interpolation, C9b dynamics-transfer (incl. GPU + CPU smokes) | **51 passed** | 11m56s | 9m44s | ✅ GREEN — xdist limited by longest single test |

Full cross-space gate: **~18 min serial → ~15 min with xdist** on the local 32-thread box.

## Speedup analysis (the "should we rent 2×H100?" question)

**No — GPUs are the wrong lever.** Measured bottlenecks are CPU-side: space-time backward-Dijkstra oracles (~9s/world), world/PRM generation, and CPU-mode deterministic trainings. The continuous suite's wall-clock is set by its **longest single smoke test** (~8–9 min of sequential world-collection + tiny trainings inside one test), which neither xdist workers nor cloud GPUs can subdivide.

Levers that would actually work, in order:
1. **On-disk labelset/world cache** (session- and run-persistent): the expensive `_collect_world_labels` results are identical across runs; caching them would cut the long smokes by an estimated 50–70%. Biggest remaining win, zero hardware.
2. **Split the mega-smokes** (e.g. C9b `run_full` smoke) into stages sharing the cache, so xdist can parallelize the pieces.
3. Cloud only as a **many-core CPU box or CI runner** for push-button validation — never GPU instances for this workload.

## Provenance
- Gate run on: RTX 5090 box, torch 2.9.0+cu130, Python 3.13, 32 threads.
- Logs: session scratchpad (`*_serial.log`, `*_xdist.log`).
- Companion docs: `HRM-v2/PORT_FIDELITY_AUDIT.md` (+ status block), `HRM-v2/RETRAIN_RESULTS.md`, `hrm-cloud/PROGRAM_AUDIT_HIERARCHY_AND_SUBSTRATE.md`.
- Remaining validation step (designed, not yet run): HRM head-to-head on our own benchmarks — existing `DeepSapientHRMBackbone` reproduces prior numbers; repaired-attention variant must be same-or-better (~1h GPU).
