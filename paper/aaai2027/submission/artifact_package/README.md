# Artifact Package: An Empirical Analysis of Transfer for Dynamic Path Planning

Anonymized artifacts for the AAAI-27 submission. Contents:

- `analysis/` -- scripts that regenerate the quoted map-level statistics from
  raw rows: `world_clustered_reanalysis.py` (C9/C9h/C9b/C10/C11 quantities),
  `c8_fixed_provider_reanalysis.py` (the fixed-provider dynamic headline),
  `c8_wastar_mcnemar.py` (exact McNemar + effort-ratio CIs + matched path
  quality for the weighted-A* control), `c8_fresh_mcnemar.py`,
  `c8_budget_curves.py`, `c8_reachable_count.py` (the deterministic
  supervised-label recount, manifest-gated), `c14_analysis.py` (factorial),
  and `c8_movingai_analysis.py` (external benchmark), each with its generated
  output committed beside it for comparison.
- `figures/` -- generators for all paper figures. `make_fig_dynamic.py`,
  `make_fig_c14.py`, and `make_fig_budget_curves.py` read every plotted number
  from the committed analysis JSONs. `make_figures.py` (integration,
  K-indexed adaptation, C11, C13 figures) embeds its plotted values as
  in-source constants; each constant is traced to the corresponding raw rows
  or result documents in this package, and provenance comments in the source
  name the origin of every array.
- `raw/` -- the raw paired evaluation rows behind every quoted quantity:
  C7 static, C8 dynamic (including the aware/blind twins), the fresh-cohort
  replication (`c8r_fresh`) and the two retrained-pipeline replications
  (`c8r_seed2001`, `c8r_seed2002`, each with the verbatim binding
  calibration), the weighted-A* development/confirmation rows with the
  tuned-weight manifest, the SIPP baseline rows, the MovingAI external rows
  with the map manifest, C9/C9h/C9b adaptation, C10 composition, C11 missions
  (evaluation, halting, state MAE), the frozen C13-M confirmation (raw rows,
  pairwise summary, gate verdict, integrity manifest with artifact hashes),
  C14 per-cell outputs with sampled-index manifests, and the C14-R
  independent-draw replicates.
- `checkpoints/` -- the two frozen source checkpoints the headline claims
  rest on: `c8_field__unet_blind.pt` (the fixed dynamic provider; SHA-256
  b8378950...) and `avgbase__hrm.pt` (the frozen static adaptation source).
  All other checkpoints are regenerable from the recorded training commands
  and seeds in `preregistrations/` and are omitted for size.
- `preregistrations/` -- the frozen design documents from C7 onward, including
  gates, budgets, and analysis grains fixed before execution, the C14
  pre-execution amendment and post-completion supersession record, and the
  2026-07-25 SIPP / MovingAI / C14-R designs.

## Reproduction

Requirements: Python 3.11+ with numpy and matplotlib.

    python analysis/world_clustered_reanalysis.py     # map-clustered statistics
    python analysis/c8_fixed_provider_reanalysis.py   # dynamic headline numbers
    # multi-pipeline replication (same script, parameterized by run directory):
    python analysis/c8_fixed_provider_reanalysis.py c8r_seed2001_eval c8r_seed2001
    python analysis/c8_fixed_provider_reanalysis.py c8r_seed2002_eval c8r_seed2002
    python figures/make_figures.py                    # supplement figures
    python figures/make_fig_dynamic.py                # paper figure 1
    python figures/make_fig_c14.py                    # paper figure 2

Each script is deterministic (fixed RNG seeds recorded in the source) and
prints the tables it regenerates; expected outputs are the committed `*.md`
files beside the scripts. **Path note:** the analysis scripts were developed
against the original repository layout and resolve raw-row paths through
constants at the top of each file; when running from this package standalone,
point those constants at the package `raw/` directories first (each script
documents its expected inputs in its docstring). The commands above are
therefore reproduction recipes, not turnkey one-liners; a path-shim pass is
planned for the camera-ready archive.
