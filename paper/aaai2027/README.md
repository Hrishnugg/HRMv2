# AAAI 2027 Submission: "The Harness, Not the Hierarchy"

**Status:** complete first full draft (anonymous-submission mode), 2026-07-20.
**Format:** AAAI 2026 author kit (per instruction, AAAI 2027 follows the same format). 7-page main-text target, references uncounted, technical appendix as separate supplementary PDF. Template usage notes: `TEMPLATE_README.md`.

## Files

| File | Role |
| --- | --- |
| `main.tex` | The paper (anonymous mode ON via `\def\aaaianonymous{true}`). |
| `supp.tex` | Technical appendix / supplementary material (protocols, full suite tables, forensics, statistics, reproducibility). |
| `references.bib` | 35 entries, every one fetched and verified programmatically against arXiv/Semantic Scholar/Crossref metadata — see `lit/lit_report*.md` for the per-entry audit. |
| `figures/make_figures.py` | Regenerates all five figure PDFs; every plotted number is traced to the master synthesis or canonical result docs (no interpolation). |
| `figures/fig1..fig5*.pdf` | Vector figures (colorblind-safe Okabe–Ito subset, validated). |
| `compile.ps1` | Build script (`pdflatex`+`bibtex` ×3, both docs). Requires MiKTeX/TeX Live — none was installed on this machine, so the draft is source-validated but not yet compiled locally. Overleaf also works: upload the directory. |
| `aaai2026.sty` / `aaai2026.bst` | Official AAAI style files (do not modify). |
| `lit/` | Literature verification reports and raw BibTeX drafts. |

## Provenance and claim discipline

- The single evidence source is `docs/experiments/MASTER_EXPERIMENT_SYNTHESIS.md` (snapshot 2026-07-20) plus canonical per-stage result documents; the narrative and claim boundaries follow `docs/experiments/analysis/PAPER_FOUNDATION_ANALYSIS.md` (its overreach-guard section lists claims the paper deliberately does NOT make).
- C13-P is cited only as preregistered-but-unlaunched future work (no outcome claims).
- Known open statistics item (also disclosed in the paper's Limitations): C9–C11 quoted intervals are record-level; world-clustered reanalysis is planned before camera-ready.

## Before submission checklist

- [ ] Install LaTeX (e.g. `winget install MiKTeX.MiKTeX`) and run `./compile.ps1`; confirm 7-page main-text fit and trim if needed (candidates: Related Work compression, Table 1 row merges).
- [ ] World-clustered reanalysis for C9/C9h/C9b/C10/C11 quoted intervals (or keep the descriptive-interval caveat wording).
- [ ] Professor review of the C13 "bounded observations on a known PRM" wording (flagged in the synthesis).
- [ ] Verify AAAI 2027 CFP specifics when released (page limit, checklist, supplementary rules) — currently assumed identical to AAAI 2026.
- [ ] Prepare anonymized code/artifact archive for supplementary upload.
