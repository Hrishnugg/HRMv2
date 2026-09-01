# C14 preregistration: label-count × world-diversity adaptation factorial

**Frozen design date:** 2026-07-23
**Status:** approved design; no implementation, data generation, or training has occurred
**Amendment v2 (2026-07-23, pre-execution):** see §7 — the static-domain world-count factor as originally written is structurally infeasible (a static world supplies ≤160 supervised states, not ~25,000); §7 records the minimal correction, adopted before any implementation, data generation, or training.
**Purpose:** turn the supervision-density account of the LoRA/full-fine-tuning crossover from a plausible interpretation into an identified mechanism, by matching adaptation runs on *supervised-state count* rather than map count across static and dynamic domains.

## 1. Scientific question and hypothesis

The C9/C9b results show a static low-label crossover (LoRA preserves the pooled prior at $K{=}1$ map; full fine-tuning overtakes by $K{=}8$–$16$) that disappears under dynamics, where one map supplies roughly 25,000 supervised $(\text{node},t)$ states. The hypothesized mechanism is that the governing variable is the **number of supervised search states $N$**, not the number of maps and not the domain.

**H-C14:** the crossover point $N^{*}$ — the smallest $N$ at which full fine-tuning's expected map-level matched expansion ratio is at least as low as LoRA's — is (a) finite and increasing in neither domain indicator nor world count once indexed by $\log N$, and (b) approximately domain-invariant: the static and dynamic $N^{*}$ estimates' 95% CIs overlap on the $\log N$ axis, while the same estimates indexed by map count do not.

## 2. Factors

| Factor | Levels |
| --- | --- |
| Supervised-state count $N$ | 256, 1{,}024, 4{,}096, 16{,}384, 65{,}536 |
| World diversity at fixed $N$ | single-world; distributed (8 worlds, $N/8$ states each) |
| Domain | static (C9 substrate, maze-dense target) ; dynamic (C9b substrate, maze-dense target) |
| Adaptation method | rank-8 LoRA (unbounded); full fine-tuning; scratch control |
| Adaptation seed | 3 |

Cells: $5 \times 2 \times 2 \times 3 \times 3 = 180$ adaptations. Sources are the frozen C7 pooled HRM base (static) and the frozen C8 field U-Net blind (dynamic) — no new source training.

## 3. Controls held fixed

- Exact-label sampling: $N$ states drawn uniformly without replacement from the target world(s)' reachable labeled states, stratified by residual-magnitude bucket exactly as in the C9 loaders; the same sampled index sets are shared across methods and seeds within a cell (methods differ only in the update rule).
- Optimization: identical optimizer, learning rate, batch size, and *total optimizer steps* across all cells (steps fixed to the largest-$N$ epoch-equivalent; smaller $N$ cycles its sample), so compute cannot proxy for labels.
- Evaluation: the fixed C9 static 30-map and C9b dynamic 20-map test cohorts at their canonical binding budgets, map-clustered bootstraps (10k, seed 20260723), success and matched-solved ratios reported separately.

## 4. Preregistered analysis

1. Per cell: map-level success delta vs the frozen zero-shot source and matched-solved median ratio with CIs.
2. Crossover estimation: for each (domain, diversity, seed), fit monotone interpolants of the LoRA and full-FT ratio-vs-$\log N$ curves and report $N^{*}$ with a map-bootstrap CI; pooled $N^{*}$ per domain via seed aggregation.
3. Direct model: map-level ratio regressed on $\log N$, method, domain, diversity, and method$\times\log N$, method$\times$domain interactions; the mechanism claim requires the method$\times\log N$ interaction to be significant and the method$\times$domain interaction to be compatible with zero.
4. Diversity readout: at fixed $N$, distributed-vs-single-world contrasts quantify how much diversity matters beyond count.

**Verdict rule:** H-C14 is *supported* only if both (a) and (b) hold under analysis 2–3; *partially supported* if (a) holds but domain invariance fails; otherwise *rejected*. All three outcomes are reportable; no cell may be rerun, resampled, or dropped after unblinding.

## 5. Compute plan and timeline

Adapters are small (rank-8 over 1–3M-parameter bases; full FT of the same bases); one adaptation is minutes-scale on the local RTX 5090, and the 180-cell grid parallelizes trivially. Primary plan: local execution after the C8-R runs; Modal burst as fallback if the local queue slips past 2026-07-26. Target: results in time for the AAAI-27 supplementary deadline (2026-07-31); the paper's submitted claims do not depend on this study and will not be edited in response to it before review.

## 6. Exclusions

No new architectures, targets, or suites; no bounded-LoRA arm (the bound question is settled by C9h); no $N$ beyond 65,536 (the dynamic single-world ceiling is ~25k states — the two largest $N$ levels in the dynamic single-world cell are marked infeasible by construction and recorded as structural missingness, itself evidence that map count caps effective supervision); no post-hoc reweighting of the preregistered regression.

## 7. Amendment v2 (2026-07-23, adopted before any execution)

**Defect found during implementation planning, before any code, data, or training existed.** §2's diversity factor (single world vs 8 worlds at every $N$) was written around the dynamic substrate's arithmetic (~25,000 supervised $(\text{node},t)$ states per world). A *static* C9-substrate world supplies at most 160 supervised states (`SCALAR_NODES_PER_WORLD = 160` on a 192-node PRM, further reduced by reachability). Under the original text, the static single-world cell is infeasible at *every* $N \geq 256$ and static distributed-8 is infeasible for $N \geq 4{,}096$ — leaving too few static points to estimate $N^{*}$ at all, i.e., the preregistered analysis (§4.2–4.3) could not run in the static domain. This is a structural-arithmetic error in the design, not an empirical outcome; it is corrected pre-execution, mirroring how construction gates elsewhere in the program (e.g., C12's $K{=}16$ drop) operate before training.

**Correction (minimal, intent-preserving).** The diversity factor is redefined per domain in terms of the *minimal world count* $w_{\min}(N)$ — the fewest worlds whose cumulative reachable labeled states reach $N$, determined by the deterministic collection stream and recorded in the manifest:

- **Concentrated:** $w_{\min}(N)$ worlds; the $N$ states are drawn from this minimal prefix. In the dynamic domain $w_{\min}(N) = 1$ for $N \leq {\sim}25\text{k}$, so this level coincides with the original "single-world" cell there; the dynamic $N{=}65{,}536$ concentrated cell remains defined by the same rule (its $w_{\min}$ becomes >1 exactly when one world cannot supply $N$, replacing the original structural-missingness bookkeeping with a measured world count — the ceiling evidence is now carried by $w_{\min}(N)$ itself).
- **Distributed:** $8 \times w_{\min}(N)$ worlds, with states drawn evenly (as evenly as reachability permits) across them.

All other design elements are unchanged: the $N$ grid, methods, seed count, shared sampled index sets within a cell, matched total optimizer steps, evaluation cohorts and budgets, the preregistered analysis, and the verdict rule. The diversity contrast remains "same $N$, 8× more worlds with ~1/8 the states per world." Sampling within the pooled reachable states is uniform without replacement (the C9 loaders' actual behavior; §3's "stratified … exactly as in the C9 loaders" resolves to uniform because those loaders do not stratify — recorded here so the implementation cannot silently choose either reading).

No result of any kind was observed before this amendment; the run has not started.
