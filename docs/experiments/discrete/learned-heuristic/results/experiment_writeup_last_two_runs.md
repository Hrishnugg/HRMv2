# Writeup of the Two Most Recent Transfer-Heuristic Experiments

**Models:** HRM 3M and ON-LSTM 3M  
**Planner:** static A* with learned residual heuristic augmentation  
**Sources:** `results_lora_fixpack.txt` and `mapscale results.txt`

## Executive summary

- Both experiments produced essentially the same planner-level conclusion: the learned models trained successfully, but the resulting heuristics behaved almost identically to the static baseline at evaluation time.
- In the original curriculum, the hard OOD B-family suites remained hard and the easy C-family suites remained easy, with HRM, ON-LSTM, and few-shot variants all tracking the baseline closely.
- In the map-scaling curriculum, the experiment became scientifically cleaner: size scaling and sparse dynamics on family A were easy, family C remained saturated, and family B emerged as the true distribution-shift bottleneck.
- The most defensible conclusion is not that LoRA failed in the abstract, but that in the current residual-heuristic pipeline the learned correction is not influencing A* enough to change search outcomes on the hard cases.

## 1. Experiment A — Original transfer curriculum with stage-wise LoRA adapters

### Purpose

This run tested whether a stage-wise transfer curriculum, combined with LoRA adapters, could improve a static A* planner by learning a residual correction to the static heuristic. The curriculum increased both map difficulty and dynamics across stages, and evaluation compared baseline static A*, zero-shot HRM 3M / ON-LSTM 3M, and few-shot variants.

### Design summary

- Training / evaluation suite family: `ID_A32_D1`, `ID_A64_D2`, `OOD_B32_D1`, `OOD_C32_D1`, `OOD_B64_D2`, `OOD_C64_D2`
- Models evaluated: `baseline_static_astar`, `onlstm_3m`, `hrm_3m`, plus few-shot K=50 and K=200 variants on the hardest suite
- Budgets: 200, 500, and 2000 node expansions per step
- Primary question: does the learned residual heuristic change planner-level behavior, especially on the hard OOD B-family suites?

### Main results

The headline result was a near-null planner outcome. On the easiest and in-distribution tasks, the learned models matched the baseline almost exactly. `ID_A32_D1` stayed at 1.00 success, while `ID_A64_D2` stayed at 0.97 success with 0.03 timeout across the baseline, HRM, and ON-LSTM. Likewise, `OOD_C32_D1` and `OOD_C64_D2` were already saturated at 1.00 success, and the learned models remained at that ceiling.

The hard suites also failed to show meaningful gains. On `OOD_B32_D1`, the baseline achieved 0.47 success and 0.53 timeout at budget 200, and both ON-LSTM and HRM matched it almost exactly. On `OOD_B64_D2`, the baseline achieved 0.34 success and 0.66 timeout, and both ON-LSTM and HRM again matched those outcomes exactly at budget 200. Even more importantly, increasing budget from 200 to 500 to 2000 raised search effort dramatically but did not change success on the hard B-family suites, indicating that the failures were not simply caused by too little search.

### Representative budget-200 outcomes


| Suite        | Baseline      | ON-LSTM 3M    | HRM 3M        | Reading                    |
| ------------ | ------------- | ------------- | ------------- | -------------------------- |
| `ID_A32_D1`  | 1.00 / 1,630  | 1.00 / 1,632  | 1.00 / 1,636  | Solved by everyone         |
| `ID_A64_D2`  | 0.97 / 6,766  | 0.97 / 6,774  | 0.97 / 6,792  | Still no gain              |
| `OOD_B32_D1` | 0.47 / 13,251 | 0.47 / 13,251 | 0.47 / 13,251 | Hard, unchanged            |
| `OOD_B64_D2` | 0.34 / 31,572 | 0.34 / 31,572 | 0.34 / 31,572 | Hardest suite; exact match |
| `OOD_C64_D2` | 1.00 / 6,596  | 1.00 / 6,605  | 1.00 / 6,606  | Already saturated          |


*Values are shown as success rate / average expansions at budget 200.*

### Few-shot adaptation

Few-shot adaptation also behaved like a null result. On the hardest suite, `OOD_B64_D2`, all four few-shot variants (`onlstm_3m_fewshotK50`, `onlstm_3m_fewshotK200`, `hrm_3m_fewshotK50`, `hrm_3m_fewshotK200`) remained at 0.34 success and 0.66 timeout across budgets, with expansions remaining extremely close to the zero-shot models and the baseline.

### Interpretation

The most plausible reading is that the learned residual heuristic was present but effectively inert. It did not materially alter node ordering or search outcomes on the suites that mattered most. This does not prove that LoRA is a bad idea. It supports a narrower conclusion: in the current pipeline, the residual target, model coupling, and A* integration did not generate a correction strong enough to change planner behavior.

### Takeaway from Experiment A

Experiment A was valuable because it established a baseline negative result cleanly: stage-wise LoRA made the transfer pipeline runnable and stable, but planner-level gains were absent. The hard OOD B-family cases remained unresolved, and few-shot adaptation did not rescue them.

## 2. Experiment B — Map-scaling curriculum with subdued obstacle scaling

### Purpose

This run reframed the curriculum around a more plausible transfer path: small static maps, then medium static maps, then medium maps with sparse dynamics. The design intentionally reduced the jump from no obstacles to many dynamic obstacles and aimed to isolate whether map scaling itself was learnable before tackling stronger distribution shift.

### Design summary

- Core suites: `ID_A32_static`, `ID_A64_static`, `ID_A64_sparseDyn`, `OOD_B64_static`, `OOD_C64_static`, `OOD_B64_sparseDyn`, `OOD_C64_sparseDyn`
- The stretch `fullDyn` stage does not appear in the current results file, so this writeup covers the core three-stage map-scale curriculum only
- Models evaluated: `baseline_static_astar`, `onlstm_3m`, `hrm_3m`, and few-shot K=50 / K=200 variants on the hardest sparse-dynamic OOD suite
- Primary question: is the difficulty really map scaling, or is it something more specific about the family-B distribution shift?

### Main results

This experiment produced a much cleaner scientific picture. The baseline itself showed that map scaling was not the core problem. `ID_A32_static`, `ID_A64_static`, and `ID_A64_sparseDyn` were all solved at 1.00 success. At budget 200, the jump from `ID_A64_static` to `ID_A64_sparseDyn` changed average steps by only 0.04 (49.58 to 49.62) and average expansions by only 8 (6,146 to 6,154), which means that adding sparse dynamics on family A barely changed the planning problem.

Family C was also easy in both static and sparse-dynamic form. The real difficulty was family B. `OOD_B64_static` sat at 0.28 success and 0.72 timeout across budgets, and `OOD_B64_sparseDyn` sat at 0.34 success and 0.66 timeout across budgets. Again, increasing budget raised search effort sharply but did not improve success, which indicates that these failures were not simply budget-limited.

### Representative budget-200 outcomes


| Suite               | Baseline      | ON-LSTM 3M    | HRM 3M        | Reading                     |
| ------------------- | ------------- | ------------- | ------------- | --------------------------- |
| `ID_A32_static`     | / 1,613       | 1.00 / 1,636  | 1.00 / 1,627  | Small static solved         |
| `ID_A64_static`     | 1.00 / 6,146  | 1.00 / 6,186  | 1.00 / 6,171  | Size scaling still easy     |
| `ID_A64_sparseDyn`  | 1.00 / 6,154  | 1.00 / 6,196  | 1.00 / 6,177  | Sparse dynamics adds little |
| `OOD_B64_static`    | 0.28 / 28,523 | 0.28 / 28,523 | 0.28 / 28,523 | B-family shift is hard      |
| `OOD_B64_sparseDyn` | 0.34 / 29,778 | 0.34 / 29,778 | 0.34 / 29,778 | Still hard, unchanged       |
| `OOD_C64_sparseDyn` | 1.00 / 6,942  | 1.00 / 6,967  | 1.00 / 6,977  | C-family remains easy       |


*Values are shown as success rate / average expansions at budget 200.*

### Few-shot adaptation

Few-shot adaptation again failed to move the hardest suite. On `OOD_B64_sparseDyn`, all few-shot variants remained at 0.34 success and 0.66 timeout, with expansions nearly identical to the zero-shot models and the baseline. That means the map-scale reframing clarified the problem, but it did not solve it.

### Interpretation

Experiment B is still a useful outcome because it isolates the true bottleneck. The revised curriculum shows that size scaling and mild dynamics are learnable in the sense that the planner already performs well there. What remains unresolved is the B-family distribution shift. The learned heuristic, even with LoRA and few-shot adaptation, did not produce measurable planner-level gains on that shift.

### Takeaway from Experiment B

This is the stronger writeup experiment. It supports the claim that the revised curriculum is cleaner and more publishable, and it identifies where the system actually breaks: not at map size transfer in general, but at a specific hard OOD family where the learned residual does not meaningfully alter search.

## 3. Cross-experiment synthesis

Taken together, the two experiments motivate a reset rather than another small patch. Experiment A showed that the original curriculum produced a broad null result. Experiment B showed that once the curriculum was cleaned up, the null result became more informative: the real problem is not generic transfer from small to large maps, and it is not sparse dynamics by itself. The real problem is a harder family-B shift that the current learned residual does not resolve.

- The present evidence does not support a claim that HRM or ON-LSTM materially improve search under the current residual-heuristic interface.
- The present evidence also does not support a claim that LoRA itself is the main issue; a more precise conclusion is that the current training target and planner coupling are not producing an influential heuristic correction.
- The next experimental generation should compare non-LoRA and LoRA under the same new curriculum, with instrumentation that measures whether predicted residuals are actually changing frontier ordering on the hard suites.

## Practical framing for a paper or memo

A defensible summary statement would be: stage-wise transfer with LoRA adapters was operationally successful, but in both the original and map-scale curricula the learned residual heuristics failed to improve planner-level performance over static A*. The map-scale curriculum nevertheless clarified the problem structure by showing that size scaling and sparse dynamics are easy for the planner, while a specific family-B distribution shift remains the dominant failure mode.

## Recommended next step

The right next move is to rebuild the experiment from the ground up around a cleaner curriculum and a cleaner comparison: a matched non-LoRA baseline versus a LoRA version, both instrumented to measure residual magnitude, positivity rate, ranking influence, and search-behavior change on hard B-family instances.