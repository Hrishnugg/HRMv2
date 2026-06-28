# C7 Integration Comparison — Pre-registered Comparisons

Binding budget per suite (lower of the calibrated band, else first budget seen): C_hard_bugtrap=24, C_hard_maze=140, C_hard_maze_dense=140, C_hard_rooms=140, C_hard_rooms_large=56, C_hard_spiral=140

_Multiplicity: BH correction is applied ONLY to the success/McNemar grid over learned arms (see `continuous_prm_c7_significance.md`). The p-values in THESE six pre-registered comparisons are UNcorrected; treat the bootstrap 95% CIs as the primary inference._

_Small-n: a p shown as `n/a (n<6)` had too few discordant/matched pairs to trust (counts / median / CI are still reported)._

## 1. field_hrm/astar vs euclid/astar (success + expansions), per suite

|Suite|Budget|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
|C_hard_bugtrap|24|24|0.458|0.750|0.292|0.016|11|0.714 [0.533, 0.800]|0.002|
|C_hard_maze|140|24|0.583|1.000|0.417|0.002|14|0.521 [0.450, 0.627]|<0.001|
|C_hard_maze_dense|140|24|0.250|0.958|0.708|<0.001|6|0.804 [0.707, 0.829]|0.031|
|C_hard_rooms|140|24|0.375|0.958|0.583|<0.001|9|0.829 [0.775, 0.885]|0.004|
|C_hard_rooms_large|56|24|0.417|0.750|0.333|0.021|9|0.839 [0.646, 1.222]|0.371|
|C_hard_spiral|140|24|0.250|0.917|0.667|<0.001|6|0.850 [0.742, 0.919]|0.031|

## 2. scalar_hrm/astar vs field_hrm/astar — the representation lever

Each arm's expansion ratio + success vs euclid, side by side (does the field beat the scalar?).

|Suite|Budget|Arm|Arm succ vs euclid (delta)|n matched|Median ratio (95% CI)|Wilcoxon p|
|---|---:|---|---:|---:|---|---:|
|C_hard_bugtrap|24|scalar_hrm|0.792 (0.333)|11|0.750 [0.500, 0.857]|0.002|
|C_hard_bugtrap|24|field_hrm|0.750 (0.292)|11|0.714 [0.533, 0.800]|0.002|
|C_hard_maze|140|scalar_hrm|1.000 (0.417)|14|0.427 [0.351, 0.675]|<0.001|
|C_hard_maze|140|field_hrm|1.000 (0.417)|14|0.521 [0.450, 0.627]|<0.001|
|C_hard_maze_dense|140|scalar_hrm|1.000 (0.750)|6|0.721 [0.636, 0.788]|0.031|
|C_hard_maze_dense|140|field_hrm|0.958 (0.708)|6|0.804 [0.707, 0.829]|0.031|
|C_hard_rooms|140|scalar_hrm|1.000 (0.625)|9|0.822 [0.632, 0.885]|0.004|
|C_hard_rooms|140|field_hrm|0.958 (0.583)|9|0.829 [0.775, 0.885]|0.004|
|C_hard_rooms_large|56|scalar_hrm|0.917 (0.500)|10|0.729 [0.393, 0.841]|0.004|
|C_hard_rooms_large|56|field_hrm|0.750 (0.333)|9|0.839 [0.646, 1.222]|0.371|
|C_hard_spiral|140|scalar_hrm|0.708 (0.458)|6|0.820 [0.739, 0.967]|0.031|
|C_hard_spiral|140|field_hrm|0.917 (0.667)|6|0.850 [0.742, 0.919]|0.031|

## 3. scalar_hrm/focal (best w) vs scalar_hrm/astar — the integration lever

Both expressed as median exp_ratio vs euclid on their matched sets (does focal help the scalar?).

_Note: w was selected post-hoc (lowest median exp-ratio); the reported focal p is optimistic (winner's curse) — treat the CI as the primary inference._

|Suite|Budget|scalar/astar ratio (CI)|best w|scalar/focal ratio (CI)|focal Wilcoxon p|
|---|---:|---|---:|---|---:|
|C_hard_bugtrap|24|0.750 [0.500, 0.857]|1.1|0.789 [0.714, 0.864]|0.002|
|C_hard_maze|140|0.427 [0.351, 0.675]|1.1|0.953 [0.926, 0.978]|0.001|
|C_hard_maze_dense|140|0.721 [0.636, 0.788]|1.1|0.977 [0.945, 0.989]|0.031|
|C_hard_rooms|140|0.822 [0.632, 0.885]|1.1|0.964 [0.946, 0.992]|0.004|
|C_hard_rooms_large|56|0.729 [0.393, 0.841]|1.1|0.846 [0.710, 0.979]|0.012|
|C_hard_spiral|140|0.820 [0.739, 0.967]|1.1|0.960 [0.929, 0.969]|0.031|

## 4. field_*/focal (best w) vs field_*/astar — focal vs additive on the field models

_Note: w was selected post-hoc (lowest median exp-ratio); the reported focal p is optimistic (winner's curse) — treat the CI as the primary inference._

|Suite|Budget|Provider|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|
|---|---:|---|---|---:|---|---:|
|C_hard_bugtrap|24|field_hrm|0.714 [0.533, 0.800]|1.1|0.800 [0.714, 0.842]|0.002|
|C_hard_bugtrap|24|field_onlstm|0.750 [0.636, 0.895]|1.1|0.864 [0.733, 1.000]|0.027|
|C_hard_bugtrap|24|field_unet|0.895 [0.750, 1.000]|1.1|0.864 [0.714, 1.050]|0.047|
|C_hard_maze|140|field_hrm|0.521 [0.450, 0.627]|1.1|0.974 [0.931, 0.992]|0.003|
|C_hard_maze|140|field_onlstm|0.572 [0.518, 0.698]|1.1|0.947 [0.927, 0.981]|0.002|
|C_hard_maze|140|field_unet|0.595 [0.522, 0.643]|1.1|0.944 [0.910, 0.988]|0.004|
|C_hard_maze_dense|140|field_hrm|0.804 [0.707, 0.829]|1.1|0.974 [0.946, 0.993]|0.031|
|C_hard_maze_dense|140|field_onlstm|0.799 [0.765, 0.914]|1.1|0.969 [0.931, 0.982]|0.031|
|C_hard_maze_dense|140|field_unet|0.838 [0.771, 0.891]|1.1|0.973 [0.942, 0.989]|0.062|
|C_hard_rooms|140|field_hrm|0.829 [0.775, 0.885]|1.1|0.964 [0.947, 0.984]|0.004|
|C_hard_rooms|140|field_onlstm|0.906 [0.812, 0.961]|1.1|0.964 [0.949, 0.992]|0.004|
|C_hard_rooms|140|field_unet|0.846 [0.770, 0.899]|1.1|0.964 [0.957, 0.985]|0.020|
|C_hard_rooms_large|56|field_hrm|0.839 [0.646, 1.222]|1.1|0.925 [0.838, 0.964]|0.008|
|C_hard_rooms_large|56|field_onlstm|0.791 [0.635, 0.882]|1.1|0.902 [0.826, 0.935]|0.002|
|C_hard_rooms_large|56|field_unet|0.998 [0.656, 1.146]|1.1|0.839 [0.740, 1.000]|0.078|
|C_hard_spiral|140|field_hrm|0.850 [0.742, 0.919]|1.1|0.959 [0.949, 0.993]|0.062|
|C_hard_spiral|140|field_onlstm|0.946 [0.922, 0.973]|1.1|0.967 [0.936, 1.000]|0.094|
|C_hard_spiral|140|field_unet|0.906 [0.820, 0.958]|1.1|0.971 [0.947, 0.993]|0.062|

## 5. Each learned arm vs oracle/astar — gap-to-ceiling

Median over the triple-matched set (euclid, oracle, arm all solved) of
`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the
euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).

|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|
|---|---:|---|---:|---:|
|C_hard_bugtrap|24|field_hrm/astar|11|0.286|
|C_hard_bugtrap|24|field_onlstm/astar|11|0.500|
|C_hard_bugtrap|24|field_unet/astar|11|0.800|
|C_hard_bugtrap|24|scalar_hrm/astar|11|0.375|
|C_hard_bugtrap|24|scalar_onlstm/astar|11|0.214|
|C_hard_maze|140|field_hrm/astar|14|0.421|
|C_hard_maze|140|field_onlstm/astar|14|0.497|
|C_hard_maze|140|field_unet/astar|14|0.509|
|C_hard_maze|140|scalar_hrm/astar|14|0.317|
|C_hard_maze|140|scalar_onlstm/astar|14|0.338|
|C_hard_maze_dense|140|field_hrm/astar|6|0.746|
|C_hard_maze_dense|140|field_onlstm/astar|6|0.746|
|C_hard_maze_dense|140|field_unet/astar|6|0.790|
|C_hard_maze_dense|140|scalar_hrm/astar|6|0.645|
|C_hard_maze_dense|140|scalar_onlstm/astar|6|0.849|
|C_hard_rooms|140|field_hrm/astar|9|0.763|
|C_hard_rooms|140|field_onlstm/astar|9|0.875|
|C_hard_rooms|140|field_unet/astar|9|0.791|
|C_hard_rooms|140|scalar_hrm/astar|9|0.753|
|C_hard_rooms|140|scalar_onlstm/astar|9|0.701|
|C_hard_rooms_large|56|field_hrm/astar|9|0.750|
|C_hard_rooms_large|56|field_onlstm/astar|10|0.695|
|C_hard_rooms_large|56|field_unet/astar|10|0.988|
|C_hard_rooms_large|56|scalar_hrm/astar|10|0.611|
|C_hard_rooms_large|56|scalar_onlstm/astar|10|0.639|
|C_hard_spiral|140|field_hrm/astar|6|0.800|
|C_hard_spiral|140|field_onlstm/astar|6|0.929|
|C_hard_spiral|140|field_unet/astar|6|0.877|
|C_hard_spiral|140|scalar_hrm/astar|6|0.759|
|C_hard_spiral|140|scalar_onlstm/astar|6|0.816|

## 6. In-distribution vs held-out — field_hrm exp_ratio + success vs euclid

In-distribution (trained): C_hard_maze, C_hard_rooms, C_hard_spiral.  Held-out (OOD): C_hard_maze_dense, C_hard_bugtrap, C_hard_rooms_large.

|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|
|---|---|---:|---:|---|---:|
|in-dist|C_hard_maze|140|14|0.521 [0.450, 0.627]|0.417|
|in-dist|C_hard_rooms|140|9|0.829 [0.775, 0.885]|0.583|
|in-dist|C_hard_spiral|140|6|0.850 [0.742, 0.919]|0.667|
|held-out|C_hard_maze_dense|140|6|0.804 [0.707, 0.829]|0.708|
|held-out|C_hard_bugtrap|24|11|0.714 [0.533, 0.800]|0.292|
|held-out|C_hard_rooms_large|56|9|0.839 [0.646, 1.222]|0.333|

## Notes

- Each comparison uses the per-suite binding budget. Arms or suites absent from this run
  are skipped with a note rather than crashing.
- Comparisons 3 and 4 pick the focal `w` with the lowest matched-median exp_ratio vs euclid;
  reporting that winner's p on the same data is optimistic — the CI is the primary inference.
- These six p-values are UNcorrected (BH applies only to the success/McNemar grid).
- The Wilcoxon p tests paired (ratio - 1) in ratio-space, matching the median ratio + CI.
- Bootstrap CIs are seeded (`np.random.default_rng(cfg.seed)`), so this analysis is reproducible.
