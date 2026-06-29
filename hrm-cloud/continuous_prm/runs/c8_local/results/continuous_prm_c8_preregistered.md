# C8 Dynamics Comparison — Pre-registered Comparisons

Binding budget per suite (lower of the calibrated band, else first budget seen): C_dyn_crossing=150, C_dyn_maze=1800, C_dyn_maze_dense=150, C_dyn_rooms=1300, C_dyn_rooms_large=400, C_dyn_spiral=2500

_Multiplicity: BH correction is applied ONLY to the success/McNemar grid over learned arms (see `continuous_prm_c8_significance.md`). The p-values in THESE six pre-registered comparisons are UNcorrected; treat the bootstrap 95% CIs as the primary inference._

_Small-n: a p shown as `n/a (n<6)` had too few discordant/matched pairs to trust (counts / median / CI are still reported)._

## 1. Time-aware learned vs euclid-time (expansions + success), per suite

Each time-aware learned arm (field_<bb>/astar, scalar_<bb>/astar) vs `euclid/astar`.

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.125 [0.082, 0.169]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/astar|10|0.200|1.000|0.800|0.008|2|0.249 [0.209, 0.289]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.213 [0.100, 0.325]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/astar|10|0.200|1.000|0.800|0.008|2|0.196 [0.091, 0.301]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/astar|10|0.400|1.000|0.600|0.031|4|0.152 [0.048, 0.357]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/astar|10|0.400|1.000|0.600|0.031|4|0.119 [0.095, 0.199]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/astar|10|0.400|1.000|0.600|0.031|4|0.329 [0.273, 0.450]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/astar|10|0.400|1.000|0.600|0.031|4|0.301 [0.254, 0.462]|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm/astar|10|0.000|0.000|0.000|n/a (n<6)|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet/astar|10|0.000|0.000|0.000|n/a (n<6)|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm/astar|10|0.000|0.000|0.000|n/a (n<6)|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm/astar|10|0.000|0.000|0.000|n/a (n<6)|0|n/a|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|10|0.300|1.000|0.700|0.016|3|0.083 [0.054, 0.094]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/astar|10|0.300|1.000|0.700|0.016|3|0.123 [0.066, 0.319]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/astar|10|0.300|1.000|0.700|0.016|3|0.130 [0.115, 0.282]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/astar|10|0.300|1.000|0.700|0.016|3|0.125 [0.086, 0.224]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/astar|10|0.400|1.000|0.600|0.031|4|0.540 [0.145, 0.743]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/astar|10|0.400|1.000|0.600|0.031|4|0.633 [0.256, 1.300]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/astar|10|0.400|0.800|0.400|n/a (n<6)|4|0.528 [0.229, 0.731]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/astar|10|0.400|0.600|0.200|0.688|2|1.323 [0.205, 2.441]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/astar|10|0.100|1.000|0.900|0.004|1|0.345 [0.345, 0.345]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|10|0.100|1.000|0.900|0.004|1|0.135 [0.135, 0.135]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|10|0.100|0.800|0.700|0.016|1|0.323 [0.323, 0.323]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|10|0.100|0.800|0.700|0.016|1|0.258 [0.258, 0.258]|n/a (n<6)|

## 2. Time-aware vs time-blind — THE SPOTLIGHT: does the future window help?

Matched expansion ratio of the time-AWARE arm vs its time-BLIND (W=0) twin (e.g. `scalar_hrm` vs `scalar_hrm_blind`, `field_unet` vs `field_unet_blind`), both astar.
Median ratio < 1 means the aware model expands fewer nodes than its blind twin (the future window helps). Success delta = aware - blind over shared worlds.

|Suite|Budget|Aware|Blind|n|Aware succ|Blind succ|Succ delta|n matched|Median ratio aware/blind (95% CI)|Wilcoxon p|
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|1.050 [0.906, 1.300]|0.445|
|C_dyn_crossing|150|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|0.964 [0.748, 1.095]|0.438|
|C_dyn_crossing|150|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|1.152 [0.759, 2.541]|0.275|
|C_dyn_crossing|150|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.166 [0.950, 1.529]|0.312|
|C_dyn_maze|1800|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|1.128 [1.003, 1.384]|0.027|
|C_dyn_maze|1800|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|0.988 [0.800, 1.085]|0.625|
|C_dyn_maze|1800|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|1.832 [1.244, 3.112]|0.014|
|C_dyn_maze|1800|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.099 [0.743, 1.633]|0.557|
|C_dyn_maze_dense|150|scalar_hrm|scalar_hrm_blind|10|0.000|0.000|0.000|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm|scalar_onlstm_blind|10|0.000|0.000|0.000|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet|field_unet_blind|10|0.000|0.000|0.000|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm|field_hrm_blind|10|0.000|0.000|0.000|0|n/a|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|0.723 [0.696, 0.835]|0.004|
|C_dyn_rooms|1300|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|0.821 [0.668, 1.013]|0.084|
|C_dyn_rooms|1300|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|1.063 [0.958, 1.303]|0.232|
|C_dyn_rooms|1300|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.033 [0.755, 1.618]|0.846|
|C_dyn_rooms_large|400|scalar_hrm|scalar_hrm_blind|10|0.800|1.000|-0.200|8|1.300 [0.986, 1.742]|0.055|
|C_dyn_rooms_large|400|scalar_onlstm|scalar_onlstm_blind|10|0.600|0.800|-0.200|6|0.900 [0.673, 1.710]|0.844|
|C_dyn_rooms_large|400|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|1.397 [1.015, 2.273]|0.027|
|C_dyn_rooms_large|400|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.002 [0.847, 1.400]|0.770|
|C_dyn_spiral|2500|scalar_hrm|scalar_hrm_blind|10|0.800|0.900|-0.100|8|0.925 [0.734, 1.063]|0.312|
|C_dyn_spiral|2500|scalar_onlstm|scalar_onlstm_blind|10|0.800|0.700|0.100|7|0.933 [0.770, 1.030]|0.375|
|C_dyn_spiral|2500|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|0.763 [0.681, 0.987]|0.020|
|C_dyn_spiral|2500|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|0.875 [0.694, 1.062]|0.160|

## 3. Additive (astar) vs focal — does C7's additive-wins hold under dynamics?

For each learned arm: its astar (additive) ratio vs euclid, and its best-w focal ratio vs euclid, side by side. Best w = lowest matched-median exp_ratio (post-hoc; the focal p is optimistic — treat the CI as primary).

|Suite|Budget|Arm|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|
|---|---:|---|---|---:|---|---:|
|C_dyn_crossing|150|field_hrm|0.125 [0.082, 0.169]|1.1|0.767 [0.735, 0.800]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind|0.211 [0.182, 0.241]|1.1|0.799 [0.771, 0.827]|n/a (n<6)|
|C_dyn_crossing|150|field_unet|0.249 [0.209, 0.289]|1.1|0.865 [0.827, 0.904]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind|0.136 [0.108, 0.164]|1.1|0.784 [0.759, 0.809]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm|0.213 [0.100, 0.325]|1.1|0.617 [0.464, 0.771]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind|0.130 [0.091, 0.169]|1.1|0.617 [0.464, 0.771]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm|0.196 [0.091, 0.301]|1.1|0.443 [0.422, 0.464]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind|0.732 [0.091, 1.373]|1.1|0.617 [0.464, 0.771]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm|0.152 [0.048, 0.357]|1.1|0.858 [0.800, 0.965]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind|0.077 [0.056, 0.267]|1.1|0.917 [0.768, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|field_unet|0.119 [0.095, 0.199]|1.1|0.919 [0.822, 0.993]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind|0.058 [0.039, 0.100]|1.1|0.872 [0.787, 0.997]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm|0.329 [0.273, 0.450]|1.1|0.895 [0.855, 0.993]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind|0.277 [0.213, 0.302]|1.1|0.947 [0.786, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm|0.301 [0.254, 0.462]|1.1|0.867 [0.824, 0.955]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind|0.340 [0.253, 0.420]|1.1|0.942 [0.788, 0.994]|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|field_hrm_blind|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|field_unet|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|field_unet_blind|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|scalar_hrm|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|scalar_hrm_blind|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|scalar_onlstm|n/a|n/a|n/a|n/a|
|C_dyn_maze_dense|150|scalar_onlstm_blind|n/a|n/a|n/a|n/a|
|C_dyn_rooms|1300|field_hrm|0.083 [0.054, 0.094]|1.1|0.831 [0.709, 0.858]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind|0.093 [0.091, 0.117]|1.1|0.968 [0.711, 0.972]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet|0.123 [0.066, 0.319]|1.1|0.899 [0.783, 0.923]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind|0.122 [0.071, 0.292]|1.1|0.924 [0.909, 0.960]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|0.130 [0.115, 0.282]|1.1|0.832 [0.756, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind|0.271 [0.161, 0.326]|1.1|0.756 [0.714, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm|0.125 [0.086, 0.224]|1.1|0.959 [0.831, 0.961]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind|0.134 [0.129, 0.283]|1.1|0.865 [0.756, 0.981]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm|0.540 [0.145, 0.743]|1.1|0.853 [0.713, 0.916]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind|0.410 [0.141, 0.776]|1.1|0.899 [0.727, 0.960]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet|0.633 [0.256, 1.300]|1.1|0.858 [0.656, 0.963]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind|0.284 [0.190, 1.131]|1.1|0.742 [0.588, 0.818]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm|0.528 [0.229, 0.731]|1.1|0.845 [0.671, 0.956]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind|0.389 [0.212, 0.574]|1.1|0.752 [0.560, 0.855]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm|1.323 [0.205, 2.441]|1.1|0.883 [0.845, 1.105]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind|1.118 [0.273, 1.216]|1.1|0.734 [0.664, 0.835]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm|0.345 [0.345, 0.345]|1.1|0.927 [0.927, 0.927]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind|0.164 [0.164, 0.164]|1.1|0.864 [0.864, 0.864]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|0.135 [0.135, 0.135]|1.1|0.991 [0.991, 0.991]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind|0.115 [0.115, 0.115]|1.1|0.931 [0.931, 0.931]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|0.323 [0.323, 0.323]|1.1|0.817 [0.817, 0.817]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind|0.422 [0.422, 0.422]|1.1|0.809 [0.809, 0.809]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|0.258 [0.258, 0.258]|1.1|0.991 [0.991, 0.991]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind|0.281 [0.281, 0.281]|1.1|0.989 [0.989, 0.989]|n/a (n<6)|

## 4. Recurrent/hierarchical vs field U-Net — do temporal models win when timing matters?

exp_ratio vs euclid (astar) side by side: the recurrent/hierarchical arms (scalar_hrm, scalar_onlstm, field_hrm, field_onlstm) vs the convolutional `field_unet`.

|Suite|Budget|Arm|n matched|Median ratio vs euclid (95% CI)|Wilcoxon p|
|---|---:|---|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|2|0.213 [0.100, 0.325]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm|2|0.196 [0.091, 0.301]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm|2|0.125 [0.082, 0.169]|n/a (n<6)|
|C_dyn_crossing|150|field_unet|2|0.249 [0.209, 0.289]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm|4|0.329 [0.273, 0.450]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm|4|0.301 [0.254, 0.462]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm|4|0.152 [0.048, 0.357]|n/a (n<6)|
|C_dyn_maze|1800|field_unet|4|0.119 [0.095, 0.199]|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm|0|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet|0|n/a|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|3|0.130 [0.115, 0.282]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm|3|0.125 [0.086, 0.224]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm|3|0.083 [0.054, 0.094]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet|3|0.123 [0.066, 0.319]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm|4|0.528 [0.229, 0.731]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm|2|1.323 [0.205, 2.441]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm|4|0.540 [0.145, 0.743]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet|4|0.633 [0.256, 1.300]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|1|0.323 [0.323, 0.323]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|1|0.258 [0.258, 0.258]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm|1|0.345 [0.345, 0.345]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|1|0.135 [0.135, 0.135]|n/a (n<6)|

## 5. Learned vs oracle — gap-to-ceiling

Median over the triple-matched set (euclid, oracle, arm all solved) of
`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the
euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).

|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|
|---|---:|---|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|2|-0.017|
|C_dyn_crossing|150|field_hrm_blind/astar|2|0.083|
|C_dyn_crossing|150|field_unet/astar|2|0.127|
|C_dyn_crossing|150|field_unet_blind/astar|2|-0.005|
|C_dyn_crossing|150|scalar_hrm/astar|2|0.085|
|C_dyn_crossing|150|scalar_hrm_blind/astar|2|-0.012|
|C_dyn_crossing|150|scalar_onlstm/astar|2|0.065|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|2|0.692|
|C_dyn_maze|1800|field_hrm/astar|4|0.083|
|C_dyn_maze|1800|field_hrm_blind/astar|4|0.042|
|C_dyn_maze|1800|field_unet/astar|4|0.098|
|C_dyn_maze|1800|field_unet_blind/astar|4|0.039|
|C_dyn_maze|1800|scalar_hrm/astar|4|0.316|
|C_dyn_maze|1800|scalar_hrm_blind/astar|4|0.231|
|C_dyn_maze|1800|scalar_onlstm/astar|4|0.287|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|4|0.319|
|C_dyn_maze_dense|150|field_hrm/astar|0|n/a|
|C_dyn_maze_dense|150|field_hrm_blind/astar|0|n/a|
|C_dyn_maze_dense|150|field_unet/astar|0|n/a|
|C_dyn_maze_dense|150|field_unet_blind/astar|0|n/a|
|C_dyn_maze_dense|150|scalar_hrm/astar|0|n/a|
|C_dyn_maze_dense|150|scalar_hrm_blind/astar|0|n/a|
|C_dyn_maze_dense|150|scalar_onlstm/astar|0|n/a|
|C_dyn_maze_dense|150|scalar_onlstm_blind/astar|0|n/a|
|C_dyn_rooms|1300|field_hrm/astar|3|0.065|
|C_dyn_rooms|1300|field_hrm_blind/astar|3|0.076|
|C_dyn_rooms|1300|field_unet/astar|3|0.106|
|C_dyn_rooms|1300|field_unet_blind/astar|3|0.105|
|C_dyn_rooms|1300|scalar_hrm/astar|3|0.113|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|3|0.257|
|C_dyn_rooms|1300|scalar_onlstm/astar|3|0.108|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|3|0.118|
|C_dyn_rooms_large|400|field_hrm/astar|4|0.504|
|C_dyn_rooms_large|400|field_hrm_blind/astar|4|0.363|
|C_dyn_rooms_large|400|field_unet/astar|4|0.595|
|C_dyn_rooms_large|400|field_unet_blind/astar|4|0.212|
|C_dyn_rooms_large|400|scalar_hrm/astar|4|0.485|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|4|0.330|
|C_dyn_rooms_large|400|scalar_onlstm/astar|2|1.382|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|3|1.133|
|C_dyn_spiral|2500|field_hrm/astar|1|0.331|
|C_dyn_spiral|2500|field_hrm_blind/astar|1|0.146|
|C_dyn_spiral|2500|field_unet/astar|1|0.117|
|C_dyn_spiral|2500|field_unet_blind/astar|1|0.096|
|C_dyn_spiral|2500|scalar_hrm/astar|1|0.309|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|1|0.410|
|C_dyn_spiral|2500|scalar_onlstm/astar|1|0.242|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|1|0.266|

## 6. In-distribution vs held-out — best learned arm exp_ratio + success vs euclid

Best learned arm (lowest pooled matched-median exp_ratio vs euclid): **field_hrm**/astar.  In-distribution (trained): C_dyn_maze, C_dyn_rooms, C_dyn_spiral.  Held-out (OOD): C_dyn_maze_dense, C_dyn_crossing, C_dyn_rooms_large.

|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|
|---|---|---:|---:|---|---:|
|in-dist|C_dyn_maze|1800|4|0.152 [0.048, 0.357]|0.600|
|in-dist|C_dyn_rooms|1300|3|0.083 [0.054, 0.094]|0.700|
|in-dist|C_dyn_spiral|2500|1|0.345 [0.345, 0.345]|0.900|
|held-out|C_dyn_maze_dense|150|0|n/a|0.000|
|held-out|C_dyn_crossing|150|2|0.125 [0.082, 0.169]|0.800|
|held-out|C_dyn_rooms_large|400|4|0.540 [0.145, 0.743]|0.600|

## Notes

- Each comparison uses the per-suite binding budget. Arms or suites absent from this run
  are skipped with a note rather than crashing.
- Comparison 2 (the spotlight) pairs each time-aware arm with its W=0 time-blind twin;
  median ratio < 1 means the future window reduces expansions.
- Comparison 3 picks the focal `w` with the lowest matched-median exp_ratio vs euclid;
  reporting that winner's p on the same data is optimistic — the CI is the primary inference.
- These six p-values are UNcorrected (BH applies only to the success/McNemar grid).
- The Wilcoxon p tests paired (ratio - 1) in ratio-space, matching the median ratio + CI.
- Bootstrap CIs are seeded (`np.random.default_rng(cfg.seed)`), so this analysis is reproducible.
