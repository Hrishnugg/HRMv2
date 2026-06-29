# C8 Dynamics Comparison — Pre-registered Comparisons

Binding budget per suite (lowest calibrated-band budget where euclid success >= 0.05, so a degenerate 0%-success edge is skipped; if no band budget qualifies, the highest band budget; else the first budget seen): C_dyn_crossing=150, C_dyn_maze=1800, C_dyn_maze_dense=3500, C_dyn_rooms=1300, C_dyn_rooms_large=400, C_dyn_spiral=2500

_Multiplicity: BH correction is applied ONLY to the success/McNemar grid over learned arms (see `continuous_prm_c8_significance.md`). The p-values in THESE six pre-registered comparisons are UNcorrected; treat the bootstrap 95% CIs as the primary inference._

_Small-n: a p shown as `n/a (n<6)` had too few discordant/matched pairs to trust (counts / median / CI are still reported)._

## 1. Time-aware learned vs euclid-time (expansions + success), per suite

Each time-aware learned arm (field_<bb>/astar, scalar_<bb>/astar) vs `euclid/astar`.

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Succ delta|McNemar p|n matched|Median ratio (95% CI)|Wilcoxon p|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.152 [0.100, 0.205]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/astar|10|0.200|1.000|0.800|0.008|2|0.185 [0.082, 0.289]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.178 [0.091, 0.265]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/astar|10|0.200|1.000|0.800|0.008|2|0.131 [0.118, 0.145]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/astar|10|0.400|1.000|0.600|0.031|4|0.432 [0.117, 0.818]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/astar|10|0.400|0.900|0.500|n/a (n<6)|4|0.168 [0.113, 0.403]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/astar|10|0.400|1.000|0.600|0.031|4|0.651 [0.294, 0.804]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/astar|10|0.400|1.000|0.600|0.031|4|0.469 [0.168, 0.523]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/astar|10|0.100|0.800|0.700|0.016|1|0.407 [0.407, 0.407]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/astar|10|0.100|0.800|0.700|0.016|1|0.523 [0.523, 0.523]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/astar|10|0.100|0.700|0.600|0.031|1|0.229 [0.229, 0.229]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|10|0.100|0.700|0.600|0.031|1|0.350 [0.350, 0.350]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|10|0.100|1.000|0.900|0.004|1|0.129 [0.129, 0.129]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/astar|10|0.100|1.000|0.900|0.004|1|0.158 [0.158, 0.158]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/astar|10|0.100|1.000|0.900|0.004|1|0.148 [0.148, 0.148]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/astar|10|0.100|1.000|0.900|0.004|1|0.190 [0.190, 0.190]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.742 [0.170, 1.315]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/astar|10|0.200|0.900|0.700|0.016|2|0.426 [0.184, 0.668]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/astar|10|0.200|1.000|0.800|0.008|2|0.628 [0.081, 1.174]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/astar|10|0.200|0.700|0.500|n/a (n<6)|2|0.901 [0.226, 1.577]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/astar|10|0.200|0.900|0.700|0.016|2|0.218 [0.188, 0.247]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|10|0.200|0.900|0.700|0.016|2|0.152 [0.118, 0.186]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|10|0.200|0.700|0.500|n/a (n<6)|2|0.434 [0.329, 0.539]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|10|0.200|0.700|0.500|n/a (n<6)|2|0.251 [0.196, 0.306]|n/a (n<6)|

## 2. Time-aware vs time-blind — THE SPOTLIGHT: does the future window help?

Matched expansion ratio of the time-AWARE arm vs its time-BLIND (W=0) twin (e.g. `scalar_hrm` vs `scalar_hrm_blind`, `field_unet` vs `field_unet_blind`), both astar.
Median ratio < 1 means the aware model expands fewer nodes than its blind twin (the future window helps). Success delta = aware - blind over shared worlds.

|Suite|Budget|Aware|Blind|n|Aware succ|Blind succ|Succ delta|n matched|Median ratio aware/blind (95% CI)|Wilcoxon p|
|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|0.962 [0.625, 1.083]|0.301|
|C_dyn_crossing|150|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|1.508 [0.909, 2.292]|0.027|
|C_dyn_crossing|150|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|0.849 [0.521, 1.038]|0.193|
|C_dyn_crossing|150|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|0.969 [0.811, 1.308]|0.922|
|C_dyn_maze|1800|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|1.064 [0.968, 1.294]|0.160|
|C_dyn_maze|1800|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|1.300 [1.044, 1.431]|0.006|
|C_dyn_maze|1800|field_unet|field_unet_blind|10|0.900|0.900|0.000|9|0.816 [0.641, 1.233]|0.359|
|C_dyn_maze|1800|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.097 [0.666, 2.363]|0.375|
|C_dyn_maze_dense|3500|scalar_hrm|scalar_hrm_blind|10|0.700|0.700|0.000|7|0.863 [0.838, 0.935]|0.016|
|C_dyn_maze_dense|3500|scalar_onlstm|scalar_onlstm_blind|10|0.700|0.900|-0.200|7|1.216 [1.053, 1.451]|0.156|
|C_dyn_maze_dense|3500|field_unet|field_unet_blind|10|0.800|0.700|0.100|7|1.036 [1.009, 1.115]|0.078|
|C_dyn_maze_dense|3500|field_hrm|field_hrm_blind|10|0.800|0.700|0.100|7|0.922 [0.687, 0.969]|0.016|
|C_dyn_rooms|1300|scalar_hrm|scalar_hrm_blind|10|1.000|1.000|0.000|10|0.886 [0.693, 1.117]|0.275|
|C_dyn_rooms|1300|scalar_onlstm|scalar_onlstm_blind|10|1.000|1.000|0.000|10|1.225 [0.798, 1.453]|0.131|
|C_dyn_rooms|1300|field_unet|field_unet_blind|10|1.000|1.000|0.000|10|0.941 [0.743, 1.683]|0.695|
|C_dyn_rooms|1300|field_hrm|field_hrm_blind|10|1.000|1.000|0.000|10|1.516 [0.652, 1.829]|0.131|
|C_dyn_rooms_large|400|scalar_hrm|scalar_hrm_blind|10|1.000|0.700|0.300|7|0.706 [0.206, 0.804]|0.078|
|C_dyn_rooms_large|400|scalar_onlstm|scalar_onlstm_blind|10|0.700|0.800|-0.100|6|1.117 [0.825, 1.692]|0.562|
|C_dyn_rooms_large|400|field_unet|field_unet_blind|10|0.900|0.800|0.100|8|0.687 [0.454, 1.039]|0.078|
|C_dyn_rooms_large|400|field_hrm|field_hrm_blind|10|1.000|0.500|0.500|5|1.480 [0.499, 2.390]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|scalar_hrm_blind|10|0.700|0.800|-0.100|7|0.853 [0.769, 1.103]|0.219|
|C_dyn_spiral|2500|scalar_onlstm|scalar_onlstm_blind|10|0.700|0.800|-0.100|7|0.853 [0.560, 1.006]|0.109|
|C_dyn_spiral|2500|field_unet|field_unet_blind|10|0.900|0.900|0.000|9|0.904 [0.721, 1.046]|0.250|
|C_dyn_spiral|2500|field_hrm|field_hrm_blind|10|0.900|0.900|0.000|9|1.187 [0.667, 1.862]|0.496|

## 3. Additive (astar) vs focal — does C7's additive-wins hold under dynamics?

For each learned arm: its astar (additive) ratio vs euclid, and its best-w focal ratio vs euclid, side by side. Best w = lowest matched-median exp_ratio (post-hoc; the focal p is optimistic — treat the CI as primary).

|Suite|Budget|Arm|astar ratio (CI)|best w|focal ratio (CI)|focal Wilcoxon p|
|---|---:|---|---|---:|---|---:|
|C_dyn_crossing|150|field_hrm|0.152 [0.100, 0.205]|1.1|0.859 [0.827, 0.892]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind|0.128 [0.100, 0.157]|1.1|0.772 [0.735, 0.809]|n/a (n<6)|
|C_dyn_crossing|150|field_unet|0.185 [0.082, 0.289]|1.1|0.877 [0.827, 0.928]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind|0.263 [0.253, 0.273]|1.1|0.639 [0.518, 0.759]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm|0.178 [0.091, 0.265]|1.1|0.660 [0.464, 0.855]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind|0.181 [0.145, 0.217]|1.1|0.828 [0.800, 0.855]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm|0.131 [0.118, 0.145]|1.1|0.641 [0.464, 0.819]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind|0.125 [0.082, 0.169]|1.1|0.611 [0.464, 0.759]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm|0.432 [0.117, 0.818]|1.1|0.972 [0.832, 0.991]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind|0.282 [0.120, 0.603]|1.1|0.859 [0.764, 0.916]|n/a (n<6)|
|C_dyn_maze|1800|field_unet|0.168 [0.113, 0.403]|1.1|0.890 [0.785, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind|0.187 [0.149, 0.268]|1.1|0.822 [0.764, 0.885]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm|0.651 [0.294, 0.804]|1.1|0.955 [0.929, 0.973]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind|0.538 [0.211, 0.572]|1.1|0.940 [0.901, 0.971]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm|0.469 [0.168, 0.523]|1.1|0.834 [0.755, 0.940]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind|0.360 [0.116, 0.546]|1.1|0.957 [0.929, 0.972]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm|0.407 [0.407, 0.407]|1.1|0.972 [0.972, 0.972]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind|0.608 [0.608, 0.608]|1.1|0.938 [0.938, 0.938]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet|0.523 [0.523, 0.523]|1.1|0.888 [0.888, 0.888]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind|0.537 [0.537, 0.537]|1.1|0.928 [0.928, 0.928]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm|0.229 [0.229, 0.229]|1.1|0.897 [0.897, 0.897]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind|0.408 [0.408, 0.408]|1.1|0.982 [0.982, 0.982]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm|0.350 [0.350, 0.350]|1.1|0.870 [0.870, 0.870]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind|0.569 [0.569, 0.569]|1.1|0.937 [0.937, 0.937]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm|0.129 [0.129, 0.129]|1.1|0.987 [0.987, 0.987]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind|0.076 [0.076, 0.076]|1.1|0.924 [0.924, 0.924]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet|0.158 [0.158, 0.158]|1.1|0.983 [0.983, 0.983]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind|0.080 [0.080, 0.080]|1.1|0.970 [0.970, 0.970]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|0.148 [0.148, 0.148]|1.1|0.979 [0.979, 0.979]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind|0.178 [0.178, 0.178]|1.1|0.729 [0.729, 0.729]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm|0.190 [0.190, 0.190]|1.1|0.926 [0.926, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind|0.075 [0.075, 0.075]|1.1|0.827 [0.827, 0.827]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm|0.742 [0.170, 1.315]|1.1|0.730 [0.598, 0.862]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind|0.078 [0.078, 0.078]|1.1|0.833 [0.701, 0.965]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet|0.426 [0.184, 0.668]|1.1|0.823 [0.721, 0.925]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind|0.408 [0.173, 0.643]|1.1|0.765 [0.664, 0.866]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm|0.628 [0.081, 1.174]|1.1|0.957 [0.942, 0.972]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind|0.956 [0.318, 1.593]|1.1|0.920 [0.855, 0.986]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm|0.901 [0.226, 1.577]|1.1|0.835 [0.731, 0.938]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind|0.800 [0.205, 1.394]|1.1|0.810 [0.656, 0.965]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm|0.218 [0.188, 0.247]|1.1|0.973 [0.971, 0.976]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind|0.097 [0.059, 0.134]|1.1|0.861 [0.833, 0.888]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|0.152 [0.118, 0.186]|1.1|0.961 [0.926, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind|0.179 [0.164, 0.194]|1.1|0.970 [0.942, 0.998]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|0.434 [0.329, 0.539]|1.1|0.970 [0.953, 0.988]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind|0.369 [0.298, 0.439]|1.1|0.928 [0.925, 0.930]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|0.251 [0.196, 0.306]|1.1|0.824 [0.810, 0.839]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind|0.388 [0.230, 0.546]|1.1|0.917 [0.892, 0.942]|n/a (n<6)|

## 4. Recurrent/hierarchical vs field U-Net — do temporal models win when timing matters?

exp_ratio vs euclid (astar) side by side: the recurrent/hierarchical arms (scalar_hrm, scalar_onlstm, field_hrm, field_onlstm) vs the convolutional `field_unet`.

|Suite|Budget|Arm|n matched|Median ratio vs euclid (95% CI)|Wilcoxon p|
|---|---:|---|---:|---|---:|
|C_dyn_crossing|150|scalar_hrm|2|0.178 [0.091, 0.265]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm|2|0.131 [0.118, 0.145]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm|2|0.152 [0.100, 0.205]|n/a (n<6)|
|C_dyn_crossing|150|field_unet|2|0.185 [0.082, 0.289]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm|4|0.651 [0.294, 0.804]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm|4|0.469 [0.168, 0.523]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm|4|0.432 [0.117, 0.818]|n/a (n<6)|
|C_dyn_maze|1800|field_unet|4|0.168 [0.113, 0.403]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm|1|0.229 [0.229, 0.229]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm|1|0.350 [0.350, 0.350]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm|1|0.407 [0.407, 0.407]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet|1|0.523 [0.523, 0.523]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm|1|0.148 [0.148, 0.148]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm|1|0.190 [0.190, 0.190]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm|1|0.129 [0.129, 0.129]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet|1|0.158 [0.158, 0.158]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm|2|0.628 [0.081, 1.174]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm|2|0.901 [0.226, 1.577]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm|2|0.742 [0.170, 1.315]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet|2|0.426 [0.184, 0.668]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm|2|0.434 [0.329, 0.539]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm|2|0.251 [0.196, 0.306]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm|2|0.218 [0.188, 0.247]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet|2|0.152 [0.118, 0.186]|n/a (n<6)|

## 5. Learned vs oracle — gap-to-ceiling

Median over the triple-matched set (euclid, oracle, arm all solved) of
`(arm_exp - oracle_exp) / (euclid_exp - oracle_exp)` — the fraction of the
euclid->oracle expansion gap left *uncaptured* (0 = matches oracle, 1 = no better than euclid).

|Suite|Budget|Arm|n triple-matched|Median uncaptured-gap fraction|
|---|---:|---|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|2|0.014|
|C_dyn_crossing|150|field_hrm_blind/astar|2|-0.014|
|C_dyn_crossing|150|field_unet/astar|2|0.053|
|C_dyn_crossing|150|field_unet_blind/astar|2|0.142|
|C_dyn_crossing|150|scalar_hrm/astar|2|0.044|
|C_dyn_crossing|150|scalar_hrm_blind/astar|2|0.048|
|C_dyn_crossing|150|scalar_onlstm/astar|2|-0.011|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|2|-0.017|
|C_dyn_maze|1800|field_hrm/astar|4|0.400|
|C_dyn_maze|1800|field_hrm_blind/astar|4|0.166|
|C_dyn_maze|1800|field_unet/astar|4|0.116|
|C_dyn_maze|1800|field_unet_blind/astar|4|0.116|
|C_dyn_maze|1800|scalar_hrm/astar|4|0.606|
|C_dyn_maze|1800|scalar_hrm_blind/astar|4|0.474|
|C_dyn_maze|1800|scalar_onlstm/astar|4|0.406|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|4|0.347|
|C_dyn_maze_dense|3500|field_hrm/astar|1|0.393|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|1|0.599|
|C_dyn_maze_dense|3500|field_unet/astar|1|0.512|
|C_dyn_maze_dense|3500|field_unet_blind/astar|1|0.526|
|C_dyn_maze_dense|3500|scalar_hrm/astar|1|0.211|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|1|0.395|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|1|0.335|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|1|0.559|
|C_dyn_rooms|1300|field_hrm/astar|1|0.113|
|C_dyn_rooms|1300|field_hrm_blind/astar|1|0.058|
|C_dyn_rooms|1300|field_unet/astar|1|0.142|
|C_dyn_rooms|1300|field_unet_blind/astar|1|0.062|
|C_dyn_rooms|1300|scalar_hrm/astar|1|0.132|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|1|0.162|
|C_dyn_rooms|1300|scalar_onlstm/astar|1|0.174|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|1|0.057|
|C_dyn_rooms_large|400|field_hrm/astar|2|0.705|
|C_dyn_rooms_large|400|field_hrm_blind/astar|1|-0.032|
|C_dyn_rooms_large|400|field_unet/astar|2|0.365|
|C_dyn_rooms_large|400|field_unet_blind/astar|2|0.346|
|C_dyn_rooms_large|400|scalar_hrm/astar|2|0.580|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|2|0.938|
|C_dyn_rooms_large|400|scalar_onlstm/astar|2|0.877|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|2|0.767|
|C_dyn_spiral|2500|field_hrm/astar|2|0.111|
|C_dyn_spiral|2500|field_hrm_blind/astar|2|-0.019|
|C_dyn_spiral|2500|field_unet/astar|2|0.043|
|C_dyn_spiral|2500|field_unet_blind/astar|2|0.072|
|C_dyn_spiral|2500|scalar_hrm/astar|2|0.371|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|2|0.293|
|C_dyn_spiral|2500|scalar_onlstm/astar|2|0.158|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|2|0.324|

## 6. In-distribution vs held-out — best learned arm exp_ratio + success vs euclid

Best learned arm (lowest pooled matched-median exp_ratio vs euclid): **field_unet**/astar.  In-distribution (trained): C_dyn_maze, C_dyn_rooms, C_dyn_spiral.  Held-out (OOD): C_dyn_maze_dense, C_dyn_crossing, C_dyn_rooms_large.

|Group|Suite|Budget|n matched|Median ratio (95% CI)|Succ delta vs euclid|
|---|---|---:|---:|---|---:|
|in-dist|C_dyn_maze|1800|4|0.168 [0.113, 0.403]|0.500|
|in-dist|C_dyn_rooms|1300|1|0.158 [0.158, 0.158]|0.900|
|in-dist|C_dyn_spiral|2500|2|0.152 [0.118, 0.186]|0.700|
|held-out|C_dyn_maze_dense|3500|1|0.523 [0.523, 0.523]|0.700|
|held-out|C_dyn_crossing|150|2|0.185 [0.082, 0.289]|0.800|
|held-out|C_dyn_rooms_large|400|2|0.426 [0.184, 0.668]|0.700|

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
