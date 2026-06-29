# C8 Heuristic Accuracy: Predicted Time-to-Go vs Oracle

Measures MAE of each C8 model's `h_table` against the exact backward-Dijkstra oracle, restricted to cells reachable within the planning horizon (`ttg <= t_max`). Cells are pooled across worlds (not averaged per-world then averaged).

Oracle-vs-oracle sanity MAE: **0.000000** (must be 0.0 -- PASSED)

## Suite: C_dyn_maze

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 13.753 | 18.283 | 0.627 | 155,977 |
| scalar_hrm | 4.432 | 6.407 | 0.202 | 155,977 |
| scalar_hrm_blind | 4.622 | 6.440 | 0.211 | 155,977 |
| scalar_onlstm | 5.509 | 8.221 | 0.251 | 155,977 |
| scalar_onlstm_blind | 4.695 | 6.499 | 0.214 | 155,977 |
| field_unet | 3.254 | 4.884 | 0.148 | 155,977 |
| field_unet_blind | 3.445 | 5.209 | 0.157 | 155,977 |
| field_hrm | 4.254 | 6.400 | 0.194 | 155,977 |
| field_hrm_blind | 3.444 | 5.203 | 0.157 | 155,977 |

*Mean oracle time-to-go (reachable cells): 21.95 steps*

## Suite: C_dyn_rooms

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 11.340 | 14.515 | 0.579 | 159,959 |
| scalar_hrm | 5.437 | 8.275 | 0.277 | 159,959 |
| scalar_hrm_blind | 4.768 | 6.839 | 0.243 | 159,959 |
| scalar_onlstm | 5.426 | 8.745 | 0.277 | 159,959 |
| scalar_onlstm_blind | 4.974 | 7.129 | 0.254 | 159,959 |
| field_unet | 2.618 | 3.904 | 0.134 | 159,959 |
| field_unet_blind | 2.892 | 4.270 | 0.148 | 159,959 |
| field_hrm | 3.757 | 6.118 | 0.192 | 159,959 |
| field_hrm_blind | 2.769 | 4.314 | 0.141 | 159,959 |

*Mean oracle time-to-go (reachable cells): 19.60 steps*

## Suite: C_dyn_spiral

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 17.500 | 24.201 | 0.666 | 136,668 |
| scalar_hrm | 5.562 | 8.376 | 0.212 | 136,668 |
| scalar_hrm_blind | 6.469 | 9.759 | 0.246 | 136,668 |
| scalar_onlstm | 5.136 | 8.126 | 0.196 | 136,668 |
| scalar_onlstm_blind | 6.314 | 9.429 | 0.240 | 136,668 |
| field_unet | 4.423 | 6.973 | 0.168 | 136,668 |
| field_unet_blind | 3.583 | 5.391 | 0.136 | 136,668 |
| field_hrm | 3.807 | 5.837 | 0.145 | 136,668 |
| field_hrm_blind | 3.455 | 5.360 | 0.132 | 136,668 |

*Mean oracle time-to-go (reachable cells): 26.26 steps*

## Suite: C_dyn_maze_dense

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 21.034 | 28.729 | 0.727 | 178,938 |
| scalar_hrm | 8.413 | 12.713 | 0.291 | 178,938 |
| scalar_hrm_blind | 9.016 | 13.205 | 0.312 | 178,938 |
| scalar_onlstm | 11.297 | 16.668 | 0.390 | 178,938 |
| scalar_onlstm_blind | 9.308 | 13.777 | 0.322 | 178,938 |
| field_unet | 10.250 | 15.744 | 0.354 | 178,938 |
| field_unet_blind | 6.815 | 10.541 | 0.236 | 178,938 |
| field_hrm | 8.768 | 13.212 | 0.303 | 178,938 |
| field_hrm_blind | 9.639 | 14.022 | 0.333 | 178,938 |

*Mean oracle time-to-go (reachable cells): 28.93 steps*

## Suite: C_dyn_crossing

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 3.492 | 4.132 | 0.286 | 181,702 |
| scalar_hrm | 11.187 | 16.424 | 0.916 | 181,702 |
| scalar_hrm_blind | 11.861 | 16.433 | 0.971 | 181,702 |
| scalar_onlstm | 11.120 | 15.811 | 0.910 | 181,702 |
| scalar_onlstm_blind | 11.549 | 16.346 | 0.945 | 181,702 |
| field_unet | 7.812 | 10.425 | 0.639 | 181,702 |
| field_unet_blind | 6.452 | 8.327 | 0.528 | 181,702 |
| field_hrm | 8.330 | 11.496 | 0.682 | 181,702 |
| field_hrm_blind | 9.194 | 11.827 | 0.752 | 181,702 |

*Mean oracle time-to-go (reachable cells): 12.22 steps*

## Suite: C_dyn_rooms_large

| model | MAE (steps) | RMSE | MAE/mean_oracle | n_cells |
|---|---|---|---|---|
| euclid | 6.349 | 8.476 | 0.405 | 149,675 |
| scalar_hrm | 20.573 | 27.605 | 1.312 | 149,675 |
| scalar_hrm_blind | 18.041 | 23.728 | 1.151 | 149,675 |
| scalar_onlstm | 12.343 | 17.097 | 0.787 | 149,675 |
| scalar_onlstm_blind | 17.967 | 23.930 | 1.146 | 149,675 |
| field_unet | 10.011 | 14.314 | 0.639 | 149,675 |
| field_unet_blind | 7.334 | 9.516 | 0.468 | 149,675 |
| field_hrm | 9.678 | 13.291 | 0.617 | 149,675 |
| field_hrm_blind | 8.849 | 10.866 | 0.564 | 149,675 |

*Mean oracle time-to-go (reachable cells): 15.68 steps*

## Aware-vs-Blind Summary

Positive delta = aware is WORSE (higher MAE); negative delta = aware is BETTER. `better` = whichever variant has lower MAE.

| suite | backbone | MAE_aware | MAE_blind | delta = aware - blind | better |
|---|---|---|---|---|---|
| C_dyn_maze | scalar_hrm | 4.432 | 4.622 | -0.190 | **aware** |
| C_dyn_maze | scalar_onlstm | 5.509 | 4.695 | +0.814 | **blind** |
| C_dyn_maze | field_unet | 3.254 | 3.445 | -0.191 | **aware** |
| C_dyn_maze | field_hrm | 4.254 | 3.444 | +0.810 | **blind** |
| C_dyn_rooms | scalar_hrm | 5.437 | 4.768 | +0.669 | **blind** |
| C_dyn_rooms | scalar_onlstm | 5.426 | 4.974 | +0.452 | **blind** |
| C_dyn_rooms | field_unet | 2.618 | 2.892 | -0.274 | **aware** |
| C_dyn_rooms | field_hrm | 3.757 | 2.769 | +0.988 | **blind** |
| C_dyn_spiral | scalar_hrm | 5.562 | 6.469 | -0.907 | **aware** |
| C_dyn_spiral | scalar_onlstm | 5.136 | 6.314 | -1.178 | **aware** |
| C_dyn_spiral | field_unet | 4.423 | 3.583 | +0.840 | **blind** |
| C_dyn_spiral | field_hrm | 3.807 | 3.455 | +0.352 | **blind** |
| C_dyn_maze_dense | scalar_hrm | 8.413 | 9.016 | -0.603 | **aware** |
| C_dyn_maze_dense | scalar_onlstm | 11.297 | 9.308 | +1.989 | **blind** |
| C_dyn_maze_dense | field_unet | 10.250 | 6.815 | +3.436 | **blind** |
| C_dyn_maze_dense | field_hrm | 8.768 | 9.639 | -0.871 | **aware** |
| C_dyn_crossing | scalar_hrm | 11.187 | 11.861 | -0.674 | **aware** |
| C_dyn_crossing | scalar_onlstm | 11.120 | 11.549 | -0.429 | **aware** |
| C_dyn_crossing | field_unet | 7.812 | 6.452 | +1.360 | **blind** |
| C_dyn_crossing | field_hrm | 8.330 | 9.194 | -0.864 | **aware** |
| C_dyn_rooms_large | scalar_hrm | 20.573 | 18.041 | +2.532 | **blind** |
| C_dyn_rooms_large | scalar_onlstm | 12.343 | 17.967 | -5.624 | **aware** |
| C_dyn_rooms_large | field_unet | 10.011 | 7.334 | +2.677 | **blind** |
| C_dyn_rooms_large | field_hrm | 9.678 | 8.849 | +0.829 | **blind** |

**Verdict:** out of 24 (suite, backbone) pairs, aware is more accurate (delta<0) in 11 and less accurate (delta>0) in 13 (mean delta = +0.248 steps). The future window makes the heuristic systematically WORSE on average - consistent with no search-expansion benefit from time-awareness.
