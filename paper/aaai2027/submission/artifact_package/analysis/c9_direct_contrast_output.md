# Direct paired full-FT $-$ LoRA contrasts (C9 raw, map-clustered)

| target | backbone | K | n maps | d success [95% CI] | d ratio [95% CI] | matched n |
|---|---|---|---|---|---|---|
| C_hard_bugtrap | hrm | 1 | 30 | -0.467 [-0.587,-0.340] | +0.225 [+0.089,+0.352] | 13 |
| C_hard_bugtrap | hrm | 16 | 30 | -0.020 [-0.120,+0.080] | -0.145 [-0.239,-0.043] | 16 |
| C_hard_bugtrap | onlstm | 1 | 30 | -0.093 [-0.207,+0.007] | +0.229 [+0.117,+0.348] | 15 |
| C_hard_bugtrap | onlstm | 16 | 30 | +0.100 [+0.013,+0.207] | -0.052 [-0.154,+0.033] | 15 |
| C_hard_maze_dense | hrm | 1 | 30 | -0.240 [-0.360,-0.133] | +0.112 [+0.052,+0.171] | 10 |
| C_hard_maze_dense | hrm | 16 | 30 | +0.027 [-0.020,+0.100] | -0.149 [-0.179,-0.117] | 10 |
| C_hard_maze_dense | onlstm | 1 | 30 | -0.047 [-0.147,+0.073] | -0.162 [-0.223,-0.089] | 10 |
| C_hard_maze_dense | onlstm | 16 | 30 | +0.000 [-0.027,+0.033] | -0.126 [-0.171,-0.083] | 10 |
| C_hard_rooms_large | hrm | 1 | 30 | -0.600 [-0.773,-0.407] | +0.290 [+0.207,+0.360] | 11 |
| C_hard_rooms_large | hrm | 16 | 30 | +0.073 [-0.027,+0.187] | -0.366 [-0.489,-0.235] | 11 |
| C_hard_rooms_large | onlstm | 1 | 30 | -0.293 [-0.433,-0.160] | +0.013 [-0.156,+0.167] | 11 |
| C_hard_rooms_large | onlstm | 16 | 30 | -0.053 [-0.167,+0.053] | -0.170 [-0.326,-0.010] | 11 |

Ratio-delta direction count (all cells): FT better 7, LoRA better 5.
Negative d ratio = full-FT expands less than LoRA (relative to euclid on triple-matched maps).