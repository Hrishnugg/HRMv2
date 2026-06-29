# C8 Dynamics Comparison — Significance

## Success: McNemar (learned arm vs euclid-time/astar), BH-corrected across the grid

_Family: learned arms only (oracle ceiling and euclid-time reference excluded). BH correction is applied to THIS success/McNemar grid only._

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|Discordant|McNemar p|BH q|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_crossing|150|field_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|20|0.300|0.600|0.300|6|0|6|0.031|0.145|
|C_dyn_crossing|150|field_hrm_blind/astar|20|0.300|0.950|0.650|13|0|13|<0.001|0.002|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|20|0.300|0.650|0.350|7|0|7|0.016|0.076|
|C_dyn_crossing|150|field_unet/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_crossing|150|field_unet/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet/focal/w=1.1|20|0.300|0.650|0.350|7|0|7|0.016|0.076|
|C_dyn_crossing|150|field_unet_blind/astar|20|0.300|0.950|0.650|13|0|13|<0.001|0.002|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|20|0.300|0.550|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_crossing|150|scalar_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|20|0.300|0.550|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_crossing|150|scalar_hrm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|20|0.300|0.650|0.350|7|0|7|0.016|0.076|
|C_dyn_crossing|150|scalar_onlstm/astar|20|0.300|0.700|0.400|9|1|10|0.021|0.103|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|20|0.300|0.600|0.300|6|0|6|0.031|0.145|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|20|0.300|0.650|0.350|7|0|7|0.016|0.076|
|C_dyn_crossing|250|field_hrm/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|field_hrm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|20|0.800|0.900|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/astar|20|0.800|0.950|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|field_unet/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|field_unet_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet_blind/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|scalar_hrm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/astar|20|0.800|0.850|0.050|3|2|5|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|field_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|20|0.300|0.350|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_maze|1800|field_unet/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|field_unet/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet/focal/w=1.1|20|0.300|0.350|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|20|0.300|0.400|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|20|0.300|0.350|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|20|0.300|0.350|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|20|0.300|0.350|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|field_hrm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|field_unet/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|20|0.800|0.850|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|20|0.800|1.000|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|20|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_hrm/astar|20|0.050|0.700|0.650|13|0|13|<0.001|0.002|
|C_dyn_maze_dense|2500|field_hrm/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_hrm/focal/w=1.1|20|0.050|0.100|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_hrm_blind/astar|20|0.050|0.650|0.600|12|0|12|<0.001|0.003|
|C_dyn_maze_dense|2500|field_hrm_blind/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_hrm_blind/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_unet/astar|20|0.050|0.750|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze_dense|2500|field_unet/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_unet/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_unet_blind/astar|20|0.050|0.750|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze_dense|2500|field_unet_blind/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|field_unet_blind/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_hrm/astar|20|0.050|0.550|0.500|10|0|10|0.002|0.010|
|C_dyn_maze_dense|2500|scalar_hrm/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_hrm/focal/w=1.1|20|0.050|0.100|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_hrm_blind/astar|20|0.050|0.650|0.600|12|0|12|<0.001|0.003|
|C_dyn_maze_dense|2500|scalar_hrm_blind/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_hrm_blind/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_onlstm/astar|20|0.050|0.450|0.400|8|0|8|0.008|0.041|
|C_dyn_maze_dense|2500|scalar_onlstm/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_onlstm/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/astar|20|0.050|0.600|0.550|11|0|11|<0.001|0.005|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/focal/w=1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/focal/w=1.1|20|0.050|0.050|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm/astar|20|0.200|0.900|0.700|14|0|14|<0.001|0.001|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|20|0.200|0.400|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|20|0.200|0.950|0.750|15|0|15|<0.001|0.001|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|20|0.200|0.400|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze_dense|3500|field_unet/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|20|0.200|0.350|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_maze_dense|3500|field_unet_blind/astar|20|0.200|0.950|0.750|15|0|15|<0.001|0.001|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|20|0.200|0.350|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_maze_dense|3500|scalar_hrm/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|20|0.200|0.400|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|20|0.200|0.800|0.600|12|0|12|<0.001|0.003|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|20|0.200|0.400|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|20|0.200|0.700|0.500|10|0|10|0.002|0.010|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|20|0.200|0.400|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|20|0.200|0.350|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|field_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|field_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|field_hrm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|20|0.300|0.500|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_rooms|1300|field_unet/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|field_unet/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|20|0.300|0.550|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1300|field_unet_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|scalar_hrm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|scalar_onlstm/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|20|0.300|0.450|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|20|0.300|1.000|0.700|14|0|14|<0.001|0.001|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|20|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|20|0.300|0.400|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|field_hrm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|20|0.750|0.850|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|field_unet/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|20|0.750|0.900|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms|1800|field_unet_blind/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|20|0.750|1.000|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/astar|20|0.750|0.900|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/astar|20|0.750|0.850|0.100|3|1|4|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/astar|20|0.750|0.900|0.150|4|1|5|n/a (n<6)|0.939|
|C_dyn_rooms_large|600|field_unet/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/astar|20|0.750|0.950|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/astar|20|0.750|0.950|0.200|5|1|6|0.219|0.637|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|20|0.750|0.750|0.000|3|3|6|1.000|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/astar|20|0.750|0.650|-0.100|1|3|4|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|20|0.750|0.900|0.150|4|1|5|n/a (n<6)|0.939|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|20|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|20|0.750|0.800|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm/astar|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm_blind/astar|20|0.950|0.950|0.000|1|1|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm_blind/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_hrm_blind/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet/astar|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet_blind/astar|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet_blind/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|field_unet_blind/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm/astar|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm_blind/astar|20|0.950|0.950|0.000|1|1|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm_blind/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_hrm_blind/focal/w=1.1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm/astar|20|0.950|0.950|0.000|1|1|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm/focal/w=1.1|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm_blind/astar|20|0.950|0.950|0.000|1|1|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm_blind/focal/w=1|20|0.950|0.950|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|900|scalar_onlstm_blind/focal/w=1.1|20|0.950|1.000|0.050|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/astar|20|0.200|0.900|0.700|14|0|14|<0.001|0.001|
|C_dyn_spiral|2500|field_hrm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/astar|20|0.200|0.900|0.700|14|0|14|<0.001|0.001|
|C_dyn_spiral|2500|field_unet/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/astar|20|0.200|0.950|0.750|15|0|15|<0.001|0.001|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|20|0.200|0.850|0.650|13|0|13|<0.001|0.002|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|20|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/astar|20|0.350|0.950|0.600|12|0|12|<0.001|0.003|
|C_dyn_spiral|3500|field_hrm/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|20|0.350|0.500|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_spiral|3500|field_hrm_blind/astar|20|0.350|0.900|0.550|11|0|11|<0.001|0.005|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|20|0.350|0.450|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet/astar|20|0.350|0.950|0.600|12|0|12|<0.001|0.003|
|C_dyn_spiral|3500|field_unet/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|20|0.350|0.600|0.250|5|0|5|n/a (n<6)|0.243|
|C_dyn_spiral|3500|field_unet_blind/astar|20|0.350|0.950|0.600|12|0|12|<0.001|0.003|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|20|0.350|0.550|0.200|4|0|4|n/a (n<6)|0.375|
|C_dyn_spiral|3500|scalar_hrm/astar|20|0.350|0.900|0.550|11|0|11|<0.001|0.005|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|20|0.350|0.500|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|20|0.350|0.900|0.550|11|0|11|<0.001|0.005|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|20|0.350|0.500|0.150|3|0|3|n/a (n<6)|0.637|
|C_dyn_spiral|3500|scalar_onlstm/astar|20|0.350|0.950|0.600|12|0|12|<0.001|0.003|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|20|0.350|0.450|0.100|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|20|0.350|0.900|0.550|11|0|11|<0.001|0.005|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|20|0.350|0.350|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|20|0.350|0.500|0.150|3|0|3|n/a (n<6)|0.637|

## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI

_Exploratory and UNcorrected: the Wilcoxon p-values below are NOT BH-corrected. Treat the bootstrap 95% CI on the median ratio as the primary inference._

|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|
|---|---:|---|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|6|0.349|[0.217, 0.458]|0.031|
|C_dyn_crossing|150|field_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|6|0.763|[0.673, 0.878]|0.031|
|C_dyn_crossing|150|field_hrm_blind/astar|6|0.231|[0.163, 0.516]|0.031|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|6|0.802|[0.699, 0.881]|0.031|
|C_dyn_crossing|150|field_unet/astar|6|0.258|[0.132, 0.769]|0.062|
|C_dyn_crossing|150|field_unet/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|field_unet/focal/w=1.1|6|0.854|[0.482, 0.980]|0.031|
|C_dyn_crossing|150|field_unet_blind/astar|6|0.191|[0.166, 0.878]|0.062|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|6|0.916|[0.645, 0.953]|0.031|
|C_dyn_crossing|150|scalar_hrm/astar|6|0.308|[0.198, 0.662]|0.031|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|6|0.838|[0.628, 0.926]|0.031|
|C_dyn_crossing|150|scalar_hrm_blind/astar|6|0.578|[0.172, 1.143]|0.219|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|6|0.771|[0.492, 0.965]|0.031|
|C_dyn_crossing|150|scalar_onlstm/astar|5|0.673|[0.292, 1.705]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|6|0.885|[0.643, 1.014]|0.094|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|6|0.644|[0.167, 1.048]|0.156|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|6|0.763|[0.495, 0.840]|0.031|
|C_dyn_crossing|250|field_hrm/astar|16|0.190|[0.115, 0.269]|<0.001|
|C_dyn_crossing|250|field_hrm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|16|0.763|[0.618, 0.833]|<0.001|
|C_dyn_crossing|250|field_hrm_blind/astar|16|0.185|[0.141, 0.319]|<0.001|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|16|0.740|[0.636, 0.826]|<0.001|
|C_dyn_crossing|250|field_unet/astar|16|0.152|[0.100, 0.270]|<0.001|
|C_dyn_crossing|250|field_unet/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet/focal/w=1.1|16|0.730|[0.647, 0.962]|<0.001|
|C_dyn_crossing|250|field_unet_blind/astar|16|0.204|[0.169, 0.333]|<0.001|
|C_dyn_crossing|250|field_unet_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet_blind/focal/w=1.1|16|0.852|[0.736, 0.923]|<0.001|
|C_dyn_crossing|250|scalar_hrm/astar|16|0.236|[0.167, 0.485]|<0.001|
|C_dyn_crossing|250|scalar_hrm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm/focal/w=1.1|16|0.786|[0.571, 0.916]|<0.001|
|C_dyn_crossing|250|scalar_hrm_blind/astar|16|0.163|[0.116, 0.380]|<0.001|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1.1|16|0.671|[0.555, 0.827]|<0.001|
|C_dyn_crossing|250|scalar_onlstm/astar|14|0.529|[0.342, 0.723]|0.025|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1.1|16|0.814|[0.607, 0.929]|<0.001|
|C_dyn_crossing|250|scalar_onlstm_blind/astar|16|0.183|[0.134, 0.365]|<0.001|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1.1|16|0.681|[0.571, 0.801]|<0.001|
|C_dyn_maze|1800|field_hrm/astar|6|0.418|[0.201, 0.642]|0.031|
|C_dyn_maze|1800|field_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|6|0.896|[0.788, 0.965]|0.031|
|C_dyn_maze|1800|field_hrm_blind/astar|6|0.054|[0.035, 0.105]|0.031|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|6|0.833|[0.779, 0.933]|0.031|
|C_dyn_maze|1800|field_unet/astar|6|0.064|[0.046, 0.088]|0.031|
|C_dyn_maze|1800|field_unet/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|field_unet/focal/w=1.1|6|0.849|[0.749, 0.938]|0.031|
|C_dyn_maze|1800|field_unet_blind/astar|6|0.093|[0.057, 0.137]|0.031|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|6|0.898|[0.803, 0.927]|0.031|
|C_dyn_maze|1800|scalar_hrm/astar|6|0.266|[0.196, 0.576]|0.031|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|6|0.864|[0.779, 0.991]|0.031|
|C_dyn_maze|1800|scalar_hrm_blind/astar|6|0.339|[0.217, 0.528]|0.031|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|6|0.882|[0.846, 0.977]|0.031|
|C_dyn_maze|1800|scalar_onlstm/astar|6|0.387|[0.250, 0.789]|0.031|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|6|0.935|[0.782, 0.985]|0.031|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|6|0.329|[0.188, 0.425]|0.031|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|6|0.874|[0.804, 0.930]|0.031|
|C_dyn_maze|2500|field_hrm/astar|16|0.241|[0.152, 0.488]|<0.001|
|C_dyn_maze|2500|field_hrm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|16|0.942|[0.881, 0.965]|<0.001|
|C_dyn_maze|2500|field_hrm_blind/astar|16|0.105|[0.060, 0.135]|<0.001|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|16|0.852|[0.822, 0.936]|<0.001|
|C_dyn_maze|2500|field_unet/astar|16|0.069|[0.059, 0.101]|<0.001|
|C_dyn_maze|2500|field_unet/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet/focal/w=1.1|16|0.957|[0.859, 0.980]|<0.001|
|C_dyn_maze|2500|field_unet_blind/astar|16|0.086|[0.051, 0.116]|<0.001|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|16|0.944|[0.898, 0.979]|<0.001|
|C_dyn_maze|2500|scalar_hrm/astar|16|0.289|[0.229, 0.390]|<0.001|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|16|0.872|[0.818, 0.935]|<0.001|
|C_dyn_maze|2500|scalar_hrm_blind/astar|16|0.353|[0.292, 0.513]|<0.001|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|16|0.939|[0.882, 0.974]|<0.001|
|C_dyn_maze|2500|scalar_onlstm/astar|16|0.405|[0.349, 0.469]|<0.001|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|16|0.942|[0.912, 0.986]|<0.001|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|16|0.300|[0.234, 0.400]|<0.001|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|16|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|16|0.923|[0.865, 0.951]|<0.001|
|C_dyn_maze_dense|2500|field_hrm/astar|1|0.337|[0.337, 0.337]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm/focal/w=1.1|1|0.991|[0.991, 0.991]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm_blind/astar|1|0.261|[0.261, 0.261]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_hrm_blind/focal/w=1.1|1|0.864|[0.864, 0.864]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet/astar|1|0.278|[0.278, 0.278]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet/focal/w=1.1|1|0.793|[0.793, 0.793]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet_blind/astar|1|0.246|[0.246, 0.246]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|field_unet_blind/focal/w=1.1|1|0.947|[0.947, 0.947]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm/astar|1|0.619|[0.619, 0.619]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm/focal/w=1.1|1|0.793|[0.793, 0.793]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm_blind/astar|1|0.472|[0.472, 0.472]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_hrm_blind/focal/w=1.1|1|0.884|[0.884, 0.884]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm/astar|1|0.678|[0.678, 0.678]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm/focal/w=1.1|1|0.884|[0.884, 0.884]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/astar|1|0.570|[0.570, 0.570]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|2500|scalar_onlstm_blind/focal/w=1.1|1|0.884|[0.884, 0.884]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/astar|4|0.331|[0.281, 0.514]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|4|0.855|[0.785, 0.991]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|4|0.462|[0.261, 0.498]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|4|0.921|[0.864, 0.955]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/astar|4|0.401|[0.278, 0.527]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|4|0.967|[0.793, 0.984]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/astar|4|0.271|[0.246, 0.343]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|4|0.950|[0.947, 0.969]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/astar|4|0.439|[0.344, 0.619]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|4|0.941|[0.793, 0.976]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|4|0.286|[0.225, 0.472]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|4|0.974|[0.884, 0.994]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|4|0.668|[0.467, 0.881]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|4|0.898|[0.884, 0.990]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|4|0.409|[0.274, 0.570]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|4|0.945|[0.884, 0.972]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|6|0.176|[0.099, 0.491]|0.031|
|C_dyn_rooms|1300|field_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|6|0.833|[0.736, 0.891]|0.031|
|C_dyn_rooms|1300|field_hrm_blind/astar|6|0.080|[0.034, 0.171]|0.031|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|6|0.833|[0.827, 0.953]|0.031|
|C_dyn_rooms|1300|field_unet/astar|6|0.096|[0.038, 0.191]|0.031|
|C_dyn_rooms|1300|field_unet/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|6|0.896|[0.690, 0.927]|0.031|
|C_dyn_rooms|1300|field_unet_blind/astar|6|0.099|[0.050, 0.214]|0.031|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|6|0.906|[0.810, 0.943]|0.031|
|C_dyn_rooms|1300|scalar_hrm/astar|6|0.200|[0.064, 0.429]|0.031|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|6|0.919|[0.814, 0.957]|0.031|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|6|0.184|[0.107, 0.470]|0.031|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|6|0.918|[0.840, 0.962]|0.031|
|C_dyn_rooms|1300|scalar_onlstm/astar|6|0.245|[0.142, 0.414]|0.031|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|6|0.907|[0.844, 0.966]|0.031|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|6|0.188|[0.105, 0.297]|0.031|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|6|0.838|[0.744, 0.932]|0.031|
|C_dyn_rooms|1800|field_hrm/astar|15|0.142|[0.112, 0.269]|<0.001|
|C_dyn_rooms|1800|field_hrm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|15|0.861|[0.827, 0.925]|<0.001|
|C_dyn_rooms|1800|field_hrm_blind/astar|15|0.089|[0.067, 0.147]|<0.001|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|15|0.850|[0.825, 0.953]|<0.001|
|C_dyn_rooms|1800|field_unet/astar|15|0.104|[0.081, 0.141]|<0.001|
|C_dyn_rooms|1800|field_unet/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|15|0.858|[0.803, 0.924]|<0.001|
|C_dyn_rooms|1800|field_unet_blind/astar|15|0.134|[0.098, 0.158]|<0.001|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|15|0.889|[0.836, 0.949]|<0.001|
|C_dyn_rooms|1800|scalar_hrm/astar|15|0.215|[0.153, 0.295]|<0.001|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|15|0.928|[0.860, 0.950]|<0.001|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|15|0.215|[0.152, 0.226]|<0.001|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|15|0.931|[0.853, 0.969]|<0.001|
|C_dyn_rooms|1800|scalar_onlstm/astar|15|0.259|[0.154, 0.336]|<0.001|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|15|0.934|[0.851, 0.960]|<0.001|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|15|0.204|[0.124, 0.252]|<0.001|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|15|0.883|[0.837, 0.957]|<0.001|
|C_dyn_rooms_large|600|field_hrm/astar|15|0.760|[0.110, 0.832]|0.002|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|15|0.892|[0.843, 0.955]|<0.001|
|C_dyn_rooms_large|600|field_hrm_blind/astar|14|0.581|[0.337, 0.782]|0.004|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|15|0.885|[0.872, 0.957]|<0.001|
|C_dyn_rooms_large|600|field_unet/astar|14|0.380|[0.132, 0.591]|0.013|
|C_dyn_rooms_large|600|field_unet/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|15|0.825|[0.739, 0.921]|<0.001|
|C_dyn_rooms_large|600|field_unet_blind/astar|15|0.228|[0.199, 0.418]|<0.001|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|15|0.862|[0.807, 0.896]|<0.001|
|C_dyn_rooms_large|600|scalar_hrm/astar|14|0.326|[0.207, 0.532]|<0.001|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|15|0.862|[0.737, 0.886]|<0.001|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|12|0.560|[0.147, 0.766]|0.042|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|15|0.869|[0.813, 0.927]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm/astar|12|0.682|[0.327, 0.899]|0.034|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|15|0.892|[0.819, 0.940]|<0.001|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|14|0.514|[0.269, 0.766]|0.007|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|15|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|15|0.943|[0.895, 0.982]|<0.001|
|C_dyn_rooms_large|900|field_hrm/astar|19|0.470|[0.129, 0.832]|<0.001|
|C_dyn_rooms_large|900|field_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|field_hrm/focal/w=1.1|19|0.894|[0.857, 0.955]|<0.001|
|C_dyn_rooms_large|900|field_hrm_blind/astar|18|0.527|[0.337, 0.716]|0.002|
|C_dyn_rooms_large|900|field_hrm_blind/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|field_hrm_blind/focal/w=1.1|19|0.913|[0.875, 0.954]|<0.001|
|C_dyn_rooms_large|900|field_unet/astar|19|0.442|[0.184, 0.654]|0.006|
|C_dyn_rooms_large|900|field_unet/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|field_unet/focal/w=1.1|19|0.847|[0.739, 0.921]|<0.001|
|C_dyn_rooms_large|900|field_unet_blind/astar|19|0.242|[0.199, 0.450]|<0.001|
|C_dyn_rooms_large|900|field_unet_blind/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|field_unet_blind/focal/w=1.1|19|0.868|[0.826, 0.896]|<0.001|
|C_dyn_rooms_large|900|scalar_hrm/astar|19|0.332|[0.225, 0.532]|<0.001|
|C_dyn_rooms_large|900|scalar_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|scalar_hrm/focal/w=1.1|19|0.864|[0.856, 0.915]|<0.001|
|C_dyn_rooms_large|900|scalar_hrm_blind/astar|18|0.630|[0.199, 0.885]|0.012|
|C_dyn_rooms_large|900|scalar_hrm_blind/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|scalar_hrm_blind/focal/w=1.1|19|0.887|[0.813, 0.927]|<0.001|
|C_dyn_rooms_large|900|scalar_onlstm/astar|18|0.833|[0.523, 0.983]|0.043|
|C_dyn_rooms_large|900|scalar_onlstm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|scalar_onlstm/focal/w=1.1|19|0.894|[0.847, 0.940]|<0.001|
|C_dyn_rooms_large|900|scalar_onlstm_blind/astar|18|0.483|[0.205, 0.665]|0.006|
|C_dyn_rooms_large|900|scalar_onlstm_blind/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|900|scalar_onlstm_blind/focal/w=1.1|19|0.937|[0.895, 0.979]|<0.001|
|C_dyn_spiral|2500|field_hrm/astar|4|0.046|[0.038, 0.073]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|4|0.894|[0.801, 0.993]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/astar|4|0.053|[0.050, 0.060]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|4|0.971|[0.949, 0.989]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|4|0.076|[0.055, 0.097]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|4|0.888|[0.818, 0.996]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/astar|4|0.087|[0.061, 0.287]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|4|0.931|[0.802, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|4|0.363|[0.064, 0.614]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|4|0.932|[0.803, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|4|0.334|[0.068, 0.609]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|4|0.912|[0.767, 0.973]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|4|0.345|[0.180, 0.639]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|4|0.813|[0.795, 0.959]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|4|0.353|[0.134, 0.552]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|4|0.901|[0.803, 0.989]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/astar|7|0.054|[0.038, 0.073]|0.016|
|C_dyn_spiral|3500|field_hrm/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|7|0.919|[0.801, 0.993]|0.016|
|C_dyn_spiral|3500|field_hrm_blind/astar|7|0.054|[0.050, 0.110]|0.016|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|7|0.965|[0.949, 0.989]|0.016|
|C_dyn_spiral|3500|field_unet/astar|7|0.078|[0.055, 0.124]|0.016|
|C_dyn_spiral|3500|field_unet/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|7|0.956|[0.864, 0.995]|0.016|
|C_dyn_spiral|3500|field_unet_blind/astar|7|0.097|[0.061, 0.287]|0.016|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|7|0.874|[0.808, 0.989]|0.016|
|C_dyn_spiral|3500|scalar_hrm/astar|7|0.377|[0.159, 0.614]|0.016|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|7|0.970|[0.878, 0.987]|0.016|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|7|0.473|[0.160, 0.545]|0.016|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|7|0.947|[0.846, 0.973]|0.016|
|C_dyn_spiral|3500|scalar_onlstm/astar|7|0.308|[0.223, 0.639]|0.016|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|7|0.853|[0.803, 0.958]|0.016|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|7|0.441|[0.203, 0.552]|0.016|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|7|1.000|[1.000, 1.000]|n/a|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|7|0.957|[0.823, 0.983]|0.016|

## Notes

- McNemar pairs each LEARNED arm against `euclid/astar` on success over shared worlds;
  gain = arm found & euclid not, loss = euclid found & arm not. `oracle` (space-time ceiling)
  and `euclid` (time-aware reference) are NOT hypotheses under test and are excluded.
- BH q-values correct ONLY across this success/McNemar grid. The expansion-Wilcoxon
  p-values are UNcorrected; the bootstrap CIs are the primary expansion inference.
- The expansion ratio uses the *matched set* (worlds euclid AND the arm both solved).
  Median ratio < 1 means the arm expands fewer nodes than euclid-time. The Wilcoxon p tests
  paired (ratio - 1) in ratio-space (matching the median ratio + CI estimand).
- A p-value is shown as `n/a (n<6)` when the McNemar discordant count or the
  expansion matched-set n is below 6 (too few pairs for a trustworthy p).
