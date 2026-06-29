# C8 Dynamics Comparison — Significance

## Success: McNemar (learned arm vs euclid-time/astar), BH-corrected across the grid

_Family: learned arms only (oracle ceiling and euclid-time reference excluded). BH correction is applied to THIS success/McNemar grid only._

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|Discordant|McNemar p|BH q|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|field_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_crossing|150|field_unet/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|field_unet/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_crossing|150|scalar_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_crossing|150|scalar_hrm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_crossing|150|scalar_onlstm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.145|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_crossing|250|field_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_unet_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|field_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|field_unet/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/focal/w=1.1|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm_blind/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm_blind/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_hrm_blind/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet_blind/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet_blind/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|field_unet_blind/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm_blind/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm_blind/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_hrm_blind/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm_blind/astar|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm_blind/focal/w=1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|150|scalar_onlstm_blind/focal/w=1.1|10|0.000|0.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm/astar|10|0.400|0.900|0.500|5|0|5|n/a (n<6)|0.340|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|10|0.400|0.900|0.500|5|0|5|n/a (n<6)|0.340|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|10|0.400|0.900|0.500|5|0|5|n/a (n<6)|0.340|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|field_hrm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms|1300|field_hrm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|10|0.300|0.600|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_rooms|1300|field_unet/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|field_unet/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms|1300|field_unet_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms|1300|scalar_hrm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms|1300|scalar_onlstm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|10|0.300|0.500|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|10|0.300|0.400|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|10|1.000|1.000|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_rooms_large|400|field_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1.1|10|0.400|0.700|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_rooms_large|400|field_unet/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_rooms_large|400|field_unet/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm/astar|10|0.400|0.800|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.180|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm/astar|10|0.400|0.600|0.200|4|2|6|0.688|1.000|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1.1|10|0.400|0.700|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|10|0.400|0.800|0.400|5|1|6|0.219|0.969|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/astar|10|0.900|0.800|-0.100|1|2|3|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|10|0.900|1.000|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|10|0.900|0.900|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.145|
|C_dyn_spiral|2500|field_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.145|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.145|
|C_dyn_spiral|2500|field_unet/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.145|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/astar|10|0.100|0.800|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|10|0.100|0.900|0.800|8|0|8|0.008|0.145|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/astar|10|0.100|0.800|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.180|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|field_hrm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_spiral|3500|field_hrm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_spiral|3500|field_unet/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|field_unet/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|10|0.300|0.700|0.400|4|0|4|n/a (n<6)|0.562|
|C_dyn_spiral|3500|field_unet_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|10|0.300|0.500|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|10|0.300|0.600|0.300|3|0|3|n/a (n<6)|0.973|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|10|0.300|0.500|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|10|0.300|0.500|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|10|0.300|1.000|0.700|7|0|7|0.016|0.145|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|10|0.300|0.300|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|10|0.300|0.600|0.300|3|0|3|n/a (n<6)|0.973|

## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI

_Exploratory and UNcorrected: the Wilcoxon p-values below are NOT BH-corrected. Treat the bootstrap 95% CI on the median ratio as the primary inference._

|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|
|---|---:|---|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|2|0.125|[0.082, 0.169]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|2|0.767|[0.735, 0.800]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/astar|2|0.211|[0.182, 0.241]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|2|0.799|[0.771, 0.827]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/astar|2|0.249|[0.209, 0.289]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/focal/w=1.1|2|0.865|[0.827, 0.904]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/astar|2|0.136|[0.108, 0.164]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|2|0.784|[0.759, 0.809]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/astar|2|0.213|[0.100, 0.325]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|2|0.617|[0.464, 0.771]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/astar|2|0.130|[0.091, 0.169]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|2|0.617|[0.464, 0.771]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/astar|2|0.196|[0.091, 0.301]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|2|0.443|[0.422, 0.464]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|2|0.732|[0.091, 1.373]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|2|0.617|[0.464, 0.771]|n/a (n<6)|
|C_dyn_crossing|250|field_hrm/astar|8|0.121|[0.082, 0.169]|0.008|
|C_dyn_crossing|250|field_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|8|0.823|[0.717, 0.926]|0.008|
|C_dyn_crossing|250|field_hrm_blind/astar|8|0.098|[0.075, 0.175]|0.008|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|8|0.693|[0.619, 0.827]|0.008|
|C_dyn_crossing|250|field_unet/astar|8|0.218|[0.083, 0.289]|0.008|
|C_dyn_crossing|250|field_unet/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet/focal/w=1.1|8|0.865|[0.679, 0.935]|0.008|
|C_dyn_crossing|250|field_unet_blind/astar|8|0.160|[0.108, 0.252]|0.008|
|C_dyn_crossing|250|field_unet_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet_blind/focal/w=1.1|8|0.795|[0.699, 0.854]|0.008|
|C_dyn_crossing|250|scalar_hrm/astar|8|0.081|[0.071, 0.100]|0.008|
|C_dyn_crossing|250|scalar_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm/focal/w=1.1|8|0.657|[0.548, 0.923]|0.008|
|C_dyn_crossing|250|scalar_hrm_blind/astar|8|0.081|[0.065, 0.159]|0.008|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1.1|8|0.630|[0.548, 0.771]|0.008|
|C_dyn_crossing|250|scalar_onlstm/astar|8|0.084|[0.067, 0.102]|0.008|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1.1|8|0.569|[0.506, 0.923]|0.008|
|C_dyn_crossing|250|scalar_onlstm_blind/astar|8|0.087|[0.072, 0.171]|0.016|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1.1|8|0.644|[0.555, 0.923]|0.008|
|C_dyn_maze|1800|field_hrm/astar|4|0.152|[0.048, 0.357]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|4|0.858|[0.800, 0.965]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/astar|4|0.077|[0.056, 0.267]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|4|0.917|[0.768, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/astar|4|0.119|[0.095, 0.199]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/focal/w=1.1|4|0.919|[0.822, 0.993]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/astar|4|0.058|[0.039, 0.100]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|4|0.872|[0.787, 0.997]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/astar|4|0.329|[0.273, 0.450]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|4|0.895|[0.855, 0.993]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/astar|4|0.277|[0.213, 0.302]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|4|0.947|[0.786, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/astar|4|0.301|[0.254, 0.462]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|4|0.867|[0.824, 0.955]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|4|0.340|[0.253, 0.420]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|4|0.942|[0.788, 0.994]|n/a (n<6)|
|C_dyn_maze|2500|field_hrm/astar|9|0.132|[0.073, 0.356]|0.004|
|C_dyn_maze|2500|field_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|9|0.884|[0.810, 0.944]|0.004|
|C_dyn_maze|2500|field_hrm_blind/astar|9|0.254|[0.056, 0.306]|0.004|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|9|0.946|[0.847, 0.988]|0.004|
|C_dyn_maze|2500|field_unet/astar|9|0.137|[0.095, 0.199]|0.004|
|C_dyn_maze|2500|field_unet/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet/focal/w=1.1|9|0.943|[0.873, 0.980]|0.004|
|C_dyn_maze|2500|field_unet_blind/astar|9|0.060|[0.047, 0.100]|0.004|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|9|0.943|[0.824, 0.980]|0.004|
|C_dyn_maze|2500|scalar_hrm/astar|9|0.302|[0.231, 0.368]|0.004|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|9|0.923|[0.855, 0.968]|0.004|
|C_dyn_maze|2500|scalar_hrm_blind/astar|9|0.275|[0.199, 0.302]|0.004|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|9|0.934|[0.932, 0.977]|0.004|
|C_dyn_maze|2500|scalar_onlstm/astar|9|0.322|[0.197, 0.422]|0.004|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|9|0.878|[0.824, 0.976]|0.004|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|9|0.316|[0.214, 0.408]|0.004|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|9|0.935|[0.851, 0.949]|0.004|
|C_dyn_maze_dense|150|field_hrm/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm_blind/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm_blind/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_hrm_blind/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet_blind/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet_blind/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|field_unet_blind/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm_blind/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm_blind/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_hrm_blind/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm_blind/astar|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm_blind/focal/w=1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|150|scalar_onlstm_blind/focal/w=1.1|0|n/a|n/a|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/astar|4|0.256|[0.179, 0.729]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|4|0.962|[0.913, 0.999]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|4|0.287|[0.127, 0.617]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|4|0.940|[0.912, 0.970]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/astar|4|0.356|[0.315, 0.732]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|4|0.895|[0.834, 0.947]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/astar|4|0.213|[0.105, 0.505]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|4|0.968|[0.885, 0.997]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/astar|4|0.444|[0.170, 0.701]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|4|0.973|[0.955, 0.977]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|4|0.442|[0.268, 0.580]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|4|0.969|[0.919, 0.989]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|4|0.561|[0.275, 0.996]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|4|0.968|[0.930, 0.993]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|4|0.657|[0.422, 0.851]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|4|0.956|[0.826, 0.971]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|3|0.083|[0.054, 0.094]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|3|0.831|[0.709, 0.858]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/astar|3|0.093|[0.091, 0.117]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|3|0.968|[0.711, 0.972]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/astar|3|0.123|[0.066, 0.319]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|3|0.899|[0.783, 0.923]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/astar|3|0.122|[0.071, 0.292]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|3|0.924|[0.909, 0.960]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/astar|3|0.130|[0.115, 0.282]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|3|0.832|[0.756, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|3|0.271|[0.161, 0.326]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|3|0.756|[0.714, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/astar|3|0.125|[0.086, 0.224]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|3|0.959|[0.831, 0.961]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|3|0.134|[0.129, 0.283]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|3|0.865|[0.756, 0.981]|n/a (n<6)|
|C_dyn_rooms|1800|field_hrm/astar|10|0.120|[0.083, 0.206]|0.002|
|C_dyn_rooms|1800|field_hrm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|10|0.855|[0.761, 0.946]|0.002|
|C_dyn_rooms|1800|field_hrm_blind/astar|10|0.107|[0.093, 0.126]|0.002|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|10|0.924|[0.846, 0.968]|0.002|
|C_dyn_rooms|1800|field_unet/astar|10|0.207|[0.131, 0.248]|0.002|
|C_dyn_rooms|1800|field_unet/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|10|0.912|[0.857, 0.980]|0.002|
|C_dyn_rooms|1800|field_unet_blind/astar|10|0.175|[0.119, 0.230]|0.002|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|10|0.925|[0.877, 0.971]|0.002|
|C_dyn_rooms|1800|scalar_hrm/astar|10|0.153|[0.130, 0.234]|0.002|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|10|0.900|[0.824, 0.921]|0.002|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|10|0.220|[0.189, 0.276]|0.002|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|10|0.850|[0.773, 0.928]|0.002|
|C_dyn_rooms|1800|scalar_onlstm/astar|10|0.130|[0.111, 0.185]|0.002|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|10|0.957|[0.831, 0.981]|0.002|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|10|0.176|[0.129, 0.226]|0.002|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|10|0.917|[0.834, 0.969]|0.002|
|C_dyn_rooms_large|400|field_hrm/astar|4|0.540|[0.145, 0.743]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/focal/w=1.1|4|0.853|[0.713, 0.916]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/astar|4|0.410|[0.141, 0.776]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1.1|4|0.899|[0.727, 0.960]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/astar|4|0.633|[0.256, 1.300]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/focal/w=1.1|4|0.858|[0.656, 0.963]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/astar|4|0.284|[0.190, 1.131]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1.1|4|0.742|[0.588, 0.818]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/astar|4|0.528|[0.229, 0.731]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1.1|4|0.845|[0.671, 0.956]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|4|0.389|[0.212, 0.574]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1.1|4|0.752|[0.560, 0.855]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/astar|2|1.323|[0.205, 2.441]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1.1|4|0.883|[0.845, 1.105]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|3|1.118|[0.273, 1.216]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1.1|4|0.734|[0.664, 0.835]|n/a (n<6)|
|C_dyn_rooms_large|600|field_hrm/astar|9|0.320|[0.145, 0.713]|0.004|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|9|0.875|[0.832, 0.946]|0.004|
|C_dyn_rooms_large|600|field_hrm_blind/astar|9|0.256|[0.141, 0.500]|0.004|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|9|0.905|[0.727, 0.934]|0.004|
|C_dyn_rooms_large|600|field_unet/astar|9|0.271|[0.242, 0.634]|0.008|
|C_dyn_rooms_large|600|field_unet/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|9|0.868|[0.796, 0.919]|0.004|
|C_dyn_rooms_large|600|field_unet_blind/astar|9|0.252|[0.152, 0.427]|0.008|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|9|0.803|[0.681, 0.963]|0.004|
|C_dyn_rooms_large|600|scalar_hrm/astar|9|0.383|[0.229, 0.731]|0.008|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|9|0.849|[0.830, 0.956]|0.004|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|9|0.287|[0.116, 0.546]|0.004|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|9|0.851|[0.676, 0.926]|0.004|
|C_dyn_rooms_large|600|scalar_onlstm/astar|7|0.814|[0.205, 0.926]|0.297|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|9|0.904|[0.845, 0.965]|0.039|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|9|0.695|[0.273, 1.216]|0.250|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|9|0.835|[0.699, 0.965]|0.004|
|C_dyn_spiral|2500|field_hrm/astar|1|0.345|[0.345, 0.345]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|1|0.927|[0.927, 0.927]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/astar|1|0.164|[0.164, 0.164]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|1|0.864|[0.864, 0.864]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|1|0.135|[0.135, 0.135]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|1|0.991|[0.991, 0.991]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/astar|1|0.115|[0.115, 0.115]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|1|0.931|[0.931, 0.931]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|1|0.323|[0.323, 0.323]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|1|0.817|[0.817, 0.817]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|1|0.422|[0.422, 0.422]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|1|0.809|[0.809, 0.809]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|1|0.258|[0.258, 0.258]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|1|0.991|[0.991, 0.991]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|1|0.281|[0.281, 0.281]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|1|0.989|[0.989, 0.989]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/astar|3|0.239|[0.171, 0.345]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|3|0.858|[0.831, 0.927]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/astar|3|0.268|[0.164, 0.461]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|3|0.877|[0.864, 0.938]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/astar|3|0.135|[0.118, 0.207]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|3|0.989|[0.957, 0.991]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/astar|3|0.196|[0.115, 0.272]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|3|0.931|[0.868, 0.975]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/astar|3|0.341|[0.323, 0.496]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|3|0.817|[0.775, 0.960]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|3|0.374|[0.320, 0.422]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|3|0.886|[0.809, 0.987]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/astar|3|0.376|[0.258, 0.404]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|3|0.983|[0.845, 0.991]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|3|0.346|[0.281, 0.392]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|3|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|3|0.984|[0.821, 0.989]|n/a (n<6)|

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
