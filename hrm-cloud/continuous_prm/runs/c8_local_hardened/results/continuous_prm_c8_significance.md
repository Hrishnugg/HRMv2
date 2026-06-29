# C8 Dynamics Comparison — Significance

## Success: McNemar (learned arm vs euclid-time/astar), BH-corrected across the grid

_Family: learned arms only (oracle ceiling and euclid-time reference excluded). BH correction is applied to THIS success/McNemar grid only._

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|Discordant|McNemar p|BH q|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C_dyn_crossing|150|field_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|field_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_crossing|150|field_unet/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|field_unet/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet/focal/w=1.1|10|0.200|0.700|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_crossing|150|field_unet_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_crossing|150|scalar_onlstm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_crossing|250|field_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
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
|C_dyn_maze|1800|field_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|field_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet/astar|10|0.400|0.900|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_maze|1800|field_unet/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet/focal/w=1.1|10|0.400|0.600|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/astar|10|0.400|0.900|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|10|0.400|0.500|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|10|0.400|1.000|0.600|6|0|6|0.031|0.184|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|10|0.400|0.400|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
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
|C_dyn_maze_dense|3500|field_hrm/astar|10|0.100|0.800|0.700|7|0|7|0.016|0.132|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.184|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet/astar|10|0.100|0.800|0.700|7|0|7|0.016|0.132|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet_blind/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.184|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.184|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.184|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|10|0.100|0.700|0.600|6|0|6|0.031|0.184|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|10|0.100|0.900|0.800|8|0|8|0.008|0.118|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|field_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|field_unet/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|10|0.100|0.300|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|10|0.100|0.400|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|10|0.100|0.400|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|10|0.100|1.000|0.900|9|0|9|0.004|0.118|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|10|0.100|0.100|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|10|0.100|0.200|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_rooms_large|400|field_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm_blind/astar|10|0.200|0.500|0.300|4|1|5|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_rooms_large|400|field_unet/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_rooms_large|400|field_unet_blind/astar|10|0.200|0.800|0.600|6|0|6|0.031|0.184|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1.1|10|0.200|0.300|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm/astar|10|0.200|1.000|0.800|8|0|8|0.008|0.118|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|10|0.200|0.700|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm/astar|10|0.200|0.700|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|10|0.200|0.800|0.600|6|0|6|0.031|0.184|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1.1|10|0.200|0.300|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/astar|10|0.800|0.900|0.100|2|1|3|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/astar|10|0.800|0.900|0.100|1|0|1|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/astar|10|0.800|0.900|0.100|2|1|3|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|10|0.800|1.000|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|10|0.800|0.800|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|2500|field_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|2500|field_unet/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/astar|10|0.200|0.700|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|10|0.200|0.800|0.600|6|0|6|0.031|0.184|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/astar|10|0.200|0.700|0.500|5|0|5|n/a (n<6)|0.321|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|10|0.200|0.800|0.600|6|0|6|0.031|0.184|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|field_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_spiral|3500|field_unet/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|field_unet/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_spiral|3500|field_unet_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|10|0.200|0.400|0.200|2|0|2|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|10|0.200|0.600|0.400|4|0|4|n/a (n<6)|0.571|
|C_dyn_spiral|3500|scalar_onlstm/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|10|0.200|0.500|0.300|3|0|3|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|10|0.200|0.900|0.700|7|0|7|0.016|0.132|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|10|0.200|0.200|0.000|0|0|0|n/a (n<6)|1.000|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|10|0.200|0.300|0.100|1|0|1|n/a (n<6)|1.000|

## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI

_Exploratory and UNcorrected: the Wilcoxon p-values below are NOT BH-corrected. Treat the bootstrap 95% CI on the median ratio as the primary inference._

|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|
|---|---:|---|---:|---:|---|---:|
|C_dyn_crossing|150|field_hrm/astar|2|0.152|[0.100, 0.205]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm/focal/w=1.1|2|0.859|[0.827, 0.892]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/astar|2|0.128|[0.100, 0.157]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_hrm_blind/focal/w=1.1|2|0.772|[0.735, 0.809]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/astar|2|0.185|[0.082, 0.289]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_unet/focal/w=1.1|2|0.877|[0.827, 0.928]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/astar|2|0.263|[0.253, 0.273]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|field_unet_blind/focal/w=1.1|2|0.639|[0.518, 0.759]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/astar|2|0.178|[0.091, 0.265]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm/focal/w=1.1|2|0.660|[0.464, 0.855]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/astar|2|0.181|[0.145, 0.217]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_hrm_blind/focal/w=1.1|2|0.828|[0.800, 0.855]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/astar|2|0.131|[0.118, 0.145]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm/focal/w=1.1|2|0.641|[0.464, 0.819]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/astar|2|0.125|[0.082, 0.169]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_crossing|150|scalar_onlstm_blind/focal/w=1.1|2|0.611|[0.464, 0.759]|n/a (n<6)|
|C_dyn_crossing|250|field_hrm/astar|8|0.078|[0.064, 0.110]|0.008|
|C_dyn_crossing|250|field_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm/focal/w=1.1|8|0.841|[0.717, 0.911]|0.008|
|C_dyn_crossing|250|field_hrm_blind/astar|8|0.080|[0.071, 0.157]|0.008|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_hrm_blind/focal/w=1.1|8|0.789|[0.695, 0.911]|0.008|
|C_dyn_crossing|250|field_unet/astar|8|0.120|[0.082, 0.218]|0.008|
|C_dyn_crossing|250|field_unet/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet/focal/w=1.1|8|0.707|[0.650, 0.928]|0.008|
|C_dyn_crossing|250|field_unet_blind/astar|8|0.202|[0.085, 0.273]|0.008|
|C_dyn_crossing|250|field_unet_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|field_unet_blind/focal/w=1.1|8|0.732|[0.649, 0.945]|0.008|
|C_dyn_crossing|250|scalar_hrm/astar|8|0.083|[0.073, 0.098]|0.008|
|C_dyn_crossing|250|scalar_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm/focal/w=1.1|8|0.696|[0.548, 0.899]|0.008|
|C_dyn_crossing|250|scalar_hrm_blind/astar|8|0.088|[0.070, 0.173]|0.008|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_hrm_blind/focal/w=1.1|8|0.789|[0.619, 0.891]|0.008|
|C_dyn_crossing|250|scalar_onlstm/astar|8|0.141|[0.102, 0.161]|0.008|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm/focal/w=1.1|8|0.814|[0.551, 0.923]|0.008|
|C_dyn_crossing|250|scalar_onlstm_blind/astar|8|0.077|[0.065, 0.110]|0.008|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_crossing|250|scalar_onlstm_blind/focal/w=1.1|8|0.650|[0.548, 0.819]|0.008|
|C_dyn_maze|1800|field_hrm/astar|4|0.432|[0.117, 0.818]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm/focal/w=1.1|4|0.972|[0.832, 0.991]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/astar|4|0.282|[0.120, 0.603]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_hrm_blind/focal/w=1.1|4|0.859|[0.764, 0.916]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/astar|4|0.168|[0.113, 0.403]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_unet/focal/w=1.1|4|0.890|[0.785, 0.994]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/astar|4|0.187|[0.149, 0.268]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|field_unet_blind/focal/w=1.1|4|0.822|[0.764, 0.885]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/astar|4|0.651|[0.294, 0.804]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm/focal/w=1.1|4|0.955|[0.929, 0.973]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/astar|4|0.538|[0.211, 0.572]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_hrm_blind/focal/w=1.1|4|0.940|[0.901, 0.971]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/astar|4|0.469|[0.168, 0.523]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm/focal/w=1.1|4|0.834|[0.755, 0.940]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/astar|4|0.360|[0.116, 0.546]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1|4|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze|1800|scalar_onlstm_blind/focal/w=1.1|4|0.957|[0.929, 0.972]|n/a (n<6)|
|C_dyn_maze|2500|field_hrm/astar|8|0.332|[0.176, 0.669]|0.008|
|C_dyn_maze|2500|field_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm/focal/w=1.1|8|0.953|[0.869, 0.983]|0.008|
|C_dyn_maze|2500|field_hrm_blind/astar|8|0.262|[0.200, 0.354]|0.008|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_hrm_blind/focal/w=1.1|8|0.910|[0.854, 0.939]|0.008|
|C_dyn_maze|2500|field_unet/astar|8|0.146|[0.113, 0.206]|0.008|
|C_dyn_maze|2500|field_unet/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet/focal/w=1.1|8|0.890|[0.807, 0.994]|0.008|
|C_dyn_maze|2500|field_unet_blind/astar|8|0.155|[0.142, 0.268]|0.008|
|C_dyn_maze|2500|field_unet_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|field_unet_blind/focal/w=1.1|8|0.881|[0.812, 0.939]|0.008|
|C_dyn_maze|2500|scalar_hrm/astar|8|0.392|[0.294, 0.741]|0.008|
|C_dyn_maze|2500|scalar_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm/focal/w=1.1|8|0.942|[0.929, 0.967]|0.008|
|C_dyn_maze|2500|scalar_hrm_blind/astar|8|0.412|[0.242, 0.557]|0.008|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_hrm_blind/focal/w=1.1|8|0.936|[0.875, 0.947]|0.008|
|C_dyn_maze|2500|scalar_onlstm/astar|8|0.396|[0.329, 0.490]|0.008|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm/focal/w=1.1|8|0.861|[0.829, 0.943]|0.008|
|C_dyn_maze|2500|scalar_onlstm_blind/astar|8|0.307|[0.230, 0.366]|0.008|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_maze|2500|scalar_onlstm_blind/focal/w=1.1|8|0.943|[0.916, 0.969]|0.008|
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
|C_dyn_maze_dense|3500|field_hrm/astar|1|0.407|[0.407, 0.407]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm/focal/w=1.1|1|0.972|[0.972, 0.972]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/astar|1|0.608|[0.608, 0.608]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_hrm_blind/focal/w=1.1|1|0.938|[0.938, 0.938]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/astar|1|0.523|[0.523, 0.523]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet/focal/w=1.1|1|0.888|[0.888, 0.888]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/astar|1|0.537|[0.537, 0.537]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|field_unet_blind/focal/w=1.1|1|0.928|[0.928, 0.928]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/astar|1|0.229|[0.229, 0.229]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm/focal/w=1.1|1|0.897|[0.897, 0.897]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/astar|1|0.408|[0.408, 0.408]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_hrm_blind/focal/w=1.1|1|0.982|[0.982, 0.982]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/astar|1|0.350|[0.350, 0.350]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm/focal/w=1.1|1|0.870|[0.870, 0.870]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/astar|1|0.569|[0.569, 0.569]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_maze_dense|3500|scalar_onlstm_blind/focal/w=1.1|1|0.937|[0.937, 0.937]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/astar|1|0.129|[0.129, 0.129]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm/focal/w=1.1|1|0.987|[0.987, 0.987]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/astar|1|0.076|[0.076, 0.076]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_hrm_blind/focal/w=1.1|1|0.924|[0.924, 0.924]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/astar|1|0.158|[0.158, 0.158]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet/focal/w=1.1|1|0.983|[0.983, 0.983]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/astar|1|0.080|[0.080, 0.080]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|field_unet_blind/focal/w=1.1|1|0.970|[0.970, 0.970]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/astar|1|0.148|[0.148, 0.148]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm/focal/w=1.1|1|0.979|[0.979, 0.979]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/astar|1|0.178|[0.178, 0.178]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_hrm_blind/focal/w=1.1|1|0.729|[0.729, 0.729]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/astar|1|0.190|[0.190, 0.190]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm/focal/w=1.1|1|0.926|[0.926, 0.926]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/astar|1|0.075|[0.075, 0.075]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1|1|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms|1300|scalar_onlstm_blind/focal/w=1.1|1|0.827|[0.827, 0.827]|n/a (n<6)|
|C_dyn_rooms|1800|field_hrm/astar|8|0.197|[0.129, 0.251]|0.008|
|C_dyn_rooms|1800|field_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm/focal/w=1.1|8|0.924|[0.850, 0.954]|0.008|
|C_dyn_rooms|1800|field_hrm_blind/astar|8|0.145|[0.076, 0.257]|0.008|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_hrm_blind/focal/w=1.1|8|0.905|[0.876, 0.955]|0.008|
|C_dyn_rooms|1800|field_unet/astar|8|0.154|[0.126, 0.268]|0.008|
|C_dyn_rooms|1800|field_unet/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet/focal/w=1.1|8|0.938|[0.845, 0.966]|0.008|
|C_dyn_rooms|1800|field_unet_blind/astar|8|0.174|[0.098, 0.201]|0.008|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|field_unet_blind/focal/w=1.1|8|0.945|[0.856, 0.955]|0.008|
|C_dyn_rooms|1800|scalar_hrm/astar|8|0.220|[0.148, 0.265]|0.008|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm/focal/w=1.1|8|0.848|[0.769, 0.963]|0.008|
|C_dyn_rooms|1800|scalar_hrm_blind/astar|8|0.236|[0.185, 0.300]|0.008|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_hrm_blind/focal/w=1.1|8|0.880|[0.766, 0.962]|0.008|
|C_dyn_rooms|1800|scalar_onlstm/astar|8|0.211|[0.163, 0.246]|0.008|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm/focal/w=1.1|8|0.934|[0.907, 0.965]|0.008|
|C_dyn_rooms|1800|scalar_onlstm_blind/astar|8|0.169|[0.133, 0.227]|0.008|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms|1800|scalar_onlstm_blind/focal/w=1.1|8|0.880|[0.776, 0.957]|0.008|
|C_dyn_rooms_large|400|field_hrm/astar|2|0.742|[0.170, 1.315]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm/focal/w=1.1|2|0.730|[0.598, 0.862]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/astar|1|0.078|[0.078, 0.078]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_hrm_blind/focal/w=1.1|2|0.833|[0.701, 0.965]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/astar|2|0.426|[0.184, 0.668]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet/focal/w=1.1|2|0.823|[0.721, 0.925]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/astar|2|0.408|[0.173, 0.643]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|field_unet_blind/focal/w=1.1|2|0.765|[0.664, 0.866]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/astar|2|0.628|[0.081, 1.174]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm/focal/w=1.1|2|0.957|[0.942, 0.972]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/astar|2|0.956|[0.318, 1.593]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_hrm_blind/focal/w=1.1|2|0.920|[0.855, 0.986]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/astar|2|0.901|[0.226, 1.577]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm/focal/w=1.1|2|0.835|[0.731, 0.938]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/astar|2|0.800|[0.205, 1.394]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_rooms_large|400|scalar_onlstm_blind/focal/w=1.1|2|0.810|[0.656, 0.965]|n/a (n<6)|
|C_dyn_rooms_large|600|field_hrm/astar|8|0.344|[0.147, 0.532]|0.016|
|C_dyn_rooms_large|600|field_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm/focal/w=1.1|8|0.854|[0.682, 0.949]|0.008|
|C_dyn_rooms_large|600|field_hrm_blind/astar|7|0.713|[0.078, 1.185]|0.469|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_hrm_blind/focal/w=1.1|8|0.823|[0.690, 0.880]|0.008|
|C_dyn_rooms_large|600|field_unet/astar|8|0.276|[0.184, 0.468]|0.008|
|C_dyn_rooms_large|600|field_unet/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet/focal/w=1.1|8|0.793|[0.681, 0.925]|0.008|
|C_dyn_rooms_large|600|field_unet_blind/astar|8|0.642|[0.259, 0.691]|0.016|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|field_unet_blind/focal/w=1.1|8|0.849|[0.681, 0.909]|0.008|
|C_dyn_rooms_large|600|scalar_hrm/astar|8|0.170|[0.109, 0.325]|0.016|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm/focal/w=1.1|8|0.932|[0.888, 0.972]|0.008|
|C_dyn_rooms_large|600|scalar_hrm_blind/astar|8|0.577|[0.325, 0.951]|0.109|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_hrm_blind/focal/w=1.1|8|0.903|[0.726, 0.962]|0.008|
|C_dyn_rooms_large|600|scalar_onlstm/astar|7|0.474|[0.253, 0.925]|0.109|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm/focal/w=1.1|8|0.929|[0.877, 0.992]|0.148|
|C_dyn_rooms_large|600|scalar_onlstm_blind/astar|8|0.566|[0.375, 0.880]|0.039|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1|8|1.000|[1.000, 1.000]|n/a|
|C_dyn_rooms_large|600|scalar_onlstm_blind/focal/w=1.1|8|0.900|[0.749, 0.965]|0.008|
|C_dyn_spiral|2500|field_hrm/astar|2|0.218|[0.188, 0.247]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm/focal/w=1.1|2|0.973|[0.971, 0.976]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/astar|2|0.097|[0.059, 0.134]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_hrm_blind/focal/w=1.1|2|0.861|[0.833, 0.888]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/astar|2|0.152|[0.118, 0.186]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet/focal/w=1.1|2|0.961|[0.926, 0.995]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/astar|2|0.179|[0.164, 0.194]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|field_unet_blind/focal/w=1.1|2|0.970|[0.942, 0.998]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/astar|2|0.434|[0.329, 0.539]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm/focal/w=1.1|2|0.970|[0.953, 0.988]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/astar|2|0.369|[0.298, 0.439]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_hrm_blind/focal/w=1.1|2|0.928|[0.925, 0.930]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/astar|2|0.251|[0.196, 0.306]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm/focal/w=1.1|2|0.824|[0.810, 0.839]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/astar|2|0.388|[0.230, 0.546]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|2500|scalar_onlstm_blind/focal/w=1.1|2|0.917|[0.892, 0.942]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/astar|2|0.218|[0.188, 0.247]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm/focal/w=1.1|2|0.973|[0.971, 0.976]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/astar|2|0.097|[0.059, 0.134]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_hrm_blind/focal/w=1.1|2|0.861|[0.833, 0.888]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/astar|2|0.152|[0.118, 0.186]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet/focal/w=1.1|2|0.961|[0.926, 0.995]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/astar|2|0.179|[0.164, 0.194]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|field_unet_blind/focal/w=1.1|2|0.970|[0.942, 0.998]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/astar|2|0.434|[0.329, 0.539]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm/focal/w=1.1|2|0.970|[0.953, 0.988]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/astar|2|0.369|[0.298, 0.439]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_hrm_blind/focal/w=1.1|2|0.928|[0.925, 0.930]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/astar|2|0.251|[0.196, 0.306]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm/focal/w=1.1|2|0.824|[0.810, 0.839]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/astar|2|0.388|[0.230, 0.546]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1|2|1.000|[1.000, 1.000]|n/a (n<6)|
|C_dyn_spiral|3500|scalar_onlstm_blind/focal/w=1.1|2|0.917|[0.892, 0.942]|n/a (n<6)|

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
