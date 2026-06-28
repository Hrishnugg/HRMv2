# C7 Integration Comparison — Significance

## Success: McNemar (learned arm vs euclid/astar), BH-corrected across the grid

_Family: learned arms only (oracle ceiling and euclid reference excluded). BH correction is applied to THIS success/McNemar grid only._

|Suite|Budget|Arm|n|Euclid succ|Arm succ|Delta|Gain|Loss|Discordant|McNemar p|BH q|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|C_hard_bugtrap|24|field_hrm/astar|24|0.458|0.750|0.292|7|0|7|0.016|0.100|
|C_hard_bugtrap|24|field_hrm/focal/w=1|24|0.458|0.458|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|24|field_hrm/focal/w=1.1|24|0.458|0.583|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|24|field_onlstm/astar|24|0.458|0.833|0.375|9|0|9|0.004|0.033|
|C_hard_bugtrap|24|field_onlstm/focal/w=1|24|0.458|0.458|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|24|field_onlstm/focal/w=1.1|24|0.458|0.583|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|24|field_unet/astar|24|0.458|0.542|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_bugtrap|24|field_unet/focal/w=1|24|0.458|0.458|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|24|field_unet/focal/w=1.1|24|0.458|0.500|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_bugtrap|24|scalar_hrm/astar|24|0.458|0.792|0.333|8|0|8|0.008|0.056|
|C_hard_bugtrap|24|scalar_hrm/focal/w=1|24|0.458|0.458|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|24|scalar_hrm/focal/w=1.1|24|0.458|0.583|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|24|scalar_onlstm/astar|24|0.458|0.750|0.292|7|0|7|0.016|0.100|
|C_hard_bugtrap|24|scalar_onlstm/focal/w=1|24|0.458|0.458|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|24|scalar_onlstm/focal/w=1.1|24|0.458|0.583|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|32|field_hrm/astar|24|0.708|0.833|0.125|4|1|5|n/a (n<6)|0.776|
|C_hard_bugtrap|32|field_hrm/focal/w=1|24|0.708|0.708|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|32|field_hrm/focal/w=1.1|24|0.708|0.833|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|32|field_onlstm/astar|24|0.708|0.917|0.208|6|1|7|0.125|0.363|
|C_hard_bugtrap|32|field_onlstm/focal/w=1|24|0.708|0.708|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|32|field_onlstm/focal/w=1.1|24|0.708|0.875|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_bugtrap|32|field_unet/astar|24|0.708|0.750|0.042|3|2|5|n/a (n<6)|1.000|
|C_hard_bugtrap|32|field_unet/focal/w=1|24|0.708|0.708|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|32|field_unet/focal/w=1.1|24|0.708|0.708|0.000|2|2|4|n/a (n<6)|1.000|
|C_hard_bugtrap|32|scalar_hrm/astar|24|0.708|0.875|0.167|5|1|6|0.219|0.529|
|C_hard_bugtrap|32|scalar_hrm/focal/w=1|24|0.708|0.708|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|32|scalar_hrm/focal/w=1.1|24|0.708|0.833|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_bugtrap|32|scalar_onlstm/astar|24|0.708|0.792|0.083|4|2|6|0.688|1.000|
|C_hard_bugtrap|32|scalar_onlstm/focal/w=1|24|0.708|0.708|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_bugtrap|32|scalar_onlstm/focal/w=1.1|24|0.708|0.958|0.250|6|0|6|0.031|0.144|
|C_hard_maze|140|field_hrm/astar|24|0.583|1.000|0.417|10|0|10|0.002|0.018|
|C_hard_maze|140|field_hrm/focal/w=1|24|0.583|0.583|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|140|field_hrm/focal/w=1.1|24|0.583|0.708|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_maze|140|field_onlstm/astar|24|0.583|1.000|0.417|10|0|10|0.002|0.018|
|C_hard_maze|140|field_onlstm/focal/w=1|24|0.583|0.583|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|140|field_onlstm/focal/w=1.1|24|0.583|0.750|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_maze|140|field_unet/astar|24|0.583|1.000|0.417|10|0|10|0.002|0.018|
|C_hard_maze|140|field_unet/focal/w=1|24|0.583|0.583|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|140|field_unet/focal/w=1.1|24|0.583|0.708|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_maze|140|scalar_hrm/astar|24|0.583|1.000|0.417|10|0|10|0.002|0.018|
|C_hard_maze|140|scalar_hrm/focal/w=1|24|0.583|0.583|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|140|scalar_hrm/focal/w=1.1|24|0.583|0.792|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_maze|140|scalar_onlstm/astar|24|0.583|1.000|0.417|10|0|10|0.002|0.018|
|C_hard_maze|140|scalar_onlstm/focal/w=1|24|0.583|0.583|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|140|scalar_onlstm/focal/w=1.1|24|0.583|0.750|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_maze|152|field_hrm/astar|24|0.958|1.000|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze|152|field_hrm/focal/w=1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|field_hrm/focal/w=1.1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|field_onlstm/astar|24|0.958|1.000|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze|152|field_onlstm/focal/w=1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|field_onlstm/focal/w=1.1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|field_unet/astar|24|0.958|1.000|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze|152|field_unet/focal/w=1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|field_unet/focal/w=1.1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_hrm/astar|24|0.958|1.000|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_hrm/focal/w=1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_hrm/focal/w=1.1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_onlstm/astar|24|0.958|1.000|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_onlstm/focal/w=1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze|152|scalar_onlstm/focal/w=1.1|24|0.958|0.958|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|field_hrm/astar|24|0.250|0.958|0.708|17|0|17|<0.001|0.001|
|C_hard_maze_dense|140|field_hrm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|field_hrm/focal/w=1.1|24|0.250|0.333|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_maze_dense|140|field_onlstm/astar|24|0.250|0.917|0.667|16|0|16|<0.001|0.001|
|C_hard_maze_dense|140|field_onlstm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|field_onlstm/focal/w=1.1|24|0.250|0.333|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_maze_dense|140|field_unet/astar|24|0.250|1.000|0.750|18|0|18|<0.001|0.001|
|C_hard_maze_dense|140|field_unet/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|field_unet/focal/w=1.1|24|0.250|0.292|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze_dense|140|scalar_hrm/astar|24|0.250|1.000|0.750|18|0|18|<0.001|0.001|
|C_hard_maze_dense|140|scalar_hrm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|scalar_hrm/focal/w=1.1|24|0.250|0.292|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze_dense|140|scalar_onlstm/astar|24|0.250|0.917|0.667|16|0|16|<0.001|0.001|
|C_hard_maze_dense|140|scalar_onlstm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|140|scalar_onlstm/focal/w=1.1|24|0.250|0.333|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_maze_dense|152|field_hrm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_maze_dense|152|field_hrm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|152|field_hrm/focal/w=1.1|24|0.750|0.917|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_maze_dense|152|field_onlstm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_maze_dense|152|field_onlstm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|152|field_onlstm/focal/w=1.1|24|0.750|0.833|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_maze_dense|152|field_unet/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_maze_dense|152|field_unet/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|152|field_unet/focal/w=1.1|24|0.750|0.792|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_maze_dense|152|scalar_hrm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_maze_dense|152|scalar_hrm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|152|scalar_hrm/focal/w=1.1|24|0.750|0.875|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_maze_dense|152|scalar_onlstm/astar|24|0.750|0.958|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_maze_dense|152|scalar_onlstm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_maze_dense|152|scalar_onlstm/focal/w=1.1|24|0.750|0.875|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_rooms|140|field_hrm/astar|24|0.375|0.958|0.583|14|0|14|<0.001|0.002|
|C_hard_rooms|140|field_hrm/focal/w=1|24|0.375|0.375|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|140|field_hrm/focal/w=1.1|24|0.375|0.417|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_rooms|140|field_onlstm/astar|24|0.375|0.667|0.292|7|0|7|0.016|0.100|
|C_hard_rooms|140|field_onlstm/focal/w=1|24|0.375|0.375|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|140|field_onlstm/focal/w=1.1|24|0.375|0.458|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_rooms|140|field_unet/astar|24|0.375|0.833|0.458|11|0|11|<0.001|0.012|
|C_hard_rooms|140|field_unet/focal/w=1|24|0.375|0.375|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|140|field_unet/focal/w=1.1|24|0.375|0.500|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_rooms|140|scalar_hrm/astar|24|0.375|1.000|0.625|15|0|15|<0.001|0.001|
|C_hard_rooms|140|scalar_hrm/focal/w=1|24|0.375|0.375|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|140|scalar_hrm/focal/w=1.1|24|0.375|0.542|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_rooms|140|scalar_onlstm/astar|24|0.375|1.000|0.625|15|0|15|<0.001|0.001|
|C_hard_rooms|140|scalar_onlstm/focal/w=1|24|0.375|0.375|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|140|scalar_onlstm/focal/w=1.1|24|0.375|0.500|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_rooms|152|field_hrm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_rooms|152|field_hrm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|152|field_hrm/focal/w=1.1|24|0.750|0.792|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_rooms|152|field_onlstm/astar|24|0.750|0.917|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_rooms|152|field_onlstm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|152|field_onlstm/focal/w=1.1|24|0.750|0.792|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_rooms|152|field_unet/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_rooms|152|field_unet/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|152|field_unet/focal/w=1.1|24|0.750|0.833|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_rooms|152|scalar_hrm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_rooms|152|scalar_hrm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|152|scalar_hrm/focal/w=1.1|24|0.750|0.875|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_rooms|152|scalar_onlstm/astar|24|0.750|1.000|0.250|6|0|6|0.031|0.144|
|C_hard_rooms|152|scalar_onlstm/focal/w=1|24|0.750|0.750|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms|152|scalar_onlstm/focal/w=1.1|24|0.750|0.833|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_rooms_large|56|field_hrm/astar|24|0.417|0.750|0.333|9|1|10|0.021|0.133|
|C_hard_rooms_large|56|field_hrm/focal/w=1|24|0.417|0.417|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|56|field_hrm/focal/w=1.1|24|0.417|0.667|0.250|7|1|8|0.070|0.248|
|C_hard_rooms_large|56|field_onlstm/astar|24|0.417|0.958|0.542|13|0|13|<0.001|0.004|
|C_hard_rooms_large|56|field_onlstm/focal/w=1|24|0.417|0.417|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|56|field_onlstm/focal/w=1.1|24|0.417|0.750|0.333|8|0|8|0.008|0.056|
|C_hard_rooms_large|56|field_unet/astar|24|0.417|0.750|0.333|8|0|8|0.008|0.056|
|C_hard_rooms_large|56|field_unet/focal/w=1|24|0.417|0.417|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|56|field_unet/focal/w=1.1|24|0.417|0.583|0.167|5|1|6|0.219|0.529|
|C_hard_rooms_large|56|scalar_hrm/astar|24|0.417|0.917|0.500|12|0|12|<0.001|0.007|
|C_hard_rooms_large|56|scalar_hrm/focal/w=1|24|0.417|0.417|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|56|scalar_hrm/focal/w=1.1|24|0.417|0.667|0.250|6|0|6|0.031|0.144|
|C_hard_rooms_large|56|scalar_onlstm/astar|24|0.417|1.000|0.583|14|0|14|<0.001|0.002|
|C_hard_rooms_large|56|scalar_onlstm/focal/w=1|24|0.417|0.417|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|56|scalar_onlstm/focal/w=1.1|24|0.417|0.625|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_rooms_large|64|field_hrm/astar|24|0.792|0.875|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_rooms_large|64|field_hrm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|64|field_hrm/focal/w=1.1|24|0.792|0.917|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_rooms_large|64|field_onlstm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_rooms_large|64|field_onlstm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|64|field_onlstm/focal/w=1.1|24|0.792|0.833|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_rooms_large|64|field_unet/astar|24|0.792|0.917|0.125|4|1|5|n/a (n<6)|0.776|
|C_hard_rooms_large|64|field_unet/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|64|field_unet/focal/w=1.1|24|0.792|0.875|0.083|3|1|4|n/a (n<6)|1.000|
|C_hard_rooms_large|64|scalar_hrm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_rooms_large|64|scalar_hrm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|64|scalar_hrm/focal/w=1.1|24|0.792|0.875|0.083|2|0|2|n/a (n<6)|0.928|
|C_hard_rooms_large|64|scalar_onlstm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_rooms_large|64|scalar_onlstm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_rooms_large|64|scalar_onlstm/focal/w=1.1|24|0.792|0.833|0.042|1|0|1|n/a (n<6)|1.000|
|C_hard_spiral|140|field_hrm/astar|24|0.250|0.917|0.667|16|0|16|<0.001|0.001|
|C_hard_spiral|140|field_hrm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|140|field_hrm/focal/w=1.1|24|0.250|0.375|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|140|field_onlstm/astar|24|0.250|0.458|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_spiral|140|field_onlstm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|140|field_onlstm/focal/w=1.1|24|0.250|0.417|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_spiral|140|field_unet/astar|24|0.250|0.583|0.333|8|0|8|0.008|0.056|
|C_hard_spiral|140|field_unet/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|140|field_unet/focal/w=1.1|24|0.250|0.375|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|140|scalar_hrm/astar|24|0.250|0.708|0.458|11|0|11|<0.001|0.012|
|C_hard_spiral|140|scalar_hrm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|140|scalar_hrm/focal/w=1.1|24|0.250|0.417|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_spiral|140|scalar_onlstm/astar|24|0.250|0.833|0.583|14|0|14|<0.001|0.002|
|C_hard_spiral|140|scalar_onlstm/focal/w=1|24|0.250|0.250|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|140|scalar_onlstm/focal/w=1.1|24|0.250|0.375|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|152|field_hrm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_spiral|152|field_hrm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|152|field_hrm/focal/w=1.1|24|0.792|0.917|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|152|field_onlstm/astar|24|0.792|0.958|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_spiral|152|field_onlstm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|152|field_onlstm/focal/w=1.1|24|0.792|0.917|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|152|field_unet/astar|24|0.792|0.958|0.167|4|0|4|n/a (n<6)|0.363|
|C_hard_spiral|152|field_unet/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|152|field_unet/focal/w=1.1|24|0.792|0.917|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|152|scalar_hrm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_spiral|152|scalar_hrm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|152|scalar_hrm/focal/w=1.1|24|0.792|0.917|0.125|3|0|3|n/a (n<6)|0.529|
|C_hard_spiral|152|scalar_onlstm/astar|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|
|C_hard_spiral|152|scalar_onlstm/focal/w=1|24|0.792|0.792|0.000|0|0|0|n/a (n<6)|1.000|
|C_hard_spiral|152|scalar_onlstm/focal/w=1.1|24|0.792|1.000|0.208|5|0|5|n/a (n<6)|0.225|

## Expansions: matched-set median ratio (arm/euclid) + Wilcoxon p + bootstrap 95% CI

_Exploratory and UNcorrected: the Wilcoxon p-values below are NOT BH-corrected. Treat the bootstrap 95% CI on the median ratio as the primary inference._

|Suite|Budget|Arm|n matched|Median ratio|95% CI|Wilcoxon p|
|---|---:|---|---:|---:|---|---:|
|C_hard_bugtrap|24|field_hrm/astar|11|0.714|[0.533, 0.800]|0.002|
|C_hard_bugtrap|24|field_hrm/focal/w=1|11|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|24|field_hrm/focal/w=1.1|11|0.800|[0.714, 0.842]|0.002|
|C_hard_bugtrap|24|field_onlstm/astar|11|0.750|[0.636, 0.895]|0.078|
|C_hard_bugtrap|24|field_onlstm/focal/w=1|11|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|24|field_onlstm/focal/w=1.1|11|0.864|[0.733, 1.000]|0.027|
|C_hard_bugtrap|24|field_unet/astar|11|0.895|[0.750, 1.000]|0.105|
|C_hard_bugtrap|24|field_unet/focal/w=1|11|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|24|field_unet/focal/w=1.1|11|0.864|[0.714, 1.050]|0.047|
|C_hard_bugtrap|24|scalar_hrm/astar|11|0.750|[0.500, 0.857]|0.002|
|C_hard_bugtrap|24|scalar_hrm/focal/w=1|11|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|24|scalar_hrm/focal/w=1.1|11|0.789|[0.714, 0.864]|0.002|
|C_hard_bugtrap|24|scalar_onlstm/astar|11|0.619|[0.500, 0.789]|0.002|
|C_hard_bugtrap|24|scalar_onlstm/focal/w=1|11|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|24|scalar_onlstm/focal/w=1.1|11|0.773|[0.714, 0.842]|<0.001|
|C_hard_bugtrap|32|field_hrm/astar|16|0.754|[0.600, 0.780]|<0.001|
|C_hard_bugtrap|32|field_hrm/focal/w=1|17|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|32|field_hrm/focal/w=1.1|17|0.800|[0.720, 0.895]|<0.001|
|C_hard_bugtrap|32|field_onlstm/astar|16|0.702|[0.635, 0.818]|0.009|
|C_hard_bugtrap|32|field_onlstm/focal/w=1|17|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|32|field_onlstm/focal/w=1.1|17|0.864|[0.731, 0.947]|0.003|
|C_hard_bugtrap|32|field_unet/astar|15|0.895|[0.866, 1.000]|0.084|
|C_hard_bugtrap|32|field_unet/focal/w=1|17|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|32|field_unet/focal/w=1.1|15|0.897|[0.720, 1.001]|0.019|
|C_hard_bugtrap|32|scalar_hrm/astar|16|0.756|[0.500, 0.838]|<0.001|
|C_hard_bugtrap|32|scalar_hrm/focal/w=1|17|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|32|scalar_hrm/focal/w=1.1|17|0.846|[0.733, 0.895]|<0.001|
|C_hard_bugtrap|32|scalar_onlstm/astar|15|0.600|[0.500, 0.750]|<0.001|
|C_hard_bugtrap|32|scalar_onlstm/focal/w=1|17|1.000|[1.000, 1.000]|n/a|
|C_hard_bugtrap|32|scalar_onlstm/focal/w=1.1|17|0.800|[0.733, 0.850]|<0.001|
|C_hard_maze|140|field_hrm/astar|14|0.521|[0.450, 0.627]|<0.001|
|C_hard_maze|140|field_hrm/focal/w=1|14|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|140|field_hrm/focal/w=1.1|14|0.974|[0.931, 0.992]|0.003|
|C_hard_maze|140|field_onlstm/astar|14|0.572|[0.518, 0.698]|<0.001|
|C_hard_maze|140|field_onlstm/focal/w=1|14|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|140|field_onlstm/focal/w=1.1|14|0.947|[0.927, 0.981]|0.002|
|C_hard_maze|140|field_unet/astar|14|0.595|[0.522, 0.643]|<0.001|
|C_hard_maze|140|field_unet/focal/w=1|14|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|140|field_unet/focal/w=1.1|14|0.944|[0.910, 0.988]|0.004|
|C_hard_maze|140|scalar_hrm/astar|14|0.427|[0.351, 0.675]|<0.001|
|C_hard_maze|140|scalar_hrm/focal/w=1|14|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|140|scalar_hrm/focal/w=1.1|14|0.953|[0.926, 0.978]|0.001|
|C_hard_maze|140|scalar_onlstm/astar|14|0.438|[0.381, 0.684]|<0.001|
|C_hard_maze|140|scalar_onlstm/focal/w=1|14|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|140|scalar_onlstm/focal/w=1.1|14|0.977|[0.949, 0.993]|0.016|
|C_hard_maze|152|field_hrm/astar|23|0.551|[0.510, 0.712]|<0.001|
|C_hard_maze|152|field_hrm/focal/w=1|23|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|152|field_hrm/focal/w=1.1|23|0.977|[0.945, 0.986]|<0.001|
|C_hard_maze|152|field_onlstm/astar|23|0.642|[0.543, 0.713]|<0.001|
|C_hard_maze|152|field_onlstm/focal/w=1|23|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|152|field_onlstm/focal/w=1.1|23|0.962|[0.928, 0.979]|<0.001|
|C_hard_maze|152|field_unet/astar|23|0.626|[0.550, 0.757]|<0.001|
|C_hard_maze|152|field_unet/focal/w=1|23|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|152|field_unet/focal/w=1.1|23|0.977|[0.926, 0.993]|<0.001|
|C_hard_maze|152|scalar_hrm/astar|23|0.510|[0.411, 0.675]|<0.001|
|C_hard_maze|152|scalar_hrm/focal/w=1|23|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|152|scalar_hrm/focal/w=1.1|23|0.961|[0.936, 0.973]|<0.001|
|C_hard_maze|152|scalar_onlstm/astar|23|0.508|[0.395, 0.725]|<0.001|
|C_hard_maze|152|scalar_onlstm/focal/w=1|23|1.000|[1.000, 1.000]|n/a|
|C_hard_maze|152|scalar_onlstm/focal/w=1.1|23|0.973|[0.952, 0.993]|<0.001|
|C_hard_maze_dense|140|field_hrm/astar|6|0.804|[0.707, 0.829]|0.031|
|C_hard_maze_dense|140|field_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|140|field_hrm/focal/w=1.1|6|0.974|[0.946, 0.993]|0.031|
|C_hard_maze_dense|140|field_onlstm/astar|6|0.799|[0.765, 0.914]|0.031|
|C_hard_maze_dense|140|field_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|140|field_onlstm/focal/w=1.1|6|0.969|[0.931, 0.982]|0.031|
|C_hard_maze_dense|140|field_unet/astar|6|0.838|[0.771, 0.891]|0.031|
|C_hard_maze_dense|140|field_unet/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|140|field_unet/focal/w=1.1|6|0.973|[0.942, 0.989]|0.062|
|C_hard_maze_dense|140|scalar_hrm/astar|6|0.721|[0.636, 0.788]|0.031|
|C_hard_maze_dense|140|scalar_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|140|scalar_hrm/focal/w=1.1|6|0.977|[0.945, 0.989]|0.031|
|C_hard_maze_dense|140|scalar_onlstm/astar|6|0.880|[0.702, 0.899]|0.031|
|C_hard_maze_dense|140|scalar_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|140|scalar_onlstm/focal/w=1.1|6|0.966|[0.942, 0.978]|0.031|
|C_hard_maze_dense|152|field_hrm/astar|18|0.738|[0.660, 0.807]|<0.001|
|C_hard_maze_dense|152|field_hrm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|152|field_hrm/focal/w=1.1|18|0.973|[0.967, 0.986]|<0.001|
|C_hard_maze_dense|152|field_onlstm/astar|18|0.782|[0.736, 0.841]|<0.001|
|C_hard_maze_dense|152|field_onlstm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|152|field_onlstm/focal/w=1.1|18|0.969|[0.963, 0.977]|<0.001|
|C_hard_maze_dense|152|field_unet/astar|18|0.823|[0.736, 0.852]|<0.001|
|C_hard_maze_dense|152|field_unet/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|152|field_unet/focal/w=1.1|18|0.978|[0.964, 0.986]|0.001|
|C_hard_maze_dense|152|scalar_hrm/astar|18|0.707|[0.594, 0.763]|<0.001|
|C_hard_maze_dense|152|scalar_hrm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|152|scalar_hrm/focal/w=1.1|18|0.976|[0.967, 0.986]|<0.001|
|C_hard_maze_dense|152|scalar_onlstm/astar|18|0.819|[0.711, 0.899]|<0.001|
|C_hard_maze_dense|152|scalar_onlstm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_maze_dense|152|scalar_onlstm/focal/w=1.1|18|0.973|[0.966, 0.983]|<0.001|
|C_hard_rooms|140|field_hrm/astar|9|0.829|[0.775, 0.885]|0.004|
|C_hard_rooms|140|field_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|140|field_hrm/focal/w=1.1|9|0.964|[0.947, 0.984]|0.004|
|C_hard_rooms|140|field_onlstm/astar|9|0.906|[0.812, 0.961]|0.004|
|C_hard_rooms|140|field_onlstm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|140|field_onlstm/focal/w=1.1|9|0.964|[0.949, 0.992]|0.004|
|C_hard_rooms|140|field_unet/astar|9|0.846|[0.770, 0.899]|0.008|
|C_hard_rooms|140|field_unet/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|140|field_unet/focal/w=1.1|9|0.964|[0.957, 0.985]|0.020|
|C_hard_rooms|140|scalar_hrm/astar|9|0.822|[0.632, 0.885]|0.004|
|C_hard_rooms|140|scalar_hrm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|140|scalar_hrm/focal/w=1.1|9|0.964|[0.946, 0.992]|0.004|
|C_hard_rooms|140|scalar_onlstm/astar|9|0.770|[0.677, 0.853]|0.004|
|C_hard_rooms|140|scalar_onlstm/focal/w=1|9|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|140|scalar_onlstm/focal/w=1.1|9|0.957|[0.946, 0.992]|0.004|
|C_hard_rooms|152|field_hrm/astar|18|0.825|[0.776, 0.885]|<0.001|
|C_hard_rooms|152|field_hrm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|152|field_hrm/focal/w=1.1|18|0.975|[0.964, 0.989]|<0.001|
|C_hard_rooms|152|field_onlstm/astar|18|0.937|[0.906, 0.959]|<0.001|
|C_hard_rooms|152|field_onlstm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|152|field_onlstm/focal/w=1.1|18|0.980|[0.963, 0.989]|<0.001|
|C_hard_rooms|152|field_unet/astar|18|0.846|[0.780, 0.899]|<0.001|
|C_hard_rooms|152|field_unet/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|152|field_unet/focal/w=1.1|18|0.976|[0.968, 0.986]|<0.001|
|C_hard_rooms|152|scalar_hrm/astar|18|0.795|[0.717, 0.867]|<0.001|
|C_hard_rooms|152|scalar_hrm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|152|scalar_hrm/focal/w=1.1|18|0.976|[0.964, 0.986]|<0.001|
|C_hard_rooms|152|scalar_onlstm/astar|18|0.719|[0.688, 0.838]|<0.001|
|C_hard_rooms|152|scalar_onlstm/focal/w=1|18|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms|152|scalar_onlstm/focal/w=1.1|18|0.976|[0.957, 0.993]|<0.001|
|C_hard_rooms_large|56|field_hrm/astar|9|0.839|[0.646, 1.222]|0.371|
|C_hard_rooms_large|56|field_hrm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|56|field_hrm/focal/w=1.1|9|0.925|[0.838, 0.964]|0.008|
|C_hard_rooms_large|56|field_onlstm/astar|10|0.791|[0.635, 0.882]|0.002|
|C_hard_rooms_large|56|field_onlstm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|56|field_onlstm/focal/w=1.1|10|0.902|[0.826, 0.935]|0.002|
|C_hard_rooms_large|56|field_unet/astar|10|0.998|[0.656, 1.146]|0.695|
|C_hard_rooms_large|56|field_unet/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|56|field_unet/focal/w=1.1|9|0.839|[0.740, 1.000]|0.078|
|C_hard_rooms_large|56|scalar_hrm/astar|10|0.729|[0.393, 0.841]|0.004|
|C_hard_rooms_large|56|scalar_hrm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|56|scalar_hrm/focal/w=1.1|10|0.846|[0.710, 0.979]|0.012|
|C_hard_rooms_large|56|scalar_onlstm/astar|10|0.737|[0.436, 0.927]|0.004|
|C_hard_rooms_large|56|scalar_onlstm/focal/w=1|10|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|56|scalar_onlstm/focal/w=1.1|10|0.939|[0.865, 0.982]|0.008|
|C_hard_rooms_large|64|field_hrm/astar|19|0.814|[0.712, 0.929]|0.015|
|C_hard_rooms_large|64|field_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|64|field_hrm/focal/w=1.1|19|0.900|[0.838, 0.944]|<0.001|
|C_hard_rooms_large|64|field_onlstm/astar|19|0.725|[0.644, 0.831]|<0.001|
|C_hard_rooms_large|64|field_onlstm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|64|field_onlstm/focal/w=1.1|19|0.881|[0.814, 0.921]|<0.001|
|C_hard_rooms_large|64|field_unet/astar|18|0.797|[0.637, 0.976]|0.010|
|C_hard_rooms_large|64|field_unet/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|64|field_unet/focal/w=1.1|18|0.839|[0.761, 0.975]|0.005|
|C_hard_rooms_large|64|scalar_hrm/astar|19|0.649|[0.436, 0.821]|<0.001|
|C_hard_rooms_large|64|scalar_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|64|scalar_hrm/focal/w=1.1|19|0.893|[0.797, 0.922]|<0.001|
|C_hard_rooms_large|64|scalar_onlstm/astar|19|0.644|[0.436, 0.810]|<0.001|
|C_hard_rooms_large|64|scalar_onlstm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_rooms_large|64|scalar_onlstm/focal/w=1.1|19|0.930|[0.865, 0.952]|<0.001|
|C_hard_spiral|140|field_hrm/astar|6|0.850|[0.742, 0.919]|0.031|
|C_hard_spiral|140|field_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|140|field_hrm/focal/w=1.1|6|0.959|[0.949, 0.993]|0.062|
|C_hard_spiral|140|field_onlstm/astar|6|0.946|[0.922, 0.973]|0.031|
|C_hard_spiral|140|field_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|140|field_onlstm/focal/w=1.1|6|0.967|[0.936, 1.000]|0.094|
|C_hard_spiral|140|field_unet/astar|6|0.906|[0.820, 0.958]|0.031|
|C_hard_spiral|140|field_unet/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|140|field_unet/focal/w=1.1|6|0.971|[0.947, 0.993]|0.062|
|C_hard_spiral|140|scalar_hrm/astar|6|0.820|[0.739, 0.967]|0.031|
|C_hard_spiral|140|scalar_hrm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|140|scalar_hrm/focal/w=1.1|6|0.960|[0.929, 0.969]|0.031|
|C_hard_spiral|140|scalar_onlstm/astar|6|0.862|[0.699, 0.973]|0.031|
|C_hard_spiral|140|scalar_onlstm/focal/w=1|6|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|140|scalar_onlstm/focal/w=1.1|6|0.949|[0.927, 0.969]|0.031|
|C_hard_spiral|152|field_hrm/astar|19|0.828|[0.748, 0.892]|<0.001|
|C_hard_spiral|152|field_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|152|field_hrm/focal/w=1.1|19|0.978|[0.960, 0.993]|0.001|
|C_hard_spiral|152|field_onlstm/astar|19|0.940|[0.922, 0.973]|<0.001|
|C_hard_spiral|152|field_onlstm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|152|field_onlstm/focal/w=1.1|19|0.978|[0.954, 0.992]|<0.001|
|C_hard_spiral|152|field_unet/astar|19|0.924|[0.864, 0.964]|<0.001|
|C_hard_spiral|152|field_unet/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|152|field_unet/focal/w=1.1|19|0.980|[0.964, 0.993]|0.001|
|C_hard_spiral|152|scalar_hrm/astar|19|0.901|[0.841, 0.953]|<0.001|
|C_hard_spiral|152|scalar_hrm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|152|scalar_hrm/focal/w=1.1|19|0.971|[0.956, 0.987]|<0.001|
|C_hard_spiral|152|scalar_onlstm/astar|19|0.887|[0.838, 0.938]|<0.001|
|C_hard_spiral|152|scalar_onlstm/focal/w=1|19|1.000|[1.000, 1.000]|n/a|
|C_hard_spiral|152|scalar_onlstm/focal/w=1.1|19|0.971|[0.956, 0.980]|<0.001|

## Notes

- McNemar pairs each LEARNED arm against `euclid/astar` on success over shared worlds;
  gain = arm found & euclid not, loss = euclid found & arm not. `oracle` (ceiling) and
  `euclid` (reference) are NOT hypotheses under test and are excluded from the family.
- BH q-values correct ONLY across this success/McNemar grid. The expansion-Wilcoxon
  p-values are UNcorrected; the bootstrap CIs are the primary expansion inference.
- The expansion ratio uses the *matched set* (worlds euclid AND the arm both solved).
  Median ratio < 1 means the arm expands fewer nodes than euclid. The Wilcoxon p tests
  paired (ratio - 1) in ratio-space (matching the median ratio + CI estimand); the bootstrap
  CI is a seeded percentile CI on the median ratio.
- A p-value is shown as `n/a (n<6)` when the McNemar discordant count or the
  expansion matched-set n is below 6 (too few pairs for a trustworthy p).
