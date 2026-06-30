# C10 — Bracketing (G0b) + RBF descriptor selectivity

Per target: per-AXIS bracketing (target descriptor interior to its OWN axis's source centroids) and the RBF weight mass landing on the target's own axis (descriptor selectivity).

## C10_maze_tgt / hrm
- axis: **maze**; same-axis sources: ['C10_maze_d0', 'C10_maze_d1', 'C10_maze_d2', 'C10_maze_d3']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.998** (selectivity: good)
- RBF weights: C10_maze_d0=0.053, C10_maze_d1=0.284, C10_maze_d2=0.393, C10_maze_d3=0.269, C10_rooms_s10=0.000, C10_rooms_s20=0.001, C10_rooms_s30=0.001, C10_rooms_s40=0.000

## C10_maze_tgt / onlstm
- axis: **maze**; same-axis sources: ['C10_maze_d0', 'C10_maze_d1', 'C10_maze_d2', 'C10_maze_d3']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.998** (selectivity: good)
- RBF weights: C10_maze_d0=0.053, C10_maze_d1=0.284, C10_maze_d2=0.393, C10_maze_d3=0.269, C10_rooms_s10=0.000, C10_rooms_s20=0.001, C10_rooms_s30=0.001, C10_rooms_s40=0.000

## C10_rooms_t25 / hrm
- axis: **rooms**; same-axis sources: ['C10_rooms_s10', 'C10_rooms_s20', 'C10_rooms_s30', 'C10_rooms_s40']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.986** (selectivity: good)
- RBF weights: C10_maze_d0=0.000, C10_maze_d1=0.000, C10_maze_d2=0.003, C10_maze_d3=0.011, C10_rooms_s10=0.000, C10_rooms_s20=0.335, C10_rooms_s30=0.398, C10_rooms_s40=0.252

## C10_rooms_t25 / onlstm
- axis: **rooms**; same-axis sources: ['C10_rooms_s10', 'C10_rooms_s20', 'C10_rooms_s30', 'C10_rooms_s40']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.986** (selectivity: good)
- RBF weights: C10_maze_d0=0.000, C10_maze_d1=0.000, C10_maze_d2=0.003, C10_maze_d3=0.011, C10_rooms_s10=0.000, C10_rooms_s20=0.335, C10_rooms_s30=0.398, C10_rooms_s40=0.252

## C10_rooms_t35 / hrm
- axis: **rooms**; same-axis sources: ['C10_rooms_s10', 'C10_rooms_s20', 'C10_rooms_s30', 'C10_rooms_s40']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.996** (selectivity: good)
- RBF weights: C10_maze_d0=0.000, C10_maze_d1=0.000, C10_maze_d2=0.001, C10_maze_d3=0.003, C10_rooms_s10=0.000, C10_rooms_s20=0.145, C10_rooms_s30=0.421, C10_rooms_s40=0.430

## C10_rooms_t35 / onlstm
- axis: **rooms**; same-axis sources: ['C10_rooms_s10', 'C10_rooms_s20', 'C10_rooms_s30', 'C10_rooms_s40']
- per-axis bracketing_ok: **True** (active-dim violations: [])
- RBF mass on own axis: **0.996** (selectivity: good)
- RBF weights: C10_maze_d0=0.000, C10_maze_d1=0.000, C10_maze_d2=0.001, C10_maze_d3=0.003, C10_rooms_s10=0.000, C10_rooms_s20=0.145, C10_rooms_s30=0.421, C10_rooms_s40=0.430
