# Fixed-provider C8 reanalysis (2026-07-23)

Primary provider: `field_unet_blind` (additive mode, binding budgets), fixed across all suites.
Statistics are map-level (n=20 maps/suite; 10k bootstraps; seed 20260723).

## Primary: field U-Net blind vs Euclid
| Suite | Euclid succ. | Blind U-Net succ. | Dsucc [95% CI] | matched median ratio [95% CI] | matched n |
|---|---:|---:|---|---|---:|
| Crossing | 0.30 | 0.95 | +0.65 [+0.45, +0.85] | 0.191 [0.166, 0.878] | 6 |
| Maze | 0.30 | 1.00 | +0.70 [+0.50, +0.90] | 0.093 [0.057, 0.137] | 6 |
| Dense maze | 0.05 | 0.75 | +0.70 [+0.50, +0.90] | 0.246 [0.246, 0.246] | 1 |
| Rooms | 0.30 | 1.00 | +0.70 [+0.50, +0.90] | 0.099 [0.050, 0.214] | 6 |
| Large rooms | 0.75 | 0.95 | +0.20 [+0.05, +0.40] | 0.228 [0.185, 0.418] | 15 |
| Spiral | 0.20 | 0.95 | +0.75 [+0.55, +0.90] | 0.087 [0.061, 0.287] | 4 |

## Aware minus blind (field U-Net twins), map-paired
| Suite | Dsucc (aware-blind) [95% CI] | D median ratio on jointly solved |
|---|---|---|
| Crossing | +0.050 [+0.000, +0.150] | +0.067 |
| Maze | +0.000 [+0.000, +0.000] | -0.029 |
| Dense maze | +0.000 [+0.000, +0.000] | +0.032 |
| Rooms | +0.000 [+0.000, +0.000] | -0.003 |
| Large rooms | -0.050 [-0.200, +0.100] | +0.145 |
| Spiral | -0.050 [-0.150, +0.000] | -0.011 |

## Secondary: per-suite best arm (post-hoc selection, labeled as such)
| Suite | Best arm | succ. | matched median ratio | matched n |
|---|---|---:|---:|---:|
| Crossing | field_unet | 1.00 | 0.258 | 6 |
| Maze | field_hrm_blind | 1.00 | 0.054 | 6 |
| Dense maze | field_unet_blind | 0.75 | 0.246 | 1 |
| Rooms | field_hrm_blind | 1.00 | 0.080 | 6 |
| Large rooms | field_unet_blind | 0.95 | 0.228 | 15 |
| Spiral | field_unet_blind | 0.95 | 0.087 | 4 |
