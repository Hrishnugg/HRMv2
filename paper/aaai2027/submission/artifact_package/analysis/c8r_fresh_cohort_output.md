# Fixed-provider C8 analysis: c8r_fresh_eval

Primary provider: `field_unet_blind` (additive mode, binding budgets), fixed across all suites.
Statistics are map-level (n=20 maps/suite; 10k bootstraps; seed 20260723).

## Primary: field U-Net blind vs Euclid
| Suite | Euclid succ. | Blind U-Net succ. | Dsucc [95% CI] | matched median ratio [95% CI] | matched n |
|---|---:|---:|---|---|---:|
| Crossing | 0.12 | 0.92 | +0.80 [+0.68, +0.90] | 0.273 [0.109, 0.516] | 6 |
| Maze | 0.12 | 0.96 | +0.84 [+0.74, +0.94] | 0.047 [0.021, 0.153] | 6 |
| Dense maze | 0.06 | 0.70 | +0.64 [+0.50, +0.78] | 0.216 [0.100, 0.281] | 3 |
| Rooms | 0.42 | 1.00 | +0.58 [+0.44, +0.72] | 0.121 [0.106, 0.172] | 21 |
| Large rooms | 0.82 | 1.00 | +0.18 [+0.08, +0.30] | 0.250 [0.188, 0.306] | 41 |
| Spiral | 0.16 | 1.00 | +0.84 [+0.74, +0.94] | 0.092 [0.065, 0.146] | 8 |

## Aware minus blind (field U-Net twins), map-paired
| Suite | Dsucc (aware-blind) [95% CI] | D median ratio on jointly solved |
|---|---|---|
| Crossing | +0.080 [+0.020, +0.160] | -0.064 |
| Maze | +0.020 [+0.000, +0.060] | +0.030 |
| Dense maze | -0.120 [-0.220, -0.040] | +0.267 |
| Rooms | +0.000 [+0.000, +0.000] | -0.027 |
| Large rooms | -0.200 [-0.320, -0.100] | +0.072 |
| Spiral | -0.040 [-0.100, +0.000] | +0.031 |

## Secondary: per-suite best arm (post-hoc selection, labeled as such)
| Suite | Best arm | succ. | matched median ratio | matched n |
|---|---|---:|---:|---:|
| Crossing | field_unet | 1.00 | 0.210 | 6 |
| Maze | field_unet | 0.98 | 0.077 | 6 |
| Dense maze | field_unet_blind | 0.70 | 0.216 | 3 |
| Rooms | field_unet | 1.00 | 0.094 | 21 |
| Large rooms | field_unet_blind | 1.00 | 0.250 | 41 |
| Spiral | field_unet_blind | 1.00 | 0.092 | 8 |
