# Fixed-provider C8 analysis: c8r_seed2002_eval

Primary provider: `field_unet_blind` (additive mode, binding budgets), fixed across all suites.
Statistics are map-level (n=20 maps/suite; 10k bootstraps; seed 20260723).

## Primary: field U-Net blind vs Euclid
| Suite | Euclid succ. | Blind U-Net succ. | Dsucc [95% CI] | matched median ratio [95% CI] | matched n |
|---|---:|---:|---|---|---:|
| Crossing | 0.12 | 1.00 | +0.88 [+0.78, +0.96] | 0.299 [0.132, 0.725] | 6 |
| Maze | 0.12 | 0.96 | +0.84 [+0.74, +0.94] | 0.059 [0.037, 0.107] | 6 |
| Dense maze | 0.06 | 0.62 | +0.56 [+0.42, +0.70] | 0.378 [0.324, 0.571] | 3 |
| Rooms | 0.42 | 1.00 | +0.58 [+0.44, +0.72] | 0.117 [0.064, 0.130] | 21 |
| Large rooms | 0.82 | 0.98 | +0.16 [+0.06, +0.26] | 0.255 [0.160, 0.407] | 41 |
| Spiral | 0.16 | 1.00 | +0.84 [+0.74, +0.94] | 0.065 [0.039, 0.171] | 8 |

## Aware minus blind (field U-Net twins), map-paired
| Suite | Dsucc (aware-blind) [95% CI] | D median ratio on jointly solved |
|---|---|---|
| Crossing | +0.000 [+0.000, +0.000] | -0.147 |
| Maze | +0.020 [+0.000, +0.060] | -0.022 |
| Dense maze | -0.060 [-0.140, +0.020] | -0.004 |
| Rooms | +0.000 [+0.000, +0.000] | -0.032 |
| Large rooms | +0.020 [+0.000, +0.060] | -0.068 |
| Spiral | +0.000 [+0.000, +0.000] | -0.008 |

## Secondary: per-suite best arm (post-hoc selection, labeled as such)
| Suite | Best arm | succ. | matched median ratio | matched n |
|---|---|---:|---:|---:|
| Crossing | field_unet | 1.00 | 0.152 | 6 |
| Maze | field_unet | 0.98 | 0.037 | 6 |
| Dense maze | field_unet_blind | 0.62 | 0.378 | 3 |
| Rooms | field_unet | 1.00 | 0.086 | 21 |
| Large rooms | field_unet | 1.00 | 0.187 | 41 |
| Spiral | field_unet | 1.00 | 0.057 | 8 |
