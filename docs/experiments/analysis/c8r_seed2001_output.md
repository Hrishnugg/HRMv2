# Fixed-provider C8 analysis: c8r_seed2001_eval

Primary provider: `field_unet_blind` (additive mode, binding budgets), fixed across all suites.
Statistics are map-level (n=20 maps/suite; 10k bootstraps; seed 20260723).

## Primary: field U-Net blind vs Euclid
| Suite | Euclid succ. | Blind U-Net succ. | Dsucc [95% CI] | matched median ratio [95% CI] | matched n |
|---|---:|---:|---|---|---:|
| Crossing | 0.12 | 0.98 | +0.86 [+0.74, +0.96] | 0.434 [0.201, 0.757] | 5 |
| Maze | 0.12 | 0.98 | +0.86 [+0.76, +0.96] | 0.059 [0.024, 0.167] | 6 |
| Dense maze | 0.06 | 0.52 | +0.46 [+0.32, +0.60] | 0.625 [0.355, 0.740] | 3 |
| Rooms | 0.42 | 1.00 | +0.58 [+0.44, +0.72] | 0.135 [0.076, 0.161] | 21 |
| Large rooms | 0.82 | 0.92 | +0.10 [-0.02, +0.22] | 0.548 [0.348, 0.653] | 39 |
| Spiral | 0.16 | 1.00 | +0.84 [+0.74, +0.94] | 0.089 [0.035, 0.164] | 8 |

## Aware minus blind (field U-Net twins), map-paired
| Suite | Dsucc (aware-blind) [95% CI] | D median ratio on jointly solved |
|---|---|---|
| Crossing | +0.000 [-0.060, +0.060] | -0.236 |
| Maze | +0.000 [+0.000, +0.000] | -0.025 |
| Dense maze | +0.060 [-0.040, +0.160] | -0.207 |
| Rooms | +0.000 [+0.000, +0.000] | -0.044 |
| Large rooms | +0.080 [+0.020, +0.160] | -0.275 |
| Spiral | +0.000 [+0.000, +0.000] | -0.012 |

## Secondary: per-suite best arm (post-hoc selection, labeled as such)
| Suite | Best arm | succ. | matched median ratio | matched n |
|---|---|---:|---:|---:|
| Crossing | field_unet | 0.98 | 0.410 | 6 |
| Maze | field_unet | 0.98 | 0.033 | 6 |
| Dense maze | field_unet | 0.58 | 0.418 | 3 |
| Rooms | field_unet | 1.00 | 0.091 | 21 |
| Large rooms | field_unet | 1.00 | 0.321 | 41 |
| Spiral | field_unet | 1.00 | 0.076 | 8 |
