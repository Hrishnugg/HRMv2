# C14 coverage contrasts: BH-adjusted sign-flip inference

| Domain | Method | N | dist-conc | p (sign-flip) | q (BH, 30) |
|---|---|---|---|---|---|
| static | full_ft | 256 | +0.233 | 0.0045 | 0.0150* |
| static | full_ft | 1024 | -0.022 | 0.5056 | 0.7984 |
| static | full_ft | 4096 | +0.011 | 1.0000 | 1.0000 |
| static | full_ft | 16384 | -0.011 | 1.0000 | 1.0000 |
| static | full_ft | 65536 | +0.011 | 1.0000 | 1.0000 |
| static | lora | 256 | -0.011 | 1.0000 | 1.0000 |
| static | lora | 1024 | +0.000 | 1.0000 | 1.0000 |
| static | lora | 4096 | +0.033 | 1.0000 | 1.0000 |
| static | lora | 16384 | -0.011 | 1.0000 | 1.0000 |
| static | lora | 65536 | -0.011 | 1.0000 | 1.0000 |
| static | scratch | 256 | +0.311 | 0.0008 | 0.0144* |
| static | scratch | 1024 | +0.033 | 0.6897 | 1.0000 |
| static | scratch | 4096 | +0.044 | 0.5049 | 0.7984 |
| static | scratch | 16384 | +0.000 | 1.0000 | 1.0000 |
| static | scratch | 65536 | -0.011 | 1.0000 | 1.0000 |
| dynamic | full_ft | 256 | +0.483 | 0.0013 | 0.0144* |
| dynamic | full_ft | 1024 | +0.450 | 0.0042 | 0.0150* |
| dynamic | full_ft | 4096 | +0.300 | 0.0074 | 0.0223* |
| dynamic | full_ft | 16384 | +0.350 | 0.0021 | 0.0144* |
| dynamic | full_ft | 65536 | +0.067 | 0.5031 | 0.7984 |
| dynamic | lora | 256 | +0.117 | 0.2530 | 0.5421 |
| dynamic | lora | 1024 | +0.117 | 0.1264 | 0.3161 |
| dynamic | lora | 4096 | +0.117 | 0.1244 | 0.3161 |
| dynamic | lora | 16384 | +0.150 | 0.2501 | 0.5421 |
| dynamic | lora | 65536 | +0.067 | 0.5012 | 0.7984 |
| dynamic | scratch | 256 | +0.467 | 0.0021 | 0.0144* |
| dynamic | scratch | 1024 | +0.417 | 0.0043 | 0.0150* |
| dynamic | scratch | 4096 | +0.450 | 0.0038 | 0.0150* |
| dynamic | scratch | 16384 | +0.483 | 0.0024 | 0.0144* |
| dynamic | scratch | 65536 | +0.050 | 0.4969 | 0.7984 |

q<0.05 cells: 10 of 30
FT/scratch cells with w_min<=2: 10/10 at q<0.05; with w_min>=4: 0/10
