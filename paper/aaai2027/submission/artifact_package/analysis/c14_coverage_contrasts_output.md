# C14 direct coverage contrasts (distributed - concentrated)

Paired per-map success differences, adaptation seeds averaged within maps, 10k map bootstraps. 30 cells; marginal intervals (one declared family per domain x method, 5 N levels each).

| Domain | Method | N | dist-conc | 95% CI | n maps |
|---|---|---|---|---|---|
| static | full_ft | 256 | +0.233* | [+0.111,+0.378] | 30 |
| static | full_ft | 1024 | -0.022 | [-0.056,+0.000] | 30 |
| static | full_ft | 4096 | +0.011 | [-0.033,+0.067] | 30 |
| static | full_ft | 16384 | -0.011 | [-0.033,+0.000] | 30 |
| static | full_ft | 65536 | +0.011 | [+0.000,+0.033] | 30 |
| static | lora | 256 | -0.011 | [-0.067,+0.044] | 30 |
| static | lora | 1024 | +0.000 | [+0.000,+0.000] | 30 |
| static | lora | 4096 | +0.033 | [+0.000,+0.100] | 30 |
| static | lora | 16384 | -0.011 | [-0.033,+0.000] | 30 |
| static | lora | 65536 | -0.011 | [-0.033,+0.000] | 30 |
| static | scratch | 256 | +0.311* | [+0.156,+0.467] | 30 |
| static | scratch | 1024 | +0.033 | [-0.033,+0.122] | 30 |
| static | scratch | 4096 | +0.044 | [+0.000,+0.122] | 30 |
| static | scratch | 16384 | +0.000 | [+0.000,+0.000] | 30 |
| static | scratch | 65536 | -0.011 | [-0.033,+0.000] | 30 |
| dynamic | full_ft | 256 | +0.483* | [+0.283,+0.700] | 20 |
| dynamic | full_ft | 1024 | +0.450* | [+0.250,+0.650] | 20 |
| dynamic | full_ft | 4096 | +0.300* | [+0.133,+0.483] | 20 |
| dynamic | full_ft | 16384 | +0.350* | [+0.183,+0.533] | 20 |
| dynamic | full_ft | 65536 | +0.067 | [+0.000,+0.183] | 20 |
| dynamic | lora | 256 | +0.117 | [+0.000,+0.267] | 20 |
| dynamic | lora | 1024 | +0.117* | [+0.017,+0.250] | 20 |
| dynamic | lora | 4096 | +0.117* | [+0.017,+0.250] | 20 |
| dynamic | lora | 16384 | +0.150 | [+0.000,+0.300] | 20 |
| dynamic | lora | 65536 | +0.067 | [+0.000,+0.183] | 20 |
| dynamic | scratch | 256 | +0.467* | [+0.250,+0.683] | 20 |
| dynamic | scratch | 1024 | +0.417* | [+0.217,+0.633] | 20 |
| dynamic | scratch | 4096 | +0.450* | [+0.250,+0.650] | 20 |
| dynamic | scratch | 16384 | +0.483* | [+0.283,+0.700] | 20 |
| dynamic | scratch | 65536 | +0.050 | [+0.000,+0.133] | 20 |

*: marginal 95% interval excludes zero.
