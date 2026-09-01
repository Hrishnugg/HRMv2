# C14-R: independent world-set replicates

## Draw 2

- dynamic|1024|conc|full_ft: -0.200 [-0.450,+0.000]
- dynamic|1024|conc|lora: +0.050 [+0.000,+0.150]
- dynamic|1024|conc|scratch: -0.300 [-0.550,-0.050]
- dynamic|1024|dist-conc|full_ft: +0.350 [+0.150,+0.550] (n=20)
- dynamic|1024|dist-conc|lora: +0.150 [+0.000,+0.300] (n=20)
- dynamic|1024|dist-conc|scratch: +0.400 [+0.200,+0.600] (n=20)
- dynamic|1024|dist|full_ft: +0.150 [+0.000,+0.300]
- dynamic|1024|dist|lora: +0.200 [+0.050,+0.400]
- dynamic|1024|dist|scratch: +0.100 [+0.000,+0.250]
- dynamic|16384|conc|full_ft: +0.000 [-0.150,+0.150]
- dynamic|16384|conc|lora: +0.000 [-0.150,+0.150]
- dynamic|16384|conc|scratch: -0.300 [-0.550,-0.050]
- dynamic|16384|dist-conc|full_ft: +0.200 [+0.050,+0.400] (n=20)
- dynamic|16384|dist-conc|lora: +0.200 [+0.050,+0.400] (n=20)
- dynamic|16384|dist-conc|scratch: +0.450 [+0.250,+0.650] (n=20)
- dynamic|16384|dist|full_ft: +0.200 [+0.050,+0.400]
- dynamic|16384|dist|lora: +0.200 [+0.050,+0.400]
- dynamic|16384|dist|scratch: +0.150 [+0.000,+0.300]
- static|256|conc|full_ft: -0.067 [-0.167,+0.000]
- static|256|conc|lora: -0.033 [-0.100,+0.000]
- static|256|conc|scratch: -0.133 [-0.267,-0.033]
- static|256|dist-conc|full_ft: +0.067 [+0.000,+0.167] (n=30)
- static|256|dist-conc|lora: +0.000 [-0.100,+0.100] (n=30)
- static|256|dist-conc|scratch: +0.133 [+0.033,+0.267] (n=30)
- static|256|dist|full_ft: +0.000 [+0.000,+0.000]
- static|256|dist|lora: -0.033 [-0.100,+0.000]
- static|256|dist|scratch: +0.000 [+0.000,+0.000]

## Draw 3

- dynamic|1024|conc|full_ft: -0.250 [-0.500,+0.000]
- dynamic|1024|conc|lora: +0.100 [+0.000,+0.250]
- dynamic|1024|conc|scratch: -0.300 [-0.550,-0.050]
- dynamic|1024|dist-conc|full_ft: +0.400 [+0.200,+0.600] (n=20)
- dynamic|1024|dist-conc|lora: +0.100 [+0.000,+0.250] (n=20)
- dynamic|1024|dist-conc|scratch: +0.450 [+0.250,+0.650] (n=20)
- dynamic|1024|dist|full_ft: +0.150 [+0.000,+0.350]
- dynamic|1024|dist|lora: +0.200 [+0.050,+0.400]
- dynamic|1024|dist|scratch: +0.150 [+0.000,+0.350]
- dynamic|16384|conc|full_ft: -0.100 [-0.300,+0.100]
- dynamic|16384|conc|lora: +0.100 [-0.100,+0.300]
- dynamic|16384|conc|scratch: -0.300 [-0.550,-0.050]
- dynamic|16384|dist-conc|full_ft: +0.300 [+0.100,+0.500] (n=20)
- dynamic|16384|dist-conc|lora: +0.100 [+0.000,+0.250] (n=20)
- dynamic|16384|dist-conc|scratch: +0.500 [+0.300,+0.700] (n=20)
- dynamic|16384|dist|full_ft: +0.200 [+0.050,+0.400]
- dynamic|16384|dist|lora: +0.200 [+0.050,+0.400]
- dynamic|16384|dist|scratch: +0.200 [+0.050,+0.400]
- static|256|conc|full_ft: -0.233 [-0.400,-0.100]
- static|256|conc|lora: -0.133 [-0.267,-0.033]
- static|256|conc|scratch: -0.500 [-0.667,-0.333]
- static|256|dist-conc|full_ft: +0.133 [+0.033,+0.267] (n=30)
- static|256|dist-conc|lora: +0.100 [+0.000,+0.200] (n=30)
- static|256|dist-conc|scratch: +0.367 [+0.200,+0.533] (n=30)
- static|256|dist|full_ft: -0.100 [-0.233,+0.000]
- static|256|dist|lora: -0.033 [-0.100,+0.000]
- static|256|dist|scratch: -0.133 [-0.267,-0.033]

## Preregistered readouts

- draw 2: R1 (dynamic conc N=1024 FT+scratch harmful CI) FAIL; R2 (dynamic dist N=1024 FT rescued) PASS; R3 (static conc N=256 harmful) FAIL; R4 (no significant LoRA drop) PASS
- draw 3: R1 (dynamic conc N=1024 FT+scratch harmful CI) FAIL; R2 (dynamic dist N=1024 FT rescued) PASS; R3 (static conc N=256 harmful) PASS; R4 (no significant LoRA drop) FAIL
