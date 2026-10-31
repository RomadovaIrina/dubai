# r10 face metrics (scripts/r10/face_metrics.py; ratios vs the source). L0 raw LatentSync .. L4 layers (25 fps vs master), F_ finals, A_ composite ablations, O_ out_stab arms

## old03
```
A_base         lipHF 0.93 lipBr 1.65 ring 1.81 ringS 52 | chinBr 0.95 chinTex 0.97 chinBrS 0 shadeHP 0.71 step 0.41 | vel 0.0018 lat 0.0005 swim 0.0023 ev 1 S 0
A_nearest      lipHF 0.93 lipBr 1.87 ring 1.97 ringS 71 | chinBr 0.97 chinTex 0.98 chinBrS 0 shadeHP 0.71 step 0.42 | vel 0.0021 lat 0.0006 swim 0.0024 ev 5 S 0
A_nocolour     lipHF 0.92 lipBr 1.56 ring 1.64 ringS 47 | chinBr 0.81 chinTex 0.97 chinBrS 0 shadeHP 0.74 step 0.44 | vel 0.0017 lat 0.0005 swim 0.0021 ev 0 S 0
A_noregion     lipHF 0.94 lipBr 1.66 ring 1.82 ringS 54 | chinBr 1.07 chinTex 0.90 chinBrS 0 shadeHP 1.01 step 0.62 | vel 0.0040 lat 0.0012 swim 0.0050 ev 38 S 0
A_nosharp      lipHF 0.81 lipBr 1.64 ring 1.77 ringS 57 | chinBr 0.95 chinTex 0.97 chinBrS 0 shadeHP 0.70 step 0.41 | vel 0.0018 lat 0.0005 swim 0.0023 ev 1 S 0
A_notex        lipHF 0.96 lipBr 1.65 ring 1.81 ringS 51 | chinBr 0.94 chinTex 0.96 chinBrS 0 shadeHP 0.71 step 0.42 | vel 0.0015 lat 0.0004 swim 0.0018 ev 0 S 0
A_rigidreg     lipHF 0.91 lipBr 1.65 ring 1.80 ringS 51 | chinBr 0.95 chinTex 0.97 chinBrS 0 shadeHP 0.72 step 0.42 | vel 0.0019 lat 0.0005 swim 0.0024 ev 0 S 0
F_fx           lipHF 0.93 lipBr 1.65 ring 1.81 ringS 52 | chinBr 0.95 chinTex 0.97 chinBrS 0 shadeHP 0.71 step 0.41 | vel 0.0018 lat 0.0005 swim 0.0023 ev 1 S 0
F_stab         lipHF 0.87 lipBr 1.60 ring 1.84 ringS 53 | chinBr 0.93 chinTex 0.96 chinBrS 0 shadeHP 0.70 step 0.40 | vel 0.0020 lat 0.0006 swim 0.0025 ev 3 S 0
L0_pre         lipHF 0.71 lipBr 1.85 ring 1.91 ringS 42 | chinBr 1.45 chinTex 0.81 chinBrS 1 shadeHP 1.12 step 0.87 | vel 0.0040 lat 0.0010 swim 0.0038 ev 31 S 0
L1_cf          lipHF 1.18 lipBr 1.76 ring 1.88 ringS 34 | chinBr 1.47 chinTex 1.60 chinBrS 4 shadeHP 1.11 step 0.88 | vel 0.0042 lat 0.0014 swim 0.0040 ev 37 S 0
L2_cfd         lipHF 0.86 lipBr 1.83 ring 1.91 ringS 43 | chinBr 1.42 chinTex 0.87 chinBrS 2 shadeHP 1.12 step 0.86 | vel 0.0039 lat 0.0011 swim 0.0037 ev 28 S 0
L3_sr          lipHF 0.92 lipBr 1.81 ring 1.87 ringS 44 | chinBr 1.39 chinTex 0.92 chinBrS 2 shadeHP 1.11 step 0.85 | vel 0.0039 lat 0.0011 swim 0.0036 ev 26 S 0
L4_rep         lipHF 0.94 lipBr 1.72 ring 1.80 ringS 34 | chinBr 1.33 chinTex 0.95 chinBrS 1 shadeHP 1.07 step 0.81 | vel 0.0042 lat 0.0012 swim 0.0039 ev 35 S 3
L4_stab        lipHF 0.72 lipBr 1.68 ring 1.98 ringS 50 | chinBr 1.29 chinTex 0.76 chinBrS 0 shadeHP 1.06 step 0.80 | vel 0.0048 lat 0.0013 swim 0.0043 ev 57 S 3
O_f1           lipHF 0.94 lipBr 1.15 ring 0.99 ringS 11 | chinBr 0.58 chinTex 0.99 chinBrS 0 shadeHP 0.63 step 0.36 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_f1c          lipHF 0.94 lipBr 1.15 ring 0.99 ringS 11 | chinBr 0.57 chinTex 0.94 chinBrS 0 shadeHP 0.59 step 0.34 | vel 0.0012 lat 0.0004 swim 0.0010 ev 0 S 0
O_f1l          lipHF 0.94 lipBr 1.14 ring 0.99 ringS 11 | chinBr 0.58 chinTex 0.98 chinBrS 0 shadeHP 0.62 step 0.36 | vel 0.0018 lat 0.0005 swim 0.0022 ev 0 S 0
O_os_sh        lipHF 0.94 lipBr 1.26 ring 1.32 ringS 21 | chinBr 0.80 chinTex 0.99 chinBrS 0 shadeHP 0.68 step 0.39 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_os_sh3       lipHF 0.94 lipBr 1.04 ring 1.09 ringS 8 | chinBr 0.67 chinTex 0.99 chinBrS 0 shadeHP 0.65 step 0.38 | vel 0.0018 lat 0.0006 swim 0.0022 ev 1 S 0
O_os_sh3f      lipHF 0.93 lipBr 0.95 ring 1.00 ringS 8 | chinBr 0.59 chinTex 0.99 chinBrS 0 shadeHP 0.62 step 0.35 | vel 0.0018 lat 0.0005 swim 0.0022 ev 0 S 0
O_os_sh3fk     lipHF 0.94 lipBr 1.15 ring 1.00 ringS 11 | chinBr 0.59 chinTex 0.99 chinBrS 0 shadeHP 0.62 step 0.35 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_os_shp       lipHF 0.68 lipBr 1.11 ring 1.17 ringS 20 | chinBr 0.57 chinTex 0.72 chinBrS 0 shadeHP 0.67 step 0.39 | vel 0.0019 lat 0.0006 swim 0.0023 ev 0 S 0
O_rs4_p0       lipHF 0.94 lipBr 1.62 ring 1.74 ringS 54 | chinBr 0.89 chinTex 0.99 chinBrS 0 shadeHP 0.71 step 0.42 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_rs4_p1       lipHF 0.94 lipBr 1.62 ring 1.74 ringS 54 | chinBr 0.89 chinTex 0.99 chinBrS 0 shadeHP 0.70 step 0.42 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_rs4_p1c      lipHF 0.94 lipBr 1.61 ring 1.74 ringS 54 | chinBr 0.89 chinTex 0.94 chinBrS 0 shadeHP 0.67 step 0.40 | vel 0.0012 lat 0.0004 swim 0.0010 ev 0 S 0
O_rs4_sh3f     lipHF 0.93 lipBr 0.95 ring 0.99 ringS 9 | chinBr 0.58 chinTex 0.99 chinBrS 0 shadeHP 0.63 step 0.36 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
O_rs4_sh3fk    lipHF 0.94 lipBr 1.15 ring 0.99 ringS 11 | chinBr 0.58 chinTex 0.99 chinBrS 0 shadeHP 0.63 step 0.36 | vel 0.0018 lat 0.0006 swim 0.0022 ev 0 S 0
```

## old05
```
F_fx           lipHF 0.97 lipBr 1.18 ring 1.08 ringS 6 | chinBr 0.96 chinTex 1.04 chinBrS 3 shadeHP 0.72 step 0.59 | vel 0.0044 lat 0.0034 swim 0.0041 ev 7 S 5
F_stab         lipHF 0.84 lipBr 1.16 ring 1.08 ringS 7 | chinBr 0.94 chinTex 1.04 chinBrS 1 shadeHP 0.64 step 0.46 | vel 0.0034 lat 0.0022 swim 0.0033 ev 2 S 0
L0_pre         lipHF 0.60 lipBr 1.22 ring 0.92 ringS 2 | chinBr 0.86 chinTex 0.71 chinBrS 0 shadeHP 1.06 step 0.76 | vel 0.0068 lat 0.0032 swim 0.0057 ev 79 S 3
L1_cf          lipHF 1.32 lipBr 1.23 ring 0.95 ringS 2 | chinBr 0.93 chinTex 2.02 chinBrS 0 shadeHP 1.07 step 0.80 | vel 0.0075 lat 0.0041 swim 0.0061 ev 100 S 7
L2_cfd         lipHF 0.83 lipBr 1.21 ring 0.93 ringS 2 | chinBr 0.87 chinTex 0.88 chinBrS 0 shadeHP 1.04 step 0.74 | vel 0.0071 lat 0.0036 swim 0.0058 ev 83 S 3
L3_sr          lipHF 1.06 lipBr 1.25 ring 0.95 ringS 3 | chinBr 0.86 chinTex 0.93 chinBrS 0 shadeHP 1.03 step 0.73 | vel 0.0072 lat 0.0037 swim 0.0059 ev 83 S 5
L4_rep         lipHF 1.07 lipBr 1.23 ring 0.98 ringS 5 | chinBr 1.00 chinTex 1.00 chinBrS 3 shadeHP 1.42 step 1.57 | vel 0.0160 lat 0.0130 swim 0.0098 ev 100 S 11
L4_stab        lipHF 0.83 lipBr 1.19 ring 0.97 ringS 3 | chinBr 0.85 chinTex 0.87 chinBrS 0 shadeHP 1.04 step 0.74 | vel 0.0081 lat 0.0045 swim 0.0064 ev 101 S 11
O_f1           lipHF 0.98 lipBr 0.92 ring 0.81 ringS 1 | chinBr 0.68 chinTex 1.09 chinBrS 0 shadeHP 0.71 step 0.54 | vel 0.0049 lat 0.0035 swim 0.0043 ev 10 S 4
O_f1c          lipHF 0.98 lipBr 0.92 ring 0.81 ringS 1 | chinBr 0.68 chinTex 1.07 chinBrS 0 shadeHP 0.70 step 0.54 | vel 0.0043 lat 0.0032 swim 0.0036 ev 7 S 3
O_os_sh3fk     lipHF 0.98 lipBr 0.91 ring 0.81 ringS 0 | chinBr 0.67 chinTex 1.09 chinBrS 0 shadeHP 0.70 step 0.53 | vel 0.0045 lat 0.0031 swim 0.0042 ev 12 S 2
O_rs4_p0       lipHF 0.99 lipBr 1.14 ring 1.03 ringS 5 | chinBr 0.92 chinTex 1.10 chinBrS 4 shadeHP 0.79 step 0.67 | vel 0.0053 lat 0.0041 swim 0.0047 ev 10 S 4
O_rs4_p1       lipHF 0.99 lipBr 1.13 ring 1.03 ringS 5 | chinBr 0.87 chinTex 1.10 chinBrS 2 shadeHP 0.79 step 0.65 | vel 0.0049 lat 0.0037 swim 0.0045 ev 11 S 4
O_rs4_p1c      lipHF 0.98 lipBr 1.13 ring 1.02 ringS 4 | chinBr 0.86 chinTex 1.09 chinBrS 2 shadeHP 0.78 step 0.65 | vel 0.0048 lat 0.0038 swim 0.0040 ev 9 S 4
```

## m03
```
A_base         lipHF 0.90 lipBr 2.14 ring 1.87 ringS 13 | chinBr 1.21 chinTex 1.06 chinBrS 0 shadeHP 1.31 step 0.81 | vel 0.0049 lat 0.0026 swim 0.0059 ev 25 S 1
A_layer_lm_smooth=2 lipHF 0.90 lipBr 2.18 ring 1.93 ringS 20 | chinBr 1.23 chinTex 1.06 chinBrS 1 shadeHP 1.32 step 0.82 | vel 0.0048 lat 0.0024 swim 0.0059 ev 26 S 1
A_layer_lm_smooth=4,mouth_region=0.25,chin_warp=on lipHF 0.90 lipBr 2.21 ring 1.94 ringS 20 | chinBr 1.07 chinTex 1.08 chinBrS 0 shadeHP 2.14 step 1.08 | vel 0.0065 lat 0.0016 swim 0.0094 ev 68 S 5
A_mouth_region=0.18,chin_warp=on lipHF 0.90 lipBr 2.13 ring 1.86 ringS 13 | chinBr 0.96 chinTex 1.06 chinBrS 0 shadeHP 2.71 step 1.29 | vel 0.0070 lat 0.0015 swim 0.0099 ev 87 S 7
A_mouth_region=0.25,chin_warp=on lipHF 0.90 lipBr 2.14 ring 1.87 ringS 13 | chinBr 1.07 chinTex 1.06 chinBrS 0 shadeHP 2.71 step 1.29 | vel 0.0065 lat 0.0018 swim 0.0089 ev 61 S 7
A_mouth_region=0.38,chin_warp=on lipHF 0.90 lipBr 2.13 ring 1.86 ringS 13 | chinBr 1.19 chinTex 1.06 chinBrS 0 shadeHP 2.46 step 1.28 | vel 0.0069 lat 0.0027 swim 0.0090 ev 72 S 5
A_nocolour     lipHF 0.90 lipBr 2.15 ring 1.84 ringS 18 | chinBr 1.19 chinTex 1.07 chinBrS 0 shadeHP 1.86 step 0.90 | vel 0.0058 lat 0.0027 swim 0.0068 ev 43 S 3
A_noregion     lipHF 0.90 lipBr 2.13 ring 1.86 ringS 13 | chinBr 1.22 chinTex 1.06 chinBrS 0 shadeHP 1.31 step 0.81 | vel 0.0049 lat 0.0026 swim 0.0059 ev 22 S 1
A_notex        lipHF 0.91 lipBr 2.15 ring 1.89 ringS 14 | chinBr 1.23 chinTex 1.08 chinBrS 0 shadeHP 1.32 step 0.81 | vel 0.0047 lat 0.0024 swim 0.0057 ev 24 S 1
A_reg_smooth=4,layer_lm_smooth=2 lipHF 0.89 lipBr 2.22 ring 1.99 ringS 18 | chinBr 1.20 chinTex 1.04 chinBrS 0 shadeHP 1.26 step 0.69 | vel 0.0040 lat 0.0018 swim 0.0055 ev 13 S 0
A_reg_smooth=4,layer_lm_smooth=3,mouth_region=0.25,chin_warp=on lipHF 0.90 lipBr 2.21 ring 1.98 ringS 18 | chinBr 1.09 chinTex 1.07 chinBrS 0 shadeHP 2.22 step 1.05 | vel 0.0065 lat 0.0014 swim 0.0089 ev 62 S 8
A_reg_smooth=4 lipHF 0.89 lipBr 2.17 ring 1.93 ringS 17 | chinBr 1.19 chinTex 1.04 chinBrS 0 shadeHP 1.26 step 0.69 | vel 0.0039 lat 0.0019 swim 0.0054 ev 16 S 0
A_reg_smooth=6 lipHF 0.89 lipBr 2.17 ring 1.93 ringS 17 | chinBr 1.19 chinTex 1.04 chinBrS 0 shadeHP 1.24 step 0.68 | vel 0.0039 lat 0.0019 swim 0.0054 ev 15 S 0
F_fx           lipHF 0.90 lipBr 2.14 ring 1.87 ringS 13 | chinBr 1.21 chinTex 1.06 chinBrS 0 shadeHP 1.31 step 0.81 | vel 0.0049 lat 0.0026 swim 0.0059 ev 25 S 1
F_stab         lipHF 0.82 lipBr 2.05 ring 1.82 ringS 13 | chinBr 1.19 chinTex 1.05 chinBrS 0 shadeHP 1.30 step 0.79 | vel 0.0050 lat 0.0027 swim 0.0060 ev 21 S 0
L0_pre         lipHF 0.72 lipBr 2.56 ring 2.14 ringS 25 | chinBr 1.51 chinTex 0.71 chinBrS 6 shadeHP 2.50 step 1.60 | vel 0.0117 lat 0.0076 swim 0.0117 ev 117 S 20
L1_cf          lipHF 1.32 lipBr 2.54 ring 2.28 ringS 28 | chinBr 1.61 chinTex 1.23 chinBrS 7 shadeHP 2.49 step 1.59 | vel 0.0115 lat 0.0070 swim 0.0113 ev 127 S 18
L2_cfd         lipHF 0.93 lipBr 2.55 ring 2.15 ringS 29 | chinBr 1.47 chinTex 0.95 chinBrS 5 shadeHP 2.47 step 1.58 | vel 0.0113 lat 0.0067 swim 0.0113 ev 130 S 18
L3_sr          lipHF 0.96 lipBr 2.52 ring 2.08 ringS 27 | chinBr 1.42 chinTex 1.02 chinBrS 4 shadeHP 2.47 step 1.58 | vel 0.0118 lat 0.0072 swim 0.0116 ev 132 S 20
L4_rep         lipHF 0.99 lipBr 2.29 ring 1.97 ringS 20 | chinBr 1.36 chinTex 1.09 chinBrS 5 shadeHP 2.41 step 1.51 | vel 0.0117 lat 0.0070 swim 0.0116 ev 133 S 22
L4_stab        lipHF 0.79 lipBr 2.27 ring 2.02 ringS 20 | chinBr 1.29 chinTex 0.93 chinBrS 3 shadeHP 2.39 step 1.49 | vel 0.0122 lat 0.0073 swim 0.0119 ev 136 S 21
O_f1           lipHF 0.93 lipBr 1.34 ring 1.04 ringS 2 | chinBr 0.80 chinTex 1.11 chinBrS 0 shadeHP 1.14 step 0.56 | vel 0.0039 lat 0.0020 swim 0.0053 ev 15 S 1
O_f1c          lipHF 0.93 lipBr 1.34 ring 1.03 ringS 2 | chinBr 0.78 chinTex 1.06 chinBrS 0 shadeHP 1.02 step 0.49 | vel 0.0029 lat 0.0016 swim 0.0030 ev 7 S 0
O_f1l          lipHF 0.93 lipBr 1.35 ring 1.05 ringS 2 | chinBr 0.79 chinTex 1.08 chinBrS 0 shadeHP 1.15 step 0.57 | vel 0.0037 lat 0.0016 swim 0.0049 ev 13 S 2
O_lipc         lipHF 0.94 lipBr 2.10 ring 1.80 ringS 12 | chinBr 1.12 chinTex 1.14 chinBrS 0 shadeHP 1.30 step 0.80 | vel 0.0049 lat 0.0026 swim 0.0059 ev 23 S 1
O_os_sh        lipHF 0.94 lipBr 1.53 ring 1.31 ringS 2 | chinBr 1.01 chinTex 1.14 chinBrS 0 shadeHP 1.27 step 0.75 | vel 0.0049 lat 0.0026 swim 0.0058 ev 26 S 1
O_os_sh3       lipHF 0.93 lipBr 1.25 ring 1.10 ringS 1 | chinBr 0.90 chinTex 1.14 chinBrS 0 shadeHP 1.23 step 0.72 | vel 0.0048 lat 0.0026 swim 0.0058 ev 22 S 1
O_os_sh3f      lipHF 0.93 lipBr 1.10 ring 0.99 ringS 1 | chinBr 0.81 chinTex 1.13 chinBrS 0 shadeHP 1.19 step 0.68 | vel 0.0048 lat 0.0026 swim 0.0058 ev 24 S 1
O_os_sh3fk     lipHF 0.93 lipBr 1.33 ring 1.02 ringS 1 | chinBr 0.81 chinTex 1.13 chinBrS 0 shadeHP 1.18 step 0.67 | vel 0.0048 lat 0.0026 swim 0.0058 ev 21 S 1
O_rs4_p0       lipHF 0.94 lipBr 2.11 ring 1.84 ringS 17 | chinBr 1.11 chinTex 1.12 chinBrS 0 shadeHP 1.25 step 0.69 | vel 0.0039 lat 0.0019 swim 0.0054 ev 14 S 1
O_rs4_p1       lipHF 0.94 lipBr 2.10 ring 1.83 ringS 14 | chinBr 1.11 chinTex 1.12 chinBrS 0 shadeHP 1.25 step 0.68 | vel 0.0039 lat 0.0020 swim 0.0054 ev 13 S 0
O_rs4_p1c      lipHF 0.93 lipBr 2.11 ring 1.83 ringS 15 | chinBr 1.09 chinTex 1.07 chinBrS 0 shadeHP 1.10 step 0.59 | vel 0.0030 lat 0.0016 swim 0.0030 ev 6 S 0
O_rs4_sh3f     lipHF 0.92 lipBr 1.11 ring 1.01 ringS 1 | chinBr 0.79 chinTex 1.11 chinBrS 0 shadeHP 1.15 step 0.57 | vel 0.0039 lat 0.0020 swim 0.0053 ev 13 S 1
O_rs4_sh3fk    lipHF 0.93 lipBr 1.34 ring 1.04 ringS 2 | chinBr 0.80 chinTex 1.11 chinBrS 0 shadeHP 1.14 step 0.56 | vel 0.0039 lat 0.0020 swim 0.0053 ev 15 S 1
O_rs4_sh3fk_cr lipHF 0.93 lipBr 1.35 ring 1.05 ringS 2 | chinBr 0.79 chinTex 1.08 chinBrS 0 shadeHP 1.15 step 0.57 | vel 0.0037 lat 0.0016 swim 0.0049 ev 13 S 2
O_rs4_sh3fk_crv lipHF 0.93 lipBr 1.35 ring 1.04 ringS 2 | chinBr 0.78 chinTex 1.07 chinBrS 0 shadeHP 1.06 step 0.51 | vel 0.0032 lat 0.0016 swim 0.0039 ev 7 S 0
O_rs4_sh3fk_crv1 lipHF 0.93 lipBr 1.34 ring 1.03 ringS 2 | chinBr 0.78 chinTex 1.06 chinBrS 0 shadeHP 1.02 step 0.49 | vel 0.0029 lat 0.0016 swim 0.0030 ev 7 S 0
```

## m18
```
A_base         lipHF 1.08 lipBr 3.01 ring 2.15 ringS 19 | chinBr 1.42 chinTex 1.04 chinBrS 2 shadeHP 1.84 step 1.08 | vel 0.0070 lat 0.0024 swim 0.0085 ev 25 S 2
A_layer_lm_smooth=2 lipHF 1.09 lipBr 3.01 ring 2.11 ringS 22 | chinBr 1.43 chinTex 1.05 chinBrS 2 shadeHP 1.84 step 1.06 | vel 0.0065 lat 0.0026 swim 0.0081 ev 20 S 3
A_layer_lm_smooth=4,mouth_region=0.25,chin_warp=on lipHF 1.10 lipBr 3.08 ring 2.15 ringS 21 | chinBr 1.16 chinTex 1.10 chinBrS 0 shadeHP 2.23 step 1.39 | vel 0.0074 lat 0.0009 swim 0.0107 ev 23 S 7
A_mouth_region=0.18,chin_warp=on lipHF 1.08 lipBr 3.00 ring 2.13 ringS 19 | chinBr 1.08 chinTex 1.10 chinBrS 0 shadeHP 2.11 step 1.29 | vel 0.0077 lat 0.0011 swim 0.0114 ev 41 S 5
A_mouth_region=0.25,chin_warp=on lipHF 1.08 lipBr 3.01 ring 2.16 ringS 19 | chinBr 1.14 chinTex 1.08 chinBrS 0 shadeHP 2.15 step 1.26 | vel 0.0069 lat 0.0010 swim 0.0098 ev 33 S 5
A_mouth_region=0.38,chin_warp=on lipHF 1.08 lipBr 3.01 ring 2.14 ringS 19 | chinBr 1.41 chinTex 1.05 chinBrS 2 shadeHP 2.08 step 1.20 | vel 0.0074 lat 0.0026 swim 0.0090 ev 33 S 2
A_nocolour     lipHF 1.08 lipBr 3.25 ring 2.35 ringS 21 | chinBr 1.53 chinTex 1.09 chinBrS 4 shadeHP 1.67 step 0.97 | vel 0.0070 lat 0.0027 swim 0.0084 ev 21 S 2
A_noregion     lipHF 1.08 lipBr 3.02 ring 2.18 ringS 20 | chinBr 1.47 chinTex 1.03 chinBrS 4 shadeHP 2.01 step 1.20 | vel 0.0077 lat 0.0029 swim 0.0091 ev 32 S 3
A_notex        lipHF 1.09 lipBr 3.02 ring 2.17 ringS 20 | chinBr 1.50 chinTex 1.06 chinBrS 3 shadeHP 1.84 step 1.08 | vel 0.0070 lat 0.0025 swim 0.0084 ev 22 S 2
A_reg_smooth=4,layer_lm_smooth=2 lipHF 1.08 lipBr 3.03 ring 2.12 ringS 22 | chinBr 1.44 chinTex 1.04 chinBrS 2 shadeHP 1.80 step 0.99 | vel 0.0061 lat 0.0025 swim 0.0078 ev 19 S 3
A_reg_smooth=4,layer_lm_smooth=3,mouth_region=0.25,chin_warp=on lipHF 1.09 lipBr 3.05 ring 2.14 ringS 23 | chinBr 1.14 chinTex 1.09 chinBrS 0 shadeHP 2.17 step 1.37 | vel 0.0075 lat 0.0009 swim 0.0105 ev 26 S 7
A_reg_smooth=4 lipHF 1.07 lipBr 3.02 ring 2.16 ringS 21 | chinBr 1.40 chinTex 1.04 chinBrS 2 shadeHP 1.79 step 0.98 | vel 0.0062 lat 0.0025 swim 0.0080 ev 23 S 3
A_reg_smooth=6 lipHF 1.07 lipBr 3.03 ring 2.17 ringS 21 | chinBr 1.40 chinTex 1.04 chinBrS 2 shadeHP 1.79 step 0.98 | vel 0.0063 lat 0.0026 swim 0.0081 ev 22 S 3
F_fx           lipHF 1.08 lipBr 3.01 ring 2.15 ringS 19 | chinBr 1.42 chinTex 1.04 chinBrS 2 shadeHP 1.84 step 1.08 | vel 0.0070 lat 0.0024 swim 0.0085 ev 25 S 2
F_stab         lipHF 0.94 lipBr 2.88 ring 2.12 ringS 18 | chinBr 1.31 chinTex 1.02 chinBrS 1 shadeHP 1.84 step 1.08 | vel 0.0077 lat 0.0034 swim 0.0091 ev 30 S 6
L0_pre         lipHF 0.77 lipBr 3.05 ring 2.08 ringS 21 | chinBr 1.55 chinTex 0.77 chinBrS 10 shadeHP 1.76 step 1.24 | vel 0.0071 lat 0.0022 swim 0.0061 ev 34 S 3
L1_cf          lipHF 1.62 lipBr 3.02 ring 2.23 ringS 23 | chinBr 2.07 chinTex 1.85 chinBrS 20 shadeHP 1.72 step 1.23 | vel 0.0076 lat 0.0025 swim 0.0066 ev 39 S 3
L2_cfd         lipHF 1.10 lipBr 3.00 ring 2.11 ringS 22 | chinBr 1.44 chinTex 1.00 chinBrS 7 shadeHP 1.76 step 1.23 | vel 0.0072 lat 0.0023 swim 0.0063 ev 34 S 4
L3_sr          lipHF 1.25 lipBr 3.03 ring 2.10 ringS 22 | chinBr 1.39 chinTex 1.10 chinBrS 3 shadeHP 1.77 step 1.23 | vel 0.0073 lat 0.0024 swim 0.0064 ev 36 S 4
L4_rep         lipHF 1.28 lipBr 2.80 ring 2.02 ringS 20 | chinBr 1.30 chinTex 1.19 chinBrS 4 shadeHP 1.82 step 1.30 | vel 0.0096 lat 0.0030 swim 0.0087 ev 44 S 6
L4_stab        lipHF 0.99 lipBr 2.72 ring 2.10 ringS 19 | chinBr 1.22 chinTex 1.00 chinBrS 2 shadeHP 1.81 step 1.29 | vel 0.0110 lat 0.0049 swim 0.0097 ev 38 S 8
O_f1           lipHF 1.13 lipBr 2.11 ring 1.34 ringS 3 | chinBr 1.10 chinTex 1.13 chinBrS 2 shadeHP 1.73 step 0.91 | vel 0.0063 lat 0.0026 swim 0.0081 ev 21 S 3
O_f1c          lipHF 1.13 lipBr 2.11 ring 1.34 ringS 3 | chinBr 1.11 chinTex 1.08 chinBrS 2 shadeHP 1.31 step 0.67 | vel 0.0045 lat 0.0021 swim 0.0036 ev 8 S 1
O_lipc         lipHF 1.14 lipBr 3.00 ring 2.15 ringS 19 | chinBr 1.42 chinTex 1.14 chinBrS 2 shadeHP 1.84 step 1.08 | vel 0.0070 lat 0.0026 swim 0.0085 ev 24 S 2
O_os_sh        lipHF 1.14 lipBr 2.34 ring 1.62 ringS 5 | chinBr 1.31 chinTex 1.14 chinBrS 2 shadeHP 1.83 step 1.06 | vel 0.0070 lat 0.0026 swim 0.0085 ev 23 S 2
O_os_sh3       lipHF 1.13 lipBr 1.91 ring 1.35 ringS 4 | chinBr 1.21 chinTex 1.14 chinBrS 2 shadeHP 1.82 step 1.05 | vel 0.0070 lat 0.0026 swim 0.0085 ev 26 S 2
O_os_sh3f      lipHF 1.13 lipBr 1.71 ring 1.24 ringS 2 | chinBr 1.13 chinTex 1.14 chinBrS 2 shadeHP 1.79 step 1.02 | vel 0.0070 lat 0.0026 swim 0.0085 ev 25 S 2
O_os_sh3fk     lipHF 1.14 lipBr 2.10 ring 1.36 ringS 3 | chinBr 1.12 chinTex 1.14 chinBrS 2 shadeHP 1.79 step 1.02 | vel 0.0070 lat 0.0026 swim 0.0086 ev 22 S 2
O_rs4_p0       lipHF 1.13 lipBr 2.97 ring 2.11 ringS 22 | chinBr 1.40 chinTex 1.13 chinBrS 2 shadeHP 1.79 step 0.99 | vel 0.0064 lat 0.0026 swim 0.0082 ev 21 S 3
O_rs4_p1       lipHF 1.14 lipBr 2.97 ring 2.12 ringS 20 | chinBr 1.41 chinTex 1.13 chinBrS 2 shadeHP 1.79 step 0.98 | vel 0.0063 lat 0.0026 swim 0.0081 ev 21 S 3
O_rs4_p1c      lipHF 1.13 lipBr 2.98 ring 2.11 ringS 19 | chinBr 1.40 chinTex 1.09 chinBrS 2 shadeHP 1.37 step 0.73 | vel 0.0045 lat 0.0022 swim 0.0036 ev 7 S 1
O_rs4_sh3f     lipHF 1.12 lipBr 1.73 ring 1.25 ringS 2 | chinBr 1.12 chinTex 1.13 chinBrS 2 shadeHP 1.73 step 0.91 | vel 0.0063 lat 0.0026 swim 0.0080 ev 21 S 3
O_rs4_sh3fk    lipHF 1.13 lipBr 2.11 ring 1.34 ringS 3 | chinBr 1.10 chinTex 1.13 chinBrS 2 shadeHP 1.73 step 0.91 | vel 0.0063 lat 0.0026 swim 0.0081 ev 21 S 3
O_rs4_sh3fk_cr lipHF 1.13 lipBr 2.10 ring 1.33 ringS 3 | chinBr 1.10 chinTex 1.11 chinBrS 2 shadeHP 1.74 step 0.91 | vel 0.0062 lat 0.0022 swim 0.0078 ev 20 S 3
O_rs4_sh3fk_crv lipHF 1.13 lipBr 2.10 ring 1.33 ringS 3 | chinBr 1.10 chinTex 1.09 chinBrS 1 shadeHP 1.46 step 0.76 | vel 0.0052 lat 0.0022 swim 0.0055 ev 12 S 3
O_rs4_sh3fk_crv1 lipHF 1.13 lipBr 2.11 ring 1.34 ringS 3 | chinBr 1.11 chinTex 1.08 chinBrS 2 shadeHP 1.31 step 0.67 | vel 0.0045 lat 0.0021 swim 0.0036 ev 8 S 1
```

## m15
```
F_fx           lipHF 0.80 lipBr 1.80 ring 1.90 ringS 15 | chinBr 1.39 chinTex 1.01 chinBrS 9 shadeHP 1.19 step 1.04 | vel 0.0069 lat 0.0036 swim 0.0082 ev 66 S 2
F_stab         lipHF 0.71 lipBr 1.68 ring 1.82 ringS 18 | chinBr 1.28 chinTex 0.99 chinBrS 5 shadeHP 1.18 step 1.03 | vel 0.0069 lat 0.0036 swim 0.0083 ev 66 S 2
L0_pre         lipHF 0.65 lipBr 2.33 ring 2.28 ringS 32 | chinBr 1.67 chinTex 0.75 chinBrS 14 shadeHP 1.85 step 1.43 | vel 0.0110 lat 0.0048 swim 0.0110 ev 105 S 28
L1_cf          lipHF 1.34 lipBr 2.42 ring 2.48 ringS 41 | chinBr 1.95 chinTex 1.39 chinBrS 21 shadeHP 1.85 step 1.42 | vel 0.0144 lat 0.0099 swim 0.0127 ev 109 S 30
L2_cfd         lipHF 0.88 lipBr 2.35 ring 2.34 ringS 37 | chinBr 1.60 chinTex 0.94 chinBrS 11 shadeHP 1.83 step 1.42 | vel 0.0141 lat 0.0097 swim 0.0123 ev 107 S 28
L3_sr          lipHF 0.93 lipBr 2.30 ring 2.27 ringS 28 | chinBr 1.54 chinTex 1.02 chinBrS 9 shadeHP 1.85 step 1.42 | vel 0.0114 lat 0.0049 swim 0.0116 ev 110 S 29
L4_rep         lipHF 0.95 lipBr 2.05 ring 2.07 ringS 20 | chinBr 1.44 chinTex 1.10 chinBrS 6 shadeHP 1.76 step 1.34 | vel 0.0114 lat 0.0053 swim 0.0115 ev 116 S 29
L4_stab        lipHF 0.74 lipBr 2.04 ring 2.17 ringS 28 | chinBr 1.36 chinTex 0.94 chinBrS 3 shadeHP 1.75 step 1.33 | vel 0.0117 lat 0.0052 swim 0.0117 ev 115 S 29
O_f1           lipHF 0.84 lipBr 1.34 ring 1.32 ringS 1 | chinBr 1.04 chinTex 1.07 chinBrS 2 shadeHP 1.22 step 1.08 | vel 0.0059 lat 0.0030 swim 0.0079 ev 39 S 3
```

## m19
```
F_fx           lipHF 0.84 lipBr 1.86 ring 2.27 ringS 14 | chinBr 1.78 chinTex 1.02 chinBrS 3 shadeHP 0.82 step 0.48 | vel 0.0041 lat 0.0019 swim 0.0051 ev 7 S 0
F_stab         lipHF 0.76 lipBr 1.74 ring 2.17 ringS 15 | chinBr 1.71 chinTex 1.01 chinBrS 1 shadeHP 0.83 step 0.49 | vel 0.0044 lat 0.0018 swim 0.0054 ev 10 S 0
L0_pre         lipHF 0.66 lipBr 2.25 ring 2.60 ringS 21 | chinBr 1.95 chinTex 0.68 chinBrS 8 shadeHP 1.52 step 0.92 | vel 0.0056 lat 0.0021 swim 0.0059 ev 19 S 1
L1_cf          lipHF 1.20 lipBr 2.30 ring 2.76 ringS 28 | chinBr 2.12 chinTex 1.06 chinBrS 12 shadeHP 1.49 step 0.90 | vel 0.0059 lat 0.0025 swim 0.0062 ev 18 S 1
L2_cfd         lipHF 0.86 lipBr 2.24 ring 2.60 ringS 21 | chinBr 1.92 chinTex 0.88 chinBrS 6 shadeHP 1.51 step 0.91 | vel 0.0056 lat 0.0023 swim 0.0059 ev 17 S 1
L3_sr          lipHF 0.97 lipBr 2.23 ring 2.54 ringS 21 | chinBr 1.87 chinTex 0.97 chinBrS 6 shadeHP 1.52 step 0.91 | vel 0.0057 lat 0.0024 swim 0.0060 ev 17 S 1
L4_rep         lipHF 0.99 lipBr 2.07 ring 2.44 ringS 15 | chinBr 1.84 chinTex 1.05 chinBrS 4 shadeHP 1.45 step 0.84 | vel 0.0058 lat 0.0027 swim 0.0059 ev 19 S 2
L4_stab        lipHF 0.79 lipBr 2.02 ring 2.44 ringS 17 | chinBr 1.88 chinTex 0.89 chinBrS 4 shadeHP 1.45 step 0.83 | vel 0.0060 lat 0.0027 swim 0.0059 ev 22 S 2
O_f1           lipHF 0.88 lipBr 1.40 ring 1.63 ringS 5 | chinBr 1.46 chinTex 1.04 chinBrS 1 shadeHP 0.74 step 0.38 | vel 0.0032 lat 0.0014 swim 0.0045 ev 1 S 0
```

## management r9 files (mgmt_r9)
```
m03_fxstab     lipHF 0.81 lipBr 2.05 ring 1.82 ringS 14 | chinBr 1.18 chinTex 1.05 chinBrS 0 shadeHP 1.31 step 0.79 | vel 0.0049 lat 0.0026 swim 0.0061 ev 26 S 0
m03_r9fix      lipHF 0.90 lipBr 2.12 ring 1.85 ringS 13 | chinBr 1.20 chinTex 1.06 chinBrS 0 shadeHP 1.31 step 0.79 | vel 0.0048 lat 0.0025 swim 0.0058 ev 21 S 1
m18_fxstab     lipHF 0.94 lipBr 2.70 ring 2.06 ringS 13 | chinBr 1.33 chinTex 1.02 chinBrS 3 shadeHP 1.72 step 1.01 | vel 0.0070 lat 0.0035 swim 0.0079 ev 27 S 4
m18_r9fix      lipHF 1.07 lipBr 2.99 ring 2.16 ringS 15 | chinBr 1.41 chinTex 1.04 chinBrS 1 shadeHP 1.73 step 1.00 | vel 0.0057 lat 0.0025 swim 0.0066 ev 24 S 1
old03_fxstab   lipHF 0.90 lipBr 1.52 ring 1.78 ringS 50 | chinBr 0.93 chinTex 0.96 chinBrS 0 shadeHP 0.67 step 0.38 | vel 0.0020 lat 0.0005 swim 0.0024 ev 1 S 0
old03_r9fix    lipHF 0.91 lipBr 1.56 ring 1.75 ringS 50 | chinBr 0.94 chinTex 0.96 chinBrS 0 shadeHP 0.68 step 0.39 | vel 0.0018 lat 0.0005 swim 0.0023 ev 1 S 0
old05_fxstab   lipHF 0.84 lipBr 1.15 ring 1.07 ringS 6 | chinBr 0.93 chinTex 1.04 chinBrS 1 shadeHP 0.64 step 0.46 | vel 0.0034 lat 0.0022 swim 0.0033 ev 3 S 0
old05_r9fix    lipHF 0.97 lipBr 1.20 ring 1.10 ringS 7 | chinBr 0.97 chinTex 1.04 chinBrS 4 shadeHP 0.73 step 0.62 | vel 0.0044 lat 0.0033 swim 0.0040 ev 9 S 4
```
