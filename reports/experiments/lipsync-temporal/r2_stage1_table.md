# Round 2 stage 1 — lower-face position arms (25 fps windows)

slide = |d_t − d_(t−1)| of the nose-base (alae / nostrils / subnasale / philtrum top) displacement output vs master on the same frame (DIS flow, inside the composite mask), px at master resolution; abs = |d_t|; corr = corr of d_x with the high-passed rigid head displacement; upper = sanity (source region, flow noise floor). shear / lf jerk / mouth resid / sharp = common metric v4 ratios vs the master window. LS s/min = LatentSync seconds per minute of LS frames (CPU shared with evals; for ranking only).

## 01 w0 (IOD 51.3 px)

| arm | nose slide mean / p95 | slide x | nose abs mean / p95 | cheek slide | corr shake | upper | SyncNet dub / mix (AV) | shear | lf jerk | mouth resid | sharp | LS s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean0 | 0.188 / 0.446 | 0.092 | 0.152 / 0.332 | 0.149 | -0.11 | 0.013 | 2.85 / 2.91 (0) | – | 1.02 | 1.02 | 0.94 | 447 |
| raw | 0.195 / 0.432 | 0.084 | 0.176 / 0.397 | 0.160 | -0.16 | 0.015 | 2.84 / 2.94 (0) | – | 1.05 | 1.04 | 0.94 | 433 |
| c032 | 0.069 / 0.170 | 0.026 | 0.077 / 0.182 | 0.067 | -0.07 | 0.013 | 2.82 / 2.85 (0) | – | 0.99 | 1.01 | 0.94 | 410 |
| c0nr32 | 0.066 / 0.159 | 0.028 | 0.076 / 0.175 | 0.067 | -0.02 | 0.013 | 2.66 / 2.79 (0) | – | 0.97 | 1.00 | 0.94 | 420 |
| flow4 | 0.195 / 0.483 | 0.090 | 0.173 / 0.385 | 0.154 | 0.04 | 0.013 | 2.98 / 3.11 (0) | – | 1.05 | 1.02 | 0.94 | 424 |
| flow4nr | 0.197 / 0.448 | 0.073 | 0.160 / 0.428 | 0.152 | -0.09 | 0.014 | 2.74 / 2.85 (0) | – | 0.98 | 1.01 | 0.94 | 433 |
| flow4nr32 | 0.065 / 0.142 | 0.029 | 0.074 / 0.193 | 0.067 | 0.08 | 0.013 | 2.71 / 2.79 (0) | – | 0.96 | 1.00 | 0.94 | 410 |
| kal | 0.219 / 0.487 | 0.093 | 0.168 / 0.380 | 0.167 | -0.15 | 0.016 | 2.96 / 3.04 (0) | – | 1.03 | 1.03 | 0.94 | 439 |
| sub3nr32 | 0.061 / 0.152 | 0.027 | 0.071 / 0.174 | 0.062 | 0.09 | 0.014 | 2.90 / 2.94 (0) | – | 0.96 | 1.00 | 0.95 | 408 |
| wrap | 0.173 / 0.448 | 0.085 | 0.133 / 0.315 | 0.152 | -0.01 | 0.015 | 1.10 / 0.90 (-3) | – | 1.05 | 1.01 | 0.98 | 159 |
| wrapnr | 0.211 / 0.512 | 0.099 | 0.148 / 0.298 | 0.181 | 0.00 | 0.013 | – / – (–) | – | – | – | – | 156 |
| wrap32 | 0.018 / 0.056 | 0.014 | 0.021 / 0.057 | 0.039 | -0.06 | 0.014 | – / – (–) | – | – | – | – | 125 |
| wrapnr32 | 0.020 / 0.049 | 0.016 | 0.023 / 0.052 | 0.045 | -0.03 | 0.015 | 1.05 / 0.82 (-3) | – | 0.99 | 1.00 | 1.00 | 129 |

## 03 w1 (IOD 128.0 px)

| arm | nose slide mean / p95 | slide x | nose abs mean / p95 | cheek slide | corr shake | upper | SyncNet dub / mix (AV) | shear | lf jerk | mouth resid | sharp | LS s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean0 | 0.176 / 0.413 | 0.091 | 0.168 / 0.331 | 0.164 | -0.22 | 0.017 | 7.50 / 7.07 (0) | – | 0.95 | 1.05 | 0.67 | 521 |
| raw | 0.202 / 0.448 | 0.108 | 0.177 / 0.346 | 0.194 | -0.13 | 0.017 | – / – (–) | – | – | – | – | 520 |
| rawnr32 | 0.101 / 0.257 | 0.039 | 0.107 / 0.235 | 0.109 | -0.18 | 0.017 | – / – (–) | – | – | – | – | 498 |
| c032 | 0.113 / 0.283 | 0.046 | 0.111 / 0.250 | 0.136 | -0.33 | 0.016 | – / – (–) | – | – | – | – | 452 |
| c0nr32 | 0.103 / 0.273 | 0.041 | 0.108 / 0.242 | 0.116 | -0.34 | 0.016 | 7.54 / 7.16 (0) | – | – | – | – | 519 |
| flow4nr32 | 0.099 / 0.254 | 0.040 | 0.108 / 0.259 | 0.108 | -0.25 | 0.015 | – / – (–) | – | – | – | – | 475 |
| kalnr32 | 0.101 / 0.242 | 0.039 | 0.106 / 0.230 | 0.117 | -0.21 | 0.016 | – / – (–) | – | – | – | – | 488 |
| sub3nr32 | 0.096 / 0.277 | 0.034 | 0.105 / 0.264 | 0.111 | -0.23 | 0.015 | – / – (–) | – | – | – | – | 455 |
| wrap | 0.117 / 0.277 | 0.062 | 0.105 / 0.224 | 0.114 | -0.00 | 0.014 | – / – (–) | – | – | – | – | 246 |
| wrapnr32 | 0.013 / 0.031 | 0.009 | 0.012 / 0.027 | 0.032 | -0.02 | 0.015 | – / – (–) | – | – | – | – | 140 |

## 03 w2 (IOD 126.7 px)

| arm | nose slide mean / p95 | slide x | nose abs mean / p95 | cheek slide | corr shake | upper | SyncNet dub / mix (AV) | shear | lf jerk | mouth resid | sharp | LS s/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean0 | 0.173 / 0.408 | 0.088 | 0.143 / 0.322 | 0.182 | -0.08 | 0.014 | – / – (–) | – | – | – | – | 508 |
| c0nr32 | 0.082 / 0.174 | 0.037 | 0.092 / 0.189 | 0.122 | -0.23 | 0.015 | – / – (–) | – | – | – | – | 502 |

Arms: `clean0` = clean0 (box ±2 anchors, mask ±3, rounded anchors, fp16 affine); `raw` = raw MediaPipe per frame (rounded, fp16); `rawnr32` = raw + sub-pixel anchors + fp32 affine; `c032` = clean0 + fp32 affine; `c0nr32` = clean0 + sub-pixel anchors + fp32 affine; `flow4` = flow-driven anchors (sigma 4); `flow4nr` = flow + sub-pixel; `flow4nr32` = flow-driven + sub-pixel + fp32; `kal` = Kalman CV+RTS (q5, r0.5); `kalnr32` = Kalman + sub-pixel + fp32; `sub3nr32` = MediaPipe every 3rd frame + flow track + sub-pixel + fp32; `wrap` = NO generation, clean0 geometry (crop -> paste only); `wrapnr` = no generation, sub-pixel; `wrap32` = no generation, fp32; `wrapnr32` = NO generation, sub-pixel + fp32; `wrapflownr32` = no generation, flow + sub-pixel + fp32; `c0nr` = clean0 + sub-pixel
