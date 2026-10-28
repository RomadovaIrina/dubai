# Regression gate: NEW `m8` vs PREV `m7`

commit ab134f4

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 01 new / prev | 05 new / prev | 02 new / prev | 04 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.860 / 7.791 = | 6.732 / 6.411 better | 4.030 / 4.157 = | None / None n/a | 6.130 / 6.212 = |
| SyncNet dub-only (whole video, info) | 6.650 / 6.630 INFO | 6.820 / 6.520 INFO | 2.630 / 2.640 INFO | None / None n/a | 4.150 / 4.250 INFO |
| SyncNet mix (whole video, info) | 6.180 / 6.240 INFO | 6.650 / 6.190 INFO | 2.080 / 2.210 INFO | None / None n/a | 3.680 / 3.750 INFO |
| dub silent while face tracked, s (audio info) | 19 / 17 INFO | 0 / 0 INFO | 10 / 10 INFO | None / None n/a | 19 / 20 INFO |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a | 0 / 0 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.570 / 1.770 = | 0.000 / 0.280 = | 1.470 / 2.400 better | 0.000 / 0.000 = | 2.530 / 2.860 = |
| v4 resid mouth | 1.189 / 1.194 = | 1.101 / 1.127 = | 0.975 / 0.997 = | None / None n/a | 1.009 / 1.014 = |
| v4 twitch mouth | 0.970 / 0.970 = | 1.060 / 1.079 = | 0.954 / 0.966 = | None / None n/a | 1.011 / 1.034 = |
| v4 shear95 lower face | 0.995 / 0.993 = | 0.931 / 0.960 = | 0.999 / 0.997 = | None / None n/a | 1.002 / 1.004 = |
| v4 resid mouth, own frames (info) | 1.188 / 1.189 INFO | 1.105 / 1.133 INFO | 0.972 / 0.997 INFO | None / None n/a | 1.028 / 1.011 INFO |
| v4 common lip-synced seconds (info) | 80.8 / 80.8 INFO | 10.4 / 10.4 INFO | 43.0 / 43.0 INFO | None / None n/a | 43.4 / 43.4 INFO |
| v4 sharpness mouth (src=1) | 0.846 / 0.852 = | 0.943 / 0.942 = | 0.963 / 0.962 = | None / None n/a | 0.937 / 0.926 = |
| slide nose-base px | 0.063 / 0.062 = | 0.078 / 0.083 = | 0.035 / 0.033 = | None / None n/a | 0.111 / 0.141 better |
| window joints ratio | 0.917 / 0.886 = | 0.918 / 0.916 = | 0.912 / 0.966 better | None / None n/a | 0.916 / 0.909 = |
| window joints LF ratio | 0.843 / 0.802 **FAIL** | 0.923 / 0.940 = | 0.839 / 0.947 better | None / None n/a | 0.886 / 0.862 = |
| bg ring lower-upper | 0.056 / 0.055 = | -0.354 / -0.353 = | -0.136 / -0.136 = | None / None n/a | -0.095 / -0.094 = |
| lip flash strong | 57 / 54 = | 2 / 1 = | 7 / 3 **FAIL** | 0 / 0 = | 17 / 14 **FAIL** |
| colour worst 1 s |dL| (half face) | 1.300 / 1.220 = | 1.090 / 1.140 = | 1.670 / 1.690 = | 0 / 0 = | 1.410 / 1.400 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.941 / 0.941 = | 0.971 / 1.016 = | 1.012 / 1.010 = | None / None n/a | 0.997 / 0.994 = |
| chin wave (lower-chin residual std, 2 s) | 1.119 / 1.121 = | 1.285 / 1.299 = | 1.003 / 0.955 = | None / None n/a | 2.151 / 2.171 = |
| chin wave worst 2 s | 1.814 / 1.688 = | 1.725 / 1.632 = | 1.999 / 2.068 = | None / None n/a | 4.137 / 4.067 = |
| below-jaw band |out-src| | 2.502 / 2.502 = | 2.330 / 2.334 = | 2.566 / 2.572 = | nan / nan = | 2.485 / 2.483 = |
| WER final mix (audio, info) | 0.029 / 0.025 INFO | 0.103 / 0.088 INFO | 0.024 / 0.037 INFO | 0.027 / 0.027 INFO | 0.045 / 0.045 INFO |
| audio hard cuts (audio, info) | 1 / 1 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO | 0.000 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.200 / -1.200 INFO | -3.000 / -2.900 INFO | -1.300 / -1.400 INFO | -1.200 / -1.300 INFO | -3.300 / -3.300 INFO |
| video s / video-min | 467.3 / 466.1 = (over limit) | 98.8 / 94.5 = | 457.0 / 460.0 = (over limit) | 37.1 / 54.0 better | 526.1 / 489.5 = (over limit) |
| peak VRAM MiB | 14602 / 14584 INFO | 15435 / 14388 INFO | 15973 / 14514 INFO | 3348 / 3332 INFO | 15822 / 15283 INFO |
| audio stream md5 | 390ce760 / 73f2cb7c INFO | 5eecaa8d / a63681f1 INFO | 575c1173 / 83b307b8 INFO | d23ad4b6 / dd923747 INFO | 501f6fcb / ccb6e611 INFO |

## Verdict: **FAIL — no publish**
- 03 window joints LF ratio: 0.843 vs 0.802 (FAIL)
- 05 lip flash strong: 7 vs 3 (FAIL)
- 04 lip flash strong: 17 vs 14 (FAIL)
