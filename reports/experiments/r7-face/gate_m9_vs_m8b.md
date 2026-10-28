# Regression gate: NEW `m9` vs PREV `m8b`

commit f5dfc93

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 05 new / prev | 04 new / prev | 03 new / prev | 01 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 3.510 / 3.911 **FAIL** | 5.320 / 6.183 **FAIL** | 6.471 / 7.861 **FAIL** | 6.105 / 6.654 **FAIL** | None / None n/a |
| SyncNet dub-only (whole video, info) | 2.350 / 2.610 INFO | 3.810 / 4.280 INFO | 5.480 / 6.620 INFO | 6.270 / 6.820 INFO | None / None n/a |
| SyncNet mix (whole video, info) | 1.870 / 2.160 INFO | 3.310 / 3.700 INFO | 5.040 / 6.110 INFO | 6.160 / 6.530 INFO | None / None n/a |
| dub silent while face tracked, s (audio info) | 10 / 10 INFO | 20 / 20 INFO | 19 / 19 INFO | 0 / 0 INFO | None / None n/a |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.630 / 2.170 better | 0.400 / 0.730 = | 0.770 / 0.770 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| v4 resid mouth | 0.989 / 0.995 = | 0.945 / 1.010 better | 1.043 / 1.165 better | 1.040 / 1.137 better | None / None n/a |
| v4 twitch mouth | 0.984 / 0.975 = | 1.024 / 1.009 = | 1.000 / 0.986 = | 1.075 / 1.079 = | None / None n/a |
| v4 shear95 lower face | 0.991 / 0.998 = | 1.004 / 1.000 = | 1.010 / 1.006 = | 0.951 / 0.939 = | None / None n/a |
| v4 resid mouth, own frames (info) | 0.988 / 0.995 INFO | 0.943 / 1.009 INFO | 1.043 / 1.166 INFO | 1.043 / 1.141 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 45.6 / 45.6 INFO | 45.1 / 45.1 INFO | 82.6 / 82.6 INFO | 10.4 / 10.4 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.967 / 0.966 = | 0.937 / 0.935 = | 0.838 / 0.834 = | 0.946 / 0.946 = | None / None n/a |
| slide nose-base px | 0.035 / 0.036 = | 0.112 / 0.111 = | 0.060 / 0.062 = | 0.084 / 0.082 = | None / None n/a |
| window joints ratio | 0.898 / 0.892 = | 0.927 / 0.930 = | 0.940 / 0.909 = | 0.893 / 0.926 = | None / None n/a |
| window joints LF ratio | 0.849 / 0.842 = | 0.878 / 0.898 = | 0.906 / 0.837 **FAIL** | 0.853 / 0.923 better | None / None n/a |
| bg ring lower-upper | -0.137 / -0.137 = | -0.095 / -0.094 = | 0.055 / 0.055 = | -0.356 / -0.359 = | None / None n/a |
| lip flash strong | 7 / 5 = | 11 / 9 = | 32 / 44 better | 3 / 4 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.700 / 1.660 = | 1.410 / 1.400 = | 1.410 / 1.330 = | 1.230 / 1.110 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 1.000 / 1.009 = | 0.992 / 1.011 = | 0.952 / 0.958 = | 0.969 / 1.006 = | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 0.965 / 0.964 = | 2.058 / 2.085 = | 1.069 / 1.110 = | 1.297 / 1.291 = | None / None n/a |
| chin wave worst 2 s | 1.890 / 1.886 = | 3.924 / 4.047 = | 1.752 / 1.845 = | 1.704 / 1.686 = | None / None n/a |
| below-jaw band |out-src| | 2.575 / 2.570 = | 2.482 / 2.485 = | 2.499 / 2.501 = | 2.348 / 2.337 = | nan / nan = |
| WER final mix (audio, info) | 0.024 / 0.024 INFO | 0.034 / 0.034 INFO | 0.020 / 0.020 INFO | 0.103 / 0.103 INFO | 0.043 / 0.043 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO |
| true peak dBTP (audio, info) | -1.300 / -1.300 INFO | -3.400 / -3.400 INFO | -1.400 / -1.400 INFO | -3.000 / -3.000 INFO | -1.300 / -1.300 INFO |
| video s / video-min | 420.6 / 420.6 = (over limit) | 438.9 / 438.9 = (over limit) | 480.0 / 480.0 = (over limit) | 94.5 / 94.5 = | 34.6 / 34.6 = |
| peak VRAM MiB | 14514 / 14514 INFO | 14370 / 14370 INFO | 15197 / 15197 INFO | 16323 / 16323 INFO | 4477 / 4477 INFO |
| audio stream md5 | f5dd3fc6 / f5dd3fc6 INFO | f3174f6b / f3174f6b INFO | edf612c0 / edf612c0 INFO | aa763a91 / aa763a91 INFO | 50418441 / 50418441 INFO |

## Verdict: **FAIL — no publish**
- 05 SyncNet on dub-speech seconds (conf at 0): 3.510 vs 3.911 (FAIL)
- 04 SyncNet on dub-speech seconds (conf at 0): 5.320 vs 6.183 (FAIL)
- 03 SyncNet on dub-speech seconds (conf at 0): 6.471 vs 7.861 (FAIL)
- 01 SyncNet on dub-speech seconds (conf at 0): 6.105 vs 6.654 (FAIL)
- 03 window joints LF ratio: 0.906 vs 0.837 (FAIL)
