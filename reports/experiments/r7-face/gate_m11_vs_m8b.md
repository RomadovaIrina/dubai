# Regression gate: NEW `m11` vs PREV `m8b`

commit 5c9841b

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 05 new / prev | 04 new / prev | 03 new / prev | 01 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 3.576 / 3.911 **FAIL** | 6.421 / 6.183 = | 7.861 / 7.861 = | 6.654 / 6.654 = | None / None n/a |
| SyncNet dub-only (whole video, info) | 2.260 / 2.610 INFO | 4.360 / 4.280 INFO | 6.620 / 6.620 INFO | 6.820 / 6.820 INFO | None / None n/a |
| SyncNet mix (whole video, info) | 1.630 / 2.160 INFO | 3.930 / 3.700 INFO | 6.110 / 6.110 INFO | 6.530 / 6.530 INFO | None / None n/a |
| dub silent while face tracked, s (audio info) | 9 / 10 INFO | 17 / 20 INFO | 19 / 19 INFO | 0 / 0 INFO | None / None n/a |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.840 / 2.170 = | 0.730 / 0.730 = | 0.770 / 0.770 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| v4 resid mouth | 0.989 / 0.998 = | 1.014 / 1.004 = | 1.165 / 1.165 = | 1.137 / 1.137 = | None / None n/a |
| v4 twitch mouth | 0.960 / 0.970 = | 1.018 / 1.009 = | 0.986 / 0.986 = | 1.079 / 1.079 = | None / None n/a |
| v4 shear95 lower face | 1.001 / 1.002 = | 1.006 / 1.000 = | 1.006 / 1.006 = | 0.939 / 0.939 = | None / None n/a |
| v4 resid mouth, own frames (info) | 0.987 / 0.995 INFO | 1.014 / 1.009 INFO | 1.166 / 1.166 INFO | 1.141 / 1.141 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 42.1 / 42.1 INFO | 43.3 / 43.3 INFO | 82.6 / 82.6 INFO | 10.4 / 10.4 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.964 / 0.967 = | 0.929 / 0.936 = | 0.834 / 0.834 = | 0.946 / 0.946 = | None / None n/a |
| slide nose-base px | 0.035 / 0.036 = | 0.113 / 0.111 = | 0.062 / 0.062 = | 0.082 / 0.082 = | None / None n/a |
| window joints ratio | 0.953 / 0.892 **FAIL** | 0.926 / 0.930 = | 0.909 / 0.909 = | 0.926 / 0.926 = | None / None n/a |
| window joints LF ratio | 0.895 / 0.842 **FAIL** | 0.901 / 0.898 = | 0.837 / 0.837 = | 0.923 / 0.923 = | None / None n/a |
| bg ring lower-upper | -0.137 / -0.137 = | -0.094 / -0.094 = | 0.055 / 0.055 = | -0.359 / -0.359 = | None / None n/a |
| lip flash strong | 6 / 5 = | 16 / 9 **FAIL** | 44 / 44 = | 4 / 4 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.660 / 1.660 = | 1.410 / 1.400 = | 1.330 / 1.330 = | 1.110 / 1.110 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.994 / 1.009 = | 0.976 / 1.011 = | 0.958 / 0.958 = | 1.006 / 1.006 = | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 1.007 / 0.964 = | 2.128 / 2.085 = | 1.110 / 1.110 = | 1.291 / 1.291 = | None / None n/a |
| chin wave worst 2 s | 1.892 / 1.886 = | 4.054 / 4.047 = | 1.845 / 1.845 = | 1.686 / 1.686 = | None / None n/a |
| below-jaw band |out-src| | 2.578 / 2.570 = | 2.482 / 2.485 = | 2.501 / 2.501 = | 2.337 / 2.337 = | nan / nan = |
| WER final mix (audio, info) | 0.057 / 0.024 INFO | 0.032 / 0.034 INFO | 0.020 / 0.020 INFO | 0.103 / 0.103 INFO | 0.043 / 0.043 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.000 INFO | 0.000 / 0.000 INFO | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO |
| true peak dBTP (audio, info) | -1.500 / -1.300 INFO | -3.100 / -3.400 INFO | -1.400 / -1.400 INFO | -3.000 / -3.000 INFO | -1.300 / -1.300 INFO |
| video s / video-min | 500.2 / 420.6 **FAIL** | 482.2 / 438.9 = (over limit) | 480.0 / 480.0 = (over limit) | 94.5 / 94.5 = | 34.6 / 34.6 = |
| peak VRAM MiB | 14514 / 14514 INFO | 18729 / 14370 INFO | 15197 / 15197 INFO | 16323 / 16323 INFO | 4477 / 4477 INFO |
| audio stream md5 | 16482c97 / f5dd3fc6 INFO | 07e82b1c / f3174f6b INFO | edf612c0 / edf612c0 INFO | aa763a91 / aa763a91 INFO | 50418441 / 50418441 INFO |

## Verdict: **FAIL — no publish**
- 05 SyncNet on dub-speech seconds (conf at 0): 3.576 vs 3.911 (FAIL)
- 05 window joints ratio: 0.953 vs 0.892 (FAIL)
- 05 window joints LF ratio: 0.895 vs 0.842 (FAIL)
- 04 lip flash strong: 16 vs 9 (FAIL)
- 05 video s / video-min: 500.2 vs 420.6 (FAIL)
