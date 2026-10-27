# Regression gate: NEW `m5` vs PREV `m4`

commit be9b9fa

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 05 new / prev | 04 new / prev |
|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.797 / 7.973 = | 4.057 / 5.079 **FAIL** | 6.215 / 5.819 better |
| SyncNet dub-only (whole video, info) | None / 5.880 MISSING | None / 3.140 MISSING | None / 3.510 MISSING |
| SyncNet mix (whole video, info) | None / 5.340 MISSING | None / 2.410 MISSING | None / 3.300 MISSING |
| dub silent while face tracked, s (audio info) | 17 / 28 INFO | 10 / 8 INFO | 20 / 23 INFO |
| |AV offset| dub | None / 0 MISSING | None / 0 MISSING | None / 0 MISSING |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.800 / 1.700 = | 2.640 / 1.740 **FAIL** | 2.600 / 0.770 **FAIL** |
| v4 resid mouth | 1.191 / 1.113 **FAIL** | 1.015 / 1.007 = | 1.006 / 0.970 = |
| v4 twitch mouth | 0.966 / 0.965 = | 0.966 / 0.978 = | 1.029 / 1.018 = |
| v4 shear95 lower face | 1.008 / 0.996 = | 0.995 / 0.995 = | 1.001 / 1.003 = |
| v4 resid mouth, own frames (info) | 1.199 / 1.118 INFO | 1.023 / 0.994 INFO | 1.011 / 0.979 INFO |
| v4 common lip-synced seconds (info) | 79.8 / 79.8 INFO | 43.4 / 43.4 INFO | 43.8 / 43.8 INFO |
| v4 sharpness mouth (src=1) | 0.796 / 0.799 = | 0.920 / 0.919 = | 0.892 / 0.887 = |
| slide nose-base px | 0.061 / 0.060 = | 0.035 / 0.039 = | 0.132 / 0.154 better |
| window joints ratio | 0.887 / 0.944 better | 0.962 / 0.923 = | 0.909 / 0.967 better |
| window joints LF ratio | 0.794 / 0.862 better | 0.942 / 0.888 **FAIL** | 0.865 / 0.963 better |
| bg ring lower-upper | 0.057 / 0.057 = | -0.135 / -0.139 = | -0.095 / -0.095 = |
| lip flash strong | 60 / 55 = | 26 / 20 **FAIL** | 14 / 10 **FAIL** |
| colour worst 1 s |dL| (half face) | 1.260 / 1.290 = | 1.460 / 1.490 = | 1.430 / 1.350 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.938 / 0.923 = | 1.015 / 1.014 = | 0.991 / 1.003 = |
| chin wave (lower-chin residual std, 2 s) | 1.136 / 1.077 = | 0.983 / 1.111 better | 2.177 / 2.095 = |
| chin wave worst 2 s | 1.678 / 1.599 = | 2.258 / 2.284 = | 4.195 / 3.629 **FAIL** |
| below-jaw band |out-src| | 2.501 / 2.502 = | 2.573 / 2.574 = | 2.484 / 2.485 = |
| WER final mix (audio, info) | 0.025 / 0.023 INFO | 0.037 / 0.108 INFO | 0.045 / 0.024 INFO |
| audio hard cuts (audio, info) | 1 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.100 / 0.000 INFO |
| true peak dBTP (audio, info) | -1.200 / -1.300 INFO | -1.400 / -1.500 INFO | -3.300 / -3.000 INFO |
| video s / video-min | 748.0 / 564.5 **FAIL** | 1048.1 / 520.9 **FAIL** | 745.3 / 620.7 **FAIL** |
| peak VRAM MiB | 14584 / 14370 INFO | 16852 / 14330 INFO | 16219 / 14450 INFO |
| audio stream md5 | 73f2cb7c / 68ad3dd7 INFO | 83b307b8 / 73a124e1 INFO | ccb6e611 / 3d411803 INFO |

## Verdict: **FAIL — no publish**
- 05 SyncNet on dub-speech seconds (conf at 0): 4.057 vs 5.079 (FAIL)
- 03 |AV offset| dub: None vs 0 (MISSING)
- 05 |AV offset| dub: None vs 0 (MISSING)
- 04 |AV offset| dub: None vs 0 (MISSING)
- 05 mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks): 2.640 vs 1.740 (FAIL)
- 04 mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks): 2.600 vs 0.770 (FAIL)
- 03 v4 resid mouth: 1.191 vs 1.113 (FAIL)
- 05 window joints LF ratio: 0.942 vs 0.888 (FAIL)
- 05 lip flash strong: 26 vs 20 (FAIL)
- 04 lip flash strong: 14 vs 10 (FAIL)
- 04 chin wave worst 2 s: 4.195 vs 3.629 (FAIL)
- 03 video s / video-min: 748.0 vs 564.5 (FAIL)
- 05 video s / video-min: 1048.1 vs 520.9 (FAIL)
- 04 video s / video-min: 745.3 vs 620.7 (FAIL)
