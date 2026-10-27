# Regression gate: NEW `m5` vs PREV `m4`

commit c858a06

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 05 new / prev | 04 new / prev | 01 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.797 / 7.973 = | 4.057 / 5.079 **FAIL** | 6.215 / 5.819 better | 6.443 / 7.783 **FAIL** | None / None n/a |
| SyncNet dub-only (whole video, info) | None / 5.880 MISSING | None / 3.140 MISSING | None / 3.510 MISSING | None / 7.960 MISSING | None / None n/a |
| SyncNet mix (whole video, info) | None / 5.340 MISSING | None / 2.410 MISSING | None / 3.300 MISSING | None / 7.660 MISSING | None / None n/a |
| dub silent while face tracked, s (audio info) | 17 / 28 INFO | 10 / 8 INFO | 20 / 23 INFO | 0 / 0 INFO | None / None n/a |
| |AV offset| dub | None / 0 MISSING | None / 0 MISSING | None / 0 MISSING | None / 0 MISSING | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.800 / 1.700 = | 2.640 / 1.740 **FAIL** | 2.600 / 0.770 **FAIL** | 0.000 / 0.000 = | 0.000 / 0.000 = |
| v4 resid mouth | 1.191 / 1.113 **FAIL** | 1.015 / 1.007 = | 1.006 / 0.970 = | 1.157 / 1.108 = | None / None n/a |
| v4 twitch mouth | 0.966 / 0.965 = | 0.966 / 0.978 = | 1.029 / 1.018 = | 1.082 / 1.038 = | None / None n/a |
| v4 shear95 lower face | 1.008 / 0.996 = | 0.995 / 0.995 = | 1.001 / 1.003 = | 0.934 / 0.930 = | None / None n/a |
| v4 resid mouth, own frames (info) | 1.199 / 1.118 INFO | 1.023 / 0.994 INFO | 1.011 / 0.979 INFO | 1.163 / 1.112 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 79.8 / 79.8 INFO | 43.4 / 43.4 INFO | 43.8 / 43.8 INFO | 10.4 / 10.4 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.796 / 0.799 = | 0.920 / 0.919 = | 0.892 / 0.887 = | 0.943 / 0.945 = | None / None n/a |
| slide nose-base px | 0.061 / 0.060 = | 0.035 / 0.039 = | 0.132 / 0.154 better | 0.089 / 0.088 = | None / None n/a |
| window joints ratio | 0.887 / 0.944 better | 0.962 / 0.923 = | 0.909 / 0.967 better | 0.929 / 0.890 = | None / None n/a |
| window joints LF ratio | 0.794 / 0.862 better | 0.942 / 0.888 **FAIL** | 0.865 / 0.963 better | 0.960 / 0.845 **FAIL** | None / None n/a |
| bg ring lower-upper | 0.057 / 0.057 = | -0.135 / -0.139 = | -0.095 / -0.095 = | -0.359 / -0.358 = | None / None n/a |
| lip flash strong | 60 / 55 = | 26 / 20 **FAIL** | 14 / 10 **FAIL** | 0 / 1 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.260 / 1.290 = | 1.460 / 1.490 = | 1.430 / 1.350 = | 1.050 / 1.050 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.938 / 0.923 = | 1.015 / 1.014 = | 0.991 / 1.003 = | 1.042 / 0.935 **FAIL** | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 1.136 / 1.077 = | 0.983 / 1.111 better | 2.177 / 2.095 = | 1.399 / 1.321 = | None / None n/a |
| chin wave worst 2 s | 1.678 / 1.599 = | 2.258 / 2.284 = | 4.195 / 3.629 **FAIL** | 1.746 / 1.837 = | None / None n/a |
| below-jaw band |out-src| | 2.501 / 2.502 = | 2.573 / 2.574 = | 2.484 / 2.485 = | 2.336 / 2.332 = | nan / nan = |
| WER final mix (audio, info) | 0.025 / 0.023 INFO | 0.037 / 0.108 INFO | 0.045 / 0.024 INFO | 0.088 / 0.078 INFO | 0.027 / 0.037 INFO |
| audio hard cuts (audio, info) | 1 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.100 / 0.000 INFO | 0.000 / 0.000 INFO | 0.200 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.200 / -1.300 INFO | -1.400 / -1.500 INFO | -3.300 / -3.000 INFO | -2.900 / -2.800 INFO | -1.300 / -1.400 INFO |
| video s / video-min | 748.0 / 564.5 **FAIL** | 1048.1 / 520.9 **FAIL** | 745.3 / 620.7 **FAIL** | 190.9 / 140.9 **FAIL** | 90.1 / 58.9 **FAIL** |
| peak VRAM MiB | 14584 / 14370 INFO | 16852 / 14330 INFO | 16219 / 14450 INFO | 14388 / 14388 INFO | 3332 / 3332 INFO |
| audio stream md5 | 73f2cb7c / 68ad3dd7 INFO | 83b307b8 / 73a124e1 INFO | ccb6e611 / 3d411803 INFO | a63681f1 / 61f79dd8 INFO | dd923747 / bfb2cc60 INFO |

## Verdict: **FAIL — no publish**
- 05 SyncNet on dub-speech seconds (conf at 0): 4.057 vs 5.079 (FAIL)
- 01 SyncNet on dub-speech seconds (conf at 0): 6.443 vs 7.783 (FAIL)
- 03 |AV offset| dub: None vs 0 (MISSING)
- 05 |AV offset| dub: None vs 0 (MISSING)
- 04 |AV offset| dub: None vs 0 (MISSING)
- 01 |AV offset| dub: None vs 0 (MISSING)
- 05 mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks): 2.640 vs 1.740 (FAIL)
- 04 mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks): 2.600 vs 0.770 (FAIL)
- 03 v4 resid mouth: 1.191 vs 1.113 (FAIL)
- 05 window joints LF ratio: 0.942 vs 0.888 (FAIL)
- 01 window joints LF ratio: 0.960 vs 0.845 (FAIL)
- 05 lip flash strong: 26 vs 20 (FAIL)
- 04 lip flash strong: 14 vs 10 (FAIL)
- 01 jaw jitter out/src: 1.042 vs 0.935 (FAIL)
- 04 chin wave worst 2 s: 4.195 vs 3.629 (FAIL)
- 03 video s / video-min: 748.0 vs 564.5 (FAIL)
- 05 video s / video-min: 1048.1 vs 520.9 (FAIL)
- 04 video s / video-min: 745.3 vs 620.7 (FAIL)
- 01 video s / video-min: 190.9 vs 140.9 (FAIL)
- 02 video s / video-min: 90.1 vs 58.9 (FAIL)
