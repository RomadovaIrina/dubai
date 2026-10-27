# Regression gate: NEW `m4` vs PREV `m3`

commit 11167c5

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 01 new / prev | 05 new / prev | 04 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.973 / 8.154 = | 7.783 / 7.764 = | 5.079 / 5.061 = | 5.819 / 5.924 = | None / None n/a |
| SyncNet dub-only (whole video, info) | 5.880 / 6.000 INFO | 7.960 / 7.990 INFO | 3.140 / 3.160 INFO | 3.510 / 3.520 INFO | None / None n/a |
| SyncNet mix (whole video, info) | 5.340 / 5.410 INFO | 7.660 / 7.710 INFO | 2.410 / 2.380 INFO | 3.300 / 3.320 INFO | None / None n/a |
| dub silent while face tracked, s (audio info) | 28 / 28 INFO | 0 / 0 INFO | 8 / 8 INFO | 23 / 23 INFO | None / None n/a |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.700 / 1.670 = | 0.000 / 0.000 = | 1.740 / 2.570 better | 0.770 / 0.770 = | 0.000 / 0.000 = |
| v4 resid mouth | 1.118 / 1.128 = | 1.108 / 1.104 = | 0.996 / 1.028 = | 0.978 / 0.987 = | None / None n/a |
| v4 twitch mouth | 0.965 / 0.970 = | 1.038 / 1.046 = | 0.976 / 0.991 = | 1.019 / 1.013 = | None / None n/a |
| v4 shear95 lower face | 0.997 / 0.962 = | 0.930 / 0.930 = | 0.996 / 0.997 = | 1.005 / 1.003 = | None / None n/a |
| v4 resid mouth, own frames (info) | 1.118 / 1.129 INFO | 1.112 / 1.108 INFO | 0.994 / 1.026 INFO | 0.979 / 0.987 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 83.1 / 83.1 INFO | 10.4 / 10.4 INFO | 44.1 / 44.1 INFO | 45.1 / 45.1 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.800 / 0.802 = | 0.945 / 0.946 = | 0.920 / 0.915 = | 0.888 / 0.888 = | None / None n/a |
| slide nose-base px | 0.060 / 0.061 = | 0.088 / None INFO | 0.039 / 0.048 = | 0.154 / 0.162 = | None / None n/a |
| window joints ratio | 0.944 / 0.935 = | 0.890 / 0.895 = | 0.923 / 0.954 = | 0.967 / 0.963 = | None / None n/a |
| window joints LF ratio | 0.862 / 0.857 = | 0.845 / 0.861 = | 0.888 / 0.958 better | 0.963 / 0.953 = | None / None n/a |
| bg ring lower-upper | 0.057 / 0.058 = | -0.358 / -0.358 = | -0.139 / -0.135 = | -0.095 / -0.096 = | None / None n/a |
| lip flash strong | 55 / 59 = | 1 / 1 = | 20 / 30 better | 10 / 9 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.290 / 1.250 = | 1.050 / 1.060 = | 1.490 / 1.560 = | 1.350 / 1.330 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.923 / 0.907 = | 0.935 / 0.963 = | 1.014 / 1.026 = | 1.003 / 1.000 = | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 1.077 / 1.533 better | 1.321 / 1.432 = | 1.111 / 1.330 better | 2.095 / 2.381 better | None / None n/a |
| chin wave worst 2 s | 1.599 / 2.219 better | 1.837 / 1.828 = | 2.284 / 2.998 better | 3.629 / 4.152 better | None / None n/a |
| below-jaw band |out-src| | 2.502 / 2.503 = | 2.332 / 2.340 = | 2.574 / 2.578 = | 2.485 / 2.485 = | nan / nan = |
| WER final mix (audio, info) | 0.023 / 0.023 INFO | 0.078 / 0.078 INFO | 0.108 / 0.108 INFO | 0.024 / 0.024 INFO | 0.037 / 0.037 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.100 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.300 / -1.300 INFO | -2.800 / -2.800 INFO | -1.500 / -1.500 INFO | -3.000 / -3.000 INFO | -1.400 / -1.400 INFO |
| video s / video-min | 564.5 / 699.6 better (over limit) | 140.9 / 188.0 better | 520.9 / 630.1 better (over limit) | 620.7 / 730.7 better (over limit) | 58.9 / 96.1 better |
| peak VRAM MiB | 14370 / 14370 INFO | 14388 / 14388 INFO | 14330 / 14330 INFO | 14450 / 14450 INFO | 3332 / 3332 INFO |
| audio stream md5 | 68ad3dd7 / 68ad3dd7 INFO | 61f79dd8 / 61f79dd8 INFO | 73a124e1 / 73a124e1 INFO | 3d411803 / 3d411803 INFO | bfb2cc60 / bfb2cc60 INFO |

## Verdict: PASS
