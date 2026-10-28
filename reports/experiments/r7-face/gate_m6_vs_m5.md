# Regression gate: NEW `m6` vs PREV `m5`

commit a5deca3

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 01 new / prev | 05 new / prev | 02 new / prev | 04 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.785 / 7.797 = | 6.443 / 6.443 = | 4.057 / 4.057 = | None / None n/a | 6.185 / 6.215 = |
| SyncNet dub-only (whole video, info) | 6.600 / 6.610 INFO | 6.560 / 6.560 INFO | 2.530 / 2.530 INFO | None / None n/a | 4.240 / 4.240 INFO |
| SyncNet mix (whole video, info) | 6.220 / 6.220 INFO | 6.120 / 6.120 INFO | 2.030 / 2.030 INFO | None / None n/a | 3.740 / 3.750 INFO |
| dub silent while face tracked, s (audio info) | 17 / 17 INFO | 0 / 0 INFO | 10 / 10 INFO | None / None n/a | 20 / 20 INFO |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a | 0 / 0 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.800 / 1.800 = | 0.000 / 0.000 = | 2.100 / 2.640 better | 0.000 / 0.000 = | 2.860 / 2.600 = |
| v4 resid mouth | 1.198 / 1.197 = | 1.157 / 1.157 = | 1.026 / 1.024 = | None / None n/a | 1.011 / 1.011 = |
| v4 twitch mouth | 0.970 / 0.970 = | 1.082 / 1.082 = | 0.969 / 0.969 = | None / None n/a | 1.028 / 1.028 = |
| v4 shear95 lower face | 1.009 / 1.008 = | 0.934 / 0.934 = | 0.997 / 0.997 = | None / None n/a | 1.000 / 1.001 = |
| v4 resid mouth, own frames (info) | 1.199 / 1.199 INFO | 1.163 / 1.163 INFO | 1.025 / 1.023 INFO | None / None n/a | 1.011 / 1.011 INFO |
| v4 common lip-synced seconds (info) | 83.8 / 83.8 INFO | 10.4 / 10.4 INFO | 45.6 / 45.6 INFO | None / None n/a | 47.0 / 47.0 INFO |
| v4 sharpness mouth (src=1) | 0.798 / 0.798 = | 0.943 / 0.943 = | 0.922 / 0.922 = | None / None n/a | 0.891 / 0.891 = |
| slide nose-base px | 0.061 / 0.061 = | 0.089 / 0.089 = | 0.035 / 0.035 = | None / None n/a | 0.132 / 0.132 = |
| window joints ratio | 0.887 / 0.887 = | 0.929 / 0.929 = | 0.961 / 0.962 = | None / None n/a | 0.909 / 0.909 = |
| window joints LF ratio | 0.794 / 0.794 = | 0.960 / 0.960 = | 0.942 / 0.942 = | None / None n/a | 0.864 / 0.865 = |
| bg ring lower-upper | 0.057 / 0.057 = | -0.359 / -0.359 = | -0.136 / -0.135 = | None / None n/a | -0.095 / -0.095 = |
| lip flash strong | 59 / 60 = | 0 / 0 = | 24 / 26 = | 0 / 0 = | 14 / 14 = |
| colour worst 1 s |dL| (half face) | 1.260 / 1.260 = | 1.050 / 1.050 = | 1.460 / 1.460 = | 0 / 0 = | 1.430 / 1.430 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.936 / 0.938 = | 1.042 / 1.042 = | 1.019 / 1.015 = | None / None n/a | 0.991 / 0.991 = |
| chin wave (lower-chin residual std, 2 s) | 1.136 / 1.136 = | 1.399 / 1.399 = | 0.983 / 0.983 = | None / None n/a | 2.177 / 2.177 = |
| chin wave worst 2 s | 1.678 / 1.678 = | 1.746 / 1.746 = | 2.252 / 2.258 = | None / None n/a | 4.190 / 4.195 = |
| below-jaw band |out-src| | 2.501 / 2.501 = | 2.336 / 2.336 = | 2.575 / 2.573 = | nan / nan = | 2.484 / 2.484 = |
| WER final mix (audio, info) | 0.025 / 0.025 INFO | 0.088 / 0.088 INFO | 0.037 / 0.037 INFO | 0.027 / 0.027 INFO | 0.045 / 0.045 INFO |
| audio hard cuts (audio, info) | 1 / 1 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO | 0.100 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.200 / -1.200 INFO | -2.900 / -2.900 INFO | -1.400 / -1.400 INFO | -1.300 / -1.300 INFO | -3.300 / -3.300 INFO |
| video s / video-min | 466.1 / 466.1 = (over limit) | 94.5 / 94.5 = | 460.0 / 460.0 = (over limit) | 54.0 / 54.0 = | 489.5 / 489.5 = (over limit) |
| peak VRAM MiB | 14584 / 14584 INFO | 14388 / 14388 INFO | 14514 / 14514 INFO | 3332 / 3332 INFO | 15283 / 15283 INFO |
| audio stream md5 | 73f2cb7c / 73f2cb7c INFO | a63681f1 / a63681f1 INFO | 83b307b8 / 83b307b8 INFO | dd923747 / dd923747 INFO | ccb6e611 / ccb6e611 INFO |

## Verdict: PASS
