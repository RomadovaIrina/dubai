# Regression gate: NEW `m7` vs PREV `m6`

commit 8b1bd0a

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 03 new / prev | 01 new / prev | 05 new / prev | 02 new / prev | 04 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 7.791 / 7.785 = | 6.411 / 6.443 = | 4.157 / 4.057 = | None / None n/a | 6.212 / 6.185 = |
| SyncNet dub-only (whole video, info) | 6.630 / 6.600 INFO | 6.520 / 6.560 INFO | 2.640 / 2.530 INFO | None / None n/a | 4.250 / 4.240 INFO |
| SyncNet mix (whole video, info) | 6.240 / 6.220 INFO | 6.190 / 6.120 INFO | 2.210 / 2.030 INFO | None / None n/a | 3.750 / 3.740 INFO |
| dub silent while face tracked, s (audio info) | 17 / 17 INFO | 0 / 0 INFO | 10 / 10 INFO | None / None n/a | 20 / 20 INFO |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a | 0 / 0 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.770 / 1.800 = | 0.280 / 0.000 = | 2.400 / 2.100 = | 0.000 / 0.000 = | 2.860 / 2.860 = |
| v4 resid mouth | 1.187 / 1.198 = | 1.127 / 1.157 = | 0.998 / 1.026 = | None / None n/a | 1.012 / 1.011 = |
| v4 twitch mouth | 0.970 / 0.970 = | 1.079 / 1.082 = | 0.968 / 0.969 = | None / None n/a | 1.034 / 1.028 = |
| v4 shear95 lower face | 0.994 / 1.009 = | 0.960 / 0.934 = | 0.998 / 0.997 = | None / None n/a | 1.004 / 1.000 = |
| v4 resid mouth, own frames (info) | 1.189 / 1.199 INFO | 1.133 / 1.163 INFO | 0.997 / 1.025 INFO | None / None n/a | 1.011 / 1.011 INFO |
| v4 common lip-synced seconds (info) | 83.8 / 83.8 INFO | 10.4 / 10.4 INFO | 45.6 / 45.6 INFO | None / None n/a | 47.0 / 47.0 INFO |
| v4 sharpness mouth (src=1) | 0.852 / 0.798 better | 0.942 / 0.943 = | 0.962 / 0.922 better | None / None n/a | 0.927 / 0.891 better |
| slide nose-base px | 0.062 / 0.061 = | 0.083 / 0.089 = | 0.033 / 0.035 = | None / None n/a | 0.141 / 0.132 = |
| window joints ratio | 0.886 / 0.887 = | 0.916 / 0.929 = | 0.966 / 0.961 = | None / None n/a | 0.909 / 0.909 = |
| window joints LF ratio | 0.802 / 0.794 = | 0.940 / 0.960 = | 0.947 / 0.942 = | None / None n/a | 0.862 / 0.864 = |
| bg ring lower-upper | 0.055 / 0.057 = | -0.353 / -0.359 = | -0.136 / -0.136 = | None / None n/a | -0.094 / -0.095 = |
| lip flash strong | 54 / 59 = | 1 / 0 = | 3 / 24 better | 0 / 0 = | 14 / 14 = |
| colour worst 1 s |dL| (half face) | 1.220 / 1.260 = | 1.140 / 1.050 = | 1.690 / 1.460 = | 0 / 0 = | 1.400 / 1.430 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 0.941 / 0.936 = | 1.016 / 1.042 = | 1.010 / 1.019 = | None / None n/a | 0.994 / 0.991 = |
| chin wave (lower-chin residual std, 2 s) | 1.121 / 1.136 = | 1.299 / 1.399 = | 0.955 / 0.983 = | None / None n/a | 2.171 / 2.177 = |
| chin wave worst 2 s | 1.688 / 1.678 = | 1.632 / 1.746 = | 2.068 / 2.252 = | None / None n/a | 4.067 / 4.190 = |
| below-jaw band |out-src| | 2.502 / 2.501 = | 2.334 / 2.336 = | 2.572 / 2.575 = | nan / nan = | 2.483 / 2.484 = |
| WER final mix (audio, info) | 0.025 / 0.025 INFO | 0.088 / 0.088 INFO | 0.037 / 0.037 INFO | 0.027 / 0.027 INFO | 0.045 / 0.045 INFO |
| audio hard cuts (audio, info) | 1 / 1 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO | 0.100 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.200 / -1.200 INFO | -2.900 / -2.900 INFO | -1.400 / -1.400 INFO | -1.300 / -1.300 INFO | -3.300 / -3.300 INFO |
| video s / video-min | 466.1 / 466.1 = (over limit) | 94.5 / 94.5 = | 460.0 / 460.0 = (over limit) | 54.0 / 54.0 = | 489.5 / 489.5 = (over limit) |
| peak VRAM MiB | 14584 / 14584 INFO | 14388 / 14388 INFO | 14514 / 14514 INFO | 3332 / 3332 INFO | 15283 / 15283 INFO |
| audio stream md5 | 73f2cb7c / 73f2cb7c INFO | a63681f1 / a63681f1 INFO | 83b307b8 / 83b307b8 INFO | dd923747 / dd923747 INFO | ccb6e611 / ccb6e611 INFO |

## Verdict: PASS
