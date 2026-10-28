# Regression gate: NEW `m10` vs PREV `m8b`

commit 8b99c78

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 05 new / prev | 04 new / prev | 03 new / prev | 01 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 3.912 / 3.911 = | 5.983 / 6.183 = | 7.304 / 7.861 **FAIL** | 6.654 / 6.654 = | None / None n/a |
| SyncNet dub-only (whole video, info) | 2.600 / 2.610 INFO | 4.210 / 4.280 INFO | 6.200 / 6.620 INFO | 6.820 / 6.820 INFO | None / None n/a |
| SyncNet mix (whole video, info) | 2.110 / 2.160 INFO | 3.680 / 3.700 INFO | 5.680 / 6.110 INFO | 6.530 / 6.530 INFO | None / None n/a |
| dub silent while face tracked, s (audio info) | 10 / 10 INFO | 20 / 20 INFO | 19 / 19 INFO | 0 / 0 INFO | None / None n/a |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.340 / 2.170 better | 0.770 / 0.730 = | 0.730 / 0.770 = | 0.000 / 0.000 = | 0.000 / 0.000 = |
| v4 resid mouth | 0.980 / 0.995 = | 1.005 / 1.010 = | 1.147 / 1.165 = | 1.137 / 1.137 = | None / None n/a |
| v4 twitch mouth | 0.961 / 0.975 = | 1.012 / 1.009 = | 0.997 / 0.986 = | 1.079 / 1.079 = | None / None n/a |
| v4 shear95 lower face | 0.996 / 0.998 = | 1.001 / 1.000 = | 0.998 / 1.006 = | 0.939 / 0.939 = | None / None n/a |
| v4 resid mouth, own frames (info) | 0.979 / 0.995 INFO | 1.004 / 1.009 INFO | 1.148 / 1.166 INFO | 1.141 / 1.141 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 45.6 / 45.6 INFO | 45.1 / 45.1 INFO | 82.6 / 82.6 INFO | 10.4 / 10.4 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.966 / 0.966 = | 0.935 / 0.935 = | 0.835 / 0.834 = | 0.946 / 0.946 = | None / None n/a |
| slide nose-base px | 0.034 / 0.036 = | 0.111 / 0.111 = | 0.062 / 0.062 = | 0.082 / 0.082 = | None / None n/a |
| window joints ratio | 0.889 / 0.892 = | 0.936 / 0.930 = | 0.920 / 0.909 = | 0.926 / 0.926 = | None / None n/a |
| window joints LF ratio | 0.839 / 0.842 = | 0.913 / 0.898 = | 0.852 / 0.837 = | 0.923 / 0.923 = | None / None n/a |
| bg ring lower-upper | -0.136 / -0.137 = | -0.094 / -0.094 = | 0.055 / 0.055 = | -0.359 / -0.359 = | None / None n/a |
| lip flash strong | 4 / 5 = | 15 / 9 **FAIL** | 47 / 44 = | 4 / 4 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.660 / 1.660 = | 1.400 / 1.400 = | 1.350 / 1.330 = | 1.110 / 1.110 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 1.002 / 1.009 = | 1.001 / 1.011 = | 0.957 / 0.958 = | 1.006 / 1.006 = | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 0.964 / 0.964 = | 2.084 / 2.085 = | 1.085 / 1.110 = | 1.291 / 1.291 = | None / None n/a |
| chin wave worst 2 s | 1.887 / 1.886 = | 4.047 / 4.047 = | 1.714 / 1.845 = | 1.686 / 1.686 = | None / None n/a |
| below-jaw band |out-src| | 2.570 / 2.570 = | 2.485 / 2.485 = | 2.501 / 2.501 = | 2.337 / 2.337 = | nan / nan = |
| WER final mix | 0.024 / 0.024 = | 0.034 / 0.034 = | 0.020 / 0.020 = | 0.103 / 0.103 = | 0.043 / 0.043 = |
| audio hard cuts | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| |LUFS + 16| | 0.000 / 0.000 = | 0.000 / 0.000 = | 0.100 / 0.100 = | 0.000 / 0.000 = | 0.200 / 0.200 = |
| true peak dBTP | -1.300 / -1.300 = | -3.400 / -3.400 = | -1.400 / -1.400 = | -3.000 / -3.000 = | -1.300 / -1.300 = |
| video s / video-min | 420.6 / 420.6 = (over limit) | 438.9 / 438.9 = (over limit) | 480.0 / 480.0 = (over limit) | 94.5 / 94.5 = | 34.6 / 34.6 = |
| peak VRAM MiB | 14514 / 14514 INFO | 14370 / 14370 INFO | 15197 / 15197 INFO | 16323 / 16323 INFO | 4477 / 4477 INFO |
| audio stream md5 | f5dd3fc6 / f5dd3fc6 INFO | f3174f6b / f3174f6b INFO | edf612c0 / edf612c0 INFO | aa763a91 / aa763a91 INFO | 50418441 / 50418441 INFO |

## Verdict: **FAIL — no publish**
- 03 SyncNet on dub-speech seconds (conf at 0): 7.304 vs 7.861 (FAIL)
- 04 lip flash strong: 15 vs 9 (FAIL)
