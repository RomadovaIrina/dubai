# Regression gate: NEW `m8b` vs PREV `m8`

commit d57e521

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | 05 new / prev | 04 new / prev | 03 new / prev | 01 new / prev | 02 new / prev |
|---|---|---|---|---|---|
| frames == source | True / True PASS | True / True PASS | True / True PASS | True / True PASS | True / True PASS |
| duration Δ s | 0.001 / 0.001 = | 0.000 / 0.000 = | 0.000 / 0.000 = | 0.001 / 0.001 = | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 3.911 / 4.030 = | 6.183 / 6.130 = | 7.861 / 7.860 = | 6.654 / 6.732 = | None / None n/a |
| SyncNet dub-only (whole video, info) | 2.610 / 2.630 INFO | 4.280 / 4.150 INFO | 6.620 / 6.650 INFO | 6.820 / 6.820 INFO | None / None n/a |
| SyncNet mix (whole video, info) | 2.160 / 2.080 INFO | 3.700 / 3.680 INFO | 6.110 / 6.180 INFO | 6.530 / 6.650 INFO | None / None n/a |
| dub silent while face tracked, s (audio info) | 10 / 10 INFO | 20 / 19 INFO | 19 / 19 INFO | 0 / 0 INFO | None / None n/a |
| |AV offset| dub | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | None / None n/a |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.170 / 1.470 **FAIL** | 0.730 / 2.530 better | 0.770 / 1.570 better | 0.000 / 0.000 = | 0.000 / 0.000 = |
| v4 resid mouth | 0.993 / 0.995 = | 1.010 / 1.010 = | 1.165 / 1.182 = | 1.137 / 1.123 = | None / None n/a |
| v4 twitch mouth | 0.976 / 0.986 = | 1.009 / 1.012 = | 0.985 / 0.987 = | 1.079 / 1.066 = | None / None n/a |
| v4 shear95 lower face | 0.998 / 1.002 = | 1.001 / 1.004 = | 1.006 / 0.997 = | 0.939 / 0.940 = | None / None n/a |
| v4 resid mouth, own frames (info) | 0.995 / 0.972 INFO | 1.009 / 1.028 INFO | 1.166 / 1.188 INFO | 1.141 / 1.105 INFO | None / None n/a |
| v4 common lip-synced seconds (info) | 44.7 / 44.7 INFO | 44.2 / 44.2 INFO | 81.6 / 81.6 INFO | 10.4 / 10.4 INFO | None / None n/a |
| v4 sharpness mouth (src=1) | 0.965 / 0.966 = | 0.934 / 0.936 = | 0.834 / 0.844 = | 0.946 / 0.946 = | None / None n/a |
| slide nose-base px | 0.036 / 0.035 = | 0.111 / 0.111 = | 0.062 / 0.063 = | 0.082 / 0.078 = | None / None n/a |
| window joints ratio | 0.892 / 0.912 = | 0.930 / 0.916 = | 0.909 / 0.917 = | 0.926 / 0.918 = | None / None n/a |
| window joints LF ratio | 0.842 / 0.839 = | 0.898 / 0.886 = | 0.837 / 0.843 = | 0.923 / 0.923 = | None / None n/a |
| bg ring lower-upper | -0.137 / -0.136 = | -0.094 / -0.095 = | 0.055 / 0.056 = | -0.359 / -0.354 = | None / None n/a |
| lip flash strong | 5 / 7 = | 9 / 17 better | 44 / 57 better | 4 / 2 = | 0 / 0 = |
| colour worst 1 s |dL| (half face) | 1.660 / 1.670 = | 1.400 / 1.410 = | 1.330 / 1.300 = | 1.110 / 1.090 = | 0 / 0 = |
| colour frames |dL|>6 | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = | 0 / 0 = |
| jaw jitter out/src | 1.009 / 1.012 = | 1.011 / 0.997 = | 0.958 / 0.941 = | 1.006 / 0.971 = | None / None n/a |
| chin wave (lower-chin residual std, 2 s) | 0.964 / 1.003 = | 2.085 / 2.151 = | 1.110 / 1.119 = | 1.291 / 1.285 = | None / None n/a |
| chin wave worst 2 s | 1.886 / 1.999 = | 4.047 / 4.137 = | 1.845 / 1.814 = | 1.686 / 1.725 = | None / None n/a |
| below-jaw band |out-src| | 2.570 / 2.566 = | 2.485 / 2.485 = | 2.501 / 2.502 = | 2.337 / 2.330 = | nan / nan = |
| WER final mix (audio, info) | 0.024 / 0.024 INFO | 0.034 / 0.045 INFO | 0.020 / 0.029 INFO | 0.103 / 0.103 INFO | 0.043 / 0.027 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO | 0 / 0 INFO | 0 / 1 INFO | 0 / 0 INFO | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.000 / 0.000 INFO | 0.000 / 0.000 INFO | 0.100 / 0.100 INFO | 0.000 / 0.000 INFO | 0.200 / 0.200 INFO |
| true peak dBTP (audio, info) | -1.300 / -1.300 INFO | -3.400 / -3.300 INFO | -1.400 / -1.200 INFO | -3.000 / -3.000 INFO | -1.300 / -1.200 INFO |
| video s / video-min | 420.6 / 457.0 = (over limit) | 438.9 / 526.1 better (over limit) | 480.0 / 467.3 = (over limit) | 94.5 / 98.8 = | 34.6 / 37.1 = |
| peak VRAM MiB | 14514 / 15973 INFO | 14370 / 15822 INFO | 15197 / 14602 INFO | 16323 / 15435 INFO | 4477 / 3348 INFO |
| audio stream md5 | f5dd3fc6 / 575c1173 INFO | f3174f6b / 501f6fcb INFO | edf612c0 / 390ce760 INFO | aa763a91 / 5eecaa8d INFO | 50418441 / d23ad4b6 INFO |

## Verdict: **FAIL — no publish**
- 05 mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks): 2.170 vs 1.470 (FAIL)
