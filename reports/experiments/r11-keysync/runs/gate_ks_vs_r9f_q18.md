# Regression gate: NEW `ks_q18` vs PREV `r9f_q18`

commit 6cabc31

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | q18 new / prev |
|---|---|
| frames == source | True / True PASS |
| duration Δ s | 0.020 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 5.196 / 7.736 **FAIL** |
| SyncNet dub-only (whole video, info) | 6.930 / 7.320 INFO |
| SyncNet mix (whole video, info) | 7.390 / 7.310 INFO |
| dub silent while face tracked, s (audio info) | 1 / 1 INFO |
| |AV offset| dub | 1 / 2 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.300 / 2.600 better |
| v4 resid mouth | 1.096 / 1.114 = |
| v4 twitch mouth | 1.084 / 1.177 better |
| v4 shear95 lower face | 0.975 / 1.028 better |
| v4 resid mouth, own frames (info) | 1.098 / 1.114 INFO |
| v4 common lip-synced seconds (info) | 19.0 / 19.0 INFO |
| v4 sharpness mouth (src=1) | 1.075 / 1.155 **FAIL** |
| slide nose-base px | None / None n/a |
| window joints ratio | 0.989 / 0.928 **FAIL** |
| window joints LF ratio | 0.980 / 0.969 = |
| bg ring lower-upper | 4.380 / -0.124 **FAIL (limit)** |
| lip flash strong | 3 / 8 better |
| colour worst 1 s |dL| (half face) | 6.080 / 1.370 **FAIL (limit)** |
| colour frames |dL|>6 | 73 / 0 **FAIL** |
| jaw jitter out/src | 0.969 / 1.049 better |
| chin wave (lower-chin residual std, 2 s) | 10.9 / 3.794 **FAIL** |
| chin wave worst 2 s | 15.6 / 5.171 **FAIL** |
| below-jaw band |out-src| | 9.837 / 2.254 **FAIL** |
| WER final mix (audio, info) | 0.056 / 0.018 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.000 / 0.000 INFO |
| true peak dBTP (audio, info) | -2.000 / -1.900 INFO |
| video s / video-min | 387.8 / 387.8 = (over limit) |
| peak VRAM MiB | 13072 / 13072 INFO |
| audio stream md5 | b16f1256 / e4087c61 INFO |

## Verdict: **FAIL — no publish**
- q18 SyncNet on dub-speech seconds (conf at 0): 5.196 vs 7.736 (FAIL)
- q18 v4 sharpness mouth (src=1): 1.075 vs 1.155 (FAIL)
- q18 window joints ratio: 0.989 vs 0.928 (FAIL)
- q18 bg ring lower-upper: 4.380 vs -0.124 (FAIL (limit))
- q18 colour worst 1 s |dL| (half face): 6.080 vs 1.370 (FAIL (limit))
- q18 colour frames |dL|>6: 73 vs 0 (FAIL)
- q18 chin wave (lower-chin residual std, 2 s): 10.9 vs 3.794 (FAIL)
- q18 chin wave worst 2 s: 15.6 vs 5.171 (FAIL)
- q18 below-jaw band |out-src|: 9.837 vs 2.254 (FAIL)
