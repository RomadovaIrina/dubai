# Regression gate: NEW `ksfn_q18` vs PREV `r9f_q18`

commit 4e6171b

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | q18 new / prev |
|---|---|
| frames == source | True / True PASS |
| duration Δ s | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 6.700 / 7.736 **FAIL** |
| SyncNet dub-only (whole video, info) | 6.010 / 7.320 INFO |
| SyncNet mix (whole video, info) | 6.240 / 7.310 INFO |
| dub silent while face tracked, s (audio info) | 1 / 1 INFO |
| |AV offset| dub | 1 / 2 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.000 / 2.600 better |
| v4 resid mouth | 1.221 / 1.114 **FAIL** |
| v4 twitch mouth | 1.203 / 1.177 = |
| v4 shear95 lower face | 1.154 / 1.028 **FAIL** |
| v4 resid mouth, own frames (info) | 1.213 / 1.114 INFO |
| v4 common lip-synced seconds (info) | 19.0 / 19.0 INFO |
| v4 sharpness mouth (src=1) | 1.115 / 1.155 **FAIL** |
| slide nose-base px | None / None n/a |
| window joints ratio | 0.980 / 0.928 **FAIL** |
| window joints LF ratio | 1.079 / 0.969 **FAIL** |
| bg ring lower-upper | -0.122 / -0.124 = |
| lip flash strong | 6 / 8 = |
| colour worst 1 s |dL| (half face) | 2.760 / 1.370 **FAIL** |
| colour frames |dL|>6 | 0 / 0 = |
| jaw jitter out/src | 1.132 / 1.049 **FAIL** |
| chin wave (lower-chin residual std, 2 s) | 9.252 / 3.794 **FAIL** |
| chin wave worst 2 s | 12.6 / 5.171 **FAIL** |
| below-jaw band |out-src| | 2.263 / 2.254 = |
| WER final mix (audio, info) | 0.018 / 0.018 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.000 / 0.000 INFO |
| true peak dBTP (audio, info) | -1.900 / -1.900 INFO |
| video s / video-min | 387.8 / 387.8 = (over limit) |
| peak VRAM MiB | 13072 / 13072 INFO |
| audio stream md5 | e4087c61 / e4087c61 INFO |

## Verdict: **FAIL — no publish**
- q18 SyncNet on dub-speech seconds (conf at 0): 6.700 vs 7.736 (FAIL)
- q18 v4 resid mouth: 1.221 vs 1.114 (FAIL)
- q18 v4 shear95 lower face: 1.154 vs 1.028 (FAIL)
- q18 v4 sharpness mouth (src=1): 1.115 vs 1.155 (FAIL)
- q18 window joints ratio: 0.980 vs 0.928 (FAIL)
- q18 window joints LF ratio: 1.079 vs 0.969 (FAIL)
- q18 colour worst 1 s |dL| (half face): 2.760 vs 1.370 (FAIL)
- q18 jaw jitter out/src: 1.132 vs 1.049 (FAIL)
- q18 chin wave (lower-chin residual std, 2 s): 9.252 vs 3.794 (FAIL)
- q18 chin wave worst 2 s: 12.6 vs 5.171 (FAIL)
