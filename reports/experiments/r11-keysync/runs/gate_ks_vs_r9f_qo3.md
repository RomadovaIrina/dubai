# Regression gate: NEW `ks_qo3` vs PREV `r9f_qo3`

commit 6cabc31

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | qo3 new / prev |
|---|---|
| frames == source | True / True PASS |
| duration Δ s | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 5.259 / 7.184 **FAIL** |
| SyncNet dub-only (whole video, info) | 1.560 / 1.580 INFO |
| SyncNet mix (whole video, info) | 6.750 / 6.390 INFO |
| dub silent while face tracked, s (audio info) | 11 / 11 INFO |
| |AV offset| dub | 2 / 1 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.900 / 1.670 = |
| v4 resid mouth | 1.080 / 1.152 better |
| v4 twitch mouth | 1.002 / 1.014 = |
| v4 shear95 lower face | 1.308 / 1.004 **FAIL** |
| v4 resid mouth, own frames (info) | 1.081 / 1.153 INFO |
| v4 common lip-synced seconds (info) | 85.0 / 85.0 INFO |
| v4 sharpness mouth (src=1) | 0.785 / 0.828 **FAIL** |
| slide nose-base px | None / None n/a |
| window joints ratio | 0.982 / 0.885 **FAIL** |
| window joints LF ratio | 1.006 / 0.801 **FAIL** |
| bg ring lower-upper | 9.011 / 0.056 **FAIL (limit)** |
| lip flash strong | 21 / 35 better |
| colour worst 1 s |dL| (half face) | 3.780 / 1.380 **FAIL** |
| colour frames |dL|>6 | 0 / 0 = |
| jaw jitter out/src | 1.192 / 0.934 **FAIL** |
| chin wave (lower-chin residual std, 2 s) | 4.836 / 1.236 **FAIL** |
| chin wave worst 2 s | 9.711 / 1.891 **FAIL** |
| below-jaw band |out-src| | 26.2 / 2.502 **FAIL** |
| WER final mix (audio, info) | 0.017 / 0.017 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.200 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.300 / -1.400 INFO |
| video s / video-min | 307.7 / 307.7 = (over limit) |
| peak VRAM MiB | 22083 / 22083 INFO |
| audio stream md5 | a4e119fd / 47c83ecd INFO |

## Verdict: **FAIL — no publish**
- qo3 SyncNet on dub-speech seconds (conf at 0): 5.259 vs 7.184 (FAIL)
- qo3 v4 shear95 lower face: 1.308 vs 1.004 (FAIL)
- qo3 v4 sharpness mouth (src=1): 0.785 vs 0.828 (FAIL)
- qo3 window joints ratio: 0.982 vs 0.885 (FAIL)
- qo3 window joints LF ratio: 1.006 vs 0.801 (FAIL)
- qo3 bg ring lower-upper: 9.011 vs 0.056 (FAIL (limit))
- qo3 colour worst 1 s |dL| (half face): 3.780 vs 1.380 (FAIL)
- qo3 jaw jitter out/src: 1.192 vs 0.934 (FAIL)
- qo3 chin wave (lower-chin residual std, 2 s): 4.836 vs 1.236 (FAIL)
- qo3 chin wave worst 2 s: 9.711 vs 1.891 (FAIL)
- qo3 below-jaw band |out-src|: 26.2 vs 2.502 (FAIL)
