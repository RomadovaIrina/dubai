# Regression gate: NEW `ksfa_qo3` vs PREV `r9f_qo3`

commit 4e6171b

Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.

| metric | qo3 new / prev |
|---|---|
| frames == source | True / True PASS |
| duration Δ s | 0.000 / 0.000 = |
| SyncNet on dub-speech seconds (conf at 0) | 5.691 / 7.184 **FAIL** |
| SyncNet dub-only (whole video, info) | 1.250 / 1.580 INFO |
| SyncNet mix (whole video, info) | 5.200 / 6.390 INFO |
| dub silent while face tracked, s (audio info) | 11 / 11 INFO |
| |AV offset| dub | 1 / 1 = |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.470 / 1.670 = |
| v4 resid mouth | 1.256 / 1.152 **FAIL** |
| v4 twitch mouth | 1.433 / 1.014 **FAIL** |
| v4 shear95 lower face | 1.253 / 1.004 **FAIL** |
| v4 resid mouth, own frames (info) | 1.255 / 1.153 INFO |
| v4 common lip-synced seconds (info) | 85.0 / 85.0 INFO |
| v4 sharpness mouth (src=1) | 0.901 / 0.828 better |
| slide nose-base px | None / None n/a |
| window joints ratio | 0.926 / 0.885 = |
| window joints LF ratio | 0.901 / 0.801 **FAIL** |
| bg ring lower-upper | 0.055 / 0.056 = |
| lip flash strong | 33 / 35 = |
| colour worst 1 s |dL| (half face) | 1.230 / 1.380 = |
| colour frames |dL|>6 | 0 / 0 = |
| jaw jitter out/src | 1.083 / 0.934 **FAIL** |
| chin wave (lower-chin residual std, 2 s) | 2.326 / 1.236 **FAIL** |
| chin wave worst 2 s | 3.666 / 1.891 **FAIL** |
| below-jaw band |out-src| | 2.502 / 2.502 = |
| WER final mix (audio, info) | 0.017 / 0.017 INFO |
| audio hard cuts (audio, info) | 0 / 0 INFO |
| |LUFS + 16| (audio, info) | 0.100 / 0.100 INFO |
| true peak dBTP (audio, info) | -1.400 / -1.400 INFO |
| video s / video-min | 307.7 / 307.7 = (over limit) |
| peak VRAM MiB | 22083 / 22083 INFO |
| audio stream md5 | 47c83ecd / 47c83ecd INFO |

## Verdict: **FAIL — no publish**
- qo3 SyncNet on dub-speech seconds (conf at 0): 5.691 vs 7.184 (FAIL)
- qo3 v4 resid mouth: 1.256 vs 1.152 (FAIL)
- qo3 v4 twitch mouth: 1.433 vs 1.014 (FAIL)
- qo3 v4 shear95 lower face: 1.253 vs 1.004 (FAIL)
- qo3 window joints LF ratio: 0.901 vs 0.801 (FAIL)
- qo3 jaw jitter out/src: 1.083 vs 0.934 (FAIL)
- qo3 chin wave (lower-chin residual std, 2 s): 2.326 vs 1.236 (FAIL)
- qo3 chin wave worst 2 s: 3.666 vs 1.891 (FAIL)
