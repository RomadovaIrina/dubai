# r8 F: m13 = candidate set for management (WATCH, order 01 03 05 04 02)

m13 = LatentSync 1.6 re-run on audio-r8-v1 (all 5) + the m11 face stack + confidence fallback fb5 (only visibly broken frames: washed pink lip blob,
lip contrast collapse, near-black cavity, open/shut jump -> held, ramped crossfade to the source mouth) + layer tone deflicker (inner-mouth tone
smoothed over time, tied to the opening; lower-face low-frequency tone). Files: <id>_r8_m13_WATCH.mp4, <id>_r8_m13_WATCH_sbs.mp4 (left m11 = final_r7, right m13).

| video | LS frames falling back | whole-frame gate | flashes m11 -> m13 | SyncNet dub-speech | pressed lips s | mouth/dub mismatch s |
|---|---:|---|---|---|---|---|
| 01 | 24.2 % | PASS | 4 -> 1 | 6.65 -> 3.63 | 0 -> 0 | 0 -> 0 |
| 03 | 10.2 % | PASS | 44 -> 29 | 7.86 -> 5.04 | 3.2 -> 2.2 | 0.77 -> 2.73 |
| 05 | 3.2 % | PASS | 6 -> 3 | 3.47 -> 2.62 | 6.95 -> 6.55 | 1.84 -> 1.87 |
| 04 | 8.7 % | PASS | 16 -> 11 | 6.42 -> 4.23 | 3.13 -> 2.0 | 0.73 -> 1.8 |
| 02 | (no face) | PASS | 0 -> 0 | - | 0 | 0 |

03 inner-mouth flicker (r8_mouth_flicker.py, 30 fps): source 1.21 / m11 2.52 / m13 1.50 (frame-to-frame HF luma); flicker events 0.63 / 4.07 / 1.22 per s;
lower face 0.32 / 0.35 / 0.33.
Gate (vs r7-quality-baseline) FAILs by design on SyncNet, on mismatch 03 / 04 and on joints LF 05: the fallback trades lip-sync for no explosions,
and the new audio changes the placement. Not separated yet: how much of the SyncNet drop is the fallback vs the deflicker vs audio-r8-v1.
