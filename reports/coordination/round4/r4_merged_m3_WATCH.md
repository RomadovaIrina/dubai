# r4 merged m3: WATCH candidate

m3 = E2E from the raw input: audio pb1 (the user's choice) + the r4_video_v1 stack with the user's rule 'while the face is visible the mouth follows the DUB' (lip-sync on dub speech and on dub-silent stretches where the source talks = closed mouth, small faces too, segments split inside dub silences >= 1 s, cap cuts at the quietest dub frame). PREV = v4-repro on this server. Known trade-offs of the rule: 05 flashes in the newly lip-synced small-face span 8.8-12 s; 04 slide in the long pb1 dub silence 21.4-27.3 s (closed mouth over a talking jaw). 05 33.7-36 s flashes = raw generator, owned by the video agent.

Files: `<id>_r4_merged_m3_WATCH.mp4`, `<id>_r4_merged_m3_WATCH_sbs.mp4` (V4 | M3). Gate verdict per video below (rows FAIL / better).

## 05: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.040 / 3.040 better |
| lip flash strong | 29 / 22 **FAIL** |
| video s / video-min | 549.9 / 923.9 better (over limit) |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.034 vs 0.038 px; 0:27.5-0:34.2: 0.056 vs 0.057 px
Worst LatentSync window joints in m3 (time, joint motion / inner motion; > 2 = a possible jump): 0:36.4 (2.4x), 0:04.4 (2.4x), 0:12.1 (2.4x), 0:38.2 (2.0x), 0:47.9 (2.0x), 0:16.5 (1.8x)
Lip flash candidates: 29 vs 22.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 3.04 s (v4-repro) -> 2.04 s; worst: 0:21.8 closed mouth, dub speaks (0.5 s), 0:49.3 talking mouth, dub silent (0.4 s), 0:49.7 talking mouth, dub silent (0.3 s), 0:38.6 talking mouth, dub silent (0.3 s)

## 04: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| slide nose-base px | 0.175 / 0.154 **FAIL** |
| lip flash strong | 12 / 18 better |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.215 vs 0.151 px; 0:39.0-0:47.6: 0.136 vs 0.158 px
Worst LatentSync window joints in m3 (time, joint motion / inner motion; > 2 = a possible jump): 0:08.5 (1.9x), 0:26.4 (1.9x), 0:15.5 (1.8x), 0:35.5 (1.8x), 0:43.7 (1.8x), 0:11.6 (1.7x)
Lip flash candidates: 12 vs 18.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.97 s (v4-repro) -> 0.73 s; worst: 0:01.8 talking mouth, dub silent (0.4 s), 0:46.4 closed mouth, dub speaks (0.4 s)

