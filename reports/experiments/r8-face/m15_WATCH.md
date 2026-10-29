# r8 F: m15 = the m13b face stack on audio-r8-v4 (recommended candidate; WATCH order 01 03 05 04 02)

How m15 is built:
- LatentSync 1.6 re-run on audio-r8-v4, then the m11 face stack.
- Fallback fb6: only visibly broken frames get a short, ramped crossfade to the source mouth.
- Inner-mouth tone deflicker dfk2 (+-15 %, sigma 1).
- Clean stack: no NC models.

The sbs shows m11 = final_r7 on the left and m15 on the right.

| video | LS frames falling back | whole-frame gate | flashes m11 -> m15 | SyncNet m11 -> m15 | pressed lips s | mismatch s |
|---|---:|---|---|---|---|---|
| 01 | 16.2 % | PASS | 4 -> 0 | 6.65 -> 4.35 | 0 -> 0 | 0 -> 0 |
| 03 | 7.4 % | PASS | 44 -> 29 | 7.86 -> 5.82 | 3.2 -> 2.0 | 0.77 -> 1.70 |
| 05 | 2.0 % | PASS | 6 -> 4 | 3.47 -> 3.36 (ok) | 6.95 -> 4.88 | 1.84 -> 0.97 |
| 04 | 4.1 % | PASS | 16 -> 2 | 6.42 -> 5.39 | 3.13 -> 2.43 | 0.73 -> 1.03 |
| 02 | (no face) | PASS | 0 | - | 0 | 0 |

04 ~17 s ("августа сегодня", the closed mouth the user flagged as "vomit"): audio-r8-v4 covers it. No pressed-lips frames are left in
15-21 s apart from one 0.03 s blip at 19.8 s.
