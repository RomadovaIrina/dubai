# DabAI round 8b: fix for the "drifting" lip-sync of final_r8 (30.10.2026)

## Diagnosis of final_r8
There is no timing error. A SyncNet offset sweep (+-15 frames, every 5-s speech window, 01 / 03 / 04 / 05) finds the best offset at 0
everywhere, with no drift.

The cause is the explosion fallback. It crossfaded the mouth partly back to the original (Russian) articulation on 6-36 % of the
lip-synced frames, spread over almost every 5-s window. On top of that, the inner-mouth tone smoothing flattened the articulation a little.
Together these lowered the lip-sync confidence everywhere, which reads as drifting sync.

## Two fixed variants (same picture pipeline as final_r8: CodeFormer detail; same audio: htdemucs / Spleeter)
- **final_r8b_m19a_nofallback**: no fallback, no tone smoothing. Lip-sync is back at the r7 level; the rare generator "explosions" (e.g. the
  01 opening second) are visible again.
- **final_r8b_m19b_tightfallback**: fallback only on clearly broken frames (01 10 %, 03 3 %, 04 2 %, 05 0.6 %), with 1-frame transitions
  and no tone smoothing. Most explosions stay hidden; about half of the lip-sync loss is recovered.

| video | SyncNet r7 / r8 / r8b-m19b / r8b-m19a | lip flashes r7 / r8 / m19b / m19a |
|---|---|---|
| 01 | 6.65 / 4.06 / 4.76 / 6.71 | 4 / 0 / 0 / 0 |
| 03 | 7.86 / 6.10 / 6.74 / 7.63 | 44 / 29 / 36 / 38 |
| 04 | 6.42 / 5.12 / 5.51 / 5.99 | 16 / 3 / 8 / 9 |
| 05 | 3.47 / 3.76 / 4.20 / 4.15 | 6 / 4 / 1 / 2 |

Whole-frame gate PASS on all. Files: `<id>_DabAI_r8_{demucs,spleeter}.mp4` + `_phone.mp4` (<= 25 MB). 02 has no face on screen, so it is
the same as final_r8.
