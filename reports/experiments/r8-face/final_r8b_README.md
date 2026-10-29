# DabAI round 8b: final set (30.10.2026)

Files: `<id>_DabAI_r8b_demucs.mp4` (main: htdemucs background bed) and `<id>_DabAI_r8b_spleeter.mp4` (the same picture and the same dub on
the Spleeter bed), for 01-05. Full quality, source resolution and fps, H.264 + AAC. Within each pair the video stream is bit-identical.

## What it is
- **Face (m19b):**
  - LatentSync 1.6, with 10 denoising steps where the dub speaks and 20 where it is silent.
  - CodeFormer adds only DETAIL (lip line, teeth, lip texture) on top of the lip-synced mouth; the mouth keeps its own shape and motion.
    CodeFormer is non-commercial; the licence risk is the user's decision and is logged.
  - Real-ESRGAN on the lips, then the r7 composite.
  - A fallback to the original mouth ONLY on clearly broken frames (washed pink blob, collapsed lips, near-black hole, sudden open/shut
    jump), with 1-frame transitions. Share of lip-synced frames: 01 10 %, 03 3 %, 04 2 %, 05 0.6 %.
- **Audio:** audio-r8-v5. The dub speaks where the source speaks, there are no cut words, and 04 has the "City of Sochi" re-take. Two
  background beds: htdemucs (main) and Spleeter.

## Why r8b and not final_r8
final_r8 felt "drifting" in lip-sync. There was no timing error: the SyncNet best offset is 0 in every 5-s window, with no drift. Its
fallback faded the mouth partly back to the original speech on 6-36 % of frames, spread across the whole video, and a mouth-tone filter
flattened the articulation. r8b keeps the fallback only on clearly broken frames and drops the tone filter.

## Numbers: final_r7 / final_r8 / final_r8b
| video | SyncNet (dub speech) | lip flashes | closed mouth while the source mouth is open, dub silent (s) | mouth / dub mismatch (s) |
|---|---|---|---|---|
| 01 | 6.65 / 4.06 / 4.76 | 4 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 03 | 7.86 / 6.10 / 6.74 | 44 / 29 / 36 | 3.2 / 3.7 / 3.87 | 0.77 / 1.87 / 1.83 |
| 04 | 6.42 / 5.12 / 5.51 | 16 / 3 / 8 | 3.13 / 2.93 / 2.90 | 0.73 / 0.70 / 1.43 |
| 05 | 3.47 / 3.76 / 4.20 | 6 / 4 / 1 | 6.95 / 4.85 / 5.08 | 1.84 / 1.67 / 2.54 |
| 02 | no face on screen | 0 | 0 | 0 |

Whole-frame gate PASS on all 5 (worst frame 0.005 %, no frame over the 0.2 % limit).

## Known leftovers
- SyncNet is still below r7 on 01, 03 and 04, where the remaining fallback shows the original mouth on broken frames. It is largest on
  01, whose first second (small face, fast head turn) breaks the generator most.
- The mouth / dub mismatch is higher on 04 / 05 than in r7 (1.4 / 2.5 s): the new, longer dub placement plus the fallback.
- The 03 "closed mouth" seconds are slightly up with CodeFormer (3.2 -> 3.9 s), most likely a landmark-measurement effect of the sharper
  lip line.
- Milder generator artefacts that are not flagged can still flash briefly.
