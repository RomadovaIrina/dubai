# DabAI round 8: management set (29.10.2026)

Files per video (01-05):
- `<id>_DabAI_r8_demucs.mp4`: the main version, htdemucs background bed (audio-r8-v5).
- `<id>_DabAI_r8_spleeter.mp4`: the same picture and the same dub, on the Spleeter background bed (audio-r8-v5-spleeter).
- `<id>_DabAI_r8_<variant>_phone.mp4`: the same, <= 25 MB (24 MB), for phones.

Both variants have a bit-identical video stream; only the background bed under the dub differs. Source resolution and fps, H.264 + AAC.

## What changed since r7 (final_r7 = m11)
Face (m18), all behind flags:
- CodeFormer is back (user decision; non-commercial licence, the risk is accepted and logged), but only as DETAIL: lip line, teeth and lip
  texture. It is applied on top of the lip-synced mouth, which keeps its own shape and motion, and averaged over neighbouring frames so it
  does not flicker. InsightFace is not used (MediaPipe alignment).
- No more "mouth explosions". Frames where the generator breaks visibly (washed pink lip blob, collapsed lips, near-black hole, sudden
  open/shut jump) fade smoothly back to the original mouth for that moment. Share of lip-synced frames that fall back: 01 15 %, 03 5 %,
  04 3 %, 05 1 %.
- The light <-> dark flicker inside the mouth is smoothed (03: frame-to-frame flicker 2.5 -> ~1.8, the source is 1.2).
- Faster generator: 10 denoising steps where the dub speaks, 20 where it is silent; CodeFormer without its face-parsing step; faster
  ESRGAN. The face stage on 03 went ~620 -> ~476 s per video-minute; the speed work continues.

Audio (from the audio agent): audio-r8-v4 dub (the dub speaks where the source speaks, no cut words, almost no dead air; the 04 ~17 s closed
mouth is covered) + the 04 "City of Sochi" re-take (14.6-15.75 s) + the htdemucs bed (v5) or the Spleeter bed.

## Numbers vs final_r7 (m11)
| video | lip flashes | SyncNet (dub speech) | closed mouth while the source mouth is open, dub silent (s) | mouth / dub mismatch (s) |
|---|---|---|---|---|
| 01 | 4 -> 0 | 6.65 -> 4.06 | 0 -> 0 | 0 -> 0 |
| 03 | 44 -> 29 | 7.86 -> 6.10 | 3.2 -> 3.7 | 0.77 -> 1.87 |
| 04 | 16 -> 3 | 6.42 -> 5.12 | 3.13 -> 2.93 | 0.73 -> 0.70 |
| 05 | 6 -> 4 | 3.47 -> 3.76 | 6.95 -> 4.85 | 1.84 -> 1.67 |
| 02 | (no face on screen) | - | 0 | 0 |

Whole-frame gate: PASS on all 5.

## Known leftovers
- Lower SyncNet on 01, 03 and 04 than r7: on the moments that fall back, the mouth follows the original speech instead of the dub. That is
  the price of removing the explosions; it is largest on 01, whose opening seconds (small face, fast head turn) are where the generator
  breaks most.
- 03 "closed mouth" seconds are slightly up with CodeFormer (3.2 -> 3.7 s). This is most likely the sharper lip line moving the landmark
  measurement, not a visibly pressed mouth; still to be checked by eye.
- The 05 window-joint smoothness metric is above its r7 tolerance (0.90 -> 1.05); no visible seam was found.
- Generator lip artefacts that are not severe enough to fall back can still show briefly. The mouth can stay open during some short
  pauses inside a dubbed phrase (the r7 leftover).
