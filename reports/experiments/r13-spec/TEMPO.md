# 7.3 tempo corridor audit (r13, 03.11)

**Spec:** "atempo up to ±30 %, then librosa; corridor ×0.75–×1.25, the rest filled with a pause".

**Method** (`scripts/r13/tempo_audit.py`, measured, not reconstructed):
- For every dubbed phrase, the same words are CTC-aligned in the raw TTS take (the "natural" pace) and in the final dub track.
- **Speech tempo** = natural / final speech duration, summed over the word durations; the pauses are excluded.
- Also per 4-word window (a local speed-up), and the final **speech rate** (syllables / s) against the video's median.
- **Validation:** on old03 the measured tempo tracks the recorded `atempo` within ±0.03 (0.888 vs 0.90, 1.067 vs 1.10, …).
- **librosa:** the current chain does not use librosa time-stretch (ffmpeg atempo only).

| video | phrases measured | speech tempo p5 / p50 / p95 | outside ×0.75–1.25 (measured) | recorded atempo outside | "rushed": rate > 1.25 × the video median (time, × median) |
|---|---:|---|---|---:|---|
| old01 | 26/27 | 0.84 / 0.97 / 1.02 | 0 | 0 | 93.96 s ×1.28 |
| old02 | 21/25 | 0.78 / 0.89 / 1.18 | 28.8 s ×0.68 (recorded 0.90) | 0 | 5.1 s ×1.56 |
| **old03** | 18/21 | 0.90 / 0.95 / 1.07 | 0 | 0 | **12.0 s ×1.37, 46.4 s ×1.33** |
| old04 | 9/13 | 0.74 / 0.84 / 1.03 | 40.9 s ×0.70 (recorded 1.0) | 0 | 44.5 s ×1.57, 17.6 s ×1.28 |
| old05 | 12/13 | 0.82 / 0.92 / 1.05 | 0 | 0 | 5.6 s ×1.27 |
| m03 | 7/7 | 0.96 / 0.98 / 1.05 | 0 | 0 | – |
| m15 | 7/7 | 0.93 / 0.96 / 1.00 | 0 | 0 | – |
| m18 | 4/4 | 0.96 / 0.98 / 1.01 | 0 | 0 | – |

## Findings
- **No phrase is stretched outside the spec corridor.** Every recorded atempo is within ×0.75–1.25. Measured speech tempo is ×0.74–1.18; the chain mostly SLOWS the voice down (median ×0.84–0.98).
- **Two measured slow-downs below ×0.75** (old02 28.8 s, old04 40.9 s) have a recorded atempo of 0.90 / 1.0. These are not stretches: the r8 retake re-said those units "at a deliberate pace", a different take. The audit compares against the first take, so it reads the slower retake as a stretch. A known limit of the method.
- **"The voice speeds up in one place" (old03).** There is no atempo violation. It is a fast Chatterbox take × a mild speed-up, heard against the slower phrases around it:
  - 12.0 s "In general," (2 words in 0.73 s: ×1.37 the median rate);
  - 46.4 s "everything will be fine." (atempo ×1.14 on an already fast take: ×1.33).

  Similar rushed phrases: old02 5.1 s, old04 44.5 s (×1.56–1.57).

## Fix proposal (behind a flag, default OFF): `DUB_PACE_FIX=1`
`scripts/r13/pace_fix.py`, hooked into `e2e.sh` after the audio, before the face (the lips follow the new line).
- **Trigger:** a phrase's final speech rate > 1.25 × the video median, OR its tempo outside ×0.75–1.25.
- **Fix:** Qwen 14B rephrases the line to fit its slot at the video's median pace (a syllable budget; the mixed-script guard). Chatterbox says it with the same speaker reference at natural tempo (atempo ≤ ×1.05). It is spliced into the same placed interval:
  - outside that interval both final mixes stay **bit-identical** (checked);
  - inside, the gap-cleaned bed + the new voice, at the original mix's level, with 50-ms crossfades;
  - the subtitles follow the new text.
- **Demo on old03:**
  - "In general," → "In brief,";
  - "everything will be fine." → "All will be well.";
  - 21 s, the rest of the mix bit-identical;
  - clip `outbox/r13_spec/old03_pace_fix_AB.mp4`: each spot before / after. In that clip the lips are the old ones, because only the audio was redone; in an E2E with the flag the face is generated after the fix.
- **Not a default:** it changes the accepted audio on 1–2 phrases per video. The user's ear decides (the AB clip).
