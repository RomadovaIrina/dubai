# r7_audio_v1 (WATCH) for m6 = m4 face + this audio

Source: the PUBLISHED r6_audio_v1 takes (handoff/in/audio-r6-v1; the user's loved audio: same words, voice, tempo, unit levels), with two r7 changes:
1. **Lip placement** (scripts/r7/lip_place.py): where the face is visible, dub blocks are moved (never stretched, tempo = r6) into the source speech
   spans. No new pauses on these takes (splits are allowed only at pb1 boundaries; none was worth it). Mean shift 0.2-0.3 s, max 0.92 s; 02 unchanged (no face).
2. **Gap clean** (scripts/r7/gap_clean.py): in dub gaps, leaked voice / separator bursts -> this video's room tone; the bed where the source speaks
   and the dub is silent is denoised (noisereduce, MIT: stationary over the room tone on 02/03/04, non-stationary 0.7 on 01/05).

| video | residue exposed dB (lower = cleaner) | on-screen talker, dub silent s | src speaks, dub silent s | dub speaks, src silent s |
|---|---|---|---|---|
| 01 | -31.3 -> -36.1 | 0 -> 0 | 6.35 -> 6.5 | 2.55 -> 2.55 |
| 02 | -32.7 -> -40.2 | 0 -> 0 | 14.8 -> 14.8 | 11.65 -> 11.65 |
| 03 | -28.2 -> -32.8 | 1.5 -> 0 | 10.15 -> 7.6 | 11.1 -> 9.15 |
| 04 | -30.8 -> -36.2 | 7.0 -> 6.4 | 11.1 -> 9.45 | 10.45 -> 8.95 |
| 05 | -22.5 -> -27.5 | 9.0 -> 7.7 | 12.65 -> 10.8 | 9.25 -> 7.25 |

Whisper (per gap, no_speech filter) hears 0 original words in the gaps of either version: the leaks were unintelligible residue.

**Gate vs r6_audio_v1: 4 rows fail (all from the placement; gap clean alone passes them), classified:**
- 03 / 04 "level jump at dub-pause onsets" 0.38 -> 4.55 / 4.9 -> 10.24 dB: METRIC: it is (mix change - SOURCE change) at a dub-pause onset; when the dub now
  stops where the speaker stops, the speaker's own speech->silence drop is subtracted. The bed processing at these onsets is unchanged (gap-clean-only arm = r6).
- 04 pause distribution W1 0.723 -> 0.835 s (tol 0.1): 9 blocks only; listen 0:12-0:19.
- 05 longest dub gap beyond the source pause 2.35 -> 3.05 s at 0:12.6 (a block moved 1 s earlier onto its source span): listen 0:12-0:17.

For the face pipeline: timing changed vs r6_audio_v1 -> re-derive dub-silence spans from these files (manifest lip_place.dub_speech, or VAD on dubbed_16k.wav).
Listen: outbox/<id>_r7_audio_v1_WATCH_AB_prev-then-new.mp4 and <id>_r7_audio_v1_WATCH_gaps_AB.m4a (every dub gap, 1 tick = r6v1, 2 ticks = r7).
