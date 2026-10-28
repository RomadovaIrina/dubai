# r7-audio status (audio agent A, 2026-10-28)

One command: `bash scripts/r7/r7_audio.sh <tag>` = r6_audio_v2 + LIP_PLACE=on + remix `--gap-clean on`; publish: `bash scripts/r7/r7_publish.sh <ver> <tag> <prev>`.

## Done (on a 7B scaffold of r6: same r6 flags, Qwen 7B because the 14B GGUF is not on this server)
- gap clean (scripts/r7/gap_clean.py): residual voice (Silero on the bed) / separator bursts in dub gaps -> room tone of the video (a floor <= -42 dBFS
  or a steady ambience, p70-p30 <= 3 dB), noisereduce (MIT) where the source speaks. Music / variable outdoor sound (01, 05): leaked voice only.
  Residue in exposed gaps (dB rel. voice): 02 -38.2 -> -41.8, 03 -28.7 -> -34.4, 04 -29.9 -> -35.9, 05 unchanged (no room tone).
- gap metric + listen clips (scripts/r7/gap_metrics.py): per gap voice / burst / voice-band excess / Whisper words; <id>_gaps_AB.m4a.
- lip placement (scripts/r7/lip_place.py): dub blocks moved (never stretched) into source speech where the face is visible; splits only at pb1
  boundaries (Whisper words + r4_pause_logic.classify), >= 0.6 s pieces. Nothing moves where no face is seen (02 bit for bit = r6).
  Speaker on screen talks, dub silent (s): 03 8.8 -> 7.3, 04 10.4 -> 8.6, 05 11.7 -> 7.45; dub over source silence 03 12.2 -> 8.1, 04 10.0 -> 8.2, 05 9.8 -> 7.1.
- Gate vs the scaffold (r7t7 vs r7s7b): 2 rows: 04 pause W1 0.643 -> 0.749 (tol 0.1, lam 0.3 arm passes but loses 05's gain), 05 dub pauses > 1 s
  5 -> 9 (source has 10: the dub now pauses where the speaker pauses).
- Cost: lip_place 5-7 s + gap_clean 2-4 s per video on top of r6 (audio stage 86-173 s/video-min measured with 2 videos sharing the GPU).

## Blocked
- The gate vs r6_audio_v1 and the WATCH / handoff need the r6_audio_v1 takes: the Qwen2.5-14B Q5_K_M GGUF (~10.5 GB > the 5 GB rule, asked the
  coordinator) or the r6v8_14b work dirs in handoff/in. Then: `bash scripts/r6/r6_audio.sh r6v1repro` ; `REUSE=r6v1repro bash scripts/r7/r7_audio.sh r7v1` ;
  `bash scripts/r7/r7_publish.sh v1 r7v1 r6v1repro` (WATCH=1 for a WATCH without READY).
- git push: the deploy key is not live yet; commits are local.
