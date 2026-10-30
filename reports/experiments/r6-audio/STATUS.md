# r6-audio status (audio agent A, 2026-10-27)
Published: r6_audio_v1 WATCH (tag r6v8_14b, Qwen 14B), outbox + handoff/out/F/audio-r6-v1 (42 MB, no READY: the gate vs pb1 fails 16 rows, classified in README.md).
Gains vs pb1: tempo-out 46-52 s -> <= 4.6 s, src-talks/dub-silent 01-04 down (03 19.1 -> 10.15), 03 face/dub-silent 12.4 -> 1.5, 04 11.1 -> 7.0, 02 WER fixed.
Open: 05 slow speaker (face/dub-silent 7.35 -> 9.0), 03 faith 4.9 -> 4.55, 02 non-silent hesitations, 03 one hard cut.
Arms and decisions: STATUS_LOG.md; all arms side by side: table_arms.md.

## v2 (after the user's verdict on v1)
v1 = audio of record (READY written). v2 WATCH = v1 takes + bed cap at dub edges: 02 12.1 s leak fixed, gate PASS vs v1. 02 hesitations = metric shift, 05 not safely fixable. See README_v2.md.

## Final (A DONE)
- Generic only: every r6 change is a default of the one-command runner `scripts/r6/r6_audio.sh`, with no per-video flags or thresholds. Thresholds are
  relative to each take or each video (take p90, the bed's own median, the video's median unit level).
- Audio of record: r6_audio_v1 (handoff/out/F/audio-r6-v1, READY). Candidate: r6_audio_v2 WATCH (= v1 takes + the edges bed cap, gate PASS vs v1, no READY).
- Reproduce: `bash scripts/r6/r6_audio.sh <tag>` -> v2 (deterministic: pb1 reproduced bit for bit). Needs `scripts/setup_spleeter.sh` and the Qwen2.5-14B Q5_K_M GGUF.
- Open: 05 slow speaker (missing English words), the pause metric's Whisper word shift (use a CTC alignment of the final track), 03 one hard cut at 19.9 s.
