# r6-audio status (audio agent A, 2026-10-27)
Published: r6_audio_v1 WATCH (tag r6v8_14b, Qwen 14B), outbox + handoff/out/F/audio-r6-v1 (42 MB, no READY: the gate vs pb1 fails 16 rows, classified in README.md).
Gains vs pb1: tempo-out 46-52 s -> <= 4.6 s, src-talks/dub-silent 01-04 down (03 19.1 -> 10.15), 03 face/dub-silent 12.4 -> 1.5, 04 11.1 -> 7.0, 02 WER fixed.
Open: 05 slow speaker (face/dub-silent 7.35 -> 9.0), 03 faith 4.9 -> 4.55, 02 non-silent hesitations, 03 one hard cut.
Arms and decisions: STATUS_LOG.md; all arms side by side: table_arms.md.

## v2 (after the user's verdict on v1)
v1 = audio of record (READY written). v2 WATCH = v1 takes + bed cap at dub edges: 02 12.1 s leak fixed, gate PASS vs v1. 02 hesitations = metric shift, 05 not safely fixable. See README_v2.md.
