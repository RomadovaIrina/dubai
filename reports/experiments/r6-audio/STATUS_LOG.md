## 2026-10-27 status 1 (audio agent A, r6)
- hypothesis: the 4 r5v4 failures come from the dry-run cost not charging what the gate counts (ramped tempo, 0.7 x dub-over), the 0.875 fill drawl, a 20 ms unit joint (02 WER) and a pause-logic word pick that classifies (next word | word after).
- changed (dcf8cf8): --dur-cost r6 (full-weight tempo outside 0.95-1.05, dub-over x1.0), --fill-slowdown 0.95, --unit-gap 0.15, r4_pause_logic picks the word before the gap START (DUB_PAUSE_PICK=r4 keeps the old one for A/B).
- failed: merging backup/r5-audio-14b-wip-1024 replaced models/ + third_party/ by its self-symlinks (ignored dirs) -> ~40 min re-download; restored.
- running: queue 1 = pb1 (baseline) -> r6v1_14b -> r6v1_7b, all 5 videos.
- next: gate both vs pb1, validate the pause pick on pb1 rows (r4 vs r6 pick), pick 14B / 7B, publish v1 or iterate.
- metric fix verified on pb1 02 (same run, same Whisper words): r4 pick 21/21 pauses INSIDE (50.1 s), every one shifted one word right
  (8.49 s after "banya." scored as "The | Russian"); r6 pick 9 INSIDE (5.5 s), 7 sentence, 3 comma, 1 strong. pb1_listen.md's INSIDE rows were mostly sentence pauses.
