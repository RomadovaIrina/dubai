## 2026-10-27 status 1 (audio agent A, r6)
- hypothesis: the 4 r5v4 failures come from the dry-run cost not charging what the gate counts (ramped tempo, 0.7 x dub-over), the 0.875 fill drawl, a 20 ms unit joint (02 WER) and a pause-logic word pick that classifies (next word | word after).
- changed (dcf8cf8): --dur-cost r6 (full-weight tempo outside 0.95-1.05, dub-over x1.0), --fill-slowdown 0.95, --unit-gap 0.15, r4_pause_logic picks the word before the gap START (DUB_PAUSE_PICK=r4 keeps the old one for A/B).
- failed: merging backup/r5-audio-14b-wip-1024 replaced models/ + third_party/ by its self-symlinks (ignored dirs) -> ~40 min re-download; restored.
- running: queue 1 = pb1 (baseline) -> r6v1_14b -> r6v1_7b, all 5 videos.
- next: gate both vs pb1, validate the pause pick on pb1 rows (r4 vs r6 pick), pick 14B / 7B, publish v1 or iterate.
- metric fix verified on pb1 02 (same run, same Whisper words): r4 pick 21/21 pauses INSIDE (50.1 s), every one shifted one word right
  (8.49 s after "banya." scored as "The | Russian"); r6 pick 9 INSIDE (5.5 s), 7 sentence, 3 comma, 1 strong. pb1_listen.md's INSIDE rows were mostly sentence pauses.

## status 2
- evidence: pb1 reproduced (03 = r5 STATUS numbers). r6v1_14b 01: tempo-out 51.9 -> 18.4 s, src-talks/dub-silent 7.2 -> 4.45, inside 17 -> 14, but sped-up 10.9 -> 18.4 s.
- cause of the speed-ups: (a) the dry run measured parts as CTC span ends, the pipeline cuts with 60/80 ms pre/post-roll (+0.14 s per part); (b) gap QA re-seeded the chosen take without the placement cost (u12 7.18 -> 7.96 s, x1.15 on 7 s).
- changed: --dur-cost r6c (cut_parts in the dry run, 40ff182), --dur-gap-qa-cost on (cost guard on gap-QA re-seeds).
- running: r6v1_14b 02-05, then queue 2 = r6v2_14b, r6v2_7b (v1 7B arm dropped: the 14B / 7B choice is made on v2).
- next: gate v1 (diagnostics), gate v2 14B vs 7B, publish the better as r6_audio_v1.
