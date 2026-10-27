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

## status 3
- r6v1_14b gate vs pb1: tempo fixed (02 50.7 -> 1.4 s, 05 26.5 -> 0), src-talks/dub-silent better on 01-03, but FAIL: 05 src-silent / face / gaps (no drawl for a slow speaker), inside pauses, 04 WER, joins, bed exposure.
- metric fixes: pause logic now classifies part boundaries from the placed part texts (Whisper pulled word starts both ways; pb1 02 21 -> 4 INSIDE); faith judge truncated long units (score 0) -> budget + retry; all tags re-scored.
- new: --dur-cost r6d = r6c + the take's own CTC pauses inside a phrase (Chatterbox hesitations) charged 2x.
- running: r6v2_14b (r6c + gap-QA cost guard) -> r6v3_14b (r6d + fill 0.9); faith re-score pb1 / v1.
- next: gate v2 / v3, then 7B on the winner config for the Qwen choice.

## status 4
- r6v2_14b (r6c + gap-QA guard, fill 0.95): unplanned speed-ups gone (01 fast 18.4 -> 0), 03 face/dub-silent 12.4 -> 0, src-talks/dub-silent 01 7.2 -> 6.35, 02 19.1 -> 14.95, 03 19.1 -> 9.6; tempo-out <= 4.3 s everywhere (pb1 23-52 s).
- r6v3 (r6d + fill 0.9) FAILED: the r6 cost charged every slowed second, so a SHORTER take was cheaper under a 0.9 floor (03 src-silent back to 17 s) -> r6e: charge only the added drawl time.
- open v2 rows: inside-phrase pauses (Chatterbox hesitations -> r6d), sentence level jumps (takes differ up to 6 dB -> remix --unit-level), bed exposure in new dub pauses, 05 face/dub-silent 7.35 -> 9.0.
- running: r6v2L (v2 remix + unit-level, no TTS), r6v4_14b (r6d + unit-level), then r6v5_14b (r6e + fill 0.9).
- next: gate v4 / v5 vs pb1; 7B on the winner; publish.

## status 5
- r6v4_14b (r6d inside-pause cost + unit-level): inside counts down on 01 / 02 / 05, but 03 / 04 lost v2's src-silent gains: the 2x inside penalty also hits good long takes and the judge
  rejected v2's winning u04 candidate this time (natural 3) -> selection outcomes vary run to run; avoiding hesitations by re-rolling is stochastic.
- remix --unit-level verified on v2's takes (r6v2L, no TTS): sentence level jump p90 below pb1 on all 5 (03 4.75 -> 1.43, pb1 2.31), pause-onset jump 03 2.57 -> 0.36.
- 02 residue-exposed: 97.6 % from ONE 0.34 s Chatterbox hesitation over a loud bed moment (12 s); 04 pause-onset jump: a 3-sample p95.
- new: --close-inside-pauses: hesitations > 0.12 s inside an English phrase shortened to 50 ms in the take (deterministic, CTC spans). Pause-logic edge-word rule.
- running: r6v5_14b (r6e + fill 0.9), then r6v6_14b (v2 + close-inside + unit-level). Next: pick v-best, 7B arm on it, publish (WATCH if the gate still fails).

## status 6
- finding: the pipeline is deterministic (r6v6 02-05 = r6v2 + unit-level bit for bit): arm-to-arm swings come from the cost change cascading (other budgets -> other candidates), not noise.
- r6v5 (r6e + fill 0.9): drawl back to pb1 level without fixing 05 face / 04 src-silent -> fill stays 0.95. 05 = slow speaker, English ~30 % shorter: r6 spreads the silence over the speaker's own pauses (several 1.5-2 s gaps) where pb1 left one 3 s gap.
- close-inside-pauses v1 never fired on 02-05 (CTC stretches the word over the pause; 02 u6 has two 0.7 s hesitations inside one 7 s part) -> it now finds the silent run itself (3da8b24). Faith judge: regex fallback (03 u7 looped).
- running: pb1b (pb1 reproduced again = gate noise floor), then r6v7_14b, r6v7_7b.
- next: gate v7 vs pb1 (+ pb1 vs pb1b), Qwen choice, publish r6_audio_v1 (WATCH if rows still fail) + handoff.

## status 7
- pb1b = pb1 bit for bit (gate PASS against itself): every remaining gate row on r6 arms is a real change, not noise.
- r6v7_14b (v2 + unit-level + close-inside on the silent run): 04 now better than pb1 (src-talks/dub-silent 12.2 -> 11.1, face 11.1 -> 7.0), 03 face 12.4 -> 0, tempo-out <= 4.6 s (pb1 23-52).
- Qwen choice on the v7 config: 14B. 14B vs 7B src-talks/dub-silent 02 14.8 vs 19.35, 03 9.5 vs 17.5, 03 face 0 vs 8.2; faith mixed (7B higher on 03 / 05, 14B on 01 / 02); 7B 04 WER .096; 7B 14 % faster (1249 vs 1450 s for 5 videos).
- pause metric: exact-zero gaps (pipeline spread pauses) are boundary pauses; pb1 INSIDE now 3/2/0/0/0. Remix --gap-bed-cap-db 10: 02 residue-exposed -20.9 -> -32.6 dB (one leaked syllable at 12.1 s in a sentence gap), bed speech heard 0.4 -> 0.15 s.
- running: r6v8_14b (v7 + close-inside 30 dB + gap-bed-cap) = the publish candidate; then publish (WATCH if rows remain) + handoff F.
