# r5-audio status (audio agent A, 2026-10-24, written before server shutdown)

## What works
- `--dur-translate on` (scripts/pilot/dur_translate.py, hooked into tts_burst.run_tts_phrase): per unit, a dry run of the REAL pb1 placement
  (flow merge, boundary-only cut DP, voiced-cut merges, burst_fit_ratios, place_groups tempo rule) on the source speech intervals costs each TTS take:
  source speech without dub + 0.7 x dub over source silence + 1.5 x speed-up (ramp 1.05->1.15) + 0.3 x slow-down (ramp 0.95->0.875).
  Above 0.25 s: Qwen per-word-budget rewordings + source-pause-ordered piece candidates, guards (numbers / names / NE / filler / script), one Qwen judge
  call per candidate (back-translation, missing / added facts, meaning + naturalness >= 4), TTS of the top 3 + plain re-seed + Chatterbox pace
  (cfg_weight 0.3 / 0.2) when under-filled; cheapest take wins, plain wins ties within 0.15 s.
- `--qwen-size 14b`: Qwen2.5-14B-Instruct GGUF Q5_K_M (Apache-2.0, model card) for the whole translation stage + candidates; much better than 7B on
  the hard sentences (05 u03 "Here it just wants to see" -> "I just want to see how our application will handle itself in this format"). PAR=1 only.
- Gate axes added: tempo outside 0.95-1.05 (s), faithfulness (14B judge, pb1 re-scored: 01 4.76, 02 4.56, 03 4.9, 04 4.5, 05 4.29).
- Tooling: scripts/r5/r5_audio.sh (one command), r5_publish.sh (gate vs pb1 -> outbox mp4 + AB + review.md + handoff F/audio-r5-vN), r5_known_check.py.

## Numbers: r5v4 (= current HEAD, 14B + dur-translate) vs pb1 reproduced on this server (same code, same metrics)
| video | src talks / dub silent (s) | dub over src silence (s) | face visible, dub silent (s) | dub gaps > 1 s beyond src pause | tempo outside 0.95-1.05 (s) | sped-up (s) | inside-phrase pauses n / s | WER |
|---|---|---|---|---|---|---|---|---|
| 01 | 7.25 -> 4.45 | 3.05 -> 3.15 | 0 -> 0 | 0 -> 0 | 64.0 -> 31.6 | 7.9 -> 8.1 | 20/7.7 -> 16/5.1 | .105 -> .077 |
| 02 | 18.25 -> 12.2 | 16.35 -> 8.45 | 0 -> 0 | 6 -> 3 | 46.9 -> 25.0 | 10.5 -> 0.5 | 22/53.1 -> 25/49.6 | .016 -> .073 |
| 03 | 19.1 -> 10.0 | 10.7 -> 14.1 | 12.4 -> 3.1 | 8 -> 2 | 46.4 -> 48.7 | 0 -> 6.3 | 15/23.3 -> 19/16.4 | .018 -> .039 |
| 04, 05 | r5v4 was rendering at shutdown | | | | | | | |
(r5v3 = previous cost, 01: src-silent 3.1, 02: 7.75 but dub-over 28.1 = filled source pauses with padding -> fixed by the v4 cost.)

## Failing gate axes (tolerances of r4_audio_gate.py)
1. dub over source silence: 03 +3.4 s (tol 2.0).
2. tempo outside 0.95-1.05: 03 +2.4 s (tol 1.0) — dominated by pb1's fill-slowdown 0.875 on every short part; sped-up seconds 0 -> 6.3 on 03.
3. pauses inside an English phrase (count, tol 0): 02 22 -> 25, 03 15 -> 19, while the SECONDS improve (53 -> 50, 23 -> 16). The metric flags almost every
   pause as INSIDE even on pb1 (02: 22 of 23) -> needs a look before it is trusted.
4. WER 02 .016 -> .073 (tol .03): 02 u6 -> u7 joint ("That's for sure." sped to x1.15 on a 0.54 s burst, u7 starts 20 ms later; ASR merged them).
Not gated yet: faithfulness of r5v4 (r5_faith.py runs at the end of r5_audio.sh).

## Speed
Audio stage ~500 s per video (01: 441 s for 97 s = ~270 s/video-min; 221 s of it = 66 extra TTS takes; Qwen 62 s). Peak VRAM ~15 GB (14B + Chatterbox + QA).
Levers: tts_top 3 -> 2, drop the pace takes where they never win, skip the plain judge (the gate does it), batch judge calls.

## Next steps
1. Finish r5v4 04 / 05, run `scripts/r5/r5_publish.sh v1 r5v4` (FORCE_WATCH=1 if the gate still fails) -> WATCH for the user's ears + handoff F/audio-r5-v1.
2. Unit joints: a minimum gap / no speed-up on the last short part when the next unit starts within 0.1 s (02 WER).
3. Tempo axis: try --fill-slowdown 0.95 with dur-translate on (the candidates now fill by words, the drawl floor may no longer be needed).
4. Re-check the pause-logic INSIDE classifier; cap dub-over-silence (weight 0.7 -> 1.0 for runs >= 1 s).
5. Speed levers above; target <= 150 s/video-min for audio.
Code: branch experiment/r5-audio (local commits; push blocked by the missing deploy key -> git bundle in outbox/r5-audio.bundle).
