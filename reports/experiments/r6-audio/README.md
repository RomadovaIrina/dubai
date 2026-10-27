# r6_audio_v1 (WATCH): duration-aware translation, round 6 (audio agent A, 2026-10-27)

**Verdict: WATCH, not gate-passing.** The gate against pb1 fails on 16 rows. Some are design trade-offs, one is a small-sample metric, and some are
real regressions; the classification is below. The user decides by ear. Handoff `handoff/out/F/audio-r6-v1/` has **no READY**.

Files (outbox): `<id>_r6_audio_v1_WATCH_over-original.mp4` (new audio over the original video), `<id>_r6_audio_v1_WATCH_AB_prev-then-new.mp4`
(the whole pb1 video, then the whole r6 video), `<id>_r6_audio_v1_WATCH_review.md` (one row per 10 s, plus every known moment from
`r4_audio_v1_known_issues.md` / `pb1_listen.md` re-measured), `r6_audio_v1_regression.md` (the full gate).

One command: `bash scripts/r6/r6_audio.sh <tag>` (Qwen2.5-14B Q5_K_M, Apache-2.0). Run tag `r6v8_14b`, code at 4fb29c0.

## What it fixes (r6_audio_v1 vs pb1, same code and metrics on this server; pb1 reproduced twice, bit for bit)

| video | src talks / dub silent (s) | face on screen talks, dub silent >= 1 s (s) | dub gaps > 1 s beyond src pause | tempo outside 0.95-1.05 (s) | sped up (s) | WER | faith (1-5) |
|---|---|---|---|---|---|---|---|
| 01 | 7.2 -> **6.35** | 0 -> 0 | 0 -> 0 | 51.9 -> **0** | 10.9 -> **0** | .086 -> .088 | 4.73 -> 4.81 |
| 02 | 19.1 -> **14.8** | 0 -> 0 | 7 -> **3** | 50.7 -> **4.6** | 9.95 -> 4.6 | .031 -> .027 | 4.35 -> 4.55 |
| 03 | 19.1 -> **10.15** | 12.4 -> **1.5** | 8 -> **3** | 46.4 -> **4.3** | 0 -> 4.3 | .018 -> .025 | 4.90 -> 4.55 |
| 04 | 12.2 -> **11.1** | 11.1 -> **7.0** | 4 -> 4 | 23.6 -> **4.25** | 0 -> 4.25 | .024 -> .045 | 4.50 -> 5.00 |
| 05 | 11.55 -> 12.65 | 7.35 -> 9.0 | 3 -> 4 | 26.5 -> **0** | 0 -> 0 | .050 -> .037 | 4.43 -> 4.29 |

The r5 failing axes:
1. **Dub over source silence** (03 +3.4 s in r5v4): 03 10.7 -> 11.1, inside tolerance. The dry-run cost now charges it at 1.0.
2. **Tempo** (03 +2.4 s in r5v4): 46-52 s -> <= 4.6 s on every video. No 0.875 drawl, and the dry run sees the real part lengths, which killed
   the unplanned x1.15 speed-ups.
3. **Pauses inside a phrase**: most "INSIDE" pauses in r4 / r5 were a metric bug (Whisper word times shifted by one word; see "metric fixes").
   Re-scored: pb1 3/2/0/0/0, r6 1/5/0/0/0. Real Chatterbox hesitations are now shortened in the take (`--close-inside-pauses`).
4. **WER 02** (.016 -> .073 in r5v4): .031 -> .027 (`--unit-gap 0.15`, sentences no longer run together).

## Gate rows that still fail (16), classified

- **By design** (the dub now pauses where the speaker pauses):
  - dub pauses > 1 s: 02 13 -> 16 (the source has 15); 05 4 -> 7 (the source has 10);
  - joins F0 jump > 3 st: 02 1 -> 3, 04 1 -> 3. These are more cuts at real source pauses; the F0 excess over the uncut take is unchanged (~0),
    so these are natural pitch resets after a comma, not splices.
- **Small-sample metric:** 04 level jump at dub-pause onsets p95, 2.58 -> 4.9. The p95 is over only 3 onsets; at 4.1 s the dub ends exactly
  where the source speech ends and the -6 dB source-speech bed duck releases.
- **Real, small:**
  - 02 residue exposed -35.2 -> -32.7 dB (tolerance 1.5). Was -20.9 before `--gap-bed-cap-db`: one leaked syllable at 12.1 s in a sentence gap.
  - 02 pauses inside a phrase 2 -> 5 (0.78 -> 1.35 s; Chatterbox hesitations of 0.13-0.45 s that are not silent enough to close).
  - 03 residue under dub -26.9 -> -24.6 dB.
  - 03 one hard cut at 19.9 s (inside the take "Today is Friday, | August twenty-eight").
  - 01 joins energy excess +1.8 dB; 01 pause distribution W1 0.15 -> 0.26 s.
- **Real, trade-off:**
  - 03 faith 4.90 -> 4.55. Duration candidates paraphrase; the judge gives 4 = "same facts, a nuance changed".
  - 05 faith 4.43 -> 4.29.
  - 05 face talks / dub silent 7.35 -> 9.0 s and gaps 3 -> 4. 05 is a slow speaker (3.5 syl/s) and the English is ~30 % shorter. r6 spreads the
    silence over the speaker's own pauses (several 1.5-2 s gaps), where pb1 drew everything out to x0.875 and left one 3 s gap. A 0.9 fill
    (r6v5) did not fix it.

## Qwen 14B vs 7B (same r6 config, all 5 videos)
**14B chosen.** Source-talks / dub-silent: 02 14.8 vs 19.35 s, 03 9.5 vs 17.5 s; 03 face 0 vs 8.2 s. Faith is mixed (7B is higher on 03 / 05, 14B
on 01 / 02). 7B has 04 WER .096. 7B is 14 % faster: 1249 vs 1450 s for 5 videos (409 s of video).

**Speed:** 14B is ~212 s per video-minute for the audio stage, including remix and metrics, at PAR=1 (pb1: 121 s at PAR=2). Peak VRAM 18.5 GB.

## Changes (branch experiment/r6-audio)
- `--dur-cost r6c`: the dry run charges the gate's own axes (every second outside 0.95-1.05, dub over silence x1.0) on the parts as `cut_parts`
  cuts them (+0.14 s per part vs CTC spans). Variants r6d / r6e were tried and rejected; see `STATUS_LOG.md`.
- `--dur-gap-qa-cost on`: a gap-QA re-seed may not raise the dry-run cost by > 0.3 s (01 u12: 7.18 -> 7.96 s take, x1.15 on 7 s).
- `--fill-slowdown 0.95`, `--unit-gap 0.15`, `--close-inside-pauses on` (silent run 30 dB below the take's p90, kept 50 ms, 10 ms crossfade).
- `r3_remix --unit-level on` (sentence level jump p90 now below pb1 on all 5) and `--gap-bed-cap-db 10`.
- Metric fixes, applied to every tag:
  - `r4_pause_logic` classifies part boundaries from the placed part texts. Inside a part it uses Whisper, with the edge-word rule and
    exact-zero (pipeline) pauses; the r4 pick scored pb1 02 at 21/21 INSIDE.
  - `r5_faith` judge token budget, retry, and regex fallback (a truncated or looping answer had scored 0).
- Tools: `scripts/r6/` (`r6_audio.sh`, `r6_publish.sh`, `r6_table.py`, `r6_repause.py`, chains).

## Next
1. 05 slow speaker: Chatterbox pace below cfg 0.2, or the judge accepting "fuller wording" more readily for slow speech.
2. Close the remaining non-silent hesitations (02), e.g. by a re-seed only for the unit that has one.
3. Faith: min judge score 5 for paraphrase candidates when the plain cost is close.
