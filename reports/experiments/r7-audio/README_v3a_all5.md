# r7_audio_v3a-all5: v3a is the default (the user's pick)

`scripts/r7/r7_audio.sh` now runs `retake.py` by default (`RETAKE=on`; `RETAKE=off` = v2):
- With `REUSE=<tag>`: retake reads those takes.
- Without REUSE: it first builds the fresh takes as `<tag>pre`.

Retake writes `<tag>src`, then the r7 chain runs on it. The faithfulness rows come from the takes' own metrics, else from `R7_PREV` (same text, so the same judge rows). The 16 kHz source is cut from `inputs/test_videos_raw`.

```
REUSE=r6v1src R7_PREV=r7v2 REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on" bash scripts/r7/r7_audio.sh r7v3a5
FORCE_WATCH=1 bash scripts/r7/r7_publish.sh v3a-all5 r7v3a5 r7v2      # + READY: the user picked v3a
```

The build took 303 s for the five videos (retake plus the chain). The full audio stage is about 138 s/video-min.

## Not the same as r7v3 on 02

r7v3's 02 was built at 16:29 with an earlier retake.py. That version rejected 02 u2, because its placement cost counted 2 cuts that weren't at a pb1 boundary. The committed retake.py (ba239f8, which also built 04 and 05) counts 1 cut and accepts it:

- old EN: "Here we have all sorts of different things, like a changing area, barrels over here, and some lounge chairs."
- new EN: "We have all sorts of things here / like a changing room area, barrels / and lounge chairs."
- judge score 4 → 5.

01, 03, 04 and 05 are sample-identical to r7v3.

## Gate vs r7v2 on 01-03: no regression

| video | dub audio | flagged gate rows | faithfulness (units < 4) | WER | pause W1 (s) | clicks in the final mix |
|---|---|---|---|---|---|---|
| 01 | sample-identical to r7v2 (the final mix is byte-identical) | none | 4.81 (1) | same | 0.261 | 45 → 45 |
| 02 | u2 re-taken | none | 4.55 (2) → **4.73 (1)** | 0.044 → **0.039** | 0.266 → 0.246 | 22 → 20 |
| 03 | sample-identical to r7v2 (the final mix is byte-identical) | none | 4.55 (0) | same | 0.306 | 21 → 21 |

Other rows on 02:
- source speaks, dub silent: 14.9 → 15.55 s (tolerance 2.0);
- dub speaks, source silent: 11.65 → 10.75 s;
- talks, dub silent: 0 s.

One new transient on 02 at 20.92 s sits 0.14 s inside a new-take piece, not at the splice (20.78 s, 10 ms fade). The total count still went down.

The gate's failing rows are the same 04/05 rows as in r7v3 (`r7_audio_v3a-all5_regression.md`); the user heard those and picked v3a.

## Handoff

- `handoff/out/F/audio-r7-v3a-all5/` (READY, changed_sentences.md).
- Outbox: `<id>_r7_audio_v3a-all5_WATCH_*` (AB = r7v2, then v3a).
