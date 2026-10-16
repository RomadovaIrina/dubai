# Agent C, P1: offline timing-policy A/B on 01 / 03 / 04 / 05 (2026-10-16)

Same raw Chatterbox takes, same translations, same burst plans as the cand5 baseline; only split → cut → placement → track changes
(`recut_offline.py`, CPU, no Qwen / TTS / GPU). Metrics: `dub_gaps.py` (source VAD on / dub VAD off, LatentSync frames with the baseline's LS
mask; the mask is gated by source ∪ dub speech and moves by ≤ 1 frame between policies), `pause_rhythm.py` (weak-boundary pauses),
`eval_cpu.py` (Whisper WER / CTC CER, CPU int8: compare policies with each other, not with the GPU table). Driver: `timing_ab.py`.
Measurement change vs P0: the dub-side Silero VAD keeps words ≥ 60 ms (default 250 ms dropped the isolated word "with" at 04 34.8 s as soon as
a policy shifted it, faking a 0.9 → 1.2 s gap); baseline restated: 01 8.6 → 8.4 s, 05 7.4 → 7.2 s, 03 / 04 unchanged.

## New layer-1 levers (opt-in, `recut_offline.py` flags; runner integration in P2)

* **spread** (`--spread pauses`, `scripts/quality/timing_spread.py`): a part shorter than its source burst no longer leaves ONE tail gap. The
  leftover is water-filled into a late start (≤ 0.2 s, minus any late start that already exists against the real source onset), pauses
  (≤ 0.25 s inserted, ≤ 0.35 s total silence with the TTS's own pause) at natural boundaries inside the part, and the tail; only as many
  pauses as needed to bring the tail under 0.3 s. Allowed boundaries: punctuation; before a clause word (and, but, so, that, which, …);
  a content word followed by a function word where Chatterbox itself already paused ≥ 0.12 s. Never after a function word / preposition /
  number, never inside a fixed phrase, never closer than 0.5 s of speech to another cut, never in voiced sound (the quietest 10 ms ≤ −25 dB
  re the speech level, ≤ −18 dB at punctuation; measured max −23 dB). Never past the source burst end. Tempo and fades are untouched.
* **flow 0.2** (existing `--sentence-flow-gap`, small value): source pauses < 0.2 s are VAD splits inside one breath, not cut points (03 23.8 /
  23.9 s: a 0.1 s split gave "the start" its own 1.1 s burst → the reviewed 0.5 s gap at 24.5 s).
* `--burst-max-lead 0.8`: **no effect** (lead is only used against speed-ups). Dropped.

## Results, sum over 01 / 03 / 04 / 05

| policy | source w/o dub s | gaps ≥ 0.35 s n / s | > 0.5 s n / s | > 1.0 s n / s | longest s | **visible LS frames** (in gaps ≥ 0.35 s) | LS frames dub-silent VAD / RMS | weak-boundary pauses | inserted pauses n / s | late start s | parts < x0.9 |
|---|---:|---|---|---|---:|---:|---|---:|---|---:|---:|
| cand5 (ref) | 32.8 | 34 / 25.2 | 21 / 19.6 | 6 / 8.4 | 2.0 | 525 | 652 / 1094 | 4 | – | – | 0 of 108 |
| no slow-down (floor 1.0) | 42.2 | 44 / 35.6 | 26 / 27.6 | 10 / 15.1 | 2.7 | 698 | 806 / 1217 | 5 | – | – | 0 |
| floor 0.875 | 30.6 | 32 / 22.6 | 18 / 16.6 | 5 / 6.8 | 1.8 | 493 | 612 / 1063 | 4 | – | – | 53 of 108 |
| floor 0.85 | 28.0 | 26 / 19.3 | 15 / 14.3 | 4 / 5.4 | 1.6 | 445 | 570 / 1029 | 4 | – | – | 53 of 108 |
| flow 0.2 | 32.6 | 32 / 24.5 | 21 / 19.7 | 6 / 8.4 | 2.0 | 518 | 647 / 1093 | 2 | – | – | 0 |
| spread: late start only | 32.6 | 27 / 18.9 | 17 / 14.5 | 3 / 4.3 | 1.8 | 392 | 657 / 1088 | 4 | – | 5.3 | 0 |
| spread: late start + pauses | 31.8 | 25 / 17.2 | 16 / 13.2 | 3 / 3.8 | 1.4 | 377 | 645 / 1082 | 4 | 7 / 1.34 | 5.3 | 0 |
| **L1 = flow 0.2 + spread** | 31.4 | 25 / 17.4 | 17 / 13.8 | 3 / 3.8 | 1.4 | **382** | 633 / 1072 | **2** | 8 / 1.52 | 4.9 | 0 of 100 |
| L1, pause 0.3 / late start 0.25 | 31.1 | 25 / 16.6 | 14 / 11.6 | 3 / 3.7 | 1.4 | 362 | 626 / 1070 | 2 | 8 / 1.53 | 5.8 | 0 |
| flow 0.3 + spread | 31.7 | 26 / 17.9 | 17 / 13.8 | 3 / 3.8 | 1.4 | 395 | 638 / 1076 | 1 | 8 / 1.52 | 4.9 | 0 |
| **L1 + floor 0.875** | 29.7 | 25 / 16.0 | 15 / 11.6 | 3 / 3.5 | 1.3 | **344** | 591 / 1047 | 2 | 6 / 1.20 | 4.5 | 53 of 100 |
| **L1 + floor 0.85** | 27.1 | 22 / 14.0 | 10 / 8.3 | 2 / 2.2 | 1.1 | **319** | 554 / 1011 | 2 | 5 / 0.99 | 4.1 | 53 of 100 |

Intelligibility (CPU Whisper int8 / wav2vec2 CTC; per video: ref → L1 → L1 + 0.875 → L1 + 0.85):

| video | cand5 WER / CTC CER / bad | L1 WER / CTC CER / bad | L1 + 0.875 WER / CTC CER / bad | L1 + 0.85 WER / CTC CER / bad |
|---|---|---|---|---|
| 01 | 0.019 / 0.051 / 0 | 0.022 / 0.05 / 0 | 0.022 / 0.051 / 0 | 0.022 / 0.052 / 0 |
| 03 | 0.009 / 0.043 / 0 | 0.009 / 0.04 / 0 | 0.009 / 0.041 / 0 | 0.009 / 0.038 / 0 |
| 04 | 0.009 / 0.048 / 0 | 0.009 / 0.042 / 0 | 0.009 / 0.051 / 0 | 0.009 / 0.058 / 0 |
| 05 | 0.012 / 0.171 / 1 | 0.012 / 0.171 / 1 | 0.0 / 0.191 / 0 | 0.012 / 0.202 / 1 |

01 WER 0.019 → 0.022 = one word in u13 ("wide-format" heard as "white format" after the flow merge re-cut that sentence); no unit
became bad (CTC CER ≥ 0.3) under any policy, no interjection, no missing final word.

## Reading

1. **Layer 1 reshapes the silence, it does not remove it** (source w/o dub 32.8 → 31.4 s), and that is the point: the visible part — lip-synced
   frames inside gaps ≥ 0.35 s — drops 525 → 382 (−27 %), gaps > 1 s 6 → 3 (8.4 → 3.8 s), the longest gap 2.0 → 1.4 s, weak-boundary pauses
   4 → 2, at x0.90 like today and with the WER unchanged. Spread does most of it; flow 0.2 removes the 24 s defect and two weak pauses.
   On 04 / 05 (2-5-word parts) spread only uses the late start: no internal boundary passes the rules.
2. **Layer 2 (the floor) is the only lever on the total**: every 0.025 of floor ≈ 2.6 s of source-without-dub over the four videos. L1 + 0.85
   gives −41 % visible frames (319), gaps > 0.5 s 21 → 10, longest 1.1 s. The price: 53 of 100 parts play below x0.9, and the CTC CER rises on
   04 (0.042 → 0.051 → 0.058) and 05 (0.171 → 0.191 → 0.202) while Whisper WER does not move — an early sign of drawn-out speech. Needs the
   human listening verdict before it can become a default.
3. **The reviewed 03 windows**: 24.5–25.0 s 0.50 s → **0.10 s** (fixed by flow 0.2). 34.3–35.5 s 1.20 s → 1.00 s (L1) → 0.80 s (L1 + 0.85),
   37.5–38.5 s 1.0 → 0.6 → 0.5 s: u04's English is 7.8 s for 11.2 s of Russian and its fit-up was rejected by the judge → layer 3 / video side.
   A 0.8 s late start at 27.8 s stays: the ASR has no word there (breath / hesitation before "Let's hope") — flagged as non-lexical for the
   video side.
4. Rejected: `--burst-max-lead` (no effect), larger pause / late-start caps (pause 0.3 / start 0.25: −20 frames more but a 0.25 s late start
   on every spread part), flow 0.3 (one weak pause less, +13 visible frames).

## Proposal (pending the listening verdict)

`--sentence-flow-gap 0.2` + `--spread pauses` (new) + `--fill-slowdown` 0.875 (or 0.85 if the clips do not sound drawn out). Everything else
as cand5. Listening clips: `/workspace/dub/outbox/audio-p1/` (README inside; original frames, dub voice only, no bed, lips NOT re-synced;
per window: original audio, cand5, L1, L1 + 0.875, L1 + 0.85; red bar = source talks while the dub is silent).

## Next (P2)
Runner integration of `--spread` (`align_units_burst` after `place_groups`, reproduced offline with `--check`), residual-gap JSON per video
for the video side (with the non-lexical flag), layer-3 design (burst-targeted literal fit-up) and its GPU request.
