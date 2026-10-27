# Audio regression gate r6_audio_v1 (commit 4fb29c0): NEW `r6v8_14b` vs PREVIOUS `pb1` (+ stem axes from `pb1`)

Tolerance = the fixed per-axis tolerance + |published previous - its reproduction by the same code on this server| (mix axes), i.e. a change is
"clearly worse" only beyond what re-running the previous version itself produces. Stem axes (WER, cuts, joins, continuity, names) compare with the reproduction.

```
r4_audio r6v8_14b commit ebb00d5 AUDIO_EXTRA='--ne-guard on --pause-boundary-only on --dur-translate on --qwen-size 14b --dur-cost r6 --fill-slowdown 0.95 --unit-gap 0.15 --dur-cost r6c --dur-gap-qa-cost on --close-inside-pauses on' REMIX_ARGS='--src-duck-db -6 --unit-level on --gap-bed-cap-db 10' REUSE=''
peak VRAM MiB  18537
```

## 01

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.001 | 0.001 | 0.001 | 0.1 |  |
| LUFS | -16.0 | -16.0 | -16.0 | 1.0 |  |
| true peak dBTP | -2.8 | -2.9 | -2.8 | -1.0 |  |
| residue under dub (dB rel. voice) | -28.96 | -29.2 | -28.96 | 1.5 |  |
| residue exposed, dub silent (dB rel. voice) | -30.83 | -31.32 | -30.83 | 1.5 |  |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | -0.73 | -0.8 | -0.73 | 1.0 |  |
| bed p95 in pauses vs source (dB) | -0.27 | -0.25 | -0.27 | 1.0 |  |
| level jump at dub-pause onsets p95 (dB) |  |  | None | 1.5 |  |
| pause distribution W1 vs source (s) | 0.153 | 0.256 | 0.153 | 0.1 | WORSE |
| dub pauses > 1 s (source 0) | 1 | 1 | 1 | 1 |  |
| source speaks, dub silent (s) | 7.2 | 6.35 | 7.2 | 2.0 |  |
| speaker on screen talks, dub silent, runs >= 1 s (s) | 0 | 0 | 0 | 1.0 |  |
| longest dub gap beyond the source's own pause there (s) | 0.8 | 0.7 | 0.8 | 0.5 |  |
| dub gaps > 1 s longer than the source pause (count) | 0 | 0 | 0 | 0 |  |
| dub speaks, source silent (s) | 2.45 | 2.55 | 2.45 | 2.0 |  |
| tempo outside 0.95-1.05 (s of dub) | 51.92 | 0.0 |  | 1.0 | better |
| faithfulness: mean Qwen judge score (1-5) | 4.731 | 4.808 |  | 0.1 |  |
| faithfulness: units scored < 4 | 3 | 1 |  | 0 | better |
| WER final mix | 0.0855 | 0.0878 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 1 | 1 |  | 1 |  |
| joins energy jump median (dB) | 10.48 | 33.46 |  | 0 |  |
| joins F0 excess over the uncut take (st) | 0.41 | 0.44 |  | 0.5 |  |
| joins energy excess over the uncut take (dB) | -0.36 | 1.45 |  | 1.5 | WORSE |
| pauses inside an English phrase (count) | 3 | 1 |  | 0 | better |
| pauses inside an English phrase (s) | 0.87 | 0.18 |  | 0.3 | better |
| sentence-to-sentence level jump p90 (dB) | 2.23 | 0.66 |  | 1.0 | better |
| sentence-to-sentence F0 jump p90 (st) | 1.98 | 2.86 |  | 1.0 |  |
| named entities misheard | 1 | 0 |  | 0 | better |

## 02

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.024 | 0.024 | 0.024 | 0.1 |  |
| LUFS | -16.1 | -16.2 | -16.1 | 1.0 |  |
| true peak dBTP | -1.3 | -1.3 | -1.3 | -1.0 |  |
| residue under dub (dB rel. voice) | -22.54 | -24.02 | -22.54 | 1.5 |  |
| residue exposed, dub silent (dB rel. voice) | -35.15 | -32.72 | -35.15 | 1.5 | WORSE |
| bed speech heard (VAD, s) | 0.3 | 0.15 | 0.3 | 1.0 |  |
| bed in pauses vs source (dB) | -0.6 | -0.79 | -0.6 | 1.0 |  |
| bed p95 in pauses vs source (dB) | -0.82 | -1.12 | -0.82 | 1.0 |  |
| level jump at dub-pause onsets p95 (dB) | 21.08 | -1.25 | 21.08 | 1.5 | better |
| pause distribution W1 vs source (s) | 0.388 | 0.269 | 0.388 | 0.1 | better |
| dub pauses > 1 s (source 15) | 13 | 16 | 13 | 1 | WORSE |
| source speaks, dub silent (s) | 19.1 | 14.8 | 19.1 | 2.0 | better |
| speaker on screen talks, dub silent, runs >= 1 s (s) | 0 | 0 | 0 | 1.0 |  |
| longest dub gap beyond the source's own pause there (s) | 2.6 | 2.5 | 2.6 | 0.5 |  |
| dub gaps > 1 s longer than the source pause (count) | 7 | 3 | 7 | 0 | better |
| dub speaks, source silent (s) | 17.3 | 11.65 | 17.3 | 2.0 | better |
| tempo outside 0.95-1.05 (s of dub) | 50.66 | 4.64 |  | 1.0 | better |
| faithfulness: mean Qwen judge score (1-5) | 4.353 | 4.545 |  | 0.1 | better |
| faithfulness: units scored < 4 | 4 | 2 |  | 0 | better |
| WER final mix | 0.0314 | 0.0272 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 1 | 3 |  | 1 | WORSE |
| joins energy jump median (dB) | 18.03 | 30.63 |  | 0 |  |
| joins F0 excess over the uncut take (st) | 0.11 | -0.03 |  | 0.5 |  |
| joins energy excess over the uncut take (dB) | 0.06 | 0.21 |  | 1.5 |  |
| pauses inside an English phrase (count) | 2 | 5 |  | 0 | WORSE |
| pauses inside an English phrase (s) | 0.78 | 1.35 |  | 0.3 | WORSE |
| sentence-to-sentence level jump p90 (dB) | 5.4 | 2.61 |  | 1.0 | better |
| sentence-to-sentence F0 jump p90 (st) | 1.53 | 2.32 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## 03

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.025 | 0.025 | 0.025 | 0.1 |  |
| LUFS | -16.2 | -16.1 | -16.2 | 1.0 |  |
| true peak dBTP | -1.4 | -1.2 | -1.4 | -1.0 |  |
| residue under dub (dB rel. voice) | -26.85 | -24.59 | -26.85 | 1.5 | WORSE |
| residue exposed, dub silent (dB rel. voice) | -23.4 | -28.18 | -23.4 | 1.5 | better |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | -0.83 | -1.03 | -0.83 | 1.0 |  |
| bed p95 in pauses vs source (dB) | -0.05 | -0.15 | -0.05 | 1.0 |  |
| level jump at dub-pause onsets p95 (dB) | 0.14 | 0.38 | 0.14 | 1.5 |  |
| pause distribution W1 vs source (s) | 0.65 | 0.224 | 0.65 | 0.1 | better |
| dub pauses > 1 s (source 10) | 14 | 11 | 14 | 1 | better |
| source speaks, dub silent (s) | 19.1 | 10.15 | 19.1 | 2.0 | better |
| speaker on screen talks, dub silent, runs >= 1 s (s) | 12.4 | 1.5 | 12.4 | 1.0 | better |
| longest dub gap beyond the source's own pause there (s) | 3.3 | 1.6 | 3.3 | 0.5 | better |
| dub gaps > 1 s longer than the source pause (count) | 8 | 3 | 8 | 0 | better |
| dub speaks, source silent (s) | 10.7 | 11.1 | 10.7 | 2.0 |  |
| tempo outside 0.95-1.05 (s of dub) | 46.38 | 4.31 |  | 1.0 | better |
| faithfulness: mean Qwen judge score (1-5) | 4.9 | 4.545 |  | 0.1 | WORSE |
| faithfulness: units scored < 4 | 0 | 0 |  | 0 |  |
| WER final mix | 0.0175 | 0.0245 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 1 |  | 0 | WORSE |
| joins F0 jump > 3 st | 5 | 5 |  | 1 |  |
| joins energy jump median (dB) | 13.39 | 8.92 |  | 0 |  |
| joins F0 excess over the uncut take (st) | -0.05 | 0.04 |  | 0.5 |  |
| joins energy excess over the uncut take (dB) | 0.08 | 0.08 |  | 1.5 |  |
| pauses inside an English phrase (count) | 0 | 0 |  | 0 |  |
| pauses inside an English phrase (s) | 0 | 0 |  | 0.3 |  |
| sentence-to-sentence level jump p90 (dB) | 2.31 | 1.43 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 1.25 | 1.63 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## 04

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.025 | 0.025 | 0.025 | 0.1 |  |
| LUFS | -16.0 | -16.1 | -16.0 | 1.0 |  |
| true peak dBTP | -3.0 | -3.3 | -3.0 | -1.0 |  |
| residue under dub (dB rel. voice) | -26.58 | -25.32 | -26.58 | 1.5 |  |
| residue exposed, dub silent (dB rel. voice) | -30.14 | -30.83 | -30.14 | 1.5 |  |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | -1.25 | -1.23 | -1.25 | 1.0 |  |
| bed p95 in pauses vs source (dB) | -0.4 | -0.71 | -0.4 | 1.0 |  |
| level jump at dub-pause onsets p95 (dB) | 2.58 | 4.9 | 2.58 | 1.5 | WORSE |
| pause distribution W1 vs source (s) | 1.329 | 0.723 | 1.329 | 0.1 | better |
| dub pauses > 1 s (source 9) | 7 | 7 | 7 | 1 |  |
| source speaks, dub silent (s) | 12.2 | 11.1 | 12.2 | 2.0 |  |
| speaker on screen talks, dub silent, runs >= 1 s (s) | 11.1 | 7.0 | 11.1 | 1.0 | better |
| longest dub gap beyond the source's own pause there (s) | 6.6 | 3.85 | 6.6 | 0.5 | better |
| dub gaps > 1 s longer than the source pause (count) | 4 | 4 | 4 | 0 |  |
| dub speaks, source silent (s) | 8.45 | 10.45 | 8.45 | 2.0 |  |
| tempo outside 0.95-1.05 (s of dub) | 23.63 | 4.25 |  | 1.0 | better |
| faithfulness: mean Qwen judge score (1-5) | 4.5 | 5.0 |  | 0.1 | better |
| faithfulness: units scored < 4 | 1 | 0 |  | 0 | better |
| WER final mix | 0.0241 | 0.0449 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 1 | 3 |  | 1 | WORSE |
| joins energy jump median (dB) | 8.97 | 11.41 |  | 0 |  |
| joins F0 excess over the uncut take (st) | 0.0 | 0.0 |  | 0.5 |  |
| joins energy excess over the uncut take (dB) | 0.0 | 0.16 |  | 1.5 |  |
| pauses inside an English phrase (count) | 0 | 0 |  | 0 |  |
| pauses inside an English phrase (s) | 0 | 0 |  | 0.3 |  |
| sentence-to-sentence level jump p90 (dB) | 1.27 | 0.5 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 1.5 | 0.77 |  | 1.0 |  |
| named entities misheard | 0 | 0 |  | 0 |  |

## 05

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.001 | 0.001 | 0.001 | 0.1 |  |
| LUFS | -16.0 | -16.0 | -16.0 | 1.0 |  |
| true peak dBTP | -1.5 | -1.4 | -1.5 | -1.0 |  |
| residue under dub (dB rel. voice) | -18.79 | -19.31 | -18.79 | 1.5 |  |
| residue exposed, dub silent (dB rel. voice) | -23.42 | -22.45 | -23.42 | 1.5 |  |
| bed speech heard (VAD, s) | 1.2 | 0.5 | 1.2 | 1.0 |  |
| bed in pauses vs source (dB) | -0.38 | -0.6 | -0.38 | 1.0 |  |
| bed p95 in pauses vs source (dB) | -0.25 | -0.22 | -0.25 | 1.0 |  |
| level jump at dub-pause onsets p95 (dB) | 3.46 | 4.74 | 3.46 | 1.5 |  |
| pause distribution W1 vs source (s) | 1.413 | 0.688 | 1.413 | 0.1 | better |
| dub pauses > 1 s (source 10) | 4 | 7 | 4 | 1 | WORSE |
| source speaks, dub silent (s) | 11.55 | 12.65 | 11.55 | 2.0 |  |
| speaker on screen talks, dub silent, runs >= 1 s (s) | 7.35 | 9.0 | 7.35 | 1.0 | WORSE |
| longest dub gap beyond the source's own pause there (s) | 4.4 | 2.35 | 4.4 | 0.5 | better |
| dub gaps > 1 s longer than the source pause (count) | 3 | 4 | 3 | 0 | WORSE |
| dub speaks, source silent (s) | 8.2 | 9.25 | 8.2 | 2.0 |  |
| tempo outside 0.95-1.05 (s of dub) | 26.49 | 0.0 |  | 1.0 | better |
| faithfulness: mean Qwen judge score (1-5) | 4.429 | 4.286 |  | 0.1 | WORSE |
| faithfulness: units scored < 4 | 2 | 2 |  | 0 |  |
| WER final mix | 0.05 | 0.0366 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 0 | 1 |  | 1 |  |
| joins energy jump median (dB) |  | 16.57 |  | 0 |  |
| joins F0 excess over the uncut take (st) |  | 0.01 |  | 0.5 |  |
| joins energy excess over the uncut take (dB) |  | 0.03 |  | 1.5 |  |
| pauses inside an English phrase (count) | 0 | 0 |  | 0 |  |
| pauses inside an English phrase (s) | 0 | 0 |  | 0.3 |  |
| sentence-to-sentence level jump p90 (dB) | 1.57 | 0.5 |  | 1.0 | better |
| sentence-to-sentence F0 jump p90 (st) | 6.18 | 3.28 |  | 1.0 | better |
| named entities misheard |  |  |  | 0 |  |

## Verdict

**NO PUBLISH:**

- 01: pause distribution W1 vs source (s) 0.153 -> 0.256
- 01: joins energy excess over the uncut take (dB) -0.36 -> 1.45
- 02: residue exposed, dub silent (dB rel. voice) -35.15 -> -32.72
- 02: dub pauses > 1 s (source 15) 13 -> 16
- 02: joins F0 jump > 3 st 1 -> 3
- 02: pauses inside an English phrase (count) 2 -> 5
- 02: pauses inside an English phrase (s) 0.78 -> 1.35
- 03: residue under dub (dB rel. voice) -26.85 -> -24.59
- 03: faithfulness: mean Qwen judge score (1-5) 4.9 -> 4.545
- 03: hard cuts (stop/restart) 0 -> 1
- 04: level jump at dub-pause onsets p95 (dB) 2.58 -> 4.9
- 04: joins F0 jump > 3 st 1 -> 3
- 05: dub pauses > 1 s (source 10) 4 -> 7
- 05: speaker on screen talks, dub silent, runs >= 1 s (s) 7.35 -> 9.0
- 05: dub gaps > 1 s longer than the source pause (count) 3 -> 4
- 05: faithfulness: mean Qwen judge score (1-5) 4.429 -> 4.286
