# Audio regression gate r4_audio_v1 (commit 005ea2a): NEW `cand2` vs PREVIOUS `ref_v4` (+ stem axes from `v4repro`)

Tolerance = the fixed per-axis tolerance + |published previous - its reproduction by the same code on this server| (mix axes), i.e. a change is
"clearly worse" only beyond what re-running the previous version itself produces. Stem axes (WER, cuts, joins, continuity, names) compare with the reproduction.

```
r4_audio cand2 commit 005ea2a AUDIO_EXTRA='' REMIX_ARGS='--src-duck-db -6' REUSE='ne3'
peak VRAM MiB  9086
```

## 01

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.001 | 0.001 | 0.001 | 0.1 |  |
| LUFS | -16.0 | -16.0 | -16.0 | 1.0 |  |
| true peak dBTP | -2.7 | -2.9 | -2.8 | -1.0 |  |
| residue under dub (dB rel. voice) | -29.77 | -28.78 | -28.77 | 2.5 |  |
| residue exposed, dub silent (dB rel. voice) | -30.92 | -30.64 | -29.56 | 2.86 |  |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | 0.19 | -0.7 | 0.07 | 1.12 |  |
| bed p95 in pauses vs source (dB) | -0.07 | -0.08 | 0.02 | 1.09 |  |
| level jump at dub-pause onsets p95 (dB) |  |  | None | 1.5 |  |
| pause distribution W1 vs source (s) | 0.236 | 0.188 | 0.188 | 0.148 |  |
| dub pauses > 1 s (source 0) | 1 | 1 | 1 | 1 |  |
| source speaks, dub silent (s) | 7.0 | 6.95 | 6.95 | 2.05 |  |
| dub speaks, source silent (s) | 2.95 | 2.6 | 2.6 | 2.35 |  |
| WER final mix | 0.0855 | 0.0855 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 2 | 2 |  | 1 |  |
| joins energy jump median (dB) | 11.97 | 11.97 |  | 1.0 |  |
| sentence-to-sentence level jump p90 (dB) | 2.33 | 2.33 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 1.59 | 1.59 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## 02

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.024 | 0.024 | 0.024 | 0.1 |  |
| LUFS | -16.3 | -16.1 | -16.1 | 1.0 |  |
| true peak dBTP | -1.4 | -1.4 | -1.4 | -1.0 |  |
| residue under dub (dB rel. voice) | -22.61 | -20.27 | -20.27 | 3.84 |  |
| residue exposed, dub silent (dB rel. voice) | -32.92 | -33.17 | -31.3 | 3.12 |  |
| bed speech heard (VAD, s) | 0.45 | 0.45 | 0.45 | 1.0 |  |
| bed in pauses vs source (dB) | -0.53 | -0.59 | -0.46 | 1.07 |  |
| bed p95 in pauses vs source (dB) | -1.0 | -1.24 | -0.98 | 1.02 |  |
| level jump at dub-pause onsets p95 (dB) | 2.15 | 3.51 | 3.5 | 2.85 |  |
| pause distribution W1 vs source (s) | 0.35 | 0.234 | 0.234 | 0.216 |  |
| dub pauses > 1 s (source 15) | 13 | 15 | 15 | 3 |  |
| source speaks, dub silent (s) | 13.25 | 11.95 | 11.95 | 3.3 |  |
| dub speaks, source silent (s) | 11.7 | 10.8 | 10.8 | 2.9 |  |
| WER final mix | 0.016 | 0.016 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 1 | 1 |  | 1 |  |
| joins energy jump median (dB) | 13.58 | 13.58 |  | 1.0 |  |
| sentence-to-sentence level jump p90 (dB) | 4.44 | 4.44 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 3.04 | 3.04 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## 03

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.025 | 0.025 | 0.025 | 0.1 |  |
| LUFS | -16.3 | -16.2 | -16.1 | 1.0 |  |
| true peak dBTP | -1.3 | -1.3 | -0.9 | -1.0 |  |
| residue under dub (dB rel. voice) | -28.04 | -26.73 | -26.7 | 2.84 |  |
| residue exposed, dub silent (dB rel. voice) | -20.11 | -22.46 | -21.25 | 2.64 |  |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | -0.11 | -0.93 | -0.11 | 1.0 |  |
| bed p95 in pauses vs source (dB) | 0.12 | -0.45 | 0.16 | 1.04 |  |
| level jump at dub-pause onsets p95 (dB) | 9.17 | 4.81 | 4.81 | 5.86 |  |
| pause distribution W1 vs source (s) | 0.284 | 0.279 | 0.279 | 0.105 |  |
| dub pauses > 1 s (source 10) | 14 | 10 | 10 | 5 |  |
| source speaks, dub silent (s) | 14.45 | 13.7 | 13.7 | 2.75 |  |
| dub speaks, source silent (s) | 7.0 | 6.5 | 6.5 | 2.5 |  |
| WER final mix | 0.0242 | 0.0242 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 4 | 4 |  | 1 |  |
| joins energy jump median (dB) | 10.605 | 10.605 |  | 1.0 |  |
| sentence-to-sentence level jump p90 (dB) | 2.5 | 2.5 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 2.24 | 2.24 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## 04

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.025 | 0.025 | 0.025 | 0.1 |  |
| LUFS | -16.0 | -16.1 | -16.0 | 1.0 |  |
| true peak dBTP | -3.1 | -2.8 | -3.2 | -1.0 |  |
| residue under dub (dB rel. voice) | -26.26 | -26.63 | -26.3 | 1.54 |  |
| residue exposed, dub silent (dB rel. voice) | -25.39 | -28.05 | -25.54 | 1.65 | better |
| bed speech heard (VAD, s) | 0.0 | 0.0 | 0.0 | 1.0 |  |
| bed in pauses vs source (dB) | -0.55 | -1.29 | -0.67 | 1.12 |  |
| bed p95 in pauses vs source (dB) | -0.57 | -1.1 | -0.83 | 1.26 |  |
| level jump at dub-pause onsets p95 (dB) | 2.51 | 1.96 | 2.36 | 1.65 |  |
| pause distribution W1 vs source (s) | 0.28 | 0.281 | 0.286 | 0.106 |  |
| dub pauses > 1 s (source 9) | 11 | 11 | 11 | 1 |  |
| source speaks, dub silent (s) | 5.65 | 5.8 | 5.9 | 2.25 |  |
| dub speaks, source silent (s) | 4.4 | 4.1 | 4.1 | 2.3 |  |
| WER final mix | 0.0337 | 0.0225 |  | 0.03 |  |
| hard cuts (stop/restart) | 1 | 1 |  | 0 |  |
| joins F0 jump > 3 st | 2 | 1 |  | 1 |  |
| joins energy jump median (dB) | 9.59 | 10.245 |  | 1.0 |  |
| sentence-to-sentence level jump p90 (dB) | 1.26 | 1.48 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 1.76 | 1.56 |  | 1.0 |  |
| named entities misheard | 1 | 0 |  | 0 | better |

## 05

| axis | previous | new | previous reproduced | tolerance | verdict |
|---|---:|---:|---:|---:|---|
| duration - source (s) | 0.001 | 0.001 | 0.001 | 0.1 |  |
| LUFS | -16.0 | -16.0 | -16.0 | 1.0 |  |
| true peak dBTP | -1.5 | -1.5 | -1.5 | -1.0 |  |
| residue under dub (dB rel. voice) | -19.2 | -20.89 | -20.88 | 3.18 |  |
| residue exposed, dub silent (dB rel. voice) | -18.59 | -22.91 | -17.63 | 2.46 | better |
| bed speech heard (VAD, s) | 1.3 | 1.05 | 1.15 | 1.15 |  |
| bed in pauses vs source (dB) | 0.67 | -1.15 | 0.02 | 1.65 | better |
| bed p95 in pauses vs source (dB) | 0.85 | -0.92 | -0.07 | 1.92 |  |
| level jump at dub-pause onsets p95 (dB) | 1.32 | -2.29 | -1.95 | 4.77 |  |
| pause distribution W1 vs source (s) | 0.681 | 1.006 | 1.006 | 0.425 |  |
| dub pauses > 1 s (source 10) | 7 | 4 | 4 | 4 |  |
| source speaks, dub silent (s) | 11.4 | 13.15 | 13.15 | 3.75 |  |
| dub speaks, source silent (s) | 9.4 | 10.6 | 10.6 | 3.2 |  |
| WER final mix | 0.0253 | 0.038 |  | 0.03 |  |
| hard cuts (stop/restart) | 0 | 0 |  | 0 |  |
| joins F0 jump > 3 st | 1 | 1 |  | 1 |  |
| joins energy jump median (dB) | 22.615 | 22.615 |  | 1.0 |  |
| sentence-to-sentence level jump p90 (dB) | 1.82 | 1.82 |  | 1.0 |  |
| sentence-to-sentence F0 jump p90 (st) | 3.47 | 3.47 |  | 1.0 |  |
| named entities misheard |  |  |  | 0 |  |

## Verdict

**PASS: no axis clearly worse on any video.**

