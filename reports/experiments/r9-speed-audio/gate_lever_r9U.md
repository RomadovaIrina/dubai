# r9 audio gate: `r9U` vs the envelope of the baseline draws (both mixes)
Result: FAIL (2 axis rows) (m04: PASS; m01: FAIL 2)

## m04 (draws: r9_m04, r9b1_m04, r9b2_m04)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.333 / 4.333 / 4.333 | 4.333 .. 4.333 | 4.333 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 0.98 / 0.9 / 0.4 | 0.4 .. 0.98 | 0.8 | inside |
| longest dub gap beyond the source pause (s) | 0.0 / 1.0 / 0.0 | 0.0 .. 1.0 | 0.0 | inside |
| pause W1 vs source (s) | 0.075 / 1.1 / 0.05 | 0.05 .. 1.1 | 0.05 | inside |
| dub speaks, source silent (s) | 0.4 / 0.1 / 0.4 | 0.1 .. 0.4 | 0.4 | inside |
| WER final mix | 0.0 / 0.0175 / 0.0 | 0.0 .. 0.0175 | 0.0 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 1 / 0 / 0 | 0 .. 1 | 1 | inside |
| dead air after words | 2 / 1 / 1 | 1 .. 2 | 1 | inside |
| gap words, Spleeter mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 2 / 0 / 1 | 0 .. 2 | 1 | inside |
| gap words, htdemucs mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |

## m01 (draws: r9_m01, r9b1_m01, r9b2_m01)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.688 / 4.562 / 4.562 | 4.562 .. 4.688 | 4.75 | better |
| source speaks, dub silent, spans > 0.3 s (s) | 0.4 / 1.1 / 2.3 | 0.4 .. 2.3 | 1.5 | inside |
| longest dub gap beyond the source pause (s) | 0.0 / 0.0 / 1.1 | 0.0 .. 1.1 | 0.0 | inside |
| pause W1 vs source (s) | 0.058 / 0.114 / 0.151 | 0.058 .. 0.151 | 0.068 | inside |
| dub speaks, source silent (s) | 1.18 / 0.98 / 0.9 | 0.9 .. 1.18 | 0.98 | inside |
| WER final mix | 0.0138 / 0.01 / 0.0135 | 0.01 .. 0.0138 | 0.0257 | FAIL |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 0 / 0 / 0 | 0 .. 0 | 1 | FAIL |
| dead air after words | 5 / 8 / 9 | 5 .. 9 | 8 | inside |
| gap words, Spleeter mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 5 / 8 / 10 | 5 .. 10 | 9 | inside |
| gap words, htdemucs mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |

