# r9 audio gate: `r9fast3` vs the envelope of the baseline draws (both mixes)
Result: FAIL (7 axis rows) (01: PASS; 02: FAIL 1; 03: FAIL 2; 04: PASS; 05: FAIL 1; m04: FAIL 3; m01: PASS)

## 01 (draws: r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5 + r9b0_01)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.808 / 4.692 / 4.692 / 4.808 / 4.808 | 4.692 .. 4.808 | 4.808 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 1.3 / 0.9 / 1.9 / 1.3 / 1.3 | 0.9 .. 1.9 | 0.5 | better |
| longest dub gap beyond the source pause (s) | 0.0 / 0.0 / 0.7 / 0.0 / 0.0 | 0.0 .. 0.7 | 0.0 | inside |
| pause W1 vs source (s) | 0.125 / 0.156 / 0.115 / 0.115 / 0.115 | 0.115 .. 0.156 | 0.128 | inside |
| dub speaks, source silent (s) | 2.96 / 2.38 / 2.66 / 3.06 / 3.06 | 2.38 .. 3.06 | 2.68 | inside |
| WER final mix | 0.0858 / 0.0646 / 0.0778 / 0.0755 / 0.0755 | 0.0646 .. 0.0858 | 0.0797 | inside |
| cut-word ends | 0.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 0.0 | 0 | inside |
| inserted mid-sentence gaps | 0.0 / 0.0 / 0.0 / 1.0 / 1 | 0.0 .. 1.0 | 0 | inside |
| dead air after words | 1.0 / 4.0 / 2.0 / 2.0 / 2 | 1.0 .. 4.0 | 2 | inside |
| gap words, Spleeter mix | 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 2 | 2 .. 2 | 2 | inside |
| gap words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |

## 02 (draws: r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5 + r9b0_02)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.636 / 4.545 / 4.727 / 4.545 / 4.636 | 4.545 .. 4.727 | 4.727 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 2.92 / 4.92 / 3.32 / 3.02 / 2.82 | 2.82 .. 4.92 | 3.02 | inside |
| longest dub gap beyond the source pause (s) | 1.35 / 2.3 / 0.7 / 1.1 / 1.1 | 0.7 .. 2.3 | 1.6 | inside |
| pause W1 vs source (s) | 0.279 / 0.297 / 0.156 / 0.313 / 0.257 | 0.156 .. 0.313 | 0.384 | FAIL |
| dub speaks, source silent (s) | 8.38 / 8.3 / 5.2 / 6.58 / 7.56 | 5.2 .. 8.38 | 7.38 | inside |
| WER final mix | 0.0311 / 0.0618 / 0.0337 / 0.0055 / 0.0053 | 0.0053 .. 0.0618 | 0.0324 | inside |
| cut-word ends | 0.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 0.0 | 0 | inside |
| inserted mid-sentence gaps | 1.0 / 3.0 / 2.0 / 2.0 / 1 | 1.0 .. 3.0 | 2 | inside |
| dead air after words | 0.0 / 2.0 / 0.0 / 1.0 / 1 | 0.0 .. 2.0 | 0 | inside |
| gap words, Spleeter mix | 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |
| gap words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |

## 03 (draws: r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5 + r9b0_03)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.545 / 4.455 / 4.545 / 4.545 / 4.545 | 4.455 .. 4.545 | 4.727 | better |
| source speaks, dub silent, spans > 0.3 s (s) | 2.2 / 4.5 / 3.1 / 3.7 / 3.7 | 2.2 .. 4.5 | 6.0 | FAIL |
| longest dub gap beyond the source pause (s) | 0.95 / 1.85 / 2.05 / 2.25 / 2.25 | 0.95 .. 2.25 | 2.5 | inside |
| pause W1 vs source (s) | 0.163 / 0.127 / 0.226 / 0.165 / 0.165 | 0.127 .. 0.226 | 0.118 | better |
| dub speaks, source silent (s) | 11.1 / 10.5 / 11.1 / 9.82 / 9.82 | 9.82 .. 11.1 | 8.7 | better |
| WER final mix | 0.0343 / 0.0296 / 0.0631 / 0.0302 / 0.0302 | 0.0296 .. 0.0631 | 0.02 | better |
| cut-word ends | 0.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 0.0 | 0 | inside |
| inserted mid-sentence gaps | 1.0 / 3.0 / 1.0 / 2.0 / 2 | 1.0 .. 3.0 | 3 | inside |
| dead air after words | 0.0 / 2.0 / 0.0 / 0.0 / 0 | 0.0 .. 2.0 | 1 | inside |
| gap words, Spleeter mix | 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 0 | 0 .. 0 | 1 | FAIL |
| gap words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |

## 04 (draws: r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5 + r9b0_04)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 5.0 / 4.8 / 5.0 / 4.8 / 5.0 | 4.8 .. 5.0 | 5.0 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 0.8 / 3.2 / 1.4 / 1.3 / 1.3 | 0.8 .. 3.2 | 0.8 | inside |
| longest dub gap beyond the source pause (s) | 0.9 / 3.6 / 2.4 / 0.8 / 0.8 | 0.8 .. 3.6 | 0.8 | inside |
| pause W1 vs source (s) | 0.302 / 0.658 / 0.407 / 0.365 / 0.365 | 0.302 .. 0.658 | 0.313 | inside |
| dub speaks, source silent (s) | 3.62 / 5.1 / 1.5 / 2.94 / 2.94 | 1.5 .. 5.1 | 4.9 | inside |
| WER final mix | 0.0337 / 0.0455 / 0.0588 / 0.0476 / 0.0357 | 0.0337 .. 0.0588 | 0.0538 | inside |
| cut-word ends | 0.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 0.0 | 0 | inside |
| inserted mid-sentence gaps | 0.0 / 0.0 / 1.0 / 0.0 / 0 | 0.0 .. 1.0 | 0 | inside |
| dead air after words | 1.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 1.0 | 0 | inside |
| gap words, Spleeter mix | 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |
| gap words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |

## 05 (draws: r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5 + r9b0_05)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.714 / 4.571 / 4.0 / 4.429 / 4.429 | 4.0 .. 4.714 | 4.571 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 3.6 / 4.1 / 4.3 / 5.22 / 5.22 | 3.6 .. 5.22 | 4.52 | inside |
| longest dub gap beyond the source pause (s) | 1.7 / 2.15 / 1.7 / 2.35 / 2.35 | 1.7 .. 2.35 | 2.75 | FAIL |
| pause W1 vs source (s) | 0.382 / 0.671 / 0.57 / 0.68 / 0.68 | 0.382 .. 0.68 | 0.666 | inside |
| dub speaks, source silent (s) | 4.78 / 5.98 / 3.68 / 3.78 / 3.78 | 3.68 .. 5.98 | 2.08 | better |
| WER final mix | 0.0602 / 0.0114 / 0.025 / 0.0 / 0.0 | 0.0 .. 0.0602 | 0.0135 | inside |
| cut-word ends | 0.0 / 0.0 / 0.0 / 0.0 / 0 | 0.0 .. 0.0 | 0 | inside |
| inserted mid-sentence gaps | 1.0 / 2.0 / 2.0 / 2.0 / 2 | 1.0 .. 2.0 | 1 | inside |
| dead air after words | 2.0 / 0.0 / 2.0 / 1.0 / 1 | 0.0 .. 2.0 | 0 | inside |
| gap words, Spleeter mix | 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 2 | 2 .. 2 | 1 | better |
| gap words, htdemucs mix | 0 | 0 .. 0 | 0 | inside |

## m04 (draws: r9_m04, r9b1_m04, r9b2_m04)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.333 / 4.333 / 4.333 | 4.333 .. 4.333 | 5.0 | better |
| source speaks, dub silent, spans > 0.3 s (s) | 0.98 / 0.9 / 0.4 | 0.4 .. 0.98 | 1.48 | FAIL |
| longest dub gap beyond the source pause (s) | 0.0 / 1.0 / 0.0 | 0.0 .. 1.0 | 0.0 | inside |
| pause W1 vs source (s) | 0.075 / 1.1 / 0.05 | 0.05 .. 1.1 | 0.175 | inside |
| dub speaks, source silent (s) | 0.4 / 0.1 / 0.4 | 0.1 .. 0.4 | 0.4 | inside |
| WER final mix | 0.0 / 0.0175 / 0.0 | 0.0 .. 0.0175 | 0.0 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 1 / 0 / 0 | 0 .. 1 | 1 | inside |
| dead air after words | 2 / 1 / 1 | 1 .. 2 | 3 | FAIL |
| gap words, Spleeter mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 2 / 0 / 1 | 0 .. 2 | 3 | FAIL |
| gap words, htdemucs mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |

## m01 (draws: r9_m01, r9b1_m01, r9b2_m01)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.688 / 4.562 / 4.562 | 4.562 .. 4.688 | 4.562 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 0.4 / 1.1 / 2.3 | 0.4 .. 2.3 | 0.9 | inside |
| longest dub gap beyond the source pause (s) | 0.0 / 0.0 / 1.1 | 0.0 .. 1.1 | 0.0 | inside |
| pause W1 vs source (s) | 0.058 / 0.114 / 0.151 | 0.058 .. 0.151 | 0.071 | inside |
| dub speaks, source silent (s) | 1.18 / 0.98 / 0.9 | 0.9 .. 1.18 | 1.18 | inside |
| WER final mix | 0.0138 / 0.01 / 0.0135 | 0.01 .. 0.0138 | 0.0167 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| dead air after words | 5 / 8 / 9 | 5 .. 9 | 4 | better |
| gap words, Spleeter mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| dead air after words, htdemucs mix | 5 / 8 / 10 | 5 .. 10 | 5 | inside |
| gap words, htdemucs mix | 0 / 0 / 0 | 0 .. 0 | 0 | inside |

