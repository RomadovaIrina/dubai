# envelope gate: `r8b5` vs the min / max of the sequential draws
Result: FAIL (2 axis rows)

## 01 (draws: r8e2e0b, r8s2d, r8s3)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.808 / 4.692 / 4.692 | 4.692 .. 4.808 | 4.808 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 1.3 / 0.9 / 1.9 | 0.9 .. 1.9 | 1.3 | inside |
| longest dub gap beyond the source pause (s) | 0.0 / 0.0 / 0.7 | 0.0 .. 0.7 | 0.0 | inside |
| pause W1 vs source (s) | 0.125 / 0.156 / 0.115 | 0.115 .. 0.156 | 0.115 | inside |
| dub speaks, source silent (s) | 2.96 / 2.38 / 2.66 | 2.38 .. 2.96 | 3.06 | inside |
| WER final mix | 0.0858 / 0.0646 / 0.0778 | 0.0646 .. 0.0858 | 0.0755 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 0 / 0 / 0 | 0 .. 0 | 1 | FAIL |
| dead air after words | 1 / 4 / 2 | 1 .. 4 | 2 | inside |

## 02 (draws: r8e2e0b, r8s2b, r8s3)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.636 / 4.545 / 4.727 | 4.545 .. 4.727 | 4.545 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 2.92 / 4.92 / 3.32 | 2.92 .. 4.92 | 3.02 | inside |
| longest dub gap beyond the source pause (s) | 1.35 / 2.3 / 0.7 | 0.7 .. 2.3 | 1.1 | inside |
| pause W1 vs source (s) | 0.279 / 0.297 / 0.156 | 0.156 .. 0.297 | 0.313 | inside |
| dub speaks, source silent (s) | 8.38 / 8.3 / 5.2 | 5.2 .. 8.38 | 6.58 | inside |
| WER final mix | 0.0311 / 0.0618 / 0.0337 | 0.0311 .. 0.0618 | 0.0055 | better |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 1 / 3 / 2 | 1 .. 3 | 2 | inside |
| dead air after words | 0 / 2 / 0 | 0 .. 2 | 1 | inside |

## 03 (draws: r8e2e0b, r8s2b, r8s3)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.545 / 4.455 / 4.545 | 4.455 .. 4.545 | 4.545 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 2.2 / 4.5 / 3.1 | 2.2 .. 4.5 | 3.7 | inside |
| longest dub gap beyond the source pause (s) | 0.95 / 1.85 / 2.05 | 0.95 .. 2.05 | 2.25 | inside |
| pause W1 vs source (s) | 0.163 / 0.127 / 0.226 | 0.127 .. 0.226 | 0.165 | inside |
| dub speaks, source silent (s) | 11.1 / 10.5 / 11.1 | 10.5 .. 11.1 | 9.82 | better |
| WER final mix | 0.0343 / 0.0296 / 0.0631 | 0.0296 .. 0.0631 | 0.0302 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 1 / 3 / 1 | 1 .. 3 | 2 | inside |
| dead air after words | 0 / 2 / 0 | 0 .. 2 | 0 | inside |

## 04 (draws: r8e2e0b, r8s2, r8s3)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 5.0 / 4.8 / 5.0 | 4.8 .. 5.0 | 4.8 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 0.8 / 3.2 / 1.4 | 0.8 .. 3.2 | 1.3 | inside |
| longest dub gap beyond the source pause (s) | 0.9 / 3.6 / 2.4 | 0.9 .. 3.6 | 0.8 | better |
| pause W1 vs source (s) | 0.302 / 0.658 / 0.407 | 0.302 .. 0.658 | 0.365 | inside |
| dub speaks, source silent (s) | 3.62 / 5.1 / 1.5 | 1.5 .. 5.1 | 2.94 | inside |
| WER final mix | 0.0337 / 0.0455 / 0.0588 | 0.0337 .. 0.0588 | 0.0476 | inside |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 0 / 0 / 1 | 0 .. 1 | 0 | inside |
| dead air after words | 1 / 0 / 0 | 0 .. 1 | 0 | inside |

## 05 (draws: r8e2e0b, r8s2, r8s3)

| axis | draws | envelope | new | verdict |
|---|---|---|---:|---|
| faith mean (judge 1-5) | 4.714 / 4.571 / 4.0 | 4.0 .. 4.714 | 4.429 | inside |
| source speaks, dub silent, spans > 0.3 s (s) | 3.6 / 4.1 / 4.3 | 3.6 .. 4.3 | 5.22 | FAIL |
| longest dub gap beyond the source pause (s) | 1.7 / 2.15 / 1.7 | 1.7 .. 2.15 | 2.35 | inside |
| pause W1 vs source (s) | 0.382 / 0.671 / 0.57 | 0.382 .. 0.671 | 0.68 | inside |
| dub speaks, source silent (s) | 4.78 / 5.98 / 3.68 | 3.68 .. 5.98 | 3.78 | inside |
| WER final mix | 0.0602 / 0.0114 / 0.025 | 0.0114 .. 0.0602 | 0.0 | better |
| cut-word ends | 0 / 0 / 0 | 0 .. 0 | 0 | inside |
| inserted mid-sentence gaps | 1 / 2 / 2 | 1 .. 2 | 2 | inside |
| dead air after words | 2 / 0 / 2 | 0 .. 2 | 1 | inside |

