# r10 face gate rows (gate_cmp.py vs the r9 + fix baseline). NB g2/g3 out_stab rows were hit by the time-base artefact (fixed in 12552e8); g4/g5 are valid

## gate
| cand | verdict | | sync | flash strong | pressed s | frame gate | stick | slide | teeth | amp | occl | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old03S | PASS |  | 7.333 -> 7.045 | 55 -> 26 | 3.2 -> 3.07 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03R | PASS |  | 7.333 -> 7.304 | 55 -> 54 | 3.2 -> 3.07 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03RS | PASS |  | 7.333 -> 7.071 | 55 -> 34 | 3.2 -> 3.27 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03RSn | FAIL |  | 7.333 -> 6.63 | 55 -> 27 | 3.2 -> 3.17 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03S | FAIL |  | 7.55 -> 7.378 | 20 -> 8 | 0.63 -> 0.7 | PASS -> PASS | 0 -> 4 | 1 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03R | PASS |  | 7.55 -> 7.563 | 20 -> 21 | 0.63 -> 0.67 | PASS -> PASS | 0 -> 1 | 1 -> 1 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03RS | FAIL |  | 7.55 -> 7.386 | 20 -> 10 | 0.63 -> 0.67 | PASS -> PASS | 0 -> 4 | 1 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03RSn | FAIL |  | 7.55 -> 7.207 | 20 -> 10 | 0.63 -> 0.73 | PASS -> PASS | 0 -> 4 | 1 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18S | PASS |  | 8.038 -> 8.01 | 14 -> 10 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 1 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18R | PASS |  | 8.038 -> 8.143 | 14 -> 16 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18RS | PASS |  | 8.038 -> 8.086 | 14 -> 11 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18RSn | PASS |  | 8.038 -> 7.907 | 14 -> 9 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old05S | FAIL |  | 3.904 -> 3.765 | 11 -> 8 | 6.98 -> 7.05 | PASS -> PASS | 1 -> 1 | 4 -> 13 | 0 -> 0 | 1 -> 1 | 0 -> 1 | None -> None |
| old05R | PENDING |  | 3.904 -> None | 11 -> None | 6.98 -> None | PASS -> None | 1 -> None | 4 -> None | 0 -> None | 1 -> None | 0 -> None | None -> None |
| old05RS | PENDING |  | 3.904 -> None | 11 -> None | 6.98 -> None | PASS -> None | 1 -> None | 4 -> None | 0 -> None | 1 -> None | 0 -> None | None -> None |
| old05RSn | PENDING |  | 3.904 -> None | 11 -> None | 6.98 -> None | PASS -> None | 1 -> None | 4 -> None | 0 -> None | 1 -> None | 0 -> None | None -> None |

## gate3
| cand | verdict | | sync | flash strong | pressed s | frame gate | stick | slide | teeth | amp | occl | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old05P1 | FAIL |  | 3.904 -> 3.968 | 11 -> 12 | 6.98 -> 7.62 | PASS -> PASS | 1 -> 1 | 4 -> 17 | 0 -> 0 | 1 -> 0 | 0 -> 0 | None -> None |
| m03P1 | FAIL |  | 7.55 -> 7.579 | 20 -> 19 | 0.63 -> 0.67 | PASS -> PASS | 0 -> 4 | 1 -> 3 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18R | PASS |  | 8.038 -> 8.143 | 14 -> 16 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18P0 | PASS |  | 8.038 -> 8.15 | 14 -> 15 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18P1 | PASS |  | 8.038 -> 8.165 | 14 -> 15 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18P1C | PASS |  | 8.038 -> 8.01 | 14 -> 14 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |

## gate4
| cand | verdict | | sync | flash strong | pressed s | frame gate | stick | slide | teeth | amp | occl | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old05R | FAIL |  | 3.904 -> 3.953 | 11 -> 11 | 6.98 -> 7.02 | PASS -> PASS | 1 -> 3 | 4 -> 6 | 0 -> 0 | 1 -> 1 | 0 -> 0 | None -> None |
| old05R2 | FAIL |  | 3.904 -> 3.94 | 11 -> 10 | 6.98 -> 7.65 | PASS -> PASS | 1 -> 1 | 4 -> 3 | 0 -> 0 | 1 -> 1 | 0 -> 0 | None -> None |
| old05F1 | FAIL |  | 3.904 -> 3.899 | 11 -> 7 | 6.98 -> 7.79 | PASS -> PASS | 1 -> 2 | 4 -> 7 | 0 -> 0 | 1 -> 1 | 0 -> 1 | None -> None |
| old05F1C | FAIL |  | 3.904 -> 3.933 | 11 -> 6 | 6.98 -> 7.59 | PASS -> PASS | 1 -> 1 | 4 -> 8 | 0 -> 0 | 1 -> 1 | 0 -> 1 | None -> None |
| m18R | PASS |  | 8.038 -> 8.143 | 14 -> 16 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18F1 | PASS |  | 8.038 -> 8.086 | 14 -> 11 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m18F1C | PASS |  | 8.038 -> 7.896 | 14 -> 11 | 0.03 -> 0.03 | PASS -> PASS | 1 -> 0 | 2 -> 2 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03R | PASS |  | 7.333 -> 7.304 | 55 -> 54 | 3.2 -> 3.07 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03F1 | PASS |  | 7.333 -> 7.061 | 55 -> 34 | 3.2 -> 3.23 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| old03F1C | FAIL |  | 7.333 -> 6.975 | 55 -> 33 | 3.2 -> 3.07 | PASS -> PASS | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03R | PASS |  | 7.55 -> 7.563 | 20 -> 21 | 0.63 -> 0.67 | PASS -> PASS | 0 -> 1 | 1 -> 1 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03F1 | PASS |  | 7.55 -> 7.386 | 20 -> 10 | 0.63 -> 0.67 | PASS -> PASS | 0 -> 0 | 1 -> 1 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
| m03F1C | PASS |  | 7.55 -> 7.253 | 20 -> 11 | 0.63 -> 0.63 | PASS -> PASS | 0 -> 1 | 1 -> 1 | 0 -> 0 | 0 -> 0 | 0 -> 0 | None -> None |
