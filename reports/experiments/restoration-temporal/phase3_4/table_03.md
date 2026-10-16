| variant | licence | mouth resid | mouth sharp | lower-face sharp | LS-seam jump (d1) | seams > 1.5x | switch resid / jerk | upper resid | frozen-mouth 24s / 35s (rank) | SyncNet mix / dub-only | post-proc s/min | edge px | fallback | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand5 (25 fps, CodeFormer) | NO (CodeFormer, buffalo_l) | 1.32 | 0.98 | 0.88 | 1.37 | 0.42 | 1.17 / 0.98 | 1.02 | 1.36 (22) / 1.47 (9) | 5.46 / 5.69 | - | - | - | rendered reference |
| clean0 (25 fps) | YES | 1.02 | 0.67 | 0.87 | 1.56 | 0.58 | 1.00 / 0.95 | 1.02 | 1.29 (26) / 1.46 (8) | 5.50 / 5.74 | - | - | - | track A recipe on this server |
| F0 source fps (clean0 composite, 12 px edge) | YES | 0.89 | 0.61 | 0.82 | 1.18 | 0.32 | 1.05 / 0.90 | 1.02 | 1.13 (49) / 1.13 (48) | 5.24 / 5.52 | 15.90 | 11.41 | 0 | audio md5 = baseline |
| F1 + C2 window-seam interpolation | YES | 0.86 | 0.60 | 0.81 | 0.98 | 0.13 | 1.05 / 0.90 | 1.02 | 1.14 (52) / 1.17 (42) | 5.29 / 5.53 | 20.00 | 11.41 | 0 | audio md5 = baseline |
| F2 + source skin texture | YES | 0.87 | 0.67 | 0.85 | 0.99 | 0.12 | 1.05 / 0.89 | 1.02 | 1.18 (40) / 1.17 (42) | 5.25 / 5.52 | 35.80 | 11.41 | 0 | audio md5 = baseline |
| F3 + lip texture | YES | 0.87 | 0.68 | 0.85 | 0.99 | 0.13 | 1.05 / 0.90 | 1.02 | 1.16 (45) / 1.17 (42) | 5.26 / 5.52 | 23.60 | 11.41 | 0 | audio md5 = baseline |
| OPT1 = F2 + still lower face in dub-silent stretches | YES | 0.87 | 0.67 | 0.85 | 0.99 | 0.13 | 1.05 / 0.90 | 1.02 | 1.17 (42) / 1.16 (47) | 5.23 / 5.51 | 26.80 | 11.41 | 0 | audio md5 = baseline |
| OPT2 = F2 + original pass-through there | YES | 0.91 | 0.69 | 0.86 | 0.98 | 0.12 | 1.05 / 0.89 | 1.02 | 1.09 (54) / 0.99 (74) | 5.18 / 5.43 | 30.50 | 11.41 | 0 | audio md5 = baseline |
| F0 with nearest-frame retiming | YES | 0.95 | 0.65 | 0.85 | 1.19 | 0.35 | 1.07 / 0.98 | 1.02 | 1.00 (62) / 1.06 (51) | 5.44 / 5.72 | 73.40 | 11.41 | 0 |  |
| F0 with nearest + rigid retiming | YES | 0.95 | 0.63 | 0.82 | 1.19 | 0.34 | 1.07 / 0.98 | 1.02 | 1.13 (44) / 1.06 (57) | 5.43 / 5.70 | 13.30 | 11.41 | 0 |  |
| F3 + sharpen 1.5 (flow retiming) | YES | 1.02 | 0.86 | 1.02 | 1.05 | 0.25 | 0.99 / 0.78 | 1.01 | - (-) / - (-) | 5.29 / 5.57 | 60.70 | 11.41 | 0 |  (frames 1380-1800 only) |
| F3 + sharpen 1.5 + lips 1.5 (flow retiming) | YES | 1.10 | 0.96 | 1.02 | 1.05 | 0.25 | 1.02 / 0.77 | 1.01 | - (-) / - (-) | 5.25 / 5.54 | 42.00 | 11.41 | 0 |  (frames 1380-1800 only) |
