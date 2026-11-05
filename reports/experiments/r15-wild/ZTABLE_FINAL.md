| video | arm | what | SyncNet | + lock SyncNet | blinks / 10 s | + lock blinks | flashes / strong | + lock flashes / strong | face GPU s/vmin |
|---|---|---|---|---|---|---|---|---|---|
| m03 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 8.774 | 8.681 | 17.78 | 4.69 | 69 / 22 | 50 / 18 | 357 |
| m03 | q2g15 | guidance 1.5 (prod 2.0) | 8.276 | 8.044 | 8.4 | 1.23 | 34 / 10 | 21 / 4 | 348 |
| m03 | q2g125 | guidance 1.25 | 7.647 | 7.425 | 3.95 | 0.74 | 16 / 4 | 13 / 2 | 350 |
| m03 | q2ts25 | LS_CFG_TSMOOTH=0.25 | 8.603 | 8.372 | 14.81 | 4.44 | 57 / 17 | 42 / 14 | 364 |
| m18 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 8.320 | 8.130 | 41.45 | 17.48 | – | – | 751 |
| m18 | q2v2 | prod + audit default fixes | 8.350 | 8.154 | 40.95 | 20.97 | – | – | 407 |
| m18 | q2as | LS_AFFINE_SMOOTH=9 | 8.260 | 8.039 | 39.95 | 17.48 | – | – | 389 |
| m18 | q2c8 | stride 8 continue | 8.588 | 8.423 | 40.95 | 18.97 | – | – | 467 |
| m18 | te2 | c8 + 2-noise latent ensemble | 8.534 | 8.310 | 40.45 | 17.48 | – | – | 739 |
| m18 | te2k | te2 + keyframes/2 | 8.542 | 8.236 | 40.95 | 18.97 | – | – | 425 |
| m18 | q2s4 | stride 4 continue | 8.443 | 8.291 | 40.95 | 20.97 | – | – | 750 |
| m18 | q2c8rf | c8 + fixed reference | 8.638 | 8.496 | 33.96 | 18.48 | – | – | 462 |
| m18 | s4c8 | student 4-step c8 | 7.583 | 7.327 | 24.97 | 12.98 | – | – | 362 |
| m18 | q2ct3 | CFG trunc 3/10 | 5.836 | 5.521 | 12.98 | 5.99 | – | – | 341 |
| m18 | q2ct5 | CFG trunc 5/10 | 6.053 | 5.708 | 12.98 | 5.99 | – | – | 347 |
| m18 | q2ct7 | CFG trunc 7/10 | 6.885 | 6.531 | 19.97 | 7.99 | – | – | 381 |
| m18 | q2g125 | guidance 1.25 | 6.624 | 6.310 | 22.47 | 6.49 | – | – | 409 |
| m18 | q2rs7 | CFG rescale 0.7 | 8.261 | 8.024 | 40.45 | 17.48 | – | – | 396 |
| m18 | q2rs10 | CFG rescale 1.0 | 8.378 | 8.167 | 40.45 | – | – | – | 360 |
| m18 | q2cfa | CF_ALIGN_FIT=lsq + CF_MOUTH=rigid | 8.359 | 8.090 | 35.95 | 17.48 | – | – | 409 |
| m18 | q2nb | AF_UPPER=nobrow | 8.268 | 8.014 | 40.45 | 18.97 | – | – | 380 |
| m18 | q2nr | LS_NOSE_RIGID=1 | 8.226 | 8.014 | 39.45 | 19.47 | – | – | 397 |
| old03 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 7.789 | 7.560 | 10.85 | -2.44 | – | – | 364 |
| old03 | q2as | LS_AFFINE_SMOOTH=9 | 7.823 | – | 9.85 | – | – | – | 357 |
