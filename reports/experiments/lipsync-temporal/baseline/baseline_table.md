| metric | 01 | 02 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|---:|
| valid | PASS | PASS | PASS | PASS | PASS |
| wall s | 695.527 | 536.761 | 1210.483 | 621.921 | 660.751 |
| audio s | 543.01 | 455.59 | 570.08 | 291.84 | 365.56 |
| LatentSync / CodeFormer s | 62.305 / 7.892 | 0.0 / 0.0 | 503.785 / 61.88 | 243.676 / 33.603 | 212.77 / 27.898 |
| LS frames | 260 | None | 2025 | 1074 | 930 |
| peak VRAM MiB (nvidia-smi) | 22471 | 8347 | 23477 | 22451 | 22479 |
| s per video-minute | 410.5 | 290.5 | 804.0 | 692.3 | 757.1 |
| SyncNet final mix / AV | 7.89 / 0 | N/A (no face) | 5.5 / 0 | 3.71 / 0 | 2.99 / 0 |
| SyncNet dub-only / AV | 8.02 / 0 | – / – | 5.73 / 0 | 3.85 / 0 | 3.42 / 0 |
| SyncNet original | 5.36 | – | 4.87 | 3.28 | 1.72 |
| Whisper WER | 0.033 | 0.064 | 0.009 | 0.009 | 0.011 |
| CTC CER | 0.07 | 0.141 | 0.043 | 0.048 | 0.126 |
| bad units | 1 | 1 | 0 | 0 | 1 |
| interjections | 1 | 0 | 0 | 0 | 0 |
| final word missing | 0 | 0 | 0 | 0 | 0 |
| atempo max / >1.15 | 1.143 / 0 | 1.15 / 0 | 1.15 / 0 | 1.025 / 0 | 1.0 / 0 |
| orig speech w/o dub s / IoU | 9.2 / 0.8779 | 9.6 / 0.7556 | 11.1 / 0.7898 | 6.1 / 0.75 | 5.4 / 0.7525 |
| LS frames dub-silent while orig speaks | 25 | 0 | 251 | 299 | 457 |
| switches (per min) | 1 (0.59) | – | 8 (5.31) | 12 (13.34) | 10 (11.44) |
| ft mouth-lm jitter ls_core | 1.047 | – | 0.903 | 0.984 | 1.026 |
| ft mouth flicker ls_core | 1.093 | – | 0.887 | 0.964 | 0.995 |
| ft upper flicker ls_core | 1.004 | – | 1.004 | 1.0 | 1.003 |
| ft upper sharp ls_core | 0.997 | – | 0.982 | 1.008 | 1.007 |
| lj mouth_warp S1_speech (n) | 1.274 (229) | – | 1.161 (1174) | 1.257 (471) | 1.099 (346) |
| lj mouth_warp S2_switch (n) | 0.992 (10) | – | 1.093 (80) | 1.133 (120) | 1.053 (100) |
| lj mouth_warp S3_head_turn (n) | 1.247 (175) | – | 1.203 (178) | 1.232 (1002) | 1.103 (844) |
| lj mouth_warp S4_closed_master (n) | – (0) | – | 1.204 (112) | 1.282 (638) | 1.226 (21) |
| lj mouth_warp S4_closed_out (n) | 1.494 (6) | – | 1.085 (953) | 1.255 (838) | 1.087 (172) |
| lj mouth_warp orig_core (n) | – (2277) | – | 1.018 (195) | 1.023 (215) | 1.009 (331) |
| lj window step mouth_diff / mouth_warp (n) | x1.467 / x1.416 (15) | – | x1.513 / x1.454 (123) | x1.308 / x1.404 (61) | x1.378 / x1.604 (54) |
| lj inner-lip gap median out / master (LS) | 0.0875 / 0.1363 | – | 0.0435 / 0.0866 | 0.0301 / 0.0431 | 0.0684 / 0.0982 |
