| metric | 01 | 02 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|---:|
| valid | PASS | PASS | PASS | PASS | PASS |
| wall s (s / video-min) | 549 (324) | 256 (139) | 881 (585) | 426 (474) | 500 (572) |
| audio s / LatentSync s / CodeFormer s | 452 / 54 / 7.0 | 221 / 0 / 0.0 | 393 / 402 / 54.3 | 157 / 219 / 29.4 | 249 / 191 / 25.6 |
| LS frames | 260 | None | 2025 | 1074 | 949 |
| peak VRAM MiB (nvidia-smi) | 22456 | 8316 | 23484 | 22436 | 22458 |
| SyncNet mix / AV | 7.89 / 0 | N/A (no face) | 5.46 / 0 | 3.69 / 0 | 3.15 / 0 |
| SyncNet dub-only / AV | 7.86 / 0 | – | 5.69 / 0 | 3.81 / 0 | 3.66 / 0 |
| original SyncNet | 5.36 | – | 4.89 | 3.24 | 1.71 |
| Whisper WER / CTC CER | 0.019 / 0.051 | 0.073 / 0.168 | 0.009 / 0.043 | 0.009 / 0.048 | 0.000 / 0.169 |
| bad units / interj / final miss | 0 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| orig speech w/o dub s / IoU | 8.6 / 0.885 | 11.6 / 0.716 | 11.1 / 0.790 | 6.1 / 0.750 | 7.4 / 0.686 |
| LS frames dub-silent while orig speaks | 24 | 0 | 251 | 299 | 520 |
| repeated words / lost numbers / cut groups | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| atempo max / >1.15 | 1.146 / 0 | 1.15 / 0 | 1.15 / 0 | 1.025 / 0 | 1.0 / 0 |
| hot inner tails/heads | 2/1 of 9 | 6/10 of 20 | 4/9 of 23 | 5/6 of 13 | 10/11 of 14 |
| LS<->orig switches (/min) | 1 (0.59) | – | 8 (5.31) | 12 (13.34) | 10 (11.44) |
| short LS <2 s / short orig gaps <2 s | 0 / 0 | – | 0 / 2 | 2 / 4 | 0 / 2 |
| mouth lm jitter / open jitter (ls_core, vs src) | 1.033 / 1.045 | – | 0.884 / 0.704 | 0.982 / 0.766 | 1.029 / 0.91 |
| mouth px flicker ls_core / switch (vs src) | 1.092 / 1.021 | – | 0.89 / 0.973 | 0.964 / 0.979 | 0.997 / 0.992 |
| upper flicker / upper sharpness (ls_core, vs src) | 1.007 / 1.005 | – | 1.004 / 0.984 | 1.0 / 1.004 | 1.002 / 1.007 |
| switch: upper-sharp step % (src) / mouth jump ratio | 98.8 (99.0) / 1.02 | – | 4.0 (4.3) / 0.99 | 12.6 (12.5) / 1.01 | 17.4 (17.2) / 1.02 |
| muxed audio stream md5 | f9c87d778b1b | d0789251154f | 6cf53a89cf73 | cb31a8f48de5 | b1447c1ea8cb |
