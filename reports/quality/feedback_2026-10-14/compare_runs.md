### 04

| metric | 12.10 | R1 | R2 | R8 | R10 |
|---|---:|---:|---:|---:|---:|
| wall s | 471.47 | 369.963 | 367.549 | 319.416 | 324.53 |
| s / video-min | 525 | 412 | 409 | 356 | 361 |
| Qwen s | 174.8 | 47.3 | 60.6 | 58.5 | 53.2 |
| TTS stage s | 137.617 | 6.159 | 6.299 | 6.222 | 13.659 |
| LatentSync s | 213.172 | 239.379 | 223.924 | 127.794 | 145.735 |
| CodeFormer s | 28.438 | 32.469 | 30.41 | 17.142 | 19.455 |
| pauses inside a sentence > 0.3 s | 12 | 0 | 1 | 0 | 0 |
| …on a weak word | 8 | 0 | 0 | 0 | 0 |
| longest pause between sentences s | 2.43 | 9.57 | 8.53 | 9.57 | 7.84 |
| parts slowed (< 1.0) | 13 | 0 | 0 | 0 | 5 |
| parts sped up | 0 | 0 | 1 | 0 | 0 |
| atempo max | 1.0 | 1.0 | 1.027 | 1.0 | 1.0 |
| next sentence shifted max s | 0.0 | 0.0 | 0.01 | 0.0 | 0.0 |
| Qwen concise rewrites | 0 | 0 | 0 | 0 | 0 |
| punctuation splits | 0 | 0 | 2 | 0 | 0 |
| source speech without dub s | 3.9 | 19.4 | 15.6 | 19.4 | 16.4 |
| dub outside source speech s | 0.9 | 11.8 | 7.8 | 11.8 | 12.3 |
| frames: dub silent, speaker talks | 66 | 463 | 368 | – | – |
| mouth opening there: output / source | 0.022 / 0.037 | 0.021 / 0.042 | 0.021 / 0.039 | None / None | None / None |
| lip motion: dub silent / dub speech | 0.008 / 0.0093 | 0.0075 / 0.0094 | 0.0075 / 0.0098 | None / None | None / None |
| WER | 0.031 | 0.009 | 0.009 | – | – |
| CTC CER | 0.068 | 0.043 | 0.046 | – | – |
| bad units | 0 | 0 | 0 | – | – |
| interjections | 0 | 0 | 0 | – | – |
| final word missing | 0 | 0 | 0 | – | – |
| SyncNet (mix) / dub-only | 3.32 / 3.42 | 2.54 / 2.79 | 2.67 / 2.92 | None / None | None / None |
| AV offset | 0 | 0 | 0 | – | – |

### 01

| metric | 12.10 | R1 | R2 | R8 | R10 |
|---|---:|---:|---:|---:|---:|
| wall s | 694.541 | 329.421 | 341.698 | 371.461 | 382.214 |
| s / video-min | 410 | 194 | 202 | 219 | 226 |
| Qwen s | 535.5 | 192.7 | 203.4 | 228.1 | 228.7 |
| TTS stage s | 355.962 | 31.681 | 33.529 | 32.478 | 35.565 |
| LatentSync s | 54.716 | 53.063 | 53.103 | 46.716 | 52.642 |
| CodeFormer s | 7.088 | 6.997 | 7.081 | 6.123 | 7.029 |
| pauses inside a sentence > 0.3 s | 1 | 0 | 0 | 0 | 0 |
| …on a weak word | 1 | 0 | 0 | 0 | 0 |
| longest pause between sentences s | 2.13 | 2.59 | 2.59 | 2.59 | 2.12 |
| parts slowed (< 1.0) | 16 | 0 | 0 | 0 | 20 |
| parts sped up | 10 | 3 | 3 | 3 | 2 |
| atempo max | 1.146 | 1.1 | 1.1 | 1.1 | 1.1 |
| next sentence shifted max s | 0.0 | 0.23 | 0.23 | 0.23 | 0.23 |
| Qwen concise rewrites | 0 | 2 | 2 | 2 | 2 |
| punctuation splits | 0 | 0 | 0 | 0 | 0 |
| source speech without dub s | 8.3 | 22.0 | 22.0 | 22.0 | 15.5 |
| dub outside source speech s | 1.9 | 2.8 | 2.8 | 2.8 | 2.9 |
| frames: dub silent, speaker talks | – | 39 | 39 | – | – |
| mouth opening there: output / source | None / None | 0.043 / 0.098 | 0.043 / 0.098 | None / None | None / None |
| lip motion: dub silent / dub speech | None / None | 0.0135 / 0.015 | 0.0135 / 0.015 | None / None | None / None |
| WER | 0.019 | 0.047 | 0.047 | – | – |
| CTC CER | 0.051 | 0.08 | 0.08 | – | – |
| bad units | 0 | 1 | 1 | – | – |
| interjections | 0 | 0 | 0 | – | – |
| final word missing | 0 | 0 | 0 | – | – |
| SyncNet (mix) / dub-only | 7.85 / 7.7 | 6.11 / 6.6 | 6.11 / 6.6 | None / None | None / None |
| AV offset | 0 | 0 | 0 | – | – |

### 03

| metric | 12.10 | R1 | R2 | R8 |
|---|---:|---:|---:|---:|
| wall s | 960.28 | 660.077 | 667.025 | 565.204 |
| s / video-min | 638 | 438 | 443 | 375 |
| Qwen s | 428.8 | 102.1 | 109.9 | 117.8 |
| TTS stage s | 332.21 | 12.22 | 12.834 | 12.369 |
| LatentSync s | 408.499 | 439.811 | 436.536 | 256.836 |
| CodeFormer s | 55.121 | 58.024 | 58.549 | 35.115 |
| pauses inside a sentence > 0.3 s | 20 | 0 | 0 | 0 |
| …on a weak word | 16 | 0 | 0 | 0 |
| longest pause between sentences s | 2.09 | 7.34 | 7.34 | 7.34 |
| parts slowed (< 1.0) | 22 | 0 | 0 | 0 |
| parts sped up | 2 | 0 | 0 | 0 |
| atempo max | 1.15 | 1.0 | 1.0 | 1.0 |
| next sentence shifted max s | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen concise rewrites | 0 | 0 | 0 | 0 |
| punctuation splits | 0 | 0 | 0 | 0 |
| source speech without dub s | 11.0 | 32.3 | 32.3 | 32.3 |
| dub outside source speech s | 1.8 | 17.0 | 17.0 | 17.0 |
| frames: dub silent, speaker talks | 195 | 767 | 767 | – |
| mouth opening there: output / source | 0.029 / 0.078 | 0.027 / 0.087 | 0.027 / 0.087 | None / None |
| lip motion: dub silent / dub speech | 0.0057 / 0.0091 | 0.0055 / 0.0087 | 0.0055 / 0.0087 | None / None |
| WER | 0.004 | 0.004 | 0.004 | – |
| CTC CER | 0.045 | 0.042 | 0.042 | – |
| bad units | 0 | 0 | 0 | – |
| interjections | 0 | 0 | 0 | – |
| final word missing | 0 | 0 | 0 | – |
| SyncNet (mix) / dub-only | 5.42 / 5.71 | 4.01 / 4.32 | 4.01 / 4.32 | None / None |
| AV offset | 0 | 0 | 0 | – |
