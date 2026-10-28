Five videos, 6.82 video-min, one video at a time on one RTX 5090.

| axis | q14 | q7 | q7j14 | q14r1 |
|---|---:|---:|---:|---:|
| audio stage s/video-min (whole r6 stage) | 110.7 | 99.0 | 102.3 | 99.4 |
|   LLM s/video-min (all LLM work) | 38.7 | 27.4 | 30.6 | 31.3 |
|     pipeline steps (load, segmentation, translation, NE, fit judge) | 4.9 | 3.7 | 3.7 | 4.7 |
|     duration stage: candidates | 13.9 | 9.5 | 9.1 | 9.6 |
|     duration stage: judge (back-translation + score) | 20.0 | 14.2 | 17.8 | 17.0 |
|     duration stage: other Qwen calls | 0.0 | -0.0 | 0.0 | -0.0 |
|   TTS stage s/video-min (Chatterbox takes + the duration-stage LLM inside it) | 91.2 | 80.7 | 84.3 | 80.4 |
| duration-stage tokens/s (generated) | 93.8 | 144.8 | 118.8 | 94.3 |
| peak VRAM MiB | 16014 | 13186 | 23814 | 16014 |
| candidates written / units switched | 310 / 19 | 352 / 16 | 349 / 25 | 225 / 17 |
| faithfulness mean (14B judge, 1-5) | 4.731 | 4.587 | 4.601 | 4.756 |
| units scored < 4 | 3 | 8 | 6 | 3 |
| named entities misheard | 0 | 0 | 0 | 0 |
| WER final mix (mean) | 0.043 | 0.04 | 0.045 | 0.048 |
| tempo outside 0.95-1.05 (s) | 19.18 | 17.49 | 12.73 | 23.74 |
| source speaks, dub silent (s) | 64.2 | 66.45 | 62.45 | 67.85 |
| dub speaks, source silent (s) | 45.95 | 47.85 | 43.7 | 43.4 |
| speaker on screen talks, dub silent (s) | 24.5 | 30.9 | 29.3 | 26.85 |
| longest dub gap beyond the source pause (s, max) | 4.05 | 3.3 | 6.05 | 4.05 |
| pause W1 vs source (s, mean) | 0.538 | 0.586 | 0.711 | 0.596 |
