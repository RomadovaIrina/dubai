# Pilot 0.8 — SyncNet on real content (final dubbed E2E outputs)

**Verdict:** `CLOSED — 5/5 final E2E outputs valid; SyncNet measured on 4/5, 1 N/A (no face); contract threshold still UNDECIDED`  
Outputs valid: **5/5** · SyncNet scoreable: **4** · N/A (no face): **1** · FAIL: **0**  
Confidence (scoreable) mean/median/min/max: **4.2725 / 3.92 / 2.51 / 6.74**  
Max |AV offset| (scoreable): **0 frames**  
Audio: current TTS intelligibility pipeline + burst alignment + optimized CodeFormer. Config: `run_clean_pipeline_05.py --target-lang en --alignment burst --video-backend optimized --codeformer optimized (runner defaults, HEAD afa9b0d5); see reports/pilot/PIPELINE_REFRESH.md`.  
Lipsync / restoration (from the runner manifests): optimized face-aware LatentSync (speech gate, RetinaFace routing), CodeFormer optimized w=1.0 batch 8 fp16 (insightface landmarks); audio alignment burst.  
**The old blocker "LatentSync fails on 01/02/05 with Face not detected" is no longer relevant**: face-aware routing passes no-face / small-face frames through, all five final E2E videos were produced and validated.  
**Contract threshold is intentionally not auto-selected:** the source spec does not define whether it should be min/mean/median/etc.

| video | pipeline valid | dubbed audio | SyncNet measurable | confidence | AV offset (frames) | original conf / offset | Δconf | status / reason |
|---|---|---|---|---:|---:|---|---:|---|
| 01_final.mp4 | yes | yes | yes | 6.74 | 0 | 5.36 / 1 | 1.38 | PASS |
| 02_final.mp4 | yes | yes | no | - | - | - / - | - | N/A_NO_FACE — SyncNet/S3FD cannot obtain a face track (no face in content: VALID_FACE=0, LatentSync segments=0) |
| 03_final.mp4 | yes | yes | yes | 5.02 | 0 | 4.89 / 2 | 0.13 | PASS |
| 04_final.mp4 | yes | yes | yes | 2.82 | 0 | 3.24 / 2 | -0.42 | PASS |
| 05_final.mp4 | yes | yes | yes | 2.51 | 0 | 1.71 / 2 | 0.8 | PASS |

Per-video pipeline facts (from the runner manifests):

| video | backend | speech gate | VALID / SMALL / NO_FACE / NO_SPEECH frames | LatentSync segments | LatentSync s | translation | CodeFormer | alignment |
|---|---|---|---|---:|---:|---|---|---|
| 01_final.mp4 | optimized | True | 261 / 177 / 1996 / 108 | 1 | 10.4 | q8_0 | optimized w=1.0 batch 8 fp16 (insightface landmarks) | burst |
| 02_final.mp4 | optimized | True | 0 / 0 / 2191 / 582 | 0 | 0.0 | q8_0 | optimized w=1.0 batch 8 fp16 (insightface landmarks) | burst |
| 03_final.mp4 | optimized | True | 1927 / 0 / 0 / 333 | 9 | 77.08 | q8_0 | optimized w=1.0 batch 8 fp16 (insightface landmarks) | burst |
| 04_final.mp4 | optimized | True | 1044 / 0 / 0 / 305 | 9 | 41.76 | q8_0 | optimized w=1.0 batch 8 fp16 (insightface landmarks) | burst |
| 05_final.mp4 | optimized | True | 914 / 94 / 0 / 303 | 8 | 34.72 | q8_0 | optimized w=1.0 batch 8 fp16 (insightface landmarks) | burst |

## Informational (from the runner manifests; not part of the SyncNet methodology)

| video | units | TTS parts | attempts | retried (ok) | bad final | fit rewritten / merged | atempo max | >1.15 / groups | atempo hist 1.0 / ≤1.05 / ≤1.10 / ≤1.15 / ≤1.20 / >1.20 | slot-fallback units | cut groups | LatentSync s (s/video-min) | LS video s | CodeFormer s (s/video-min) | CF faces | peak VRAM MiB (nvsmi) | total wall s | valid |
|---|---:|---:|---:|---|---:|---|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|---|
| 01 | 29 | 35 | 42 | 3 (3 ok) | 0 | 0 / 3 | 1.2 | 1 / 32 | 23 / 2 / 0 / 6 / 1 / 0 | 0 | 0 | 76.0 (44.87) | 10.4 | 7.8 (4.63) | 260 | 25178 | 382.127 | PASS |
| 02 | 12 | 19 | 24 | 3 (3 ok) | 0 | 0 / 1 | 1.2 | 1 / 18 | 15 / 0 / 0 / 2 / 1 / 0 | 0 | 1 | 0.0 (0.0) | 0.0 | 0.0 (0.0) | 0 | 8252 | 267.237 | PASS |
| 03 | 10 | 21 | 26 | 2 (1 ok) | 1 | 0 / 2 | 1.15 | 0 / 19 | 17 / 0 / 0 / 2 / 0 / 0 | 0 | 0 | 435.5 (289.27) | 77.08 | 57.1 (37.91) | 1927 | 25156 | 746.174 | PASS |
| 04 | 5 | 14 | 16 | 1 (1 ok) | 0 | 0 / 1 | 1.15 | 0 / 13 | 12 / 0 / 0 / 1 / 0 / 0 | 0 | 0 | 245.0 (272.69) | 41.76 | 31.9 (35.47) | 1044 | 24156 | 436.054 | PASS |
| 05 | 7 | 11 | 11 | 0 (0 ok) | 0 | 0 / 0 | 1.025 | 0 / 11 | 10 / 1 / 0 / 0 / 0 / 0 | 0 | 0 | 204.8 (234.6) | 34.72 | 26.1 (29.94) | 868 | 24156 | 392.695 | PASS |

Per-video result pages: `reports/pilot/refresh_videos/<id>.md`.
