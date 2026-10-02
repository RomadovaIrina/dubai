# Pilot 0.6 refresh — current video contour timing (LatentSync optimized + CodeFormer optimized), 2026-10-02

Video: `04_final.manifest.json` source 53.900 s · GPU NVIDIA GeForce RTX 5090 / sm_120 · HEAD `afa9b0d5`  
Samples: 2 production runs (e2e + 1 video-stage repeat(s) on the same audio/work dir) + 1 sweep pass (informational). Formula: `stage_seconds / source_duration_seconds * 60`. Target: combined <= 300 s/video-min.

```text
LatentSync seconds        = 247.09   (median; samples 244.97, 249.20)
CodeFormer seconds        = 31.15   (median; samples 31.87, 30.44)
LatentSync s/video-min    = 247.09 / 53.90 * 60 = 275.0
CodeFormer s/video-min    = 31.15 / 53.90 * 60 = 34.7
combined video contour    = 309.7 s/video-min
target                    = 300 s/video-min
verdict                   = FAIL_TARGET

peak VRAM (nvidia-smi, device-wide, whole run)   = 24156 MiB   (informational)
CodeFormer faces                                 = 1044 (eligible LATENT_SYNC frames 1044 of 1349 master frames)
CodeFormer seconds / processed face              = 0.0298
```

| sample | kind | LS stage s | LS verify s | LS s/video-min | LS s/lipsynced-min | CF stage s | CF faces | CF s/video-min | CF s/face | combined s/video-min | video stage total s | pipeline total s | nvsmi peak MiB | CF torch peak MiB | valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e2e | e2e | 244.97 | 15.326 | 272.7 | 351.968 | 31.87 | 1044 | 35.5 | 0.0305 | 308.2 | 315.8 | 436.054 | 24156 | 13117 | PASS |
| r2 | video_stage_repeat | 249.20 | 16.209 | 277.4 | 358.047 | 30.44 | 1044 | 33.9 | 0.0292 | 311.3 | 322.2 | 328.003 | 23909 | 10981 | PASS |
| sweep (0.7) | video stage, 5 CF instances | 251.225 | - | 279.657 | - | 29.399 | 1044 | 32.726 | - | - | - | - | - | 9528 (torch) | - |

## old legacy -> new optimized

| metric | old legacy (2026-09-15) | new optimized (this refresh) |
|---|---:|---:|
| LatentSync s/video-min | 403.224 (whole video, CLI) | 275.0 (speech-gated 77% of frames; 352.0 per lip-synced min) |
| CodeFormer s/video-min | 222.17 (upstream CLI, 1349 faces, 512x936) | 34.7 (in-memory, 1044 faces, geometry preserved) |
| combined s/video-min | 625.39 | 309.7 |
| target 300 | FAIL_TARGET | FAIL_TARGET |
| CodeFormer s/face | 0.1481 | 0.0298 |
| peak VRAM MiB | LS 20607 / CF 2198 (separate processes) | 24156 (one process, LatentSync + CodeFormer resident) |

Notes: LatentSync time = stage wall incl. LatentSync's own detector verify pass and in-memory segment prep, excl. model load (reported separately); CodeFormer time = in-memory stage wall on the LATENT_SYNC frames only (model load reported separately); the current pipeline lip-syncs only speech-gated VALID_FACE frames, so s/video-min is also given per lip-synced minute for comparison with the old whole-video number; VRAM informational only (0.4 not reopened).

Evidence: `reports/pilot/refresh_0.6_current_pipeline.json`, manifests listed in the samples; media under /tmp/pilot_refresh (not tracked).
