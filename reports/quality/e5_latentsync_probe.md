# E5 — LatentSync inference-quality probe (04, LS segment 1 = 1.44–16.36 s, 373 frames, baseline dubbed audio, same detections)

One parameter group per variant vs the frozen config (DeepCache 5, 20 steps, guidance 1.5, batch 2, fp16, seed 1247). Metrics: mouth/upper-face Laplacian sharpness vs master, mouth flicker vs master, PSNR vs the baseline variant output, upstream SyncNet on the segment clip. Clips: result_videos/probes/. Script: scripts/quality/ls_probe.py; raw: /tmp/dabai_quality/candidates/ls_probe/04_seg001_probe.json

| variant | changed | wall s | UNet calls | peak MiB | mouth sharp | upper sharp | flicker | PSNR vs base mean/min | SyncNet | AV |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| dc5 | baseline | 95.1 | 260 | 22530 | 0.552 | 0.704 | 0.967 | - / - | 4.45 | 0 |
| dc3 | {'deepcache_interval': 3} | 116.0 | 260 | 21910 | 0.55 | 0.706 | 0.959 | 55.44 / 53.82 | 4.5 | 0 |
| dcoff | {'deepcache': False} | 202.7 | 260 | 16658 | 0.551 | 0.708 | 0.96 | 53.52 / 50.9 | 4.53 | 0 |
| s25 | {'steps': 25} | 109.6 | 325 | 21914 | 0.554 | 0.706 | 0.968 | 63.68 / 60.55 | 4.4 | 0 |
| s30 | {'steps': 30} | 124.1 | 390 | 22556 | 0.555 | 0.707 | 0.968 | 61.05 / 57.99 | 4.42 | 0 |
| g10 | {'guidance': 1.0} | 73.7 | 260 | 19732 | 0.555 | 0.704 | 0.969 | 56.18 / 47.53 | 2.9 | 0 |
| g20 | {'guidance': 2.0} | 99.0 | 260 | 21916 | 0.55 | 0.704 | 0.971 | 55.97 / 45.74 | 5.49 | 0 |
| b1 | {'window_batch_size': 1} | 100.5 | 480 | 19732 | 0.552 | 0.704 | 0.967 | 76.99 / 72.93 | 4.38 | 0 |

## Reading
- DeepCache 5 / 3 / off, steps 20 / 25 / 30, batch 2 / 1: outputs are near-identical (PSNR 53–77 dB between variants), sharpness / flicker / SyncNet within noise (4.38–4.53). None of them is a quality lever; the face softness (mouth 0.55, upper 0.70 of the master) is intrinsic to LatentSync's face-square paste-back, not to the sampler. → REJECTED as quality candidates (DeepCache off costs 2.1x time for nothing; batch 1 costs 1.06x).
- Guidance is the only lever: 1.0 → SyncNet 2.90 (lip motion collapses), 1.5 → 4.45, **2.0 → 5.49** with identical sharpness and flicker and no visible over-articulation on the mouth strips (frames/04_probe_guidance_mouth.png). Guidance 2.5 / 3.0 queued on the same segment; a full-video A/B follows if 2.0–2.5 holds up.
