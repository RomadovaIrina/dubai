# Phase 2 — cheapest A/B on the 03 window (track B, 2026-10-16)

Window: 03 frames 1050-1506 (42.0-60.2 s; switches, continuous speech, the largest head turn of 03). Common input: the LatentSync raw
frames of `decompose_video_stage.py` (cand5 settings). Every variant uses the cand5 mouth-only composite (same smoothed detector boxes)
and the cand5 4-frame crossfade; the muxed audio is the baseline's (video-only experiment). Metrics: `mouth_jitter_hr.py` (ratios vs the
source, LS-core frames unless noted), SyncNet with the dub voice only on the window (`window_syncnet.py`). Data: `phase2_03_window/`.

| variant | license | mouth texture resid | sharpness | pixel change d1 | window-boundary jump d1 | boundaries > 1.5x | switch windows resid / jerk | lower-face ring resid | SyncNet dub-only |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| source | – | 1.00 | 1.00 | 1.00 | 0.99 | 0.21 | 1.00 / 1.00 | 1.00 | 0.72 (original lips on the dub) |
| **cand5** (CodeFormer per frame) | NO | 1.43 | 0.99 | 0.95 | 1.21 | 0.16 | 1.23 / 1.11 | 1.06 | 6.12 |
| no restorer | YES | 1.05 | 0.61 | 0.80 | 1.29 | 0.37 | 1.03 / 1.15 | 0.96 | 6.32 |
| **C2** boundary interpolation, no restorer | YES | **1.01** | 0.60 | 0.78 | **0.59** | **0.00** | 1.03 / 1.15 | 0.94 | **6.31** |
| GFPGAN v1.4 per frame (random / fixed noise: identical) | YES\* | 1.17 | 0.77 | 0.82 | 1.21 | 0.26 | 1.16 / 1.15 | 1.00 | 6.01 / 6.08 |
| RestoreFormer per frame | YES\* | 1.57 | 1.09 | 1.05 | 1.14 | 0.16 | 1.39 / 1.28 | 1.49 | 6.31 |
| C1+C2 CodeFormer detail, alpha 0.35 / 0.5 gain 1.3 | NO | 1.04 / 1.16 | 0.69 / 0.90 | 0.79 / 0.86 | 0.60 / 0.61 | 0.00 | 1.13-1.22 / 1.14-1.16 | 0.93 / 0.97 | 6.14 / 6.09 |
| **C1+C2 GFPGAN v1.4 detail**, alpha 0.5 gain 1.3 | YES\* | **1.01** | 0.71 | 0.76 | **0.59** | **0.00** | 1.18 / 1.17 | 0.89 | 6.00 |
| C1+C2 RestoreFormer detail, alpha 0.5 gain 1.3 | YES\* | 1.29 | 0.98 | 0.94 | 0.64 | 0.00 | 1.42 / 1.39 | 1.25 | 6.19 |

Control: re-running the composite from the stored stages with recomputed boxes reproduces cand5 exactly (`CTRL_cf` = `S3_final`, every metric).

## Reading
* **C2 removes the 16-frame window jump** (boundary change 1.29 -> 0.59 of the inner frames, no boundary above 1.5x) at no SyncNet cost
  (6.32 -> 6.31). It is a pure post-process (flow-interpolated frames at t = 1/3, 2/3 between the last and first frame of adjacent windows,
  only inside the paste-back mask). Its phase profile spreads the step over three frames (phase 15 / 0 / 1: 2.95 / 1.59 / 2.90 vs inner ~2.7).
* **C1 (motion-compensated detail filter) removes the restorer's shimmer** (CodeFormer 1.43 -> 1.04-1.16; GFPGAN 1.17 -> 1.01) but
  averaging misaligned detail costs sharpness (CodeFormer 0.99 -> 0.69-0.90). It is the knob between "calm" and "crisp".
* **No per-frame restorer is both crisp and calm**: CodeFormer 1.43 / 0.99, RestoreFormer 1.57 / 1.09, GFPGAN 1.17 / 0.77. GFPGAN's
  random StyleGAN noise is not the cause (fixed noise: identical numbers). RestoreFormer also changes the lower-face skin (ring 1.49).
* **Every restorer costs lip sync on this window** (SyncNet 6.31-6.33 without one; CodeFormer 5.98 before the composite / 6.12 after,
  GFPGAN 6.01-6.08, C1+C2 GFPGAN 6.00); AV offset 0 everywhere.
* What is left: the lower face inside the paste ellipse (chin stubble, skin) is softer than the source in C2 / C1+C2 (sharpness 0.60-0.71,
  visible in `p2-sbs-03-46s-52s_mouthcrop_halfspeed.mp4`).

## Next (Phase 3, pre-approved)
1. **Source-texture transfer outside the lips**: keep LatentSync's low frequencies (shape, articulation) everywhere in the ellipse, take
   the high-frequency skin / stubble texture from the SOURCE frame (flow-aligned to the LatentSync geometry) outside a lip mask built from
   commercial-safe mouth landmarks (YuNet / MediaPipe); lips + teeth stay LatentSync (+ optional stabilised restorer detail). Expected:
   sharpness of the skin back to ~1.0 with the source's own (steady) texture, no new popping.
2. Then the four scoreable videos (01 / 03 / 04 / 05) with the winner(s) via the video-stage re-run, full metric set, audio md5 identity.
