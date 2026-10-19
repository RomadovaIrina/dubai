# `scripts/quality/mouth_jitter_hr.py` — high-resolution, face-stabilised mouth / face jitter metric

Proposed as the **common twitch metric** for both tracks. CPU only, deterministic, no new pip dependency (opencv-python >= 4.8 + numpy;
the 230 KB YuNet model is fetched to `models/yunet/` on first use). 03 full video (2257 frames): ~100 s on this box.

```bash
python scripts/quality/mouth_jitter_hr.py --video OUT.mp4 --reference RUN/work/work_<id>/master_25fps.mp4 --out x.json \
    [--manifest RUN/final/<id>_final.manifest.json] [--frames a:b ...] [--tag name] [--series] [--sheet DIR]
```
`--reference` = the 25 fps master the output was built from (a source with another frame rate is resampled). `--manifest` (optional) adds
the LS core / switch / original classes and the LatentSync window-boundary analysis.

## Why
`face_temporal.py` (64x64 grey crops of a fixed box, insightface landmarks of the OUTPUT) rates cand5 as calm as the source (03 LS-core
mouth flicker 0.89, landmark jitter 0.88), but the reviewer sees the mouth twitch constantly. Looking at face-stabilised crops
(`03` frames 380-408, sheet in `artifacts/restoration-temporal/metric_validation`) shows what is missed: the LatentSync mouth snaps between
shapes within one frame (393 open -> 394 pressed shut -> 395 skewed lower lip) and the lip texture changes from frame to frame.

## What it measures
* **Stabilisation (same warp for both videos):** YuNet (OpenCV FaceDetectorYN, MIT code + weights) eyes + nose of the REFERENCE, smoothed
  over time, similarity -> canonical face (IOD 96 px, crop 250x317). Stage 2 removes the hand-held camera / head shake left in the crops: DIS
  flow on the reference's rigid region (brows / eyes / nose bridge), chained, high-passed (sigma 4 frames), the same correction applied to
  both videos. Mouth / jaw never drive the alignment. Result: reference upper-face motion 0.26 px/frame.
* **Per region (mouth box from the reference's mouth corners; lower-face ring around it = paste-back mask edge; upper face = control),
  video and reference on identical crops:**
  `resid` (appearance change NOT explained by optical flow: texture popping), `d1` (pixel change), `acc` (pixel 2nd difference), `vel` /
  `jerk` (DIS flow relative to the head), `sharp` (|Laplacian|), `impulse` = sum(acc)/sum(d1) (1-frame snaps), `pops` (frames above the
  reference's 98th percentile of d1 / acc, normalised so reference = 1).
* **LatentSync window boundaries** (`--manifest`): LatentSync 1.6 denoises independent 16-frame windows from each LS segment start; per phase
  (0..15) mouth means, phase-0 / inner ratio, and per boundary the d1 jump vs the previous window's inner median.

## Validation
Controls: identity (master vs master) = 1.000 for every value; pass-through frames 1.00-1.02; upper face of every mouth-only composite
1.00-1.02 (it IS the source there); decomposition stage `master` through the same lossless encoder = 1.00.

cand5 vs source, whole videos (`metric_validation/<id>_cand5_full.json`):

| video | LS window boundary d1 jump (phase 0 / inner): cand5 (source, same frames) | resid at boundaries: cand5 (source) | boundaries with a > 1.5x d1 jump: cand5 (source) | LS-core mouth resid (texture popping) | old face_temporal LS-core mouth flicker |
|---|---|---|---|---:|---:|
| 01 | 1.34 (0.93) | 1.45 (1.03) | 50 % (13 %), n=16 | 1.35 | 1.09 |
| 03 | 1.37 (0.97) | 1.43 (0.99) | 42 % (17 %), n=124 | 1.32 | 0.89 |
| 04 | 1.25 (0.98) | 1.36 (1.00) | 30 % (16 %), n=64 | 1.36 | 0.96 |
| 05 | 1.32 (1.09) | 1.53 (1.09) | 34 % (14 %), n=56 | 1.20 | 1.00 |

Worst-moment clip windows (face_temporal's worst switches + worst in-LS windows, +-1.5 s; `<id>_cand5_worstclips.json`): old metric
0.96-1.06, new mouth resid 1.13-1.40 on 18 of 21 windows (01: pops 4-7x the source); the three exceptions (05 18-21 s, 24-29 s) are
windows dominated by pass-through frames.

Per-phase mouth d1 in 03 (LS core): cand5 0.96 / 0.72 / 0.67 ... 0.73 / 0.92 at phases 0 / 1 / 2 ... 14 / 15, source flat 0.87-1.03 —
a jump every 16 frames = every 0.64 s, i.e. "constantly".

## Limits
* The source is a reference for *naturalness*, not ground truth: a dubbed mouth says different words, so `vel` / `d1` < 1 means less
  articulation, not "better". Twitch evidence = `resid`, the boundary jump, `impulse`, `pops`; detail = `sharp`.
* YuNet needs a face of ~10-300 px; faces bigger than ~300 px should be downscaled (not needed for 01-05). 01 has a face in 17 % of frames.
* YuNet / DIS: commercial-safe for a metric (MIT / Apache, no learned flow weights); YuNet's training data is WIDER FACE (CC BY-NC-ND) —
  see `LICENSES.md`.
