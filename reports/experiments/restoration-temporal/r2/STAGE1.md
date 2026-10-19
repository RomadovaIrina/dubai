# Round 2, track B (composite side), stage 1: lower-face position

Server dub-B3 (RTX 5090), branch `experiment/r2-composite-position` (local). Inputs: canonical work dirs (audio md5 of every clean0
output = canonical `final/<id>_final.audio_stream.md5`: 03 6cf53a89, 04 cb31a8f4, 05 b1447c1e, 01 f9c87d77), regenerated
`master_25fps.mp4` (= `make_master`, frame counts = manifests: 2542 / 2260 / 1349 / 1311), clean0 by the CLEAN0.md recipe. Slide = the
common `slide_metric.py` (track A's npz + rigid) unless noted; px at source resolution.

## 1. Main finding: at 25 fps there is (almost) no slide; at the source frame rate there is a large one, from the master's clock

* **25 fps, clean0, whole videos** (nose-base slide = |d_t - d_t-1|, mean / p95 / max):
  03 0.17 / 0.39 / 0.91 px (IOD 130), 04 0.24 / 0.50 / 1.12 px (IOD 155, the strongest shake), 05 0.23 / 0.49 px (IOD 46),
  01 0.23 / 0.54 px (IOD 51). The generated lower face sits where the source head is to well under 1 % IOD in every frame: LatentSync
  inpaints in the crop, but the visible (source) upper half of the crop anchors the generated half. Upper-face check: 0.01 px.
* **The 25 fps master lags its own clock by ~60 ms.** `make_master` = `ffmpeg -i SRC -r 25` (output option). It keeps the first 12
  source frames 1:1 and then drops one in six, so master frame m shows source time m/25 - 47...73 ms (mean -60 ms) on 03 / 04 / 05
  (01, VFR ~25 fps: -7 ms, no drops). Exact per-frame map: frame-index video through the same command
  (`artifacts/.../r2/timing/<id>_master_timing.json`), identical to pixel matching of the master against the source on 04.
* **Consequence for the source-fps composite** (`fps_composite.py`, yesterday's MAIN / F0..F3, stage 3): the LatentSync layer for source
  time t is taken at master position 25 t (right for the lips: LatentSync syncs frame m to audio m/25), but its HEAD POSE is from
  t - 60 ms, and the mask is placed for that time too. The upper face is the true frame t, so the pasted lower face trails the head by
  head speed x 60 ms. That is the defect the user saw: the whole lower face (mouth, base of the nose, cheeks) slides against the eyes /
  glasses, worst on camera shake, and the nose looks displaced (clean0's mask has alpha 1.0 at the nose tip, alae and subnasale, so
  the pasted generated nose is what moves).

## 2. Source-fps fix: keep the mouth audio-synced, take the geometry from the true time (`fps_composite.py --master-timing --register`)

mu(k) = tau^-1(t_k) = the master position whose TRUE time is the source frame's. Mask / landmarks from mu, and the LatentSync layer is
moved from its own pose (master position 25 t_k) to the shown frame's: T1 = landmark similarity (eyes / nose bridge / nose tip),
T2 = T1 + DIS RANSAC similarity on the stable lower-face parts against the source frame, T3 = T2 + smooth flow on the paste ring.
Measured at 30 fps: output frame k vs source frame k, regions from landmarks at the true time (`r2_slide_fps.sh`).

Frame-exact cuts of output and source (`r2_cut_frames.py`; OpenCV frame seeking drifted by several frames on 04 39-47.6 s, 03 60-68 s
and 05 in the first pass: upper-face "error" 0.6-6 px on untouched pixels. Every row below has upper face 0.012-0.024 px).
Columns: nose-base offset |d| mean / slide mean / slide p95 (common metric), cheeks slide, independent phase-correlation nose-base slide.

| window (30 fps), IOD | F0 = yesterday's geometry | T1 = true time + landmark rigid | **T3 = T1 + DIS similarity + ring** |
|---|---|---|---|
| 04 25-31.6 s, 158 | 13.03 / 4.90 / 9.17, ch 4.00, pc 5.22 | 2.76 / 2.89 / 5.68, ch 1.62, pc 2.68 | **0.41 / 0.37 / 1.31**, ch 0.08, pc 0.34 |
| 04 39-47.6 s, 152 | 10.88 / 4.27 / 9.01, ch 3.45, pc 4.38 | 2.44 / 2.55 / 5.71, ch 1.55, pc 2.50 | **0.27 / 0.26 / 0.68**, ch 0.09, pc 0.28 |
| 03 46-56.4 s, 129 | 1.21 / 0.64 / 1.55, ch 0.39, pc 0.58 | 0.69 / 0.58 / 1.60, ch 0.33, pc 0.51 | **0.15 / 0.14 / 0.39**, ch 0.07, pc 0.17 |
| 03 60-68 s, 128 | 1.09 / 0.57 / 1.22, ch 0.38, pc 0.56 | 0.64 / 0.54 / 1.20, ch 0.35, pc 0.51 | **0.12 / 0.10 / 0.24**, ch 0.08, pc 0.17 |
| 03 21.6-37.6 s (human 24-25, 35-36), 132 | 0.88 / 0.51 / 1.22, ch 0.31, pc 0.52 | 0.63 / 0.50 / 1.26, ch 0.30, pc 0.48 | **0.13 / 0.10 / 0.26**, ch 0.07, pc 0.16 |
| 05 42-49.36 s, 49 | 2.43 / 1.08 / 2.50, ch 0.83, pc 1.15 | 0.64 / 0.65 / 1.34, ch 0.42, pc 0.71 | **0.13 / 0.09 / 0.19**, ch 0.05, pc 0.28 |
| 05 27.5-34.2 s, 47 | 2.21 / 1.03 / 2.41, ch 0.73, pc 1.09 | 0.63 / 0.63 / 1.33, ch 0.40, pc 0.69 | **0.17 / 0.13 / 0.32**, ch 0.08, pc 0.29 |
| 01 0-10.4 s (master only -7 ms off), 50 | 0.33 / 0.35 / 1.04, ch 0.19, pc 0.42 | 0.20 / 0.21 / 0.46, ch 0.12, pc 0.31 | **0.10 / 0.09 / 0.22**, ch 0.07, pc 0.24 |

T3 puts the pasted lower face within 0.1-0.4 px of the source on every shaky window (25 fps clean0 itself: 0.17-0.24 px), F0 is 4-13x
worse in slide and up to 30x in offset. `mouth_jitter_hr.py` on 04 25-31.6 s (30 fps, vs the source; ratios, 1.0 = source): mouth motion
relative to the head (vel) F0 3.29 -> T3 1.23, mouth jerk 2.85 -> 1.49, mouth pixel acceleration 1.71 -> 0.94, lower-face vel / jerk
1.34 / 1.40 -> 1.00 / 1.03, lower-face shear 1.15 -> 1.04; sharpness unchanged (mouth 0.73 -> 0.75, lower face 0.88 -> 0.90), resid
0.93 -> 0.90. An image-based rigid prior (`--register-prior image`) is implemented but not needed (04: 0.37 vs 0.34 px slide).

Independent estimator (phase correlation of gradient patches, no flow; 04 25-31.6 s): nose-base slide F0 5.23 -> T1 2.71 -> T2 1.08 ->
T3 0.34 px (p95 10.1 -> 0.75); philtrum left-right slide 7.2 -> 1.04 px. Upper face 0.02 px in every variant (encoder noise).
SyncNet dub-only, 04 25.1-31.5 s (6.4 s, noisy): clean0 25 fps 4.06; 30 fps F0 3.75, T1 3.36, T3 3.71, T3 + nearest retime 3.57:
the geometry fix costs no lip-sync; the 25 -> 30 retime itself costs ~0.3 (yesterday ~0.2).
Post-processing, whole video incl. decode / encode (12 threads, CPU shared): F0 35-41 s/video-min, T3 42-65 s/video-min.

## 3. 25 fps registration arms (the brief's items 2-4), 03 46-56.4 s

| variant | common metric nose-base slide (mean / p95) | cheeks | independent check | mask centre vs head (mean / p95) |
|---|---|---|---|---|
| clean0 | 0.190 / 0.421 | 0.173 | 0.147 | 0.63 / 1.64 |
| translation | 0.099 / 0.245 | 0.064 | 0.145 | 0.63 / 1.64 |
| similarity | 0.102 / 0.230 | 0.048 | 0.149 | 0.63 / 1.64 |
| similarity + ring | 0.044 / 0.122 | 0.027 | 0.107 | 0.63 / 1.64 |
| nosecut mask (no registration) | 0.019 (nose = source) | 0.004 | 0.070 | 0.78 / 1.85 |
| head-locked nosecut | | | | 0.65 / 1.51 |

At 25 fps registration moves numbers that are already sub-pixel; the independent check barely moves (0.147 -> 0.107 with the ring),
held-out fits generalise only partly (fit on cheeks -> nose base 0.137). Not worth its 47-117 s / LS-minute at 25 fps. The nose cut
(item 4) keeps the source nose: worth keeping at any frame rate (no generated nose = nothing nose-shaped can move).

## 3b. Whole-video SyncNet at the source frame rate (final mix / dub-only; `syncnet_fps.py`, fps=25 rendition of the 30 fps output)

| video | cand5 (13.10 report) | clean0 25 fps (this server) | F0 30 fps (yesterday's geometry) | T3 30 fps (ring incl. chin) | T2 30 fps (no ring) | T3u 30 fps (ring above the mouth) |
|---|---|---|---|---|---|---|
| 01 | 7.89 / 7.86 | 7.99 / 7.91 | 7.61 / 7.49 | 7.21 / 7.11 | **7.70 / 7.72** | 7.64 / 7.56 |
| 03 | 5.46 / 5.69 | 5.49 / 5.75 | 5.24 / 5.48 | 5.25 / 5.51 | running | running |
| 04 | 3.69 / 3.81 | 3.86 / 3.98 | 3.53 / 3.55 | 3.63 / 3.77 | running | running |
| 05 | 3.15 / 3.66 | 3.20 / 3.73 | 2.91 / 3.30 | 2.91 / 3.26 | running | running |

* clean0 at 25 fps is at or above cand5 on all four videos. Every 30 fps variant loses ~0.25-0.45 against it: the mouth is flow-
  interpolated from 25 to 30 fps (yesterday's finding; stage 3).
* The geometry fix is SyncNet-neutral or positive on 03 / 04 / 05 (04 +0.22 dub-only), but the full-ring T3 costs 0.38 on 01: the ring
  band includes the chin / jaw, and pulling the generated jaw towards the SOURCE jaw puts the original speech's articulation back into
  the dub (01's face is small: IOD 50, the chin is within the ring). `--ring-region upper` (default now: ring only above the mouth-corner
  line) recovers most of it (7.56), no ring (T2) is best for lip-sync (7.72) at a slide of 0.14 px (T3u 0.09 px) on 01.

## 4. Open / next
* Whole-video SyncNet (mix / dub-only) of F0 vs T3 at source fps on 01 / 03 / 04 / 05 against clean0 25 fps and cand5: running.
* Stage 2 / 3 combination: T3 on top of the other server's generator-input fix (fp32 AlignRestore, flow-driven anchors, head-locked mouth
  mask) once its frames are copied; MAIN extras (C2 seams, source skin texture, sharpen) + the nose cut at 30 fps.
* Pipeline alternative: master with `-vf fps=25` (nearest frame, no systematic offset) or keep `-r 25` and pass the timing map (done here).
  The first changes the generator input (other server / coordinator decision). With either, the layer still needs moving to the shown
  frame's pose for the source frames between master frames (one in six for 30 fps), which is what --register does.

## Clips (`/workspace/dub/artifacts/restoration-temporal/sbs/r2/`)
* `<id>_<t0>-<t1>_B_ORIG-CLEAN0-OLD30-FIX30_full[_half].mp4`: ORIGINAL | CLEAN0 25 fps | OLD 30 fps (F0 clock) | FIX 30 fps (T3), for
  04 25-31.6 / 39-47.6 s, 03 23-26.5 / 34-37 s (human-marked), 05 42-49.36 s
* `<id>_<t0>-<t1>_B_crop2x_ORIG-OLD30-FIX30[_half].mp4`: head-tracked 2x nose / mouth crop (head motion cancelled, yellow ticks = source
  nose tip): in OLD the pasted nose and mouth sit right of the source nose line; in FIX they sit on it.
