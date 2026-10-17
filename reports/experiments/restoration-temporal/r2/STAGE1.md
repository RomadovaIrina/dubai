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

| window (30 fps) | F0 = yesterday's geometry: offset / slide / slide p95 | T1 | T3 |
|---|---|---|---|
| 04 25-31.6 s | 13.03 / 4.89 / 9.13 | 2.75 / 2.89 / 5.72 | **0.39 / 0.37 / 1.32** |
| 04 39-47.6 s | 16.37 / 5.66 / 11.36 | 7.17 / 3.65 / 7.61 | 6.36 / 2.83 / 5.76 (prior fails, see 4) |
| 03 46-56.4 s | 1.21 / 0.64 / 1.59 | 0.69 / 0.58 / 1.62 | **0.14 / 0.13 / 0.37** |
| 03 60-68 s | 1.61 / 0.67 / 1.43 | 1.08 / 0.70 / 1.39 | 0.70 / 0.44 / 0.97 |
| 03 21.6-37.6 s (incl. human 24-25, 35-36) | 0.88 / 0.51 / 1.23 | 0.62 / 0.50 / 1.29 | **0.12 / 0.10 / 0.26** |
| 05 42-49.36 s (IOD 46) | 3.66 / 1.41 / 3.34 | 1.78 / 1.01 / 2.09 | 1.51 / 0.76 / 1.65 |
| 05 27.5-34.2 s | 3.48 / 1.35 / 2.98 | 1.76 / 1.06 / 2.33 | 1.53 / 0.81 / 1.79 |

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

## 4. Open / next
* T3 fails on 04 39-47.6 s (6.4 px) and is partial on 05 (1.5 px at IOD 46): the landmark prior (MediaPipe VIDEO mode, fast motion)
  is poor there, and half-resolution flow is coarse for 46 px faces. Next: an image-based prior (rigid upper-face flow between the
  source frame the LatentSync layer came from, j(m), and the shown frame k), full-resolution flow for small faces, then the nose cut
  and the head-locked mask at 30 fps.
* Not yet measured for the 30 fps variants: mouth resid / sharpness / shear (`mouth_jitter_hr.py` was starved by thread
  oversubscription and stopped; rerun with OpenCV threads capped), whole-video SyncNet (01 / 05 vs cand5), 01 at 30 fps (its master is
  only -7 ms off, so the slide should be small there; 01's SyncNet loss has another cause).
* Cheaper alternative for the pipeline: make the master with `-vf fps=25` (nearest frame, mean error 0) or keep `-r 25` and pass the
  timing map, as done here. The first changes the generator input (other server / coordinator decision).

## Clips (`/workspace/dub/artifacts/restoration-temporal/sbs/r2/`)
* `04_25.0-31.6_B_ORIG-CLEAN0-OLD30-FIX30_full[_half].mp4`: ORIGINAL | CLEAN0 25 fps | OLD 30 fps (F0 clock) | FIX 30 fps (T3)
* `04_25.0-31.6_B_crop2x_ORIG-OLD30-FIX30[_half].mp4`: head-tracked 2x nose / mouth crop (head motion cancelled, yellow ticks = source
  nose tip): in OLD the pasted nose and mouth sit right of the source nose line; in FIX they sit on it.
