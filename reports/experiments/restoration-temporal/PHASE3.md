# Phase 3 — clean line on top of clean0: source frame rate, window seams, sharpness, frozen mouth (track B, 2026-10-16)

Rules in force: CLEAN-ONLY / final licence rule (file licences decide; NC and unlicensed weights quarantined; GFPGAN / RestoreFormer stay
quarantined by the user's choice), MediaPipe face stack (clean-base 326e807 merged as c0b37d8), no restorer, every video under
`/workspace/dub/artifacts/` only. Customer-spec constraints tracked per variant: processing norm (s per video-minute), no jerks at seams
(8.5), source resolution AND frame rate (9.4), 10-15 px feathered alpha edge (8.4), de-blur by clean means (8.3), never crash (2 % rule).

## clean0 reproduced on Server B (03, `candidate_video_stage.py`, CLEAN0.md recipe, on the cand5 work dir)

| | Server A (CLEAN0.md) | Server B |
|---|---:|---:|
| validation | PASS | PASS (2260 frames, 25 fps) |
| SyncNet final mix / dub-only / AV | 5.45 / 5.70 / 0 | **5.50 / 5.74 / 0** |
| muxed audio | bit-identical | bit-identical (md5 6cf53a89… = baseline) |
| mouth texture resid (LS core, vs master) | 1.02 | **1.025** |
| mouth sharpness | 0.71 | 0.67 (v4 metric, same definition) |
| video stage wall | 714 s (shared CPU) | 428 s |

clean0 01 / 04 / 05: see `PHASE3_full.md` (in progress).

## P0a — source frame rate (`scripts/quality/fps_composite.py`)

All five sources are 30 fps (01 VFR ~25 fps average, 02 / 05 near-30 VFR); every pipeline output so far was 25 fps CFR (the LatentSync
master is muxed), i.e. one source frame in six dropped. The new composite keeps the SOURCE frames and timestamps (PyAV, source time base
and pts, VFR included) and only the masked lower face comes from LatentSync, flow-retimed from the 25 fps master to every source timestamp.

Judder measured with the common metric against the 30 fps source (upper face = untouched source content, so any change is cadence):

| output | upper-face jerk | upper-face pixel acceleration | frames above the source's p98 change (d1 / acc) |
|---|---:|---:|---|
| cand5 (25 fps) shown at 30 fps | 1.36 | 1.31 | 3.0x / 3.7x |
| clean0 (25 fps) shown at 30 fps | 1.36 | 1.31 | 3.0x / 3.6x |
| **F0 source-fps composite** | **1.03** | **1.08** | 1.2x / 1.4x (encoder noise) |

## 03 variants on the source frame rate (ratios vs the source at its own fps; LS core unless noted)

| variant | licence | mouth resid | mouth sharp | lower-face sharp | LS-seam jump (d1) | seams > 1.5x | switch resid / jerk | upper resid | frozen-mouth 24s / 35s (rank) | SyncNet mix / dub-only | post-proc s/min | edge px | fallback | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cand5 (25 fps, CodeFormer) | NO (CodeFormer, buffalo_l) | 1.32 | 0.98 | 0.88 | 1.37 | 0.42 | 1.17 / 0.98 | 1.02 | 1.36 (22) / 1.47 (9) | 5.46 / 5.69 | - | - | - | rendered reference |
| clean0 (25 fps) | YES | 1.02 | 0.67 | 0.87 | 1.56 | 0.58 | 1.00 / 0.95 | 1.02 | 1.29 (26) / 1.46 (8) | 5.50 / 5.74 | - | - | - | track A recipe on this server |
| F0 source fps (clean0 composite, 12 px edge) | YES | 0.89 | 0.61 | 0.82 | 1.18 | 0.32 | 1.05 / 0.90 | 1.02 | 1.13 (49) / 1.13 (48) | 5.24 / 5.52 | 15.90 | 11.41 | 0 | audio md5 = baseline |
| F1 + C2 window-seam interpolation | YES | 0.86 | 0.60 | 0.81 | 0.98 | 0.13 | 1.05 / 0.90 | 1.02 | 1.14 (52) / 1.17 (42) | 5.29 / 5.53 | 20.00 | 11.41 | 0 | audio md5 = baseline |
| F2 + source skin texture | YES | 0.87 | 0.67 | 0.85 | 0.99 | 0.12 | 1.05 / 0.89 | 1.02 | 1.18 (40) / 1.17 (42) | 5.25 / 5.52 | 35.80 | 11.41 | 0 | audio md5 = baseline |
| F3 + lip texture | YES | 0.87 | 0.68 | 0.85 | 0.99 | 0.13 | 1.05 / 0.90 | 1.02 | 1.16 (45) / 1.17 (42) | 5.26 / 5.52 | 23.60 | 11.41 | 0 | audio md5 = baseline |
| OPT1 = F2 + still lower face in dub-silent stretches | YES | 0.87 | 0.67 | 0.85 | 0.99 | 0.13 | 1.05 / 0.90 | 1.02 | 1.17 (42) / 1.16 (47) | 5.23 / 5.51 | 26.80 | 11.41 | 0 | audio md5 = baseline |
| OPT2 = F2 + original pass-through there | YES | 0.91 | 0.69 | 0.86 | 0.98 | 0.12 | 1.05 / 0.89 | 1.02 | 1.09 (54) / 0.99 (74) | 5.18 / 5.43 | 30.50 | 11.41 | 0 | audio md5 = baseline |
| F0 with nearest-frame retiming | YES | 0.95 | 0.65 | 0.85 | 1.19 | 0.35 | 1.07 / 0.98 | 1.02 | 1.00 (62) / 1.06 (51) | 5.44 / 5.72 | 73.40 | 11.41 | 0 |  |
| F0 with nearest + rigid retiming | YES | 0.95 | 0.63 | 0.82 | 1.19 | 0.34 | 1.07 / 0.98 | 1.02 | 1.13 (44) / 1.06 (57) | 5.43 / 5.70 | 13.30 | 11.41 | 0 |  |
| F3 + sharpen 1.5 (flow retiming) | YES | 1.02 | 0.86 | 1.02 | 1.05 | 0.25 | 0.99 / 0.78 | 1.01 | - (-) / - (-) | 5.29 / 5.57 | 60.70 | 11.41 | 0 |  (frames 1380-1800 only) |
| F3 + sharpen 1.5 + lips 1.5 (flow retiming) | YES | 1.10 | 0.96 | 1.02 | 1.05 | 0.25 | 1.02 / 0.77 | 1.01 | - (-) / - (-) | 5.25 / 5.54 | 42.00 | 11.41 | 0 |  (frames 1380-1800 only) |

Notes: post-processing s / video-min are upper bounds (measured while CPU-heavy jobs ran in parallel; identical work read 13 s and 73 s);
SyncNet of source-fps outputs is computed on an fps=25 rendition (`syncnet_fps.py`); `(frames 1380-1800 only)` rows come from the sweep window.

Reading:
1. **Frame rate (9.4):** the source-fps composite removes the 1-in-6 judder (upper-face jerk 1.36 -> 1.03 vs the source) and keeps audio,
   frame count and timestamps; 0 fallbacks; edge 11.4 px (clean0 / cand5: 7.7 px, below spec 8.4).
2. **Retiming:** flow interpolation blends two mouth shapes and costs ~0.2 SyncNet (5.24 / 5.52); nearest master frame keeps the exact
   LatentSync shapes (5.44 / 5.72), the rigid head alignment puts them at the source frame's head position (5.43 / 5.70, clean0 5.50 / 5.74).
3. **Seams (8.5):** C2 brings the 16-frame window jump from 1.56 (clean0) to 0.98-1.05 of the inner frames (source 0.99), seams > 1.5x
   58 % -> 12-25 % (source 17 %), no SyncNet cost.
4. **Sharpness (8.3):** source skin texture alone: lower face 0.85, mouth 0.67; + global unsharp 1.5: 1.02 / 0.86 (resid 1.03); + extra
   1.5 on the lip contour: **1.02 / 0.96** (resid 1.10; CodeFormer cand5: resid 1.32, sharp 0.98). Sharpening does not cost SyncNet.
5. **Frozen mouth:** OPT2 (original pass-through in dub-silent stretches) removes the frozen-mouth signal at 35 s (0.99) but costs SyncNet
   (5.18 / 5.43: the source lips move without the dub voice); OPT1 (still lower face) changes the jaw line outside the metric's ring
   (index unchanged) -> human decision on the clips (`sbs_index.md`, p3 / p4 clips at 24-25 s and 35-36 s). Both take a gap list (`--gaps`).
