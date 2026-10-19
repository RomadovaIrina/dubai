# Round 3, track B: face stability / composite at the source frame rate

Branch `experiment/r3-face`. Changelog: `CHANGES.md`. Outputs: `/workspace/dub/artifacts/r3-face/`. User clips: `/workspace/dub/outbox/`.
Generator for every row: A's round-2 LatentSync settings (flow anchors, sub-pixel, fp32 AlignRestore, head-locked mask). No restorer, no new model.

## R3 recipe
1. Master `-vf fps=25` (`--master-timing-mode fps`): mean clock error 0 instead of −60 ms. The generated jaw/chin inherits the input
   frame's jaw, so a late master makes the lower face lag even after head registration.
2. `fps_composite.py` at the source frame rate:
   - **geometry:** landmarks of every source frame plus the source-rate rigid head motion (`--source-lm`); DIS similarity and upper-ring registration (`--register sim_ring`);
   - **colour:** low-frequency colour match to the source on the upper mask skin (`--colour-match upper`);
   - **window joints:** C2 boundary interpolation plus the excess low-frequency step spread over ±4 frames (`--seam-lf 4`);
   - **sharpness:** closed-loop per-video unsharp to 0.95 of the source, with source skin texture (`--sharpen-auto 0.95 --skin-texture source`);
   - **feather:** fixed 12 px edge plus a lip guard (`--lip-guard on`).

## Table 1: arms on the canonical cand5 dub (reference numbers only; this dub's bed is not licence-clean, so no clips use it)
Column key:
- **slide:** nose-base slide mean / max p95 over the shaky windows (common slide metric, px).
- **band_up … chin:** excess motion-compensated residual of the pasted region vs the source (r3_flicker.py, grey levels; 0 = moves exactly like the source).
- **colour:** Lab colour-offset flicker.
- **mouth … lf shear:** v4 ratios vs the source on the shaky windows.
- **seams:** v4 window-boundary jumps > 1.5x, with the source's share in brackets.

T3 = round-2 FIX30. Sdets = source-frame geometry. F_ = on-time master.

| video | variant | slide | offset | cheeks | band_up | skin_up | chin | colour | sharp_r3 | mouth vel | mouth jerk | mouth resid | mouth sharp | lf jerk | lf shear | SyncNet mix / dub | edge px | s/min (CPU shared) | seams >1.5x (src) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 04 | T3 | 0.278 / 1.25 | 0.30 | 0.077 | 0.25 | 0.18 | 2.67 | 0.57 | 0.90 | 1.06 | 1.31 | 0.79 | 0.78 | 1.22 | 1.10 | 3.39 / 3.54 | 11.41 | 53.2 |  |
| 04 | Sdets | 0.236 / 0.67 | 0.28 | 0.079 | 0.24 | 0.17 | 2.39 | 0.58 | 0.90 | 0.96 | 1.08 | 0.80 | 0.78 | 1.13 | 1.09 | 3.48 / 3.65 | 11.41 | 53.8 |  |
| 04 | F_Sdets | 0.140 / 0.34 | 0.14 | 0.066 | -0.01 | -0.10 | 0.49 | 0.38 | 0.90 | 0.84 | 0.83 | 0.78 | 0.78 | 0.99 | 0.99 | 3.53 / 3.69 | 11.41 | 46.4 |  |
| 04 | F_cl | 0.115 / 0.25 | 0.12 | 0.059 | 0.04 | -0.01 | 0.72 | 0.18 | 1.02 | 0.86 | 0.86 | 0.87 | 0.93 | 1.00 | 0.98 | 3.48 / 3.62 | 11.41 | 69.5 | 18 % (9 %) |
| 03 | F_Sdets | 0.073 / 0.20 | 0.07 | 0.062 | 0.00 | -0.08 | 0.25 | 0.29 | 0.71 | 0.70 | 0.64 | 0.87 | 0.62 | 0.93 | 0.95 | 5.24 / 5.47 | 11.41 | 41.0 | 12 % (22 %) |
| 03 | F_cl | 0.059 / 0.16 | 0.06 | 0.054 | 0.07 | 0.01 | 0.47 | 0.15 | 0.93 | 0.70 | 0.65 | 0.99 | 0.90 | 0.93 | 0.93 | 5.28 / 5.51 | 11.41 | 110.1 | 8 % (22 %) |
| 05 | T3 | 0.101 / 0.29 | 0.12 | 0.054 | -0.41 | -0.35 | 0.63 | 0.78 | 0.85 | 0.91 | 1.00 | 0.73 | 0.81 | 1.05 | 1.01 | 2.84 / 3.18 | 11.52 | 35.6 | 32 % (5 %) |
| 05 | F_Sdets | 0.055 / 0.13 | 0.07 | 0.035 | -0.34 | -0.41 | -0.26 | 0.54 | 0.84 | 0.83 | 0.82 | 0.74 | 0.82 | 0.95 | 0.96 | 2.84 / 3.25 | 11.52 | 29.4 | 26 % (5 %) |
| 05 | F_cmU | 0.052 / 0.13 | 0.06 | 0.034 | -0.32 | -0.37 | -0.21 | 0.40 | 0.84 | 0.83 | 0.81 | 0.77 | 0.81 | 0.94 | 0.96 | 2.77 / 3.27 | 11.52 | 133.1 | 26 % (5 %) |
| 05 | F_auto | 0.054 / 0.14 | 0.06 | 0.033 | -0.26 | -0.32 | -0.12 | 0.39 | 0.94 | 0.83 | 0.81 | 0.79 | 0.89 | 0.95 | 0.97 | 2.77 / 3.24 | 11.52 | 118.3 | 32 % (5 %) |
| 05 | F_cl | 0.047 / 0.12 | 0.07 | 0.029 | -0.10 | -0.17 | -0.02 | 0.38 | 1.03 | 0.83 | 0.83 | 0.85 | 0.97 | 0.95 | 0.98 | 2.73 / 3.12 | 11.52 | 44.6 | 32 % (5 %) |
| 01 | F_cl | 0.085 / 0.22 | 0.10 | 0.055 | 0.01 | -0.10 | 0.42 | 0.42 | 1.00 | 0.76 | 0.78 | 0.98 | 1.00 | 0.96 | 0.89 | 7.48 / 7.45 | 11.49 | 25.3 | 25 % (19 %) |

## Table 2: final R3 on C's clean audio-r2 (Spleeter bed): the deliverable
c2_L30 = experiment: LatentSync run at 30 fps on the source frames (no master, no retime).

| video | variant | slide | offset | cheeks | band_up | skin_up | chin | colour | sharp_r3 | mouth vel | mouth jerk | mouth resid | mouth sharp | lf jerk | lf shear | seams >1.5x (src) | SyncNet mix / dub | edge px | s/min (CPU shared) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | c2_R3f | 0.081 / 0.20 | 0.09 | 0.058 | 0.00 | -0.12 | 0.38 | 0.42 | 0.94 | 0.75 | 0.77 | 0.97 | 0.96 | 0.96 | 0.91 | 19 % (19 %) | 7.38 / 7.60 | 11.49 | 10.6 |
| 03 | c2_R3f | 0.056 / 0.16 | 0.06 | 0.054 | 0.06 | 0.00 | 0.43 | 0.15 | 0.91 | 0.70 | 0.64 | 0.98 | 0.89 | 0.92 | 0.92 | 8 % (22 %) | 5.35 / 5.65 | 11.41 | 73.4 |
| 04 | c2_R3f | 0.118 / 0.29 | 0.12 | 0.061 | 0.03 | -0.01 | 0.71 | 0.18 | 1.00 | 0.85 | 0.85 | 0.86 | 0.91 | 0.99 | 0.99 | 23 % (9 %) | 3.45 / 3.68 | 11.41 | 78.4 |
| 04 | c2_L30 | 0.098 / 0.23 | 0.10 | 0.058 | 0.05 | 0.01 | 0.43 | 0.16 | 1.04 | 0.86 | 0.90 | 0.92 | 0.95 | 1.02 | 1.00 |  | 3.38 / 3.63 | 11.41 | 95.8 |
| 05 | c2_R3f | 0.046 / 0.12 | 0.07 | 0.030 | -0.12 | -0.18 | -0.05 | 0.38 | 1.00 | 0.84 | 0.82 | 0.85 | 0.95 | 0.95 | 0.97 | 32 % (5 %) | 2.56 / 3.13 | 11.52 | 42.9 |

Reference SyncNet for A's 25 fps output on the same audio (mix / dub-only): 04 3.66 / 3.88, 05 2.67 / 3.21.

### Spec 8.4 feather and lip coverage (final R3, C audio)
| video | IOD px | edge 10-90 % px (median / min / max) | lips alpha min | frames lips < 0.98 | lips to ramp, min px | fallback frames |
|---|---|---|---|---|---|---|
| 01 | 50 | 11.49 / 11.49 / 11.50 | 0.998 | 0 | 7.6 | 0 |
| 03 | 130 | 11.41 / 11.39 / 11.44 | 1.000 | 0 | 36.6 | 0 |
| 04 | 155 | 11.41 / 11.39 / 11.43 | 1.000 | 0 | 16.4 | 0 |
| 05 | 46 | 11.52 / 11.46 / 11.63 | 0.987 (without lip guard: 0.776, 62 frames < 0.98) | 0 | 5.1 | 0 |

## Findings
* **Lag:** the `-r 25` master's 60 ms clock lag was still in the generated jaw. On 04, the on-time master takes:
  * chin excess 2.39 → 0.49;
  * jaw band 2.75 → 0.62;
  * slide 0.25 → 0.14 px;
  * SyncNet dub-only up 0.04 at 30 fps and 0.16 at 25 fps.
* **Slide on the shaky windows:** 0.05–0.12 px, against 0.10–0.28 for round 2's T3.
* **Window joints (r3_joints.py, on the output):** A's 25 fps output 1.38x the inner frames (58 % > 1.5x low-frequency); R3 0.93–0.95.
* **Sharpness vs SyncNet:** mouth sharpness 0.89–0.97 of the source. The last few percent cost ~0.1 SyncNet on 05, so the target is 0.95.
* **30 fps LatentSync (04):**
  * steadier than the 25 fps path: chin excess 0.43 vs 0.71, joints 0.82 vs 0.95, slide 0.10 vs 0.12;
  * no better in SyncNet: 3.63 vs 3.68 dub-only;
  * costs 396 s/video-min of video stage (vs ~265): over budget.
* **SyncNet at 30 fps:** every 30 fps output scores ~0.2 below A's 25 fps output, including the native 30 fps generation. Not separated yet: 25 fps re-sampling of real 30 fps content vs content.
* **Composite cost:** ~20 s/video-min with 12 threads, mostly decode / encode (profile: ~14 ms/frame of single-thread work). The s/min columns above were measured with the CPU shared and are upper bounds.
