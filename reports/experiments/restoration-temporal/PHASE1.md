# Phase 1 — prior work, diagnosis, decomposition, plan (track B, 2026-10-16)

## 1. What is already implemented (video side), and where it stands

| lever | status in cand5 | evidence |
|---|---|---|
| speech gate (VAD source ∪ dub, margin 0.24 s) | default; merge gap **1.2** (opt-in, in cand5); 2.0 = option (04: 14 -> 4 switches, +17 % LS frames), not in cand5 | restore_review 2026-10-12, feedback_2026-10-13 A |
| in-memory LS segments, RetinaFace router (stride 3), LatentSync verify pass | default | PIPELINE_REFRESH, 0.5 |
| LatentSync DeepCache 5 / batch 2 / 20 steps / guidance 1.5 | default; `--ls-steps` opt-in (12 steps: human sees lip twitches) | e5_latentsync_probe (steps / DeepCache no quality lever; guidance 2.0 -> SyncNet +1 on one segment, not adopted — generator = track A) |
| 4-frame linear crossfade at LS <-> original switches | default | face_aware_latentsync_accel |
| mouth-only paste-back (E4: feathered lower-face ellipse, top 0.52, feather 0.08) | opt-in, **in cand5** | restore_review V2 (upper sharp 0.82 -> 0.86) |
| mask bbox smoothing ±3 | opt-in, in cand5 (no measured effect) | feedback A |
| LatentSync landmark smoothing ±2 (affine crop input) | opt-in, in cand5 (03 lip jitter 0.95 -> 0.89) | feedback A |
| CodeFormer optimized (w 1.0, fp16, batch 8, insightface 5 points) | default of `refresh_e2e.sh`; weight sweep done (w < 1 costs SyncNet) | codeformer_week2, 0.7 |
| CodeFormer **mouth scope** (restore the raw LS frame, composite after) | opt-in, in cand5 (removes the eye / brow re-render: 01 upper sharpness 4.45 -> 1.01) | feedback A |

Audio (not this track, kept bit-identical by every video-only experiment): acoustic cuts, long-pause rule, adaptive fades, TTS QA + gap
QA (cand5); per-word QA, ordered translation, sentence-flow gap, Qwen on GPU — opt-in, not in cand5.

## 2. What really remains (measured on the reproduced cand5, `metric_mouth_jitter_hr.md`)

| problem | number | source stage |
|---|---|---|
| **mouth jump every 0.64 s** — LatentSync denoises independent 16-frame windows | boundary-frame mouth change 1.25-1.37x the inner frames (source 0.93-1.09); texture residual at boundaries 1.36-1.53x (source ~1.0); 30-50 % of boundaries jump > 1.5x (source 13-17 %) — 01 / 03 / 04 / 05 | LatentSync (present in the raw output) |
| **mouth texture popping** frame to frame | LS-core mouth texture residual 1.20-1.36 (01-05); decomposition: LS raw 1.05 -> + CodeFormer 1.43 | **CodeFormer** (per-frame codebook re-synthesis) |
| soft LS mouth (why CodeFormer is there) | mouth sharpness LS raw 0.61 of the source, + CodeFormer 0.99 | LatentSync |
| under-articulation | mouth motion relative to the head 0.77 (LS raw), opening dynamics 0.70 (03) | LatentSync (track A) |
| LS <-> original switches | 03 5.3 / min, 04 13.3, 05 11.4; after the crossfade the switch windows read resid 1.23 / jerk 1.11 (CF) and 1.03 / 1.15 (no CF) | gate / transitions (secondary) |
| licenses | CodeFormer = NC; InsightFace buffalo_l = NC (also used by CodeFormer alignment, paste-back mask, landmark smoothing) | `LICENSES.md` |
| metric blind spot | face_temporal reads 0.89-1.09 where the new metric shows the jumps / popping | fixed: `scripts/quality/mouth_jitter_hr.py` @ 25aedba |

## 3. Decomposition (03 window 42.0-60.2 s, frames 1050-1506, same frames, `decomp_03_w1050_1506/`)

`scripts/quality/decompose_video_stage.py` re-ran the cand5 video stage on the window (LS pieces on the run's 16-frame grid; the fully
contained segment reproduces the run output within encoder noise, mean |Δ| 1.52 levels). Mouth, LS-core frames, ratio vs the source:

| stage | texture residual | pixel change d1 | motion vs head | sharpness | window-boundary jump d1 / resid (source 0.99 / 0.97) | switch windows resid / jerk |
|---|---:|---:|---:|---:|---|---|
| S0 LatentSync raw | **1.05** | 0.80 | 0.77 | **0.61** | **1.29 / 1.45** | 1.13 / 1.25 |
| S1 + CodeFormer | **1.43** | 0.95 | 0.65 | **0.99** | 1.21 / 1.30 | 1.46 / 1.32 |
| S2 mouth-only composite | 1.43 | 0.95 | 0.70 | 0.99 | 1.21 / 1.30 | 1.46 / 1.40 |
| S3 + crossfade = cand5 | 1.43 | 0.95 | 0.70 | 0.99 | 1.21 / 1.30 | **1.23 / 1.11** |
| S3nc cand5 without CodeFormer | 1.05 | 0.80 | 0.79 | 0.61 | 1.29 / 1.45 | 1.03 / 1.15 |

Attribution: (1) **LatentSync** creates the periodic window-boundary jumps, the softness and the under-articulation; (2) **CodeFormer**
creates the frame-to-frame mouth texture popping (+36 % residual) while restoring the detail, and re-renders the upper face (1.29, removed
by the composite); (3) **paste-back** is neutral for the mouth (lower-face ring 1.01 without CF, 1.08 with); (4) **transitions** are
mostly handled by the crossfade. CodeFormer IS a meaningful source -> but another per-frame restorer will pop the same way, so the core of
this track is temporal consistency of the restored mouth, plus the boundary jumps.

## 4. Plan — candidates in order, and why none repeats earlier work

| # | candidate | expected effect | new? |
|---|---|---|---|
| C1 | **temporal detail stabilisation**: keep LatentSync's (steady) low frequencies, take the restorer's high-frequency detail and filter it with a DIS-flow motion-compensated, occlusion-aware recursive filter in the mouth ROI | resid -> ~1.05 with sharpness ~0.9 | yes — only per-frame CodeFormer settings were ever tried |
| C2 | **window-boundary smoothing**: motion-compensated blend of the mouth ROI over the 2-3 frames around every 16-frame LS boundary | boundary jump -> ~1.0 | yes (post-process; the root fix — overlapping windows inside LatentSync — is generator work, flagged for the coordinator) |
| C3 | **restorer swap**: GFPGAN v1.4, RestoreFormer(++), VQFR (YES\*) and "none" vs CodeFormer (NO), each alone and with C1 | commercial-safe detail; C1 decides the popping | yes — only CodeFormer was tested |
| C4 | **commercial-safe landmarks** for the paste-back mask / restorer alignment / smoothing (MediaPipe Face Landmarker, YuNet) | license fix, must not regress | yes |
| C5 | transition shape (eased / longer crossfade, switch at a closed-mouth frame) | small (switch windows already ~1.1) | partly new; lower priority |

Not doing: LatentSync sampler / steps / guidance (track A; E5 done), CodeFormer weight sweeps (week 2 / 0.7 done), heavy diffusion VSR
(SeedVR2 / FlashVSR) unless C1-C3 fail, any NC model (KEEP, PGTFormer, DicFace, CoTracker ...) without the coordinator's go.
Each candidate: same window first (`decompose_video_stage.py` stage frames as the common input), full metric set (mouth_jitter_hr,
face_temporal, SyncNet dub-only, upper-face control), muxed audio md5 unchanged; then the four scoreable videos.
