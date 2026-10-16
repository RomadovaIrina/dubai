# clean0 — commercially clean video stage (track A, 2026-10-16)

**clean0 = cand5 audio + LatentSync 1.6 on the clean MediaPipe face stack, no restorer, mouth-only composite.**
Flags (opt-in, defaults unchanged): `--face-stack mediapipe --codeformer off` on top of the cand5 flags.

```bash
scripts/setup_mediapipe.sh                     # /venv/mediapipe: mediapipe==0.10.21 (own env, subprocess; /venv/dabai pins untouched)
scripts/download_models.sh                     # now also models/mediapipe/face_landmarker.task (sha256-pinned); NC weights only with DUB_ALLOW_NC=1
# full E2E
DUB_EXTRA_ARGS="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on --gate-merge-gap-s 1.2 \
  --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2 --face-stack mediapipe --codeformer off" \
  bash scripts/pilot/refresh_e2e.sh <ids>
# video stage only, on an existing work dir (how the 03 numbers below were made)
python scripts/quality/candidate_video_stage.py --id 03 --tag clean0 --manifest <run>/final/03_final.manifest.json --out <dir> --mux final_mix \
  --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off --face-stack mediapipe
```

## What changed (scripts/pilot/clean_face_stack.py)

| part of the video stage | cand5 | clean0 |
|---|---|---|
| face router (VALID / SMALL / NO_FACE, w ≥ 50, h ≥ 80, aspect 0.2-1.5) | RetinaFace R50 (facexlib; WIDER FACE NC data) | MediaPipe face oval of the same pass, same thresholds |
| LatentSync crop / affine (3 anchors), verify pass, `--ls-landmark-smooth`, mouth-mask bbox (`--mask-smooth`) | InsightFace buffalo_l det_10g + 2d106det (NC) | MediaPipe Face Landmarker (Apache-2.0), VIDEO mode, one pass over the master in `/venv/mediapipe`; 478 points mapped to the 106-point indices LatentSync reads (`MAP106`, same named regions), constant anchor correction to LatentSync's InsightFace training alignment (`clean_face_stack_calib.json`, eye-distance units, median over 01/03/04/05) |
| LatentSync's own FaceDetector | loads buffalo_l (auto-downloads it if missing) | replaced by a stub; detections are always replayed |
| restorer | CodeFormer (S-Lab NC) | none (`--codeformer off`) |
| proof | – | NC files moved to `/workspace/dub/quarantine_nc/` (README: sha256 + reason); `guard_nc()` makes the InsightFace / LatentSync FaceDetector / facexlib loaders raise; after the run `find` = 0 NC files in the tree (nothing re-downloaded) |

Licences: `LICENSES.md` §0. LatentSync weights are Open RAIL++-M: its use restrictions must go into our ToS.

## 03 — clean0 vs cand5 (same dubbed audio and gate; video stage re-run on the cand5 work dir)

| metric | cand5 | clean0 |
|---|---:|---:|
| validation (duration / resolution / frames / decodable) | PASS | PASS |
| LS segments / LS frames / switches | 4 / 2025 / 8 | 4 / 2025 / 8 (MediaPipe routing = RetinaFace routing on 03) |
| muxed audio | final_mix | **bit-identical** (decoded md5 9d7c38ec…1d61) |
| SyncNet final mix / dub-only / AV offset | 5.50 / 5.73 / 0 | **5.45 / 5.70 / 0** |
| crop anchor acceleration px, as LatentSync receives it (±2 smoothing, rounded) mean / p95 | 0.474 / 0.833 | **0.352 / 0.833** (-26 %) |
| mouth-mask bbox acceleration px (±2) mean / p95 | 0.268 / 0.60 | **0.144 / 0.35** (-46 %) |
| common metric `mouth_jitter_hr.py`, whole 03, mouth (video / source): unexplained appearance change (resid) | 1.29 | **1.02** |
| … pixel acceleration / twitch index (jerk / vel) / velocity | 0.92 / 1.064 / 0.72 | **0.75 / 1.057** / 0.78 |
| … LS window start vs inside: mouth d1 step, share of window starts with a > 1.5x jump (source 17 %) | x1.38, 41 % | x1.54, **60 %** |
| … human-marked 600-625 + 875-900: resid / twitch / acc | 1.24 / 1.22 / 0.97 | **0.96 / 1.14 / 0.74** |
| … human-marked: window starts > 1.5x jump | 1/3 | 3/3 |
| … mouth sharpness (mean abs Laplacian) | 0.98 | **0.71** (no restorer) |
| … upper face resid / jerk | 1.02 / 1.02 | 1.02 / 1.02 |
| `lipsync_jitter.py` v3 (MediaPipe IMAGE-mode landmarks): mouth warping error ls_core / S1 speech / S3 head turn / S4 closed (source) | 1.31 / 1.35 / 1.40 / 1.34 | **1.07 / 1.12 / 1.11 / 1.06** |
| … lip-trajectory energy 6-12.5 Hz vs source | 1.11 | 1.06 |
| … window-seam step (mouth_warp) | x1.49 | x1.66 |
| video stage wall s (LatentSync s) | 504 LS in the E2E run | 714 (596 LS; MediaPipe pass 55 s) — **not a valid speed number**: the 20-thread common-metric job shared the CPU |

Reading:
1. Removing CodeFormer from the mouth path takes the mouth's motion-unexplained texture change from 1.29x to 1.02x the source
   (it re-synthesised the lips / teeth every frame; L0 on 40-58 s showed the same without any lip-sync: 0.96 -> 1.29). The price is a
   softer mouth (sharpness 0.71x) -> a clean restorer / sharpening is track B's question.
2. The clean crop is calmer than the InsightFace crop and keeps lip-sync (SyncNet -0.05 / -0.03, AV 0).
3. What remains is LatentSync's own **16-frame window seam** (every 0.64 s), now the most visible generator artefact (60 % of window
   starts jump > 1.5x; all three seams inside the human-marked windows). Next on the clean base: overlapping windows with per-step fusion
   (`scripts/quality/ls_stable.py`, S1), then the reference-frame (leakage) arm.
4. The human-marked moments also contain a content cause the video stage cannot fix: the dub is shorter than the source burst (24-25 s)
   or silent while the source talks (34.22-35.50 s), so a closed generated mouth sits in a talking face.

Side-by-side clips (ORIGINAL | CAND5 | CLEAN0, full + 2x lower-face zoom, incl. 24-25 s and 35-36 s): `/workspace/dub/artifacts/lipsync-temporal/sbs/`
(index: `clean0/sbs_index.md`).

## 04 — full E2E runner smoke (`refresh_e2e.sh` with the clean0 flags; audio regenerated, so audio metrics are a new take)

| | cand5 (Phase 0) | clean0 E2E |
|---|---:|---:|
| validation | PASS | PASS (router `mediapipe`, CodeFormer OFF, 1349/1349 face frames) |
| LS segments / frames | 6 / 1074 | 6 / 1074 |
| wall s (s per video-minute) | 622 (692) | 595 (662) — audio = CPU Qwen dominates |
| LatentSync s (s/video-min) | 244 (271) | 229 (255) |
| MediaPipe face-stack pass s (s/video-min) | – | 40 (44): CPU, depends only on the master -> can run in parallel with the audio stage (like `--video-prep parallel`), off the critical path |
| peak VRAM MiB (nvidia-smi) | 22451 | 21565 |

Output: `/workspace/dub/outbox/clean0-e2e-04/04_clean0_e2e.mp4`.
