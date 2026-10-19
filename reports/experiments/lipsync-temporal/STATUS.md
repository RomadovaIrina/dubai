# Track A — lip-sync / temporal generation (`experiment/lipsync-temporal`) — STATUS (PAUSED 2026-10-16 ~23:20 UTC, user decision)

Research question: is LatentSync itself the main source of the visible mouth / face jitter, or what is around it? Answer so far:
**both, in two separable parts** — (1) CodeFormer's per-frame re-synthesis of the mouth (around the generator; removed in clean0) and
(2) LatentSync's independent 16-frame windows (a jump every 0.64 s; inside the generator; fixable by per-step fusion, budget version open).

## Done (all pushed)

| item | where | SHA |
|---|---|---|
| Phase 0: cand5 reproduced on server A, 5/5 PASS, metrics | `PHASE0.md`, `baseline/` | 1088a73 |
| Phase 1: prior work, diagnosis, candidate shortlist, licence audit | `PHASE1_PLAN.md`, `LICENSES.md`, `research/` | 73f0578 |
| jitter-by-situation metric (`lipsync_jitter.py`, v3 = MediaPipe landmarks), seam bookkeeping | `scripts/quality/` | d4d1c94, db440fd |
| **clean0** = cand5 audio + LatentSync 1.6 on the MediaPipe face stack, no restorer (accepted, = branch clean-base) | `CLEAN0.md` | 326e807 |
| clean by default (`--face-stack mediapipe`, `--codeformer off`, `--background off`, `guard_nc()`, tripwires, clean smoke tests) | | 05dae5c, 65ff44c |
| NC weights quarantined (buffalo_l, CodeFormer, ParseNet, RetinaFace R50, HDemucs) | `/workspace/dub/quarantine_nc/README.md` | – |
| frame map source <-> 25 fps master (spec 9.4) + composite note | `FRAME_MAP.md`, `scripts/pilot/frame_map.py` | db440fd |
| head pose + voice-only pre-check (spec 5.4/5.5), robustness / 2 % status (spec 11) | `scripts/pilot/clean_face_stack.py` | de61988 |
| two faces: tracking + diarization-speaker -> face (spec 8.2) | same | 396835d |
| seam A/B on the clean base + fusion ceiling + bridge mode | `SEAMS.md`, `seams_table_v2.md` | 44e5d70, 6a8e214, this commit |
| speed notes (MediaPipe subsampling / GPU delegate) | `SPEED_NOTES.md` | 6a8e214 |

## Key numbers

03, clean0 vs cand5 (same audio, full video): mouth unexplained appearance change 1.29 -> **1.02** of the source (human-marked 24-25 s +
35-36 s: 1.24 -> 0.96), SyncNet dub-only 5.73 -> 5.70 (AV 0), crop jitter -26 %, mask-box jitter -46 %, mouth sharpness 0.98 -> 0.71
(no restorer), audio bit-identical. 04 full E2E clean0: PASS, LatentSync 255 s/video-min, MediaPipe pass 44 s/video-min.

Seam A/B, 03 window B (21.6-37.6 s, phase-aligned, contains the human-marked moments) — seam jump = mouth d1 at the recorded seam / local
median (source at the same frames in brackets); full table `seams_table_v2.md` (window A in `seams_table_v1.md`):

| arm | LatentSync cost vs clean0 | seam jump median | seams > 1.5x | mouth resid / twitch | human-marked resid | SyncNet dub-only |
|---|---|---|---|---|---|---|
| clean0 (16-frame windows) | – | 1.71 (0.83) | 67 % (8 %) | 0.99 / 1.08 | 0.94 | 5.64 |
| 24-frame windows | +1 % | 1.74 | 56 % | 0.99 / 1.08 | 0.91 | 5.94 |
| 24f + 1-frame continuation | +7 % | 1.71 | 59 % | 0.99 / 1.08 | 0.92 | 6.18 |
| 24f + 2-frame continuation | +12 % | 1.66 | 53 % | 0.98 / 1.08 | 0.89 | 6.09 |
| 16f + 4-frame continuation | +29 % | 1.51 | 53 % | 1.00 / 1.06 | 0.92 | 5.80 |
| **per-step fusion, stride 8, no DeepCache (ceiling)** | **+322 %** | **0.96 (0.87)** | **8 % (6 %)** | **0.89 / 1.05** | **0.82** | 5.88 |
| **bridge 8 frames x last 4 steps (budget design)** | **+10 %** | **1.28** | **29 %** | **0.94 / 1.05** | **0.89** | 5.62 |

Reading: continuation / longer windows do not fix the seam; per-step fusion does (seams at source level, mouth calmer, SyncNet up) but
costs 4.2x LatentSync. **The budget design works at exactly the +10 % budget**: bridge 8 frames x last 4 DDIM steps
(`--bridge-len 8 --bridge-steps 4`: low-noise re-denoise of a short window centred on each seam, blended back) halves the jumpy seams
(67 % -> 29 %, median jump 1.71 -> 1.28), gives the lowest twitch index of the budget arms (1.05, = fusion ceiling) and the lowest
human-marked twitch (1.07), SyncNet unchanged (5.62). Only window B measured; larger bridges (16x4 / 16x8) and window A not run.

## Not done / in progress at the pause

* TBDub (Apache-2.0, fully allowed) is installed (`/venv/tbdub`, student weights) but **not run** (was next in the queue). Input clips
  and dub audio for both 03 windows are prepared (`artifacts/lipsync-temporal/p2c_seams/ref_{A,B}.mp4`, `work_03_*_C0/drive_16k.wav`).
* 05 per-frame pose-gate A/B (off / 30 / 20 deg) and the two-face render + "right face untouched" check: scripts ready
  (`artifacts/lipsync-temporal/pose05/run.sh`, `twoface/run.sh`), **not run**. Two-face speaker mapping verified offline (r 0.50 vs -0.05).
* Bridge arms 16x4 / 16x8, the window-A bridges, sbs clips of the bridge arm (no new renders at the pause): not run. The BR8k4 window clip is in `outbox/p2c-03winB-BR8k4/`.

## Next steps I would take

1. Bridge 8x4 on window A + bridges 16x4 / 16x8 on both windows; take the best within ~+10-20 % to full 03 and
   decide with a human look (sbs at 24-25 / 35-36 s and a long speech stretch).
2. TBDub student vs clean0 on both 03 windows (same gate / composite), with s/video-min against the 300 s/min norm.
3. 05 pose gate, two-face render, then a clean0 + seam-fix E2E regression on 01 / 03 / 04 / 05 (+ 02) on the canonical work dirs.
4. Speed phase: MediaPipe in parallel with the audio stage / every 2nd-3rd frame; seam fix cost compensation.

## Viewable outputs (laptop sync)

`/workspace/dub/outbox/`: clean0-03 (full 03), clean0-e2e-04, p2-03win-* (cand5-stack diagnostics before CLEAN-ONLY), p2c-03winA/B-*
(clean-base seam arms incl. F8nodc fusion ceiling). Side-by-side clips: `/workspace/dub/artifacts/lipsync-temporal/sbs/` (index.md).
