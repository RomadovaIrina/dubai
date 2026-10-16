# Track B — restoration / temporal stabilization (`experiment/restoration-temporal`) — STATUS (program PAUSED 2026-10-16 ~23:20 UTC)

| phase | state | evidence |
|---|---|---|
| 0 baseline reproduction (cand5) | done, accepted | `env_phase0.md`, `baseline_phase0.md`, `baseline/` |
| 1 prior work, plan, licences | done, accepted | `PHASE1.md`, `LICENSES.md`, `LICENSES_desk_research.md` |
| 2 decomposition + first candidates (03 window) | done, accepted | `PHASE2.md`, `phase2_03_window/`, `metric_mouth_jitter_hr.md` |
| human-marked windows 24-25 s / 35-36 s | done, accepted | `human_windows_03.md` (dub-silent stretches: frozen mouth in a talking face + seam snap) |
| 3 clean line on clean0 (source fps, seams, sharpness, frozen mouth) | done | `PHASE3.md` |
| 4 main line on 01 / 03 / 04 / 05 + 02 | done (one open issue: SyncNet on 01 / 05) | `PHASE4.md`, `phase4/` |
| 5 final report | not started | – |

Base: clean-base 326e807 merged (c0b37d8). Rules: final licence rule (file licences decide), NC / unlicensed weights in
`/workspace/dub/quarantine_nc/` (buffalo_l, CodeFormer, ParseNet, RetinaFace R50, HDEMUCS); GFPGAN / RestoreFormer kept quarantined by the
user's choice; every video under `/workspace/dub/artifacts/` only (review clips and copies: `artifacts/restoration-temporal/sbs/`, index
`sbs_index.md`); GPU jobs under `flock /workspace/dub/gpu.lock`.

## Recommended main-line flags (clean, no restorer, no model in the post-process)

```bash
# 1) clean0 video stage (track A recipe, CLEAN0.md): --face-stack mediapipe --codeformer off on top of the cand5 flags
# 2) LatentSync raw frames of the same run: scripts/quality/decompose_video_stage.py --manifest <clean0 manifest> --frames all --out <dir>
# 3) source-fps composite:
python scripts/quality/fps_composite.py --manifest <clean0 manifest> --ls-raw <dir>/S0_ls_raw.mp4 --out <out>.mp4 \
  --retime nearest_rigid --boundary on --skin-texture source --lip-texture on --ls-sharpen 1.5 --ls-sharpen-lips 1.5 --edge-px 12
#    frozen-mouth option for dub-silent stretches (human decision pending): add --silence still|orig [--gaps <track C residual-gap JSON>]
```
(Production integration: step 2 is a research convenience — the composite should consume the LatentSync frames in-process instead.)

## Done (highlights)
* Common metric `scripts/quality/mouth_jitter_hr.py` (v4 + fps-aware), adopted by both tracks; it explains the human marks.
* Decomposition: LatentSync = 16-frame window seams + softness; CodeFormer = texture popping (+36 %); paste-back neutral.
* Source-fps composite: 25 fps judder removed (upper-face jerk 1.36 -> 1.03), spec 9.4 met on all five videos, audio bit-identical.
* C2 seam interpolation: seam jump 1.34-1.56 -> 1.04-1.20; feather 7.7 px -> 11.4 px (spec 10-15).
* Clean sharpness recovery (source texture + unsharp): mouth 1.02-1.19 / lower face 0.95-1.04 of the source vs clean0 0.67-0.87 / 0.87-0.97.

## Open / next steps (when resumed)
1. SyncNet drop on 01 (-0.7) / 05 (-0.3 / -0.6) vs cand5: score clean0 01 / 04 / 05 and MAIN with `--ls-sharpen 0` to locate the cause.
2. Adaptive sharpening amount (target sharpness ~1.0 per video; 01 / 05 are over-sharpened at the fixed 1.5 + 1.5).
3. Strong seams on 04 / 05 still above the source (33 % / 28 % vs 16 % / 12 %): compare with track A's generator-side seam fix.
4. Frozen-mouth option: human verdict on OPT1 vs OPT2 (p3 / p4 clips), then feed track C's residual-gap JSON via `--gaps`.
5. Clean processing-cost measurement on a quiet box (post-process numbers so far are upper bounds: 13-80 s / video-min under contention).
