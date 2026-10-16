# Phase 4 — main line on 01 / 03 / 04 / 05 (+ 02 pass-through), Server B, 2026-10-16 (program paused right after)

Main line (`scripts/quality/fps_composite.py`, clean: MediaPipe stack, no restorer, no model in the post-process):

```bash
python scripts/quality/fps_composite.py --manifest <clean0 manifest> --ls-raw <decompose S0_ls_raw.mp4> --out <id>_MAIN.mp4 \
  --retime nearest_rigid --boundary on --skin-texture source --lip-texture on --ls-sharpen 1.5 --ls-sharpen-lips 1.5   # --edge-px 12 default
```
Inputs: clean0 runs (`candidate_video_stage.py`, CLEAN0.md recipe) on the cand5 work dirs of Server B, clean decompositions
(`decompose_video_stage.py --frames all --face-stack mediapipe`; each reproduces its clean0 output within encoder noise, mean |Δ| 1.5-1.7).
Runner: `scripts/quality/run_phase4.sh` (spec checks, common metric vs the SOURCE at its own fps, SyncNet under the GPU lock on an fps=25
rendition). Outputs: `/workspace/dub/artifacts/restoration-temporal/phase4/` (copies in `sbs/phase4/`).

## Spec checks (9.4 frame rate / resolution, audio identity, robustness, 8.4 feather)

| video | source fps (avg) -> output | frames source = output | resolution | duration delta s | audio md5 = cand5 baseline | fallback frames | edge px (10-90 %) | LS frames touched |
|---|---|---|---|---:|---|---:|---:|---:|
| 01 (VFR) | 25.003 -> 25.011 | 2542 = 2542 | 480x848 = | <= 0.15 | yes | 0 | 11.5 | 260 |
| 02 (no face) | 29.982 -> 29.991 | 3324 = 3324 | 464x848 = | <= 0.15 | yes | 0 | – | 0 (pass-through) |
| 03 | 30.000 -> 30.011 | 2710 = 2710 | 464x848 = | <= 0.15 | yes | 0 | 11.4 | 2434 |
| 04 | 30.000 -> 30.018 | 1617 = 1617 | 464x848 = | <= 0.15 | yes | 0 | 11.4 | 1298 |
| 05 (VFR) | 29.924 -> 29.942 | 1567 = 1567 | 464x848 = | <= 0.15 | yes | 0 | 11.5 | 1034 |

All five PASS (`phase4/<id>_MAIN.checks.json`). Every earlier pipeline output (cand5, clean0) was 25 fps CFR.

## Quality (LS-core mouth / lower face, ratios vs the source; clean0 vs its 25 fps master)

| video | mouth resid MAIN (clean0) | mouth sharpness MAIN (clean0) | lower-face sharpness MAIN (clean0) | 16-frame seam jump MAIN (clean0) | seams > 1.5x MAIN / clean0 / source | upper face resid | SyncNet mix / dub-only MAIN | cand5 |
|---|---|---|---|---|---|---:|---|---|
| 01 | 1.22 (1.03) | 1.19 (0.87) | 1.04 (0.95) | 1.05 (1.38) | 13 % / 50 % / 19 % | 0.99 | 7.23 / 7.12 | 7.89 / 7.86 |
| 03 | 1.12 (1.03) | 1.03 (0.67) | 1.02 (0.87) | 1.04 (1.56) | 16 % / 58 % / 18 % | 1.02 | **5.50 / 5.78** | 5.46 / 5.69 |
| 04 | 1.03 (0.96) | 1.02 (0.80) | 0.98 (0.95) | 1.20 (1.35) | 33 % / 50 % / 16 % | 1.02 | 3.62 / 3.63 | 3.69 / 3.81 |
| 05 | 1.05 (0.91) | 1.16 (0.86) | 0.95 (0.97) | 1.07 (1.34) | 28 % / 47 % / 12 % | 1.05 | 2.81 / 3.04 | 3.15 / 3.66 |

AV offset 0 on every scoreable output.

## Reading
* Spec 9.4 (source frame rate / resolution), audio identity, 8.4 (10-15 px edge) and robustness: met on all five videos.
* Seams (8.5): the 16-frame window jump drops from 1.34-1.56 (clean0) to 1.04-1.20; 03 / 01 reach the source's level, 04 / 05 still have
  more strong seams than the source (33 % / 28 % vs 16 % / 12 %).
* Sharpness (8.3): mouth 1.02-1.19 and lower face 0.95-1.04 of the source (clean0 0.67-0.87 / 0.87-0.97). The fixed sharpening amount
  over-sharpens the softer sources (01 1.19, 05 1.16) -> next: an adaptive amount that targets ~1.0 per video.
* **Open issue — lip-sync on 01 / 05:** SyncNet 03 is at cand5 (5.50 / 5.78 vs 5.46 / 5.69) but 01 (-0.66 / -0.74), 05 (-0.34 / -0.62) and
  04 (-0.07 / -0.18) are lower. Not separated yet (program paused): candidates are the clean face stack itself (clean0 01 / 04 / 05 were
  not SyncNet-scored on this server), the over-sharpening of 01 / 05, and the source-timestamp retiming of the VFR sources (01, 05).
  First step when resumed: SyncNet of clean0 01 / 04 / 05 and of MAIN without sharpening (`--ls-sharpen 0`).
* Frozen-mouth options (OPT1 still lower face / OPT2 original pass-through) were rendered on 03 only (MAIN base) for the human review.
