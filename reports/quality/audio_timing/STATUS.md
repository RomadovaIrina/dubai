# Agent C (English dub timing, server B, CPU) — STATUS at pause, 2026-10-16 ~23:20 UTC

Program paused by the user. Branch `experiment/audio-timing` (from `clean-base` 326e807). GPU minutes used: **0** (the granted layer-3 GPU
step was cancelled before it started).

## Done
* P0 `p0_inventory.md`: what exists / default / opt-in; baseline residual gaps on 01/03/04/05. Key finding: every part that ends before a
  gap ≥ 0.35 s already plays at the x0.90 slow-down floor, so tempo alone cannot fill the gaps.
* P1 `p1_timing_ab.md`: offline A/B on the same raw TTS (12 policies × 4 videos, `timing_ab.py`, `dub_gaps.py`, `timing_spread.py`,
  CPU WER/CER `eval_cpu.py`). New lever `--spread pauses` (leftover silence of a part → late start ≤ 0.2 s + pauses ≤ 0.25 s at phrase
  boundaries) + `--sentence-flow-gap 0.2`.
  Four videos: LatentSync frames inside gaps ≥ 0.35 s 525 → 382 (L1) → 344 (L1 + floor 0.875) → 319 (L1 + 0.85); gaps > 1 s 6 → 3; longest
  2.0 → 1.3 s; weak-boundary pauses 4 → 2; WER unchanged; CTC CER +0.01…+0.03 on 04/05 with the lower floors. Reviewed 03 24.5 s gap
  0.5 → 0.1 s; 03 34.6 s stays 0.9 s (translation length).
* P2 `p2_proposal.md`: `--spread` integrated in `run_clean_pipeline_05.py` (default off; offline recut reproduces P1 sample-exact);
  residual-gap lists for the video side in `residual/` (cand5, L1 + 0.875, L1 + 0.85 per video, frame indices + context); layer-3 design.
* Layer 3 (partial, CPU only): `refit_offline.py` plan → Qwen per-burst literal rewrite (v3: only short pieces, word range, boundary-repeat
  check, faithfulness + fluency judges). Rewrites in `/workspace/dub/artifacts/audio-timing/refit/<id>/rewrite.json` (log
  `refit/rewrite.log`; copies in `reports/quality/audio_timing/refit/`); NOT synthesised. Result: **6 of 17 sentences passed all guards** (03 u03 / u04 / u08,
  04 u05, 05 u03, 01 u14); rejections: added information 3, not fluent 3, boundary repeat 1, length not met 4. The run finished at 23:17 UTC.
* Clean background bed `bed_p1.md` + `LICENSES.md`: on 03 Spleeter 2stems (MIT; weights = release asset of the MIT repo, counsel note) has
  no detectable voice residue and continuous ambience (−0.1 dB vs HDEMUCS −3.0 dB); UMX-HQ leaks the Russian voice; UMX-SE / room tone are
  clean but dip 2–3 dB under speech.

## Recommended flags (on top of cand5 audio flags)
`--sentence-flow-gap 0.2 --spread pauses --fill-slowdown 0.875` (0.85 only if the listening test accepts the slower speech).
Bed: Spleeter 2stems instead of HDEMUCS, pending the listening verdict and a licence confirmation for the release-asset weights.

## Listening material (laptop sync)
`/workspace/dub/outbox/audio-p1/` (50 clips + README: original / cand5 / L1 / L1 + 0.875 / L1 + 0.85, original frames, dub voice only,
lips NOT re-synced), `/workspace/dub/outbox/audio-bed-p1/` (21 clips + README: 7 beds incl. no-bed and the NC reference, 03).

## Open / next steps
1. Human verdict: floor 0.875 vs 0.85 (clips audio-p1), bed Spleeter vs room tone (clips audio-bed-p1).
2. Layer 3: review the accepted rewrites by hand before any synthesis (2 of the first 5 accepted are wrong: 03 u08 ends "…and I wish",
   04 u05 "что вперёд" → "That's great."); then the GPU step (~10 min, ≤ 6 GB, `gpu_refit.sh` under `flock /workspace/dub/gpu.lock`) and
   `refit_offline.py build` (per-unit acceptance vs L1).
3. One E2E GPU run of the recommended flags (the offline recut is the controlled A/B; a full run re-synthesises everything).
4. Bed on 01/02/04/05, then `--bed-separator spleeter|roomtone` in `background_bed.separate_background` (HDEMUCS only behind the NC guard).
5. Hand the residual lists (`residual/*_L1_0.875.residual.json`) to the video side (silent face / pass-through).
