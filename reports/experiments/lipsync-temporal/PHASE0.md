# Track A — Phase 0: cand5 baseline reproduced on server A (2026-10-16)

Server, versions, setup, weights, smoke and inputs: `STATUS.md` (RTX 5090 32 GB, driver 590.48.01, torch 2.7.1+cu128, 23.0 vCPU quota,
HEAD 559d700 = the coordinator's SHA, setup / download / smoke 11/11 PASS, input sha256 OK).

## Baseline choice

No human-reviewed candidate newer than cand5 exists at 559d700: the commits after 4fbfbf3 (cand5) are speed work (Qwen on the GPU,
resident worker, `--video-prep parallel` measured harmful) and opt-in flags without a human review (`--ordered-translation`,
`--tts-word-qa`, `--ls-steps`); the iteration-3 "sentence flow" listening test was audio-only on listening clips, not a full candidate.
-> baseline = cand5 exactly as specified (Qwen on the CPU), command in `/workspace/dub/artifacts/baseline/run_baseline.sh`:

```bash
PILOT_REFRESH_ROOT=/workspace/dub/artifacts/baseline/<id> DUB_EXTRA_ARGS="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive \
  --tts-gap-qa on --gate-merge-gap-s 1.2 --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2" \
  bash scripts/pilot/refresh_e2e.sh <id>          # 03 first, then 01 02 04 05; strictly sequential, nothing else on the GPU
```

Work dirs and outputs: `/workspace/dub/artifacts/baseline/<id>/{final,work/work_<id>,logs}`. Metrics: `e2e_report.py --syncnet`,
`face_temporal.py`, and the new `lipsync_jitter.py` (`/workspace/dub/artifacts/baseline/metrics/`); small copies in `baseline/` here.

## Results (exclusive GPU, 23 vCPU quota, DUB_THREADS 16)

| metric | 01 | 02 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|---:|
| valid (incl. no-OOM check) | PASS | PASS | PASS | PASS | PASS |
| wall s (s per video-minute) | 696 (410) | 537 (290) | 1210 (804) | 622 (692) | 661 (757) |
| audio stages s (CPU Qwen) | 543 | 456 | 570 | 292 | 366 |
| LatentSync / CodeFormer s · LS frames | 62 / 8 · 260 | 0 / 0 · – | 504 / 62 · 2025 | 244 / 34 · 1074 | 213 / 28 · 930 |
| peak VRAM MiB (nvidia-smi) | 22471 | 8347 | 23477 | 22451 | 22479 |
| SyncNet final mix / AV offset | 7.89 / 0 | no face | 5.50 / 0 | 3.71 / 0 | 2.99 / 0 |
| SyncNet dub-only / AV offset (original) | 8.02 / 0 (5.36) | – | 5.73 / 0 (4.87) | 3.85 / 0 (3.28) | 3.42 / 0 (1.72) |
| Whisper WER / CTC CER | 0.033 / 0.070 | 0.064 / 0.141 | 0.009 / 0.043 | 0.009 / 0.048 | 0.011 / 0.126 |
| bad units / interjections / final word missing | 1 / 1 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| atempo max / > 1.15 | 1.143 / 0 | 1.150 / 0 | 1.150 / 0 | 1.025 / 0 | 1.000 / 0 |
| source speech without dub s / VAD IoU | 9.2 / 0.878 | 9.6 / 0.756 | 11.1 / 0.790 | 6.1 / 0.750 | 5.4 / 0.753 |
| LS frames dub-silent while the original speaks | 25 | 0 | 251 | 299 | 457 |
| LS <-> original switches (per min) | 1 (0.6) | – | 8 (5.3) | 12 (13.3) | 10 (11.4) |
| face_temporal ls_core: mouth-lm jitter / mouth flicker / upper flicker / upper sharp | 1.05 / 1.09 / 1.00 / 1.00 | – | 0.90 / 0.89 / 1.00 / 0.98 | 0.98 / 0.96 / 1.00 / 1.01 | 1.03 / 1.00 / 1.00 / 1.01 |

Muxed audio: the decoded PCM of every output = its `final_mix_aac.m4a` (md5 identical, `baseline/sha256_baseline_outputs.txt`) -> a
video-stage experiment that muxes the same file keeps the audio bit-identical.

Against the cand5 numbers in `reports/quality/feedback_2026-10-13/README.md` (other box, shared GPU): same quality class — SyncNet dub-only
8.02 / 5.73 / 3.85 / 3.42 vs 7.86 / 5.79 / 3.92 / 3.70, switches/min 0.6 / 5.3 / 13.3 / 11.4 vs 0.6 / 5.3 / 15.6 / 11.4, upper face
1.00 vs 1.00-1.01. The audio side differs per take (WER 01 0.033 vs 0.016, 02 0.064 vs 0.090, source-without-dub 01 9.2 vs 7.2 s,
05 5.4 vs 6.8 s): CPU Qwen is path-dependent (documented 2026-10-12), so the texts / TTS takes of two runs differ. This is exactly why
both tracks should compare on one canonical work dir per video (rule 7) — this box's work dirs are ready to be the canonical set if wanted.
Runtime is valid (exclusive GPU) but on a 23-vCPU quota with CPU Qwen; the 03 video stage alone is 504 s LatentSync (≈ 4.0 LS frames/s).

## Jitter by situation (new metric `scripts/quality/lipsync_jitter.py`, output / master, 1.0 = as calm as the source)

`mouth_warp` = DIS optical-flow warping error in a head-stabilised mouth ROI (change that smooth motion does not explain).

| mouth_warp ratio (frames) | 01 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|
| S1 inside continuous dubbed speech | 1.27 (229) | 1.16 (1174) | 1.26 (471) | 1.10 (346) |
| S2 LS <-> original switch ±5 frames | 0.99 (10) | 1.09 (80) | 1.13 (120) | 1.05 (100) |
| S3 head turn / nod | 1.25 (175) | 1.20 (178) | 1.23 (1002) | 1.10 (844) |
| S4 closed mouth in the source | – (0) | 1.20 (112) | 1.28 (638) | 1.23 (21) |
| control: pass-through core | – | 1.02 (195) | 1.02 (215) | 1.01 (331) |
| **LatentSync 16-frame window seam: mouth_diff / mouth_warp step (seams)** | ×1.47 / ×1.42 (15) | ×1.51 / ×1.45 (123) | ×1.31 / ×1.40 (61) | ×1.38 / ×1.60 (54) |
| inner-lip gap median, output / source (LS frames) | 0.088 / 0.136 | 0.044 / 0.087 | 0.030 / 0.043 | 0.068 / 0.098 |

Frame ranges of every situation: `baseline/<id>_baseline.lipsync_jitter.json` -> `situation_ranges_s`; worst 1 s windows per
situation -> `worst_1s_by_situation`. Reading in `PHASE1_PLAN.md` §2.
