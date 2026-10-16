# Phase 0 — cand5 baseline reproduced on Server B (2026-10-16)

Environment: `env_phase0.md` (RTX 5090 exclusive, 15.36 vCPU quota, HEAD `559d700`, setup / models / smoke 11/11 PASS).
Baseline = cand5 (`reports/quality/feedback_2026-10-13/README.md`). Checked before running: the commits after 4fbfbf3 (speed levers, identity
check, llama CUDA build, Qwen on GPU, resident worker, iteration-3 report, 559d700 ordered translation / per-word TTS QA / `--ls-steps`) are all
opt-in or speed-only; none defines a newer human-reviewed full candidate. The only human A/B after cand5 is `--sentence-flow-gap 1.5` on
audio-only listening clips ("better than 0, not yet ideal", iteration3 report) — an audio option, not a candidate; not used.
Qwen stays on the CPU (user decision 2026-10-12), so the audio side is exactly cand5.

Runs: `baseline/baseline_run.sh 03`, then `01 02 04 05` (sequential, GPU idle before each, nothing else on the GPU), metrics with
`baseline/baseline_metrics.sh` (e2e_report --syncnet, face_temporal, pause_rhythm, audio_stages, md5 of the muxed audio stream) run between /
after the pipeline runs, never concurrently. Outputs and work dirs (not in git): `/workspace/dub/artifacts/baseline/<id>/{final,work,logs}`;
the work dir `work/work_<id>/` holds the dub track, master_25fps and every audio-stage file for video-only re-runs.

## Table (SyncNet "dub-only" = same frames with the dub voice only, comparable to the 2026-10-12/13 reports)

| metric | 01 | 02 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|---:|
| valid | PASS | PASS | PASS | PASS | PASS |
| wall s (s / video-min) | 549 (324) | 256 (139) | 881 (585) | 426 (474) | 500 (572) |
| audio s / LatentSync s / CodeFormer s | 452 / 54 / 7.0 | 221 / 0 / 0.0 | 393 / 402 / 54.3 | 157 / 219 / 29.4 | 249 / 191 / 25.6 |
| LS frames | 260 | 0 (no face) | 2025 | 1074 | 949 |
| peak VRAM MiB (nvidia-smi, device) | 22456 | 8316 | 23484 | 22436 | 22458 |
| SyncNet mix / AV | 7.89 / 0 | N/A (no face) | 5.46 / 0 | 3.69 / 0 | 3.15 / 0 |
| SyncNet dub-only / AV (cand5 ref) | 7.86 / 0 (7.86) | – | 5.69 / 0 (5.79) | 3.81 / 0 (3.92) | 3.66 / 0 (3.70) |
| original SyncNet | 5.36 | – | 4.89 | 3.24 | 1.71 |
| Whisper WER / CTC CER (cand5 ref WER) | 0.019 / 0.051 (0.016) | 0.073 / 0.168 (0.090) | 0.009 / 0.043 (0.004) | 0.009 / 0.048 (0.009) | 0.000 / 0.169 (0.000) |
| bad units / interjections / final word missing | 0 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 |
| source speech w/o dub s (cand5 ref) / VAD IoU | 8.6 (7.2) / 0.885 | 11.6 (10.0) / 0.716 | 11.1 (9.3) / 0.790 | 6.1 (5.0) / 0.750 | 7.4 (6.8) / 0.686 |
| LS frames dub-silent while the original speaks | 24 | 0 | 251 | 299 | 520 |
| repeated words / lost numbers / cut groups | 1* / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| atempo max / > 1.15 | 1.146 / 0 | 1.15 / 0 | 1.15 / 0 | 1.025 / 0 | 1.0 / 0 |
| hot inner tails / heads | 2/1 of 9 | 6/10 of 20 | 4/9 of 23 | 5/6 of 13 | 10/11 of 14 |
| LS <-> original switches (/min) (cand5 ref /min) | 1 (0.59) (0.6) | – | 8 (5.31) (5.3) | 12 (13.34) (15.6) | 10 (11.44) (11.4) |
| short LS < 2 s / short original gaps < 2 s | 0 / 0 | – | 0 / 2 | 2 / 4 | 0 / 2 |
| mouth landmark jitter / opening jitter, LS core, vs source | 1.033 / 1.045 | – | 0.884 / 0.704 | 0.982 / 0.766 | 1.029 / 0.910 |
| mouth pixel flicker LS core / switch window, vs source | 1.092 / 1.021 | – | 0.890 / 0.973 | 0.964 / 0.979 | 0.997 / 0.992 |
| upper-face flicker / sharpness, LS core, vs source | 1.007 / 1.005 | – | 1.004 / 0.984 | 1.000 / 1.004 | 1.002 / 1.007 |
| switch: upper-sharp step % (source) / mouth jump ratio | 98.8 (99.0) / 1.02 | – | 4.0 (4.3) / 0.99 | 12.6 (12.5) / 1.01 | 17.4 (17.2) / 1.02 |
| muxed audio stream md5 (reference for video-only re-runs) | f9c87d778b1b | d0789251154f | 6cf53a89cf73 | cb31a8f48de5 | b1447c1ea8cb |

\* 01: CTC hears "twenty" twice in "20-25" — the known decoder artifact from the 2026-10-12 review, not a TTS repeat.

## Reading

* **Reproduced.** Face / video numbers match the cand5 report within run noise: SyncNet dub-only 7.86 / 5.69 / 3.81 / 3.66 vs 7.86 / 5.79 /
  3.92 / 3.70, AV offset 0 everywhere, 03 switches 8 (= cand5), 05 10 (= cand5), 04 12 (cand5 14), face_temporal ratios within ±0.03
  (03 lip jitter 0.884 vs 0.89, mouth flicker 0.890 vs 0.90, upper flicker 1.004 vs 1.00).
* **Audio differs a little, as expected with CPU Qwen** (path-dependent greedy decoding, documented 2026-10-12): source speech without dub
  is 0.6-1.8 s higher per video than the cand5 report, 02 WER lower (0.073 vs 0.090), 05 has one marginal CTC-bad unit (CER 0.311, Whisper
  WER 0 — "take off and start saying something"). Same quality class; no old regression (interjections 0, final word missing 0, atempo
  > 1.15: 0, cut groups 0). For cross-server comparisons the coordinator's canonical work dirs (same dub audio) are the reference.
* **Runtime (exclusive GPU, 15.4 vCPU):** 324 / 139 / 585 / 474 / 572 s per video-minute (median 474). The video stage of 03 is
  LatentSync 402 s + CodeFormer 54 s for 2025 LS frames (CodeFormer = 26.8 ms/frame).
* **Observation that shapes Phase 1 (not a conclusion yet):** on every scoreable video the existing face_temporal metrics already rate the
  cand5 mouth as calm as the source (LS-core mouth flicker 0.89-1.09, landmark jitter 0.88-1.03, switch windows 0.97-1.02) — yet the
  2026-10-13 human review says the mouth twitches constantly. The current metrics (64x64 grey crops of a fixed box, insightface landmarks on
  the output) are probably blind to what the reviewer sees; the decomposition in Phase 2 needs higher-resolution, face-stabilised mouth
  measurements before any restorer is judged.
