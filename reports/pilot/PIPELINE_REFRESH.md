# Pipeline refresh — pilot items 0.6 / 0.7 / 0.8 re-measured on the current production candidate (2026-10-02)

```text
HEAD:   afa9b0d5dd0f7490c1955826b009b1ba9f40eedc  (origin/tts-intelligibility-fix, "Add interim TTS intelligibility v2 E2E results"; remote HEAD identical at clone time)
GPU:    NVIDIA GeForce RTX 5090, sm_120, 32607 MiB, driver 580.126.09 (CUDA 13.0 driver) · torch 2.7.1+cu128 · /venv/dabai Python 3.11
host:   clean container, cgroup CPU quota 13.6 vCPU, 1 TB RAM, 35 GB disk (4.5 GB free after everything)
setup:  scripts/setup_dabai.sh (stages 0-9, exit 0; first attempt died on a 30 s uv download timeout -> rerun with UV_HTTP_TIMEOUT=900), scripts/download_models.sh (exit 0)
third_party: latentsync a229c394, CodeFormer b33cc7d6 (same as the previous measurements)
smoke:  11/11 PASS  (the 10 legacy components + the new tts-intelligibility check; reports/smoke.md 2026-10-02T16:41:37Z)
runner: run_clean_pipeline_05.py --target-lang en --alignment burst --video-backend optimized --codeformer optimized   (runner defaults:
        fp16 token-ID guard, natural chunking, TTS QA/retry, duration-aware re-translation, atempo cap 1.15 / hard 1.2, burst alignment,
        optimized LatentSync (gate 0.24/0.6, RetinaFace stride 3, DeepCache 5, batch 2), optimized CodeFormer w 1.0 / batch 8 / fp16 / insightface landmarks)
E2E:    5/5 outputs valid (/tmp/pilot_refresh/final/<id>_final.mp4 + manifest + nvsmi log); wall 386 / 269 / 750 / 438 / 396 s (01..05); GPU idle before every run
```

## 0.6 — video contour timing (04, formula stage_s / source_s * 60)

```text
OLD (2026-09-15, legacy):  LatentSync CLI whole video 403.2 s/video-min + upstream inference_codeformer.py CLI 222.2 s/video-min = 625.4 s/video-min   (FAIL_TARGET, 2.08x)
NEW (current optimized):   LatentSync 247.1 s (median of e2e 245.0 / video-stage repeat 249.2 / sweep pass 251.2) = 275.0 s/video-min
                           CodeFormer  31.2 s (31.9 / 30.4; sweep w1.0 29.4)                                         =  34.7 s/video-min
                           combined                                                                                   = 309.7 s/video-min
TARGET: <= 300 s/video-min
VERDICT: FAIL_TARGET — 3 % over budget (LatentSync alone 275 = 92 % of the budget); 2.0x faster than the legacy contour
informational: peak VRAM 24156 MiB (e2e, nvidia-smi device-wide; 23909 repeat; 25230 in the 5-instance sweep) — 0.4 not reopened;
               CodeFormer 1044 faces (speech-gated LATENT_SYNC frames, 77 % of 1349), 0.0298 s/face (legacy 0.1481 s/frame, 512x936 output);
               LatentSync per lip-synced minute 352 s (old whole-video 403); verify pass 15-16 s is inside the LatentSync number
evidence: reports/pilot/refresh_0.6_current_pipeline.{json,md}
```

## 0.7 — current optimized CodeFormer vs SyncNet (04, one LatentSync pass, A/B in memory, strict policy unchanged)

```text
OLD (legacy CLI, LatentSync-only input, own audio):   before 3.21/0 -> w0.3 2.85 (-0.36) · w0.5 2.97 (-0.24) · w0.7 3.06 (-0.15) · w0.9 3.17 (-0.04) · w1.0 3.17 (-0.04)   NO_STRICT_SAFE_WEIGHT
NEW (optimized in-memory, same LS frames, dubbed audio): before 2.90/0 -> w0.3 2.51 (-0.39) · w0.5 2.61 (-0.29) · w0.7 2.69 (-0.21) · w0.9 2.73 (-0.17) · w1.0 2.82 (-0.08)
      AV offset 0 -> 0 at every w; geometry 464x848 / 1349 frames / 53.96 s preserved at every w (legacy changed it to 512x936);
      CodeFormer 29.4-30.1 s per weight (32.7-33.5 s/video-min); production w1.0 re-assembled bit-identical to the pipeline's own output
VERDICT: formal 0.7 = FAIL (NO_STRICT_SAFE_WEIGHT) — confidence still decreases monotonically with lower w; best effort w = 1.0 (-0.08 = -2.8 %)
note: the A baseline (2.90) is not the legacy 3.21: it is the gated/burst pipeline output with the DUBBED track; the comparison that matters is A vs B on identical frames
evidence: reports/pilot/refresh_0.7_current_codeformer.{json,md}  (script scripts/pilot/refresh_07_codeformer_sweep.py)
```

## 0.8 — SyncNet on the five CURRENT final E2E outputs

```text
pipeline validation 5/5 PASS · SyncNet scoreable 4/5 · N/A_NO_FACE 1 (02: VALID_FACE 0, LatentSync segments 0 — valid dubbed output) · FAIL 0
old per-video (0.8 2026-09-18, slot alignment, CodeFormer OFF):   01 6.84/0 · 02 N/A · 03 4.65/0 · 04 2.76/0 · 05 2.61/0   (orig 5.36/1 · – · 4.89/2 · 3.24/2 · 1.71/2)
new per-video (burst + TTS v2 + CodeFormer optimized w1.0):       01 6.74/0 · 02 N/A · 03 5.02/0 · 04 2.82/0 · 05 2.51/0   Δ vs original +1.38 · – · +0.13 · -0.42 · +0.80
aggregate (scoreable):  old mean 4.215 / median 3.705 / min 2.61 / max 6.84   ->   new mean 4.273 / median 3.92 / min 2.51 / max 6.74 ; max |AV offset| 0
verdict: CLOSED — 5/5 valid, 4/5 measured, 1 N/A; contract threshold still UNDECIDED (spec defines no aggregation rule)
informational (manifests): TTS parts 35/19/21/14/11, QA retries 3/3/2/1/0 (success 3/3/1/1/–; 1 part still bad on 03 = one-word "forward"),
   atempo max 1.20/1.20/1.15/1.15/1.025 (groups > 1.15: 1/1/0/0/0, none > 1.2, no slot fallback, 1 cut group on 02),
   LatentSync 76.0/0/435.5/245.0/204.8 s, CodeFormer 7.8/0/57.1/31.9/26.1 s (260/0/1927/1044/868 faces), peak VRAM 25178/8252/25156/24156/24156 MiB
   intelligibility of the final audio (independent ASR, scripts/diag/eval_run.py, unit-mean): 01 WER 0.018 / CTC CER 0.066 / bad units 0 · 02 WER 0.089 / CTC CER 0.150 / bad units 2 · 03 WER 0.110 / CTC CER 0.099 / bad units 1 · 04 WER 0.024 / CTC CER 0.052 / bad units 0 · 05 WER 0.051 / CTC CER 0.215 / bad units 2
evidence: reports/pilot/refresh_0.8_current_e2e.{json,md}, per-video pages reports/pilot/refresh_videos/<id>.{md,json}
```

## OTHER PILOT ITEMS

```text
0.1 0.2 0.3 0.4 0.5 0.9 0.10 0.11:  not rerun — unaffected by the TTS/alignment/CodeFormer changes or intentionally excluded (no FlashAttention probe,
no Qwen Q5/Q8, no CPU benchmarks, no cold start, no sequential S, no Blackwell re-validation, no VRAM residency suite, no formal 0.5).
The smoke run is a prerequisite only, not a re-measurement of 0.1/0.2. Production settings were not changed for any measurement.
```

## regressions

- **0.6 still FAIL_TARGET** (309.7 vs 300): LatentSync alone needs 275 s/video-min on 04 (77 % of frames lip-synced); CodeFormer adds 35.
- **0.7 still FAIL** under the strict policy: every weight lowers SyncNet confidence (w 1.0: -0.08); the direction and monotonicity of the legacy result are reproduced on the optimized implementation.
- **vs the interim week-3 burst E2E numbers** (not formal pilot evidence): SyncNet 04 3.20 -> 2.82, 05 2.94 -> 2.51, 03 5.88 -> 5.02, 01 6.45 -> 6.74. The TTS-intelligibility v2 changes (cap 1.15/1.2, natural chunking, fewer/longer parts) leave more of the original speech without dubbed speech (coverage IoU 03 0.61, 04 0.54, 05 0.38, 02 0.46 — `dub_coverage.py`), which is what SyncNet sees. Versus the formal old 0.8 (slot alignment) the dataset is neutral-to-better (mean 4.215 -> 4.273).
- 04 and 05 stay below their originals' SyncNet (04 2.82 vs 3.24) — 05 is above (2.51 vs 1.71).

## improvements

- Video contour 625 -> 310 s/video-min on 04 (LatentSync 403 -> 275, CodeFormer 222 -> 35; CodeFormer per face 0.148 -> 0.030 s, geometry preserved).
- 0.8: 03 +0.37 (4.65 -> 5.02) and above its original; AV offset 0 on every scoreable output; dataset median 3.705 -> 3.92; 5/5 valid, no LatentSync failure, no "Face not detected".
- TTS: fp16 token guard 0 violations on all runs; QA retries fixed 8 of 9 flagged parts; no part above atempo 1.2 (previous interim: up to 1.6).
- Environment reproduced from scratch on a clean box in ~25 min (setup 5 min, weights 10 min, smoke 8 min), smoke 11/11.

## unexpected findings

- `scripts/setup_dabai.sh` stage 8 appends `ACTIVE_VENV=`/`HF_HOME=` to `/workspace/.env` without a newline check: the provided `.env` had no trailing newline, so `ACTIVE_VENV=dabai` was glued onto the HF_TOKEN line (token invalid for pyannote). Fixed by inserting the newline; the setup script was not changed.
- First setup attempt failed on uv's 30 s per-file download timeout (nvidia-cudnn 693 MB); `UV_HTTP_TIMEOUT=900` fixed it.
- `reports/pip-freeze.lock.txt` drifted on rebuild: 7 patch-level bumps (charset-normalizer, fonttools, gdown, ipykernel, onnx, platformdirs, sqlalchemy) and `whisperx`/`nltk` absent (they were manual `--no-deps` installs of Phase C, not in the setup script). Torch/CUDA/transformers/diffusers pins unchanged, smoke 11/11.
- The runner banner prints "(CodeFormer disabled)" even with `--codeformer optimized` (cosmetic, `run_clean_pipeline_05.py` line 660); manifests record the real configuration. The 0.8 benchmark's hard-coded "CodeFormer OFF" descriptions were replaced by text derived from the manifests (methodology untouched).
- `run_smoke.sh` has 11 tests, not 10 (tts-intelligibility was added with the TTS fix).
- Peak VRAM is 24.2-25.2 GB on the face videos (previous reports: 22.5-23.5 GB); the sweep with 5 CodeFormer instances resident peaked at 25.2 GB — still 7 GB below the 32 GB card.
- 35 GB disk is the practical limit of this box: venv 8.2 GB + weights 19 GB + base image; the CodeFormer smoke dumps 754 MB of PNGs (deleted); 4.5 GB free at the end.
- 03: the last unit "вперед" -> "forward" (0.42 s slot) fails TTS QA after 3 attempts ("forward forward", 2.2 s) and runs 0.9 s past the end of the timeline before being cut — one-word parts remain the weak spot of Chatterbox; see reports/pilot/refresh_03_analysis.md.
