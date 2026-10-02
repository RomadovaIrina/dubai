# Pilot 0.7 refresh — CURRENT optimized CodeFormer vs SyncNet (video 04, 2026-10-02)

Verdict (strict policy unchanged: Δconf >= -0, |AV offset| worsening <= 0): **formal 0.7 = FAIL** (NO_STRICT_SAFE_WEIGHT); best effort w = 1.0 (Δconf -0.08)

Method: ONE LatentSync pass (current optimized backend, 9 LATENT_SYNC segments, 1044 frames) on the dubbed track of the E2E run; A = those frames assembled with CodeFormer OFF; B = the same frames restored in memory by `codeformer_accel.CodeFormerAccel` (batch 8, fp16, insightface landmarks) at each w, same crossfade / audio / encoder. SyncNet: upstream evaluator, unchanged. Production output (w 1) re-assembled bit-identical to the pipeline's own file: True.

A (CodeFormer OFF): SyncNet confidence **2.9**, AV offset **0** frames; 464x848, 1349 frames, 53.96 s

| w | SyncNet before | SyncNet after | Δconf | offset before | offset after | Δ\|offset\| | strict pass | CodeFormer s | s/face | CF s/video-min | faces | geometry / frames / duration vs A |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.3 | 2.9 | 2.51 | -0.39 | 0 | 0 | +0 | NO | 30.12 | 0.0288 | 33.5 | 1044 | 464x848 1349f 53.96s OK |
| 0.5 | 2.9 | 2.61 | -0.29 | 0 | 0 | +0 | NO | 29.70 | 0.0284 | 33.1 | 1044 | 464x848 1349f 53.96s OK |
| 0.7 | 2.9 | 2.69 | -0.21 | 0 | 0 | +0 | NO | 29.38 | 0.0281 | 32.7 | 1044 | 464x848 1349f 53.96s OK |
| 0.9 | 2.9 | 2.73 | -0.17 | 0 | 0 | +0 | NO | 29.49 | 0.0282 | 32.8 | 1044 | 464x848 1349f 53.96s OK |
| 1 | 2.9 | 2.82 | -0.08 | 0 | 0 | +0 | NO | 29.40 | 0.0282 | 32.7 | 1044 | 464x848 1349f 53.96s OK |

LatentSync (this pass): stage 251.225 s = 279.657 s/video-min, verify 16.447 s, load 18.196 s; torch peak 9528 MiB during the video stage (LatentSync + 5 CodeFormer instances resident).

## Comparison with the legacy 0.7 (2026-09-15, upstream CLI, output 512x936)

old before 3.21 / offset 0; old verdict `CLOSED — NO_STRICT_SAFE_WEIGHT`

| w | old conf after (legacy CLI) | old Δconf | new conf after (optimized) | new Δconf |
|---:|---:|---:|---:|---:|
| 0.3 | 2.85 | -0.36 | 2.51 | -0.39 |
| 0.5 | 2.97 | -0.24 | 2.61 | -0.29 |
| 0.7 | 3.06 | -0.15 | 2.69 | -0.21 |
| 0.9 | 3.17 | -0.04 | 2.73 | -0.17 |
| 1 | 3.17 | -0.04 | 2.82 | -0.08 |

Evidence: this file + `reports/pilot/refresh_0.7_current_codeformer.json`; media in `/tmp/pilot_refresh/r07` (not tracked).
