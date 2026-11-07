# r12 LS (LatentSync speed + sync) hand-over (02.11.2026, server r12-l, destroyed after this push)

- **Branch:** `experiment/r12` = 430e3a1 (k5 / k7) + merges of `origin/experiment/r10-face` (6543991) and `origin/experiment/r10-base` (7d61ec5).
  - The one conflict was in `fps_composite.py`: both flag sets are kept (`--cm-mouth-excl` for k5, FACE's `--colour-mouth-flat`), all default off.
- **New:**
  - `E2E_SYNC=1|k5|k7` in `scripts/r9/e2e.sh`.
  - Levers `LS_CFG_TRUNC`, `LS_CFG_DELTA` (default off).
  - `scripts/r12/ls_base.sh`, `ls_arm.sh` (`QUICK=1` = SyncNet only), `bench_fp8.py`.
  - Gate rows in `gate_rows.md`.

## Method
- **GPU time:** face GPU-busy s/vmin = NVML sampler (`scripts/r10/gpu_sampler.py`), generator + k5 composite, warm process, filtered by job.
  - r12-l had no pynvml in `/venv/main`, so I installed `nvidia-ml-py`. Without it the sampler silently writes nothing.
- **Sync:** SyncNet = the gate's `sync_speech`.
- **Arms:** each arm re-runs only the generator on the baseline's audio (`ls_arm.sh`).
  - No flock around the generator: `r4_video.sh` takes `gpu.lock` itself, so a nested lock deadlocks.

## Baselines on r12-l
E2E_FX=1 E2E_SYNC=1, gate vs the r9 + fix composite of the same layers (SyncNet / strong flashes):

| video | r9 + fix | k5 | gate |
|---|---|---|---|
| m18 | 7.743 / 7 | **8.304** / 5 | PASS; step 0 verified (k5 8.33 ± 0.05) |
| m03 | 7.62 / 17 | 8.086 / 22 | stick 0 → 2 (the eye-accepted closures) |
| old03 | 7.216 / 36 | 7.496 / 47 | occluder ring 0 → 1 (flashes rise more than on r11-q; eye check) |

## Levers (face GPU-busy s/vmin = generator + composite; SyncNet)

| lever | m18 | m03 | old03 | verdict |
|---|---|---|---|---|
| **k5 production control (warm)** | **302** (257 + 45) / 8.27 | **278** (253 + 25) / 8.13 | **274** (256 + 17) / 7.47 | reference |
| CFG truncation `LS_CFG_TRUNC=3 / 5 / 7` | gen 212 / 210 / 233; SyncNet **5.88 / 6.02 / 6.83** | | | FAIL; k=7 without DeepCache 6.64, so the loss is genuine |
| no CFG (`--guidance 1.0`) | gen 182 / 5.51 | | | FAIL: LS needs guidance 2.0 on every step |
| CFG delta reuse on the DeepCache skip steps `LS_CFG_DELTA=1` | gen 221 / 7.78 | | | FAIL (−0.5) |
| no DeepCache | gen 554 / 7.94 | | | DeepCache costs no sync and saves about 55 % |
| UNet compile `LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1` (warmed on m18, then unseen) | gen 275 cold / 8.26 | gen **234** (−7.5 %) / 8.10 | gen 254 (≈0) / –; wall 2742 s (re-tracing) | not worth it |
| FP8 linears (torchao 0.12, `bench_fp8.py`, full UNet call at the production shape) | fp16 697 → fp8 989 ms eager; compiled 612 → 853 ms | | | slower on the 5090: dropped |
| `--steps 8` (silent 20) | 281 / 8.32 | | | small (−21) |
| `--steps-silent 10` (ss10) | 284 / **8.40** | | | small gain, sync OK |
| **`LS_RATE=2/3`** | **215 (179 + 36) / 8.30** (vs 302 / 8.27) | | | **likely winner**: −29 % face GPU, SyncNet unchanged; needs m03 / old03 + eye |
| `--steps 6` / `--steps 5` | 263 / 6.50; 248 / 7.06 | | | FAIL |
| `--deepcache-interval 10` | 290 / 5.73 | | | FAIL |

Generator breakdown (m18 control, LatentSync call 68 s for 382 frames):
- UNet 33.9 s;
- VAE + conditioning 17.9 s;
- bridges about 8 s;
- affine + restore 8.3 s.

Plus, per process: LatentSync load 21.7 s and face-stack MediaPipe 36.8 s (CPU). The r10 resident gen worker removes the load.

## NEXT (tomorrow)
1. `LS_RATE=2/3` (+ `--steps-silent 10`) on the UNSEEN m03 and old03: QUICK arms, then the full gate vs k5, then a 2-panel eye clip (k5 | k5 + LS_RATE 2/3), full length.
   - The r10 note: half-rate (1/2) failed the flash gate; 2/3 passed on m03 / m19 / old03 under r9 + fix.
2. If it passes: E2E with audio (`E2E_FX=1 E2E_SYNC=1 E2E_LS_RATE=2/3`) on the 3 videos plus the gen worker queue (`scripts/r10/worker.py --gen-worker`). Target < 300 GPU-s/vmin E2E.
3. **Not worth more time:** CFG truncation / delta, FP8, fewer than 8 steps, a longer DeepCache interval, the dynamic compile.

## Resume (new box)
```
git checkout experiment/r12; bash scripts/r9/r9_models.sh; uv pip install --python /venv/main/bin/python nvidia-ml-py
setsid nohup bash scripts/r12/ls_base.sh m18 m03 old03 > /workspace/dub/r12/logs/base.out 2>&1 &     # k5 baselines + gate bases (~2.5 h)
QUICK=1 bash scripts/r12/ls_arm.sh m03 r23 "" LS_RATE=2/3                                            # then without QUICK for the full gate
QUICK=1 bash scripts/r12/ls_arm.sh m03 r23s "--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 10 --codeformer-parse off --codeformer-batch 16" LS_RATE=2/3
```
