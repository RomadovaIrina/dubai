# r15 K: LatentSync UNet TensorRT with the r6 student (step 250 EMA)

Branch `experiment/r15-ls-trt` (from `experiment/r14-distill` 56715cd). Server r15-k (RTX 5090), TensorRT 10.16 in /venv/dabai.

## What
- `scripts/r15/ls_trt_gen.py` = the production face generator (`candidate_video_stage.py`) with the student merged (as `student_gen.py`: 4 steps, guidance 1.0, no DeepCache, bridge-steps 1), plus:
  - `LS_TRT_CAPTURE=<dir>`: the UNet inputs per call shape;
  - `LS_TRT_DIR=<dir>`: the UNet forward swapped for TensorRT engines, per shape; other shapes fall back to PyTorch, counted;
  - `LS_TRT_FREE=1`: frees the PyTorch weights (only when every shape has an engine);
  - `LS_TRT_TIMING=1`: CUDA-synchronised UNet time.
- `scripts/r15/ls_trt.py`: per captured shape, loads the UNet as the generator does (stage2_512 + latentsync_unet.pt, fp16) and merges the LoRA, then fp16 ONNX (no constant folding, repacked external data) → TensorRT FP16 (norms in fp32) → PSNR / timing vs PyTorch.
- `scripts/r15/ls_run.sh <arm> <id>`: the generator stage alone, with the r4_video.sh default args, on the q2 dub (artifacts/r4a/r9_q2_<id>_htd from r15-w).

## Results
Per-call identity and speed (engine vs PyTorch eager fp16, the captured inputs at 3 timesteps):

| shape (sample) | PSNR dB | TRT ms | torch ms | × |
|---|---|---|---|---|
| 2×13×16×64×64 | 77.6 / 75.3 / 72.6 | 267 | 362 | 1.35 |
| 1×13×16×64×64 | 77.9 / 75.2 / 73.0 | 127 | 182 | 1.44 |
| 1×13×12×64×64 | 77.9 / 74.9 / 71.7 | 97 | 141 | 1.45 |
| 1×13×8×64×64 (sliced from the 16-frame capture: the 1-step bridges never reach 3 timesteps) | 77.9 / 75.1 / 72.4 | 64 | 97 | 1.52 |

Full generator runs (the same inputs, seed, args):

| video | UNet calls (TRT) | UNet s PyTorch → TRT | raw LatentSync layer PSNR TRT vs PyTorch | LatentSync stage s/vmin PyTorch → TRT* | nvidia-smi peak MiB PyTorch → TRT |
|---|---|---|---|---|---|
| m03 (40.5 s) | 185 (185) | 49.5 → 36.1 (−27 %) | 76.4 dB | 204 → 185 | 14586 → 24904 |
| old03 (90.3 s) | 416 (390; 26 rare 3–11-frame calls fall back) | 106.2 → 78.4 (−26 %) | 75.6 dB | 240 → 163 | 14566 → 27288 |

\* The stage wall times are NOT a clean comparison: the runs overlapped with CPU-heavy KeySync composite renders on the same box. The CUDA-timed UNet column is the reliable one.
At 4 student steps the UNet is only ~25–35 % of the LatentSync stage (the VAE, audio features, affine / mask work make up the rest), so the expected stage gain is ~−8..−10 %.

## Verdict
- TensorRT reproduces the student exactly (≥ 72 dB per call, 75–76 dB on the rendered layer) and cuts the UNet time by ~26 %.
- But it is a small end-to-end gain at 4 steps.
- With one engine per shape it costs MORE memory (4 × 2.4 GB of engine weights). For production: ONE engine with a dynamic frame / batch profile (the motion-module reshapes must then trace with symbolic frames), or engine weight sharing / refit.
- Licences: LatentSync (Apache-2.0) + TensorRT; no change.

## Reproduce
```bash
# capture (PyTorch student run), engines, TRT run
LS_TRT_CAPTURE=/workspace/dub/lstrt/cap bash scripts/r15/ls_run.sh pt m03
python scripts/r15/ls_trt.py --cap /workspace/dub/lstrt/cap --out /workspace/dub/lstrt/eng --lora /workspace/dub/r15k/r6_step_00250_ema.pt
LS_TRT_DIR=/workspace/dub/lstrt/eng bash scripts/r15/ls_run.sh trt m03      # + LS_TRT_FREE=1 when every call shape has an engine
```

## Кратко (RU)
TensorRT для LatentSync (ученик r6): картинка идентична исходному PyTorch (75–76 дБ), UNet быстрее на ~26 %. Но на 4 шагах UNet — лишь треть стадии, поэтому вся стадия ускоряется примерно на 10 %. Памяти уходит больше (4 движка по 2,4 ГБ). Для прода нужен один движок с переменной формой.
