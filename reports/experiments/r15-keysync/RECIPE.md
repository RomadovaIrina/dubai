# KeySync fast: the run recipe (r15 K)

"KeySync fast" consists of:
- the production KeySync keyframes (8 AYS steps, 3-way CFG, DeepCache 2:0) as a TensorRT FP16 engine;
- the interpolation UNet replaced by the guided LCM-LoRA student (1 row, 4 steps; 2 steps is the faster variant), as a TensorRT FP16 engine with the LoRA merged;
- the VAE decode compiled (torch.compile, timesteps = 1 = the authors' per-frame math, 8 frames per call).

Everything after the KeySync stage (SyncNet AV shift, lag fix, paste / cfhf finals) is unchanged.

## 1. Environment (one RTX 5090, driver ≥ 595, CUDA 12.8 wheels)
```bash
# the repo: scripts/keysync/*.py, scripts/keysync/r15/*  ->  /workspace/dub/keysync
bash scripts/keysync/r15/k_setup.sh          # keysync 5e2503b + keysync_5e2503b_sm120_r15.patch, venv (torch 2.7.1 cu128), weights 13.15 GB (HF toninio19/keysync)
. /workspace/dub/keysync/venv/bin/activate
uv pip install tensorrt-cu12==10.16.1.11 onnx==1.18.0
```
Student weights:
- `lcm_r1_lora_fp16.pt` (231 MB) with `lora_meta.json` in the SAME directory (`ks_student.merge_lora` reads the meta next to the checkpoint).
- Server copy: `/workspace/dub/r15k/`.

## 2. TensorRT engines (build once per GPU model + driver + TensorRT version; plans are not portable)
The engines are built from captured production UNet inputs (`kf.pt`, `interp.pt`, ~30 MB; server copy `/workspace/dub/r15k/trt_cap/`).
To recapture, run prod once with `KS_CAPTURE=<dir>`.
```bash
cd /workspace/dub/keysync/code
python ../ks_trt.py --cap ../trt/cap --which kf --out ../trt/eng                                   # keyframes: full + shallow (DeepCache), batch 42, ~2 min
python ../ks_trt.py --cap ../trt/cap --which interp --out ../trt/eng_s --lora <dir>/lcm_r1_lora_fp16.pt --no-dc   # student: full only, batch 28, ~2 min
ln -sfn ../eng/kf ../trt/eng_s/kf
```
Each run prints `TRT_VERIFY` (PSNR vs PyTorch, expect ≥ 56 dB).

## 3. Run (resident worker; one process for a queue of videos)
```bash
cd /workspace/dub/keysync/code; . ../venv/bin/activate
KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk python ../ks_fast.py --jobs jobs.txt --out OUT \
  --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s \
  --interp-lora <dir>/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile
```
- `jobs.txt`: `<tag> <face crop: 25 fps, 512x512> <dub voice, 16 kHz mono>` per line. The crop and box come from `ks_prep.py` / `ks_prep_src.sh`; for 2 faces use `KS_BOX_K=1.25 KS_BOX_M=0.4`.
- `--interp-steps 2` = "KeySync fast" (fast2c, eye-approved on m03). `--interp-steps 4` = fast4c (179 / 169 / 170 s/vmin; old03 sync −0.6 vs prod, flicker a bit lower).
- `KS_DC=2:0` applies to the keyframe engine only. The student runs without DeepCache.
- `--compile unet` does nothing when both UNets are TRT. `--vae-compile` compiles the decoder: ~4 min autotune on a cold machine, seconds once `/tmp/torchinductor_root` is warm. Keep the inductor cache in the image.
- Peak VRAM 12.4 GB. The output is `OUT/<tag>/<tag>_<tag>.mp4` (512² 25 fps); the KS_JOB line gives the per-stage timing.

## 4. Post (unchanged)
- As evaluated: `k_eval.sh` = SyncNet AV shift → `ks_tsmooth.py --alpha 0.8` → `ks_paste.py` (KS_SHIFT, KS_HF 3, KS_SHARP 0.3, KS_HF_LM = source landmarks).
  - Optional `KS_KEEP_NOSE=0.04`: the source nose pixels (removes all nose shimmer; for the eye).
- Or the production r13 cfhf finals chain on the fast output: `bash k_cfhf.sh <id> fast2 <render>` (= `ks_final.sh` with the r13_s8final env; how `outbox/r15_k/full/` was made).

## Numbers (warm, face GPU-s per video-minute, m18 / m03 / old03)
prod 478 / 469 / 474 → fast4c 179 / 169 / 170 → **fast2c 148 / 137 / 137** (SyncNet 7.61 / 8.14 / 7.28 vs prod 7.19 / 7.81 / 7.50).
