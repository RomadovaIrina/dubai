# r15 K: KeySync speed pilot (TensorRT + few-step distillation)

Server r15-k (vast 59279756, RTX 5090, Japan). Branch `experiment/r15-keysync` (from `experiment/r13-keysync` 2f88248).
Scripts: `scripts/keysync/r15/` (they run from `/workspace/dub/keysync`, like the r12/r13 KeySync scripts), the updated
`scripts/keysync/ks_fast.py`, and the full KeySync code patch `scripts/keysync/keysync_5e2503b_sm120_r15.patch` (= r13 + a per-model DeepCache switch).

## TL;DR
- **Distillation is the lever; TensorRT is a small extra.** The guided LCM-LoRA student of the interpolation UNet (1 row instead of 3 CFG rows,
  4 or 2 steps instead of 9) plus TensorRT on both UNets takes the KeySync face stage from **~474 to ~215 (4-step) / ~180 (2-step)
  face GPU-s/vmin** (−55 % / −62 %) on one 5090.
- **Quality by the metrics is mixed:**
  - m18 / m03: better than prod (SyncNet +0.2..+0.4, lips flicker −3..−8 %).
  - old03: worse (SyncNet −0.59 at 4 steps, −0.26 at 2 steps).
  - The eye must decide. Clips in `outbox/r15_k/`.
- **TensorRT alone:** −9..−10 % end to end (prod already runs torch.compile). Output identical (40.8–45.4 dB pixel PSNR vs prod).
  Peak VRAM 27.7 → 12.4 GB, and no compile warm-up.
- **~140 is not reached.** At 4 steps on m18 the remaining split is: interpolation VAE decode 80, keyframes 36, encode 19, save 12, UNet 70 s/vmin.
  The next levers:
  - a working TensorRT VAE decoder (mine built an output-less network; not debugged in time);
  - the same distillation on the keyframe UNet.

| arm | face GPU-s/vmin m18 / m03 / old03 | SyncNet speech@0 m18 / m03 / old03 | lips flicker m18 / m03 / old03 | m18 split /vmin: interp UNet / interp decode / kf / enc / save |
|---|---|---|---|---|
| prod (r13 default: 9 evals × 3 CFG rows, DeepCache 2:0, torch.compile) | 478 / 469 / 474 | 7.188 / 7.813 / 7.502 | 3.60 / 2.68 / 4.35 | 313 / 90 / 42 / 18 / 10 |
| trt (prod + TensorRT FP16 on both UNets) | 435 / 421 / 425 | 7.225 / 7.862 / 7.509 | 3.58 / 2.71 / 4.30 | 272 / 90 / 37 / 19 / 10 |
| lcm4 (student 4 steps, torch.compile = lever 2 alone) | 247 / 236 / 237 | 7.356 / 8.126 / 6.871 | 3.33 / 2.56 / 4.06 | 79 / 90 / 45 / 19 / 10 |
| **fast4** (student 4 steps + TRT both UNets + 4-frame decode) | **223 / 210 / 211** | **7.419 / 8.160 / 6.917** | 3.32 / 2.58 / 4.06 | 70 / 80 / 36 / 19 / 12 |
| **fast2** (`fast2v`: the same at 2 steps) | **187 / 176 / 178** | **7.601 / 8.139 / 7.245** | 3.41 / 2.62 / 4.18 | 36 / 80 / 36 / 19 / 11 |

Lower-face ("low") tone flicker:
- prod: 1.40 / 0.82 / 0.55
- fast4: 1.30 / 0.79 / 0.64
- fast2: 1.32 / 0.87 / 0.87 (old03 is worse at 2 steps)

The `fast2v` arm was meant to include the TRT VAE decoder. The engine failed, so it ran with the 4-frame torch decode, the same as fast4.

## Round 2 (after the eye: m18 fast4 "rather good than bad"; m03 fast2 "the right side is great")
**"KeySync fast" = fast2c: 148 / 137 / 137 face GPU-s/vmin (m18 / m03 / old03), prod 478 / 469 / 474 (3.4× faster), peak VRAM 12.4 GB.**
The recipe is in `RECIPE.md`.

| arm | face GPU-s/vmin m18 / m03 / old03 | SyncNet speech@0 | lips flicker | m18 split /vmin: interp UNet / interp decode / kf / enc / save |
|---|---|---|---|---|
| prod | 478 / 469 / 474 | 7.188 / 7.813 / 7.502 | 3.60 / 2.68 / 4.35 | 313 / 90 / 42 / 18 / 10 |
| fast4c (4-step + TRT + compiled decode) | 179 / 169 / 170 | 7.427 / 8.173 / 6.922 | 3.32 / 2.58 / 4.06 | 70 / 42 / 33 / 19 / 10 |
| **fast2c (2-step + TRT + compiled decode)** | **148 / 137 / 137** | **7.606 / 8.142 / 7.276** | 3.41 / 2.62 / 4.18 | 36 / 42 / 33 / 19 / 12 |

- **The decode lever:** `torch.compile` of the VAE decoder with timesteps fixed to 1, the authors' per-frame math (`--vae-compile --vae-batch 8`).
  - 52 → 26 ms per frame, 75 dB vs eager. The r12 compile failure was the dynamic `timesteps` argument.
  - The interpolation decode is 90 → 42 s/vmin.
- **The TRT VAE decoder:**
  - Fixed: constant folding had baked > 2 GB of tensors into node attributes, so TensorRT parsed an empty network. Export without folding, attributes moved to external data.
  - Builds at 69 dB, but it is SLOWER than torch (66.6 vs 52.5 ms per frame; the unfolded graph carries 3.2 GB of materialised constants). Not used.
- **Full-length finals through the production r13 cfhf chain:** `outbox/r15_k/full/` (`k_cfhf.sh`, i.e. `ks_final.sh` with the r13_s8final env;
  the r13 artefact layout rebuilt from the source + the q2 dub).
  - Files: `<v>_keysync_fast2.mp4`, `<v>_keysync_prodKS.mp4`, `<v>_prodKS_vs_fast2_full.mp4`.
  - Gate SyncNet on the finals, prod → fast2: m18 7.16 → 7.52, m03 6.94 → 7.23, old03 7.44 → 7.14.
  - m03: the mother is the dubbed face (speaker crop), the child untouched.
  - fast2 here = the fast2v renders (what the eye saw). The packaged fast2c matches them at 43–46 dB (only the decode numerics differ).
- **Nose-tip tremble (the m18 note) is not the student:**
  - Landmark jitter of the nose tip vs a head anchor (`k_nose.py`, 1e-3 IOD per frame, raw layers):

    | | source | prod | fast4 | fast2 |
    |---|---|---|---|---|
    | m18 | 3.99 | 4.47 | 4.17 | 4.49 |
    | m03 | 3.34 | 3.67 | 3.48 | 3.57 |
    | old03 | 1.33 | 1.55 | 1.57 | 1.56 |

  - On the pasted finals (30 fps), m18 source / prod / fast4 = 3.29 / 3.27 / 3.28.
  - Pixel shimmer inside the nose hull (`k_noseflick.py`, added vs the source, p95): prod 1.63, trt 1.56, fast4 1.34, fast2 1.64. All of them are below the source's own grain on average.
  - So it is shared KeySync behaviour: the nose lies inside KeySync's generated box, and KeySync moves it ~2 % IOD.
  - `ks_paste.py KS_KEEP_NOSE=0.04` puts the source nose pixels back (shimmer p95 → 0.07). Landmark-wise it reads worse (mediapipe's nose point follows the mouth). Option clip: not made yet.
- **Checkpoint:** run r1 final step 2984, `/workspace/dub/r15k/lcm_r1_lora_fp16.pt` + `lora_meta.json`.

## Setup and measurement
- Held out: old03 (`qo3`), m03 (speaker crop), m18 (`q18`). The dub voice = r15's q2 dub (`r9_q2_<v>_htd/dubbed_16k.wav` from r15-w);
  r13 used an older dub, so the absolute SyncNet differs from r13's table. Compare arms only within this report.
- Face GPU-s/vmin = the KeySync stage of `ks_fast.py` (resident worker, warm; landmarks excluded = computed off the critical path, as in prod).
- Quality = the same post for every arm (`k_eval.sh`): SyncNet AV shift of the raw output, then α 0.8 lag smoothing, then the raw paste with the r13 detail
  transfer (KS_HF 3 / KS_SHARP 0.3) and the q2 mix. Then SyncNet mean conf at offset 0 over the dub-speech seconds
  (= `scripts/r11/sync_speech.py`), and the lower-face tone flicker vs the source (`tone_flicker.py`, lips / low regions, grey levels per frame).
- Training clips (non-test): test_videos_raw 01 / 04 / 05 and 6 mamakris clips (02 and one mamakris clip had no detectable face).
  tr_mk5 = the validation clip. Total ~7 min, 804 training segments of 14 frames.

## Baseline = the production KeySync face stage (r13 default)
`ks_fast.py --cfg _s8 --vae-half --compile unet`, KS_DC=2:0, KS_LMK_CACHE. 8 AYS steps on both models (9 UNet evaluations), full 3-way CFG (3 rows),
DeepCache 2:0.

| video | face GPU-s/vmin | SyncNet speech@0 | lips flicker mean / p95 | low flicker |
|---|---|---|---|---|
| m18 | 478 | 7.188 | 3.60 / 9.63 | 1.40 |
| m03 | 469 | 7.813 | 2.68 / 7.85 | 0.82 |
| old03 | 474 | 7.502 | 4.35 / 14.63 | 0.55 |

Where the time goes (m18, s per video-minute): interpolation UNet 314, interpolation VAE decode 90, keyframes 42 (UNet 33 + decode 9),
VAE encode 18, save 10. **Even with free UNets about 160 s/vmin would remain** (decode + encode + keyframes + save).

Production calls the interpolation UNet in 28-frame chunks (KS_ICHUNK 2 = 2 segments of 14), so a 3-row CFG step is 3 × n_seg / 2 calls.
The keyframe UNet is called with 42 frames (3 rows × 14 keyframes).

## Lever 1: TensorRT FP16 (`ks_trt.py`)
- ONNX (TorchScript exporter, fp32 graph) → TensorRT 10.16 FP16, normalisation layers pinned to fp32, fixed shapes (kf 42, interp 28).
  Two engines per model reproduce the production DeepCache split: `full` (x, t, ctx) → (out, h) and `shallow` (x, t, ctx, h) → out.
  A drop-in `TRTUNet` runs the same DeepCache state machine as the keysync patch. Build: ~2 min per model, 2.9 GB plan each.
- Gotcha: TensorRT 10.16 failed to import the exporter's ~1000 loose external tensors (`onnx__MatMul_*`). Repacking into one `.data` file fixed it.

Per UNet call, on the captured production inputs (the first chunk at sigma 0 / 1 / 2):

| model | torch eager (autocast fp16) | TRT FP16 | output vs PyTorch |
|---|---|---|---|
| interpolation full (28 frames) | 409 ms | 278 ms (1.47×) | 61.3 dB PSNR, rel L2 3.2 % |
| interpolation shallow | 40.5 ms | 35.1 ms | 61.5 dB |
| keyframe full (42 frames) | 618 ms | 417 ms (1.48×) | 58.3 dB, rel L2 3.8 % |
| keyframe shallow | 61.4 ms | 52.3 ms | 56.6 dB |

Production uses torch.compile (max-autotune), which is already ~1.2× faster than eager. So end to end, TRT gains only ~1.15× on the UNet stages:

| video | prod s/vmin | TRT s/vmin | SyncNet prod → TRT | lips flicker prod → TRT | raw-output pixel PSNR TRT vs prod |
|---|---|---|---|---|---|
| m18 | 478 | 435 (−9 %) | 7.188 → 7.225 | 3.60 → 3.58 | 44.6 dB |
| m03 | 469 | 421 (−10 %) | 7.813 → 7.862 | 2.68 → 2.71 | 45.4 dB |
| old03 | 474 | 425 (−10 %) | 7.502 → 7.509 | 4.35 → 4.30 | 40.8 dB |

- Bonus: peak VRAM 27.7 → 12.4 GB (the fp32 PyTorch UNets are freed), and no compile warm-up (the first prod video took 914 s with max-autotune).
- Batched VAE decode (`--vae-batch N`: N frames per call with timesteps=1 = exactly the authors' per-frame decode): 14 frames OOM'd next to the TRT engines.
  At 4 frames per call the interpolation decode goes 30 → 27 s on m18 (~−10 %).
- TensorRT VAE decoder (`ks_trt_vae.py`, interpolation model, per-frame math): the ONNX export ran (27 s), but TensorRT reported
  "Network must have at least one output" on build, twice. Not debugged in time. It is the next lever: the decode is now the largest single stage.

## Lever 2: LCM-LoRA distillation of the INTERPOLATION UNet (`ks_lcm.py`)
Why the interpolation UNet and not the keyframe UNet: interpolation sampling is ~80 % of the KeySync time and keyframes ~8 % (r13 profile, confirmed above).
- Teacher: the production interpolation UNet with the 3-way AudioRef CFG (audio 5, ref 2). One Euler step on the production AYS-8 grid
  [700, 40.0, 11.3, 4.97, 1.79, 0.785, 0.264, 0.051, 0.002], i.e. exactly the production sampler's step, without DeepCache.
- Student: the same UNet + LoRA r64 on all 530 Linear / Conv2d layers (115 M params), ONE row (the full condition, the CFG baked in).
  f(x, s) = c_skip x + c_out D(x, s) with a steep boundary (f = x at s_min, f ≈ D above 0.01). The target is the student itself
  at the teacher's point (stop-grad, as LCM-LoRA, no EMA). Pseudo-Huber loss inside the generated mask; the keyframe ends are excluded.
- Data: x0 = the real latents of each 14-frame segment, the keyframes = the real latents at the segment ends (how the interpolation model was trained),
  each clip with its own audio (`ks_dump.py`: the authors' `sample()` with the keyframe model stubbed and the sampler replaced by a dumper).
- Inference (`ks_student.py`): the LoRA merged into the weights (zero runtime cost). The LCM sampler runs 4 steps (s0 s2 s4 s6) or 2 steps (s0 s4),
  one row, with fresh re-noise. The keyframes stay on the production sampler.
- Run r0 (aborted after 50 steps): a ×10 boundary scaling gave c_skip + c_out ≈ 1.24 at small sigma (the loss at the last grid step was 10× the others).
  Fixed, then restarted as r1.

### Training (r1)
- 2,984 steps in 92 min (1.8 s/it, batch = one 14-frame segment, 14.3 GB), AdamW lr 1e-4, LoRA r64 / α64.
- The loss fell 0.035 → ~0.02; at the top sigma (where the CFG has to be learned) 0.10 → ~0.05.
- Validation = 6 segments of tr_mk5: the student's N-step sample vs the production 8-step CFG sample from the SAME noise (latent PSNR in the mask).
  It stays flat at 20.2–21.4 dB (4-step) / 20.4–21.5 dB (2-step) from step 0.
  - This metric is capped by the fresh re-noise of the multistep LCM sampler (even a perfect student lands on a different sample), and 6 segments swing ±0.7 dB.
  - What does move: the student's PSNR vs the real lips goes from 18.5 (cond-only, step 0) to 17.6–18.0 = the teacher's 17.75, i.e. it takes on the CFG behaviour.
- The FINAL checkpoint is used: `/workspace/dub/keysync/lcm/r1/lora_last.pt` (441 MB fp32); fp16 copy `/workspace/dub/r15k/lcm_r1_lora_fp16.pt`; meta `lora_meta.json` next to it.
- The student's TRT engine (LoRA merged) matches PyTorch at 58–60 dB, 281 ms per 28-frame call (eager 417 ms).

### Lever 2 alone (lcm4, torch.compile, no TRT)
- 247 / 236 / 237 s/vmin (−48 %).
- The interpolation UNet stage drops 313 → 79 s/vmin on m18 (4.0×): 4 evaluations × 1 row vs 9 × 3 rows with DeepCache.
- SyncNet m18 / m03 above prod, old03 −0.63.

## Combined (fast4 / fast2) and what is left
- fast4 = 2.2× and fast2 = 2.6× faster than prod.
- The remaining time is mostly NOT the UNet. On m18 fast4: interpolation decode 80, UNet 70, keyframes 36, encode 19, save 12 s/vmin.
- Towards ~140:
  1. A working TRT VAE decoder (~80 → ~30?).
  2. The same guided LCM on the keyframe UNet (36 → ~12).
  3. 2-step if the eye accepts it.
  Estimated fast2 + 1 + 2 ≈ 110–130 s/vmin. Not measured.
- **Quality risks for the eye:**
  - old03: SyncNet −0.6 (4-step) / −0.3 (2-step) vs prod. Both pasted finals are aligned (global SyncNet offset 0), so this is real.
  - On m03 at 17–21 s the student's chin / under-lip skin looks smoother: a small mark visible in prod is gone in a sampled frame.
  - The training data is small (~7 min of 9 clips, own audio, real latents). More clips (DVIDS etc.) and dubbed-audio pairs should help old03-like faces.

## Clips
`/workspace/dub/outbox/r15_k/` (7 files, 13 MB): 0.5x mouth 2-panels, the same post on both sides, the q2 mix slowed.
- `<v>_mouth_prodKS_vs_fast4_<w>s_half_speed.mp4`: prod | 4-step student + TRT (old03 3–7, m03 17–21, m18 2–6).
- `<v>_mouth_prodKS_vs_fast2_<w>s_half_speed.mp4`: prod | 2-step.
- `m18_mouth_prodKS_vs_trt_2-6s_half_speed.mp4`: prod | TensorRT (identical).
- Full-length pasted finals (not in the outbox): `/workspace/dub/keysync/fast/<arm>/eval_<id>/paste.mp4`.

## How to run
- Setup: `scripts/keysync/r15/k_setup.sh` (KeySync venv + 13 GB weights), then `k_inputs.sh test|mamakris`.
- Prod: `k_base.sh`.
- TRT: `k_trt.sh`, then `ks_fast.py ... --trt <eng dir>`.
- Distillation: `k_train.sh` (dump + `ks_lcm.py`).
- Student: `ks_fast.py ... --interp-lora <ckpt> --interp-steps 4|2 [--trt <eng dir with the student engine>] [--vae-batch 4]`.
- Eval: `k_eval.sh <arm> <id>`; table: `k_table.py`; clips: `k_clip.sh`.

## Process notes
- Rental from ~14:47 to ~19:30 server time, ≈ 4.7 GPU-h. The prod compile warm-up alone was 15 min (max-autotune).
- Every GPU job ran under `gpu.lock`. Audio was not touched. No downloads over 5 GB except the KeySync weights (13.15 GB, reported first).

## Licence note
KeySync (antonibigata/keysync; weights toninio19/keysync, tagged Apache-2.0 on HF) is fine-tuned from Stable Video Diffusion.
- SVD is under the Stability AI Community License: commercial use needs a Stability registration, and an enterprise licence above $1M annual revenue.
- KeySync's training data includes CelebV (non-commercial research use).
- The distilled LoRA here is a derivative of the KeySync weights, so it inherits all of this. The decision is the user's (01.11: "не думаю что больше миллиона долларов в год будет").
- TensorRT (NVIDIA licence, deployment allowed) adds no non-commercial restriction.

## Кратко для Ирины (RU)
- KeySync удалось ускорить в 2,2–2,6 раза: ~215 (4 шага) или ~180 (2 шага) GPU-с на минуту видео вместо ~475. Цель ~140 пока не достигнута.
- Главный рычаг — дистилляция модели интерполяции: одна «строка» вместо тройного CFG и 4 или 2 шага вместо 9. TensorRT добавляет ещё ~10 % и экономит память, на картинку не влияет.
- По метрикам: на m18 и m03 синхрон лучше прода (+0,2..+0,4), мерцание губ чуть меньше. На old03 хуже (−0,6 на 4 шагах, −0,3 на 2 шагах). Нужен глаз: клипы 0,5x в outbox/r15_k.
- На m03 у быстрой версии кожа под губой немного глаже (родинка в одном кадре пропала) — проверить глазом.
- Чтобы дойти до ~140: ускорить декодер VAE (сейчас это самая долгая часть; движок TensorRT для него не собрался, не успел отладить) и так же дистиллировать модель ключевых кадров.
- Лицензия: KeySync сделан на основе SVD (лицензия Stability Community: регистрация, до $1 млн выручки в год) и обучен в том числе на CelebV (некоммерческий датасет). Наш LoRA — производная от весов KeySync, ограничения те же. Решение за вами.

## Раунд 2, кратко для Ирины (RU)
- «KeySync fast» (2 шага + TensorRT + ускоренный декодер) — 148 / 137 / 137 GPU-с на минуту видео (m18 / m03 / old03) вместо ~475, то есть в 3,4 раза быстрее. Цель ~140 достигнута.
- Полные видео через продовую цепочку KeySync (fast2, прод и бок о бок) лежат в outbox/r15_k/full. На m03 дублируется мама, ребёнок не тронут.
- Синхрон на финалах: m18 и m03 лучше прода, old03 чуть хуже (−0,3).
- Дрожание кончика носа — не от ученика: у прода оно такое же. Есть опция «нос из исходника» (KS_KEEP_NOSE).
- Чекпоинт: lcm_r1_lora_fp16.pt (шаг 2984) и lora_meta.json, в /workspace/dub/r15k. Рецепт запуска — RECIPE.md.
