# r15 K: FINAL REPORT (KeySync speed + "KeySync fast" + plain paste; LatentSync TRT) - 2026-11-06

Server r15-k (vast 59279756, RTX 5090, driver 595.99.02, CUDA 12.8 wheels, TensorRT 10.16.1.11).

**Code (no push; bundles):**
- `experiment/r15-keysync`: head **see BRANCHES.txt** (424b149 + this report), from `experiment/r13-keysync` 2f88248 → `bundles/r15-keysync.bundle`.
- `experiment/r15-ls-trt`: head **97e691c**, from `experiment/r14-distill` 56715cd → `bundles/r15-ls-trt.bundle`.
- Detailed reports: `REPORT.md` (KeySync, all rounds), `RECIPE.md` (exact commands), `REPORT_ls_trt.md`.

## 1. Eye verdicts (the user, 05-06.11)
- fast4 m18: "rather good than bad", nose tip trembles.
- fast2 m03 (512 crop clip): "the right side is f*cking great".
- Full videos through the r13 cfhf composite: a mess. The face floats / slides: m03 at 3 / 7 / 12 s, m18 at ~3 s. old03 is waxy (stubble lost). The m03 cheek bleeds into the child. A black-hole mouth at m03 25 s.
  The raw 512 crop is "the best".
- **Plain paste** (the crop pasted back with the exact inverse transform): m03 + old03 "охуенно", no shaking. **m18: the face shakes like jelly, the mask lags the head.** Sync and mouth motion are good on all three.
- Still visible on plain (= the generator itself): the smeared mouth at m03 0:03; the black-hole mouth at m03 0:25.

## 2. What was done (numbers: face GPU-s per video-minute, one RTX 5090, warm)
| arm | m18 / m03 / old03 s/vmin | SyncNet speech@0 (raw-paste eval) m18 / m03 / old03 |
|---|---|---|
| prod KeySync (r13: 8 AYS steps, 3-way CFG, DeepCache 2:0, torch.compile) | 478 / 469 / 474 | 7.19 / 7.81 / 7.50 |
| + TensorRT FP16 both UNets | 435 / 421 / 425 | 7.23 / 7.86 / 7.51 |
| LCM-LoRA student (interpolation UNet), 4 steps, torch | 247 / 236 / 237 | 7.36 / 8.13 / 6.87 |
| fast4c = student 4 steps + TRT + compiled VAE decode | 179 / 169 / 170 | 7.43 / 8.17 / 6.92 |
| **fast2c = KeySync fast (2 steps)** | **148 / 137 / 137** | **7.61 / 8.14 / 7.28** |

Gate SyncNet on the FULL finals (speech seconds, offset 0, q2 dub):

| | m18 | m03 | old03 |
|---|---|---|---|
| prodKS cfhf | 7.16 | 6.94 | 7.44 |
| fast2 cfhf | 7.52 | 7.23 | 7.14 |
| **fast2 PLAIN** | **7.97** | **8.58** | **7.68** |

Notes:
- **The student:** LCM-LoRA (r64, 115 M params) on the KeySync interpolation UNet, guided distillation (the 3-way CFG baked in, 1 row). 2,984 steps / 92 min on 9 non-test clips. Checkpoint = the final step. The keyframes stay prod.
- **TensorRT:** −10 % only (prod already compiled), VRAM 27.7 → 12.4 GB, identical output (41–45 dB).
- **The decode lever:** `torch.compile` of the VAE decoder at timesteps=1 (`--vae-compile --vae-batch 8`), 1.9×, identical (75 dB). The TRT VAE engine was slower than torch.
- **Why the cfhf composite broke the video:** `ks_fix`'s explosion guard has false positives. It swaps the dubbed face for the source face with a ±10-frame cross-fade (m18 3.68 s; m03 4.3 / 7.9 / 8.7 / 11.7 s). Plus the 25 → 30 fps MCI retime and the tsmooth flow warp.
- **Plain paste** (`ks_plain.py`):
  - the static inverse crop, 25 fps 1:1;
  - a feathered face-oval mask minus other faces (`k_faces2.py`);
  - one constant Lab colour match per video;
  - optional `--tex 2` (source stubble HF, flow-warped) and `--cavity 0.8` (black-hole lift). Their eye verdict is pending (the `fast2plainfx` videos).
- **LatentSync TRT (r6 student step 250):**
  - identical output (75–76 dB on the rendered layer), UNet −26 %;
  - but the LatentSync stage only ~−10 % (204 → 185 s/vmin on m03);
  - more VRAM: 4 per-shape engines × 2.4 GB. Not worth it as-is.

## 3. Open problems + ranked NEXT STEPS
1. **Plain paste = the new default** (done for m03 / old03; the eye approves).
   **SMART PASTE for m18 (the user's plan for tomorrow).** Measured cause (`k_lag_head.py`, eye / nose-bridge anchor of the raw 512 KeySync output vs the source crop):
   - KeySync itself does NOT lag: best lag 0 frames on all 3 (velocity correlation m18 0.983, m03 0.995, old03 0.79; amplitude ratio 0.99–1.05; mean anchor offset 0.4–1.1 px).
   - The lag is the paste's **AV shift**: KeySync frame i is shown at master frame i + 1 (shift 1 on m18 / m03, 0 on old03), so the WHOLE pasted face trails the head by 40 ms.
     m18's face is large (720 px box) and fast: 3.4 px/frame average, p95 ≈ 9 px at source scale. That's the jelly. m03 moves less on screen (2.2 / 5.5 px); old03 has shift 0.
   - Fixes, cheapest first:
     - (i) **Shift the AUDIO, not the video.** Delay the dub wav by the measured AV offset (40 ms = 1 frame) BEFORE KeySync, then paste with `--shift 0`. Zero geometric lag by construction; sync kept.
       Cost: one KeySync re-run (fast2c ≈ 140 s/vmin). Command below.
     - (ii) Per-frame similarity alignment of the shifted KeySync frame onto the CURRENT source frame, on stable landmarks only (eye corners 33 / 263 / 133 / 362, nose bridge 168, 6): `cv2.estimateAffinePartial2D`, temporally smoothed. Then the mask.
     - (iii) A tighter lower-face / jaw mask (the rest from the source); the shifted pixels then cover less moving area.
     - (iv) Gate (ii) / (iii) by head speed so m03 / old03 stay plain.
2. **Mouth interior / teeth (black hole).** Share of open-mouth frames with a near-black, toothless cavity (`k_cavity.py`, m03): source 3.1 %, **prodKS 20 %, fast2 16 %**. So it is KeySync itself, not the student. Options:
   - (a) the lift that already exists (`--cavity 0.8`): 0 GPU, no teeth;
   - (b) a teeth / interior bank from the sharpest source open-mouth frames, warped into the generated cavity (as agent F's `src_detail.py` bank): CPU only, ~1 day;
   - (c) more KeySync steps only on wide-open windows: GPU +10–20 %;
   - (d) a LatentSync mouth-interior blend on those frames: needs both generators.
3. **Mouth sharpness at m03 0:03** (the generator's own smear): the source detail transfer inside the lips (F's `src_detail.py`, gated by structure agreement), or more steps on that window. Evaluate on the plain paste.
4. **Stubble / beard** (old03): `ks_plain.py --tex 2` exists (`*_fast2plainfx.mp4`); eye check pending.
5. **Nose-tip tremor:** shared by prod KeySync (not the student); measured jitter fast4 ≤ prod. `KS_KEEP_NOSE=0.04` (source nose pixels) exists in `ks_paste.py`; not yet ported to `ks_plain.py` (easy: the same mask logic).
6. LatentSync TRT: one dynamic-shape engine (frames 3–16, batch 1–2) instead of 4 fixed ones; only then is it worth it (memory).

## 4. Exact commands (r15-k layout; scripts in the bundle under scripts/keysync/ and scripts/keysync/r15/; they run from /workspace/dub/keysync)
```bash
# KeySync env + 13.15 GB weights (HF toninio19/keysync: WavLM-Base+.pt, keyframe_dub.pt, interpolation_dub.pt)
bash scripts/keysync/r15/k_setup.sh; . /workspace/dub/keysync/venv/bin/activate; uv pip install tensorrt-cu12==10.16.1.11 onnx==1.18.0
# inputs: the static 512 crop + box (ks_prep_src.sh; m03: KS_BOX_K=1.25 KS_BOX_M=0.4) and the dub voice 16 kHz
bash k_inputs.sh test|mamakris
# TRT engines (from the captured UNet inputs trt/cap/{kf,interp}.pt; recapture: KS_CAPTURE=<dir> on a prod run)
cd code; python ../ks_trt.py --cap ../trt/cap --which kf --out ../trt/eng
python ../ks_trt.py --cap ../trt/cap --which interp --out ../trt/eng_s --lora <dir>/lcm_r1_lora_fp16.pt --no-dc; ln -sfn ../eng/kf ../trt/eng_s/kf
# KeySync fast (fast2c)
KS_DC=2:0 KS_LMK_CACHE=../lmk python ../ks_fast.py --jobs jobs.txt --out OUT --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s \
  --interp-lora <dir>/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile
# the plain paste (k_plain.sh <id> <render> <name> wraps it; --others for multi-face, --tex 2 --cavity 0.8 optional)
KS_CMS=1 python ks_plain.py --ks OUT/<id>/<id>_<id>.mp4 --master master_25fps.mp4 --box data/box_<id>.json --lm source_lm_F.npz \
  [--others lm/m03_faces2.npz] --shift <-av_offset of the raw output> --audio final_mix_aac.m4a --out FINAL.mp4
# NEXT STEP 1 (i): delay the dub instead of shifting the video (m18: av_offset -1 -> 40 ms), re-run KeySync, paste with --shift 0
ffmpeg -i dubbed_16k.wav -af adelay=40:all=1 -ar 16000 -ac 1 data/audios/q18.wav   # then the fast2c run + ks_plain.py --shift 0
# head-lag check: python k_lag_head.py --src lm/<id>_crop.mediapipe.npz --ks <KeySync output>.mediapipe.npz
```
LatentSync TRT: `REPORT_ls_trt.md` (`scripts/r15/ls_run.sh`, `ls_trt.py`, `ls_trt_gen.py` on `experiment/r15-ls-trt`).

## 5. Artifacts (what is where; NOT copied into HANDOFF_K because of the r15-k bandwidth)
- **KeySync student LoRA:** r15-k `/workspace/dub/r15k/lcm_r1_lora_fp16.pt` (231 MB) + `lora_meta.json` (r64 / α64, 530 modules), already on the laptop.
  fp32 + every 250-step checkpoint: `/workspace/dub/keysync/lcm/r1/` (5.6 GB, 12 × 441 MB; the final = `lora_last.pt` = `lora_pick.pt`, step 2984).
  **Irreplaceable** (92 GPU-min).
- **TRT engines** (being pulled by the coordinator to `C:\repos\chkpts\keysync_trt_r15`; rebuild ≈ 2 min each, but only on the SAME GPU / driver / TRT version):
  - `/workspace/dub/keysync/trt/eng/{kf,interp}/{full,shallow}.plan` (3.09 GB + 37 MB each);
  - `trt/eng_s/interp/full.plan` (student, 3.09 GB);
  - `trt/vae/dec.plan` (0.63 GB, slower than torch: unused);
  - `trt/cap/{kf,interp}.pt` (28 MB, the export inputs; also in `/workspace/dub/r15k/trt_cap/`);
  - the ONNX graphs (`*.onnx` + `.onnx.data`, regenerable).
- **LatentSync TRT engines:** `/workspace/dub/lstrt/eng/<shape>/unet.plan` (4 × 2.56 GB) + `lstrt/cap/` (30 MB). Regenerable in ~6 min.
- **KeySync base weights:** HF `toninio19/keysync` `checkpoints/{WavLM-Base+.pt, keyframe_dub.pt, interpolation_dub.pt}` = 13.15 GB (`dl.py`; skip svd_no_emb / VSR / SAM2).
  Code: `github.com/antonibigata/keysync` @ 5e2503b + `scripts/keysync/keysync_5e2503b_sm120_r15.patch`; then `sed` the `_s6` / `_s8` configs (k_setup.sh does it).
  hubert-base-ls960 is fetched by transformers.
- **Distillation data dump** `/workspace/dub/keysync/dump/*.pt` (9 clips, ~1.3 GB): regenerable with `ks_dump.py` (~6 min).
- **Raw KeySync fast2 renders + everything ks_plain needs:** in `HANDOFF_K/paste_inputs/` (small), so the smart paste can be developed WITHOUT a GPU.
- **Licence:** KeySync is SVD-derived (Stability Community licence: registration, enterprise above $1M revenue) and was trained on data including CelebV (NC). The LoRA inherits this; the user's decision.

## 6. Кратко (RU)
- KeySync fast (ученик 2 шага + TensorRT + ускоренный декодер): ~140 GPU-с на минуту видео вместо ~475.
- Простая вклейка 512-кропа обратно в рамку: m03 и old03 — без тряски, синхрон лучше прода на всех трёх видео (m18 7,97, m03 8,58, old03 7,68).
- m18 «желе»: сам KeySync голову не задерживает (задержка 0 кадров). Отставание даёт сдвиг вклейки на 1 кадр (40 мс), который нужен для синхрона. Решение на завтра №1: сдвинуть звук на 40 мс до генерации и вклеивать без сдвига. Дальше — выравнивание по глазам и носу и маска поуже.
- «Чёрная дыра» во рту — от самого KeySync (у прода 20 % открытых кадров, у ученика 16 %, у исходника 3 %). Варианты: банк зубов из исходника, больше шагов на открытом рту.
- Чекпоинт ученика — lcm_r1_lora_fp16.pt (шаг 2984), на локалке. Движки TensorRT пересобираются за 2 минуты (команды выше).
