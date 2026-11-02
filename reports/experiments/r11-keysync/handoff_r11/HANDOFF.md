# KeySync speed: handoff (02.11; r11-s / r11-q destroyed)

## Where things stand
- Last pushed: experiment/r10-base 7d61ec5. It contains:
  - ks_fast.py v1, ks_fast.sh, ks_eval.sh;
  - the keysync patch with DeepCache, SDPA 4-D, the fast path, the AYS fewer-steps change;
  - REPORT.md with the lever table 971 → 450.
- **Not pushed, rebuild from the files in this folder:** the guidance modes, the DeepCache reset on a mode switch, compile-mode, cudnn-bench / VAE channels_last, the fp8 option, the gate sheet tool.
  - The patch scripts edit `/workspace/dub/keysync/ks_fast.py` and `code/`. Apply them in this order:
    1. guide_patch.py
    2. guide2.py
    3. guide3.py
    4. cmode.py
    5. k5.py
    6. fp8.py
  - The code patches 1-6 (dc_patch, dc_patch2, sdpa4d, fastpath, the CPU skip mask, KS_LMK_BS) are already in 7d61ec5's keysync_5e2503b_sm120.patch.

## Results after 7d61ec5
All on m18, warm resident worker, KeySync stage only.
- Stack: 8 steps + KS_DC=2:0 + --vae-half + --compile unet.
- Gate base: SyncNet at offset 0 after the shift 7.68, flashes 5-6.

| KS_IGUIDE (interpolation only) | s/vmin | SyncNet | flashes | verdict |
|---|---|---|---|---|
| none (best stack) | 450 | 7.71 | 5 | base |
| cond | 266 | 6.76 | 2 | FAIL sync |
| **trunc:3** | **338** | **7.41** | **4** | **PASS** |
| trunc:6 (pre-fix) | ~379 | 7.71 | 7 | borderline |
| 2way (drop uncond) | 357 | 7.89 | 10 | FAIL flashes |
| mix:3 | 393 | 7.93 | 10 | FAIL flashes |
| ucc | 671 | broken | - | dropped (defeats DeepCache) |

- Kernel track (r11-q): CUDA graphs (max-autotune) 484 vs 474 without, no gain.
- FP8 (torchao 0.11, `--fp8`) ran to completion; its number was not read before the shutdown.

## In flight when the servers died (re-run these)
```
IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=trunc:2 bash ks_fast.sh ig2_trunc2 --cfg _s8 --vae-half --compile unet
IDS="q18 q18" KS_DC=3:0 KS_IGUIDE=trunc:3 bash ks_fast.sh ig2_t3dc3 --cfg _s8 --vae-half --compile unet
IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh ig2_t3vae --cfg _s8 --vae-half --compile unet --vae-cl --cudnn-bench --decoding-t 4
IDS="q18 q18" KS_DC=2:0 bash ks_fast.sh q_fp8 --cfg _s8 --vae-half --compile unet --fp8
```
- Then the finalist on old03 (IDS="q18 q18 qo3") plus the open-mouth sheet:
  `ks_sheet.py --lm <r9g_qo3 timing>/qo3_source_lm_F.npz --t0 60.7 --n 8 --step 0.2 ...`

## Where the 338 run's time goes (m18, 113 s)
| stage | s |
|---|---|
| interpolation sampling | 56 |
| VAE decode | 29 (~87 s/vmin) |
| keyframes | 13 |
| landmarks | ~6 |
| VAE encode | ~5 |
| save | 3 |

- Next lever: the VAE decode. Try decoding_t > 1 at fp16, channels_last, a tiled or compiled decoder (the plain torch.compile of the temporal decoder fails on `timesteps`).

## Reinstall on a new box
1. `git clone https://github.com/antonibigata/keysync code` (5e2503b), then `git apply` reports/experiments/r11-keysync/scripts/keysync_5e2503b_sm120.patch.
   - Then add the untracked configs: `sed "s/num_steps: 10/num_steps: N/"` on keyframe/interpolation.yaml → `*_sN.yaml`.
2. Set up the venv:
   - `uv venv -p 3.11 venv`
   - `uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url .../cu128`
   - `uv pip install --no-deps -r req_ks.txt` (requirements.txt minus torch / nvidia / triton / xformers / deepspeed / bitsandbytes / dlib / face-recognition / insightface / onnx*)
   - `uv pip install omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 face-alignment==1.4.1` (plus torchao==0.11.0 for fp8)
   - No xformers: the FA3 build fails on sm_120; the patch routes to 4-D SDPA.
3. Download the weights, about 13 GB: from HF toninio19/keysync, `checkpoints/{WavLM-Base+.pt, keyframe_dub.pt, interpolation_dub.pt}`. Skip svd_no_emb, VSR and SAM2.
4. Licence: SVD-derived weights under the Stability Community License. Commercial use needs Stability registration and revenue < $1M; user decision 01.11.
