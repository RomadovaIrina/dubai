# r12 KeySync: HANDOFF (2026-11-02, end of day, server r12-k destroyed after this push)

Branch `experiment/r12-keysync` (off r10-base 7d61ec5). The scripts are in `scripts/keysync/`, a snapshot of `/workspace/dub/keysync` on r12-k.
The numbers are in `reports/experiments/r12-keysync/{finals,finals_v2,finals_auto,speed,lagfix}/`.

## Deliverables (management set)
| video | file | note |
|---|---|---|
| m18 | `outbox/r12_ks/final_v2/m18_KeySync.mp4` | α 0.8 lag fix |
| m03 | `outbox/r12_ks/final_v2/m03_KeySync.mp4` | α 0.8 lag fix, speaker crop |
| old03 | `outbox/r12_ks/final/old03_KeySync.mp4` | v1, no smoothing |

The outbox is on r12-k only. Copy it off before the server is destroyed if it is still needed.

The finals pipeline is `scripts/keysync/ks_final.sh <id> <src> [name]`:
1. KeySync at 10 steps on the dub voice.
2. SyncNet AV shift, then `ks_layer`.
3. `ks_fix --guard 3 --chin tex` (from r10-face scripts/r11, copied to `r11face/`).
4. `ks_tsmooth` (`KS_TSMOOTH`, default 0.8; `0` = off; `auto` = adaptive).
5. Raw paste with the demucs mix, encoded to ≤ 25 MB.

## Lag fix (ks_tsmooth.py)
How it works: a flow-aligned temporal smoothing of the lower-face region on the 25 fps layer. Frame t is warped by α × the mean of its flows to t−1 and t+1. A one-frame jerk is pulled toward its neighbours; smooth motion is untouched.

Jerk = the ks_lag spike rate. SyncNet = speech seconds, offset 0, 25 fps.

| video | v1 (none) jerk / SyncNet | α 0.8 jerk / SyncNet | auto jerk / SyncNet | source jerk |
|---|---|---|---|---|
| m18 | 16.1 % / 7.455 | 13.0 % / 7.458 | 14.1 % / 7.455 | 11.0 % |
| m03 | 15.5 % / 7.624 | 16.5 % / 7.636 | 16.7 % / 7.614 | 16.1 % |
| old03 | – / 7.261 | 15.7 % / 6.971 | 18.0 % / 7.162 | 24.3 % |

- The user accepted α 0.8 by eye on m18 ("как будто сильно лучше").
- old03 never had excess jitter, so a fixed α only costs sync there.
- **auto** sets α_t = 0.8 · clip((rK − rS) / 0.1, 0, 1), gaussian-smoothed. rK and rS are the layer's and the source's (master_25fps landmarks) spike rates in a 25-frame window.
  - The mean α it picked: m18 0.52, m03 0.32, old03 0.30.
  - **It does NOT pass the coordinator's gate:** old03 7.162 < 7.2, and m18 jerk 14.1 % vs 13 %.
  - Ideas to try next: a larger ramp or excess threshold (e.g. only when rK > 1.3·rS), per-frame spike gating instead of a window, a longer window.
- The default stays **0.8**. auto is behind `KS_TSMOOTH=auto`.

## Speed arms (KeySync only, s per video-minute, warm resident worker, 1×5090)
- Eval: `ks_eval.sh`, i.e. the SyncNet AV shift, α 0.8 smoothing on the raw 512² crop, then paste.
- Each cell: SyncNet / strong flashes / jerk.
- This eval smooths the raw crop rather than the guarded layer, so its absolute SyncNet sits below the finals path (f_base m03 7.58 here vs 7.64 in the final). The arm-vs-arm comparison is like for like.

| arm | s/vmin | m18 | m03 | old03 |
|---|---|---|---|---|
| f_base: 10 steps, the authors' settings | ~990 | 7.446 / 5 / 13.0 % | 7.583 / 2 / 11.9 % | 7.026 / 16 / 18.1 % |
| **f_s8: 8 steps + DC2:0 + fp16 VAE + compile UNet, full 3-way CFG** | **~496** | **7.567 / 7 / 13.6 %** | **7.619 / 5 / 11.8 %** | see `speed/f_s8_eval_ts_qo3.json` if it landed |
| f_t3: f_s8 + trunc:3 interp guidance | ~385 (old03 cold 433) | 7.155 / 4 / 12.5 % | 7.071 / 4 / 8.2 % | 6.633 / 8 / 15.7 % |
| f_t2: trunc:2 | ~351 | 6.841 / 4 / 12.0 % | 6.904 (unsmoothed) | – |
| f_t3dc3: trunc:3 + DC3 | ~316 | 6.095 / 8 | 6.471 / 9 | – |

Findings:
- **The truncated-guidance arms fail m03:**
  - f_t3 is −0.51 vs f_base on m03, with and without the smoothing (same shift).
  - It is −0.29 on m18 and −0.39 on old03.
  - The r11 "trunc:3 PASS" was measured on m18 / old03 without the smoothing; it does not generalise.
- **f_s8 (full CFG) holds sync** on m18 and m03 (≥ f_base) at ~2× the speed of f_base. It is the fastest KeySync that keeps sync so far.
- ~496 s/vmin is still above the ≤ 300 GPU-s/vmin E2E target.

## Not run (fixed in ks_fast.py, queued, server stopped)
- `--vae-cl`: channels_last now applies to the 2-D convs only. The temporal decoder has 5-D Conv3d weights, which crashed it.
- `--fp8`: torchao PerRow needs bf16 weights and the UNets are fp16. Now PerTensor, on Linear layers with dims % 16 == 0. Untested.
- The planned arms on the f_s8 base (`scripts/keysync/r12_speed4.sh`):
  - f_s6 (6 steps + DC2, full CFG)
  - f_s8dec (`--vae-cl --cudnn-bench --decoding-t 4`)
  - f_s8fp8 (`--fp8`)
- The r11 stage profile, for where the time goes: interpolation sampling ≈ 60 %, VAE decode ≈ 25 %, face_alignment landmarks ≈ 10 %.

## Multi-face videos (m03)
- The default crop (`ks_prep.py`: side = 1.7·face_h + 1·move) included the child. face_alignment then returned 2 faces, and KeySync animated the child.
- Two fixes:
  1. The keysync code patch `sgm/util.py` `ensure_landmarks_shape` keeps the largest face.
  2. A speaker-only crop: `KS_BOX_K=1.25 KS_BOX_M=0.4` in `ks_prep.py`.
- Generic follow-up: pick the speaking face with an active-speaker check before the crop.

## Code
- The full keysync (antonibigata/keysync 5e2503b) patch for sm_120 + r12 is `scripts/keysync/keysync_5e2503b_sm120_r12_final.patch`, applied with `git apply` in a clean clone. It covers:
  - SDPA instead of xformers with 4-D q/k/v
  - the skip-attention mask fast path
  - the weights_only allowlist
  - the optional face-ID imports and the WavLM path
  - AYS with fewer steps
  - the `KS_DC` DeepCache, the `KS_IGUIDE` guidance modes, `KS_ICHUNK`, `KS_LMK_BS`
  - the multi-face fix
  - the `interpolation_s6` / `s8` configs
- Setup: `scripts/keysync/setup_ks.sh` + `req_ks.txt`. Weights: `dl.py` (~13 GB, HF token from env; never print it).

## NEXT
1. On f_s8: run f_s8dec, f_s8fp8 and f_s6 (`r12_speed4.sh`) on m18 / m03 / old03, with the same eval and gate (SyncNet ≥ f_base − 0.3, flashes no worse).
2. Get under ~300 s/vmin without truncation. Candidates:
   - FP8 / int8 on the interpolation UNet
   - a cheaper VAE decode (decoding_t 4 in fp16; a tiny temporal decoder)
   - batching the CFG branches
   - DeepCache 2:1 on the keyframes only
   - landmarks from the r9 mediapipe pass instead of face_alignment (saves ~10 %)
3. Adaptive smoothing: tighten the trigger so old03 stays ≥ 7.2 and m18 near 13 %; then a mouth-crop 2-panel of old03 (v1 | auto).
4. Lip-corner defect sheet (management note "небольшой дефект с кончиками губ"); KeySync + k5 composite sync.
