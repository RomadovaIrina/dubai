# Distillation pilot: LatentSync 1.6 → a 4-step student (r13, 03.11)

**Question:** "does it even make sense?" **Answer:** yes, it is worth a real run. It is not junk, and not yet production quality.

## Setup (≈ 2.7 h of GPU in total)
- **Code** (`scripts/r13/distill/`) reuses LatentSync's own UNet, scheduler, StableSyncNet, mel and loss code. No LatentSync / generator / composite file is edited: the production generator is wrapped at runtime via `R10_GEN_PY`.
- **Data:**
  - Window conditioning (mask, masked + reference latents, whisper audio, mel) is captured from the **production generator** (our MediaPipe alignment) on the training clips only: 01, 04, 05, m15 + 4 other mamakris clips. 02 and one mamakris clip have no face; the 60-fps 1072×1904 clip was skipped (30 min to capture).
  - That gave **746 windows**; the teacher ran on 300 of them.
  - **old03, m03, m18 never seen.**
- **Teacher:** vanilla LatentSync 20-step DDIM, CFG 2.0, no DeepCache (6 s / window, 4.6 GB).
- **Student:** the same UNet + a LoRA, rank 32, on all 288 attention projections (15 M params). 4 DDIM steps, 1 pass, **no CFG**.
  - Loss = x0 MSE to the teacher + 0.05 × the stable-SyncNet loss on the decoded x0 (LatentSync's stage-2 recipe).
  - 1500 steps, lr 1e-4, batch 1 window: 37 min, peak 17.7 GB.
- **Eval:** the FULL production pipeline (k5 composite) through `dabai_job`, with the student merged into the generator.
  - SyncNet = the gate's `sync_speech`.
  - Flashes = `r3_flash` strong (the gate rule).
  - The control = the same 4-step / no-CFG sampler with the ORIGINAL weights.

## Results (held-out videos)

| | old03 | m03 | m18 |
|---|---:|---:|---:|
| SyncNet, k5 production (10 / 20 steps, CFG 2.0, DeepCache) | 7.58 | 8.16 | 8.35 |
| **SyncNet, student (4 steps, no CFG)** | **6.43** | **7.98** | **7.96** |
| SyncNet, control: original weights, 4 steps, no CFG | – | 6.56 | 5.56 |
| strong flashes, k5 / student | 39 / 29 | 18 / 10 | 8 / 7 |
| **face GPU-busy s/vmin**, k5 → student (face_gen + face_comp) | 266 → **167** | 261 → **180** | 287 → **195** |
| whole job GPU-busy s/vmin, k5 → student | 320 → 219 | 323 → 242 | 349 → 251 |

Held-out windows of the training clips:
- **Latent MSE to the teacher:**
  - untrained 4-step 0.073;
  - student best 0.058 at step 750, then drifting to 0.079 at step 1500 (small data, constant lr);
  - for scale, production's 10-step CFG sampling is at 0.010.
- **Stable-SyncNet similarity:** teacher 0.80, untrained 0.86, student 0.80–0.84. Not informative: it even rates the blurry untrained 4-step output higher.

## Reading it
- **The training clearly works on lip-sync.** Against the plain 4-step sampler the student gains **+1.4 (m03) and +2.4 (m18) SyncNet**, from 37 minutes on 300 windows of 9 short clips. On m03 / m18 it ends up within 0.2–0.4 of production.
- **old03 stays 1.15 below k5**: the bearded face, the hardest one, and none like it in the tiny training set.
- **Speed: −35 % face GPU time.** The whole job lands at 219–251 s/vmin, **under the 300 norm** on all three, without any other lever.
- **Look:** mouths are plausible and stable, with no artefacts. They are **less articulate**: smaller openings, flatter teeth / inner mouth (2-panels). The fewer "flashes" are partly that.
- **The latent-MSE curve says the run is far from converged.** It improves, then drifts with a constant lr. A real run needs more data and a schedule.

## Is it worth a real run? Yes, and what I'd change
1. **Data, the main lever:** thousands of windows, not 300.
   - A licensed talking-head set; the user wants to see the sources first.
   - Or simply all our pipeline's own inputs as they arrive.
   - Teacher targets precomputed: 6 s / window → ~2.5 GPU-h for 1500 windows.
2. **Training:** a 3–5k-step cosine-decay lr; EMA of the LoRA weights; a validation set of whole held-out VIDEOS scored by SyncNet, not latent MSE (which tracks quality poorly here).
3. **Losses:**
   - LatentSync's full stage-2 set: + LPIPS and the pixel loss inside the mouth mask, against the "dull mouth";
   - a higher sync weight (0.1–0.2);
   - the consistency target (teacher-trajectory states instead of re-noised endpoints) for sharper detail.
4. **Steps:** a **6-step** student as the safer middle ground: still CFG-free, ~−25 % face GPU.
5. **Gate:** the r10 / r12 face gate (SyncNet −0.3, flashes, stick / slide, pressed lips) + the user's eye on 2-panels, as for every face lever.

**Clips** (`outbox/r13_spec/distill/`):
- `old03_mouth_k5_vs_student4_half_speed.mp4` and `m03_mouth_k5_vs_student4_half_speed.mp4` (0.5×);
- the normal-speed versions, plus m18.
