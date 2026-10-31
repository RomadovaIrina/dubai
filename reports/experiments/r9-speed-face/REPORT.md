# r9 FACE speed: REPORT (30–31.10.2026)

Goal: ≤ 300 s per video-minute end-to-end on one RTX 5090 without degrading the user-approved "r9 + fix" face. Branch: `experiment/r9-speed-face` (pushed).

All numbers are FACE ONLY (generator + composite; the audio stage is reused from a finished handoff), one job on the GPU.

**Clocks:**
- **wall** = elapsed s per video-minute;
- **GPU-busy** = s per video-minute with nvidia-smi utilisation > 10 % (100 ms samples; the first runs backfilled from 500 ms).

**Gate vs the r9 + fix output of the same video:**
- SyncNet dub speech ≥ −0.3;
- strong flashes ≤ +2;
- pressed lips ≤ +0.5 s;
- the whole-frame gate PASS;
- stick / unstick and slides ≤ +1 per video (+2 over 60 s) (coordinator rule 30.10);
- occluder ring ≤ base;
- fallback ≤ +1 %.

Tools: `scripts/r9/face_only.sh` (runner), `scripts/r9/gate_cmp.py` (gate).

## 1. Result on r9-a, m03 (40.5 s, 720p30): the recommended stack

| variant | wall s/vmin (gen + comp) | GPU-busy s/vmin | gate vs b0 |
|---|---|---|---|
| **b0**: r9 + fix (E2E_FX look) | **982** (677 + 305) | 276 | reference: sync 7.61, flashes 20, pressed 0.67, stick 0 / slide 1, fallback 3.92 % |
| S: stack (below, incl. mouth-only decode) | 635 (429 + 206) | 266 | FAIL: stick 0 → 2 (the events appear in the composite / fallback; the generated layers have the same counts). Rest PASS. |
| S2: S without mouth-only decode | 642 (437 + 205) | 278 | PASS on every row measured (stick within +1: dropping the decode fixes S), whole-frame gate not finished |
| **S + VAE compile (warm cache) = RECOMMENDED** | **619** (416 + 203) | **245** | **PASS on every row**: sync 7.43, flashes 17, pressed 1.10 (+0.43), stick 1 / slide 0, fallback 3.92 %, frame gate PASS |
| S + VAE compile (first run, cold cache) | 810 | n/a | identical metrics to the warm run (deterministic). The ~120 s compile is paid once per machine (inductor disk cache). |
| S + half-rate generation | **552** (346 + 206) | **158** | FAIL by the stack's own stick +2 (same as S). Sync 7.34 (−0.275, inside −0.3 but thin); flashes 21, pressed 0.73. |

**Recommended stack** (environment of `face_only.sh` / e2e, all behind flags, defaults unchanged):

    GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast CFD_EXTRA=--threads=12 FAST_MPREUSE=1 LS_DEC_ROW0=15 FC_EXTRA=--pre-threads=12 \
    LS_VAE_COMPILE=max-autotune-no-cudagraphs

That is 982 → 619 s/vmin wall (−37 %) on r9-a, gate PASS.

**Half-rate** (LS_HALF_RATE=1) is the largest single lever left: −67 s/vmin wall on top of S, and the GPU drops to 158 s/vmin. Its own cost in the gate is small: vs S, sync −0.16, flashes +3, stick ±0. The thin sync margin needs a second video before it is adopted. The m19 check was queued at hand-over (section 4).

## 2. Lever by lever (measured alone; r9-g = 830 s/vmin baseline machine, r9-a = 982)

| lever | switch | what it does | effect | gate |
|---|---|---|---|---|
| L1 | `GEN_COMP=0` | the generator's own source-fps composite is not run (the fx composite redoes it); only the layer MediaPipe it wrote is kept | m03 827 → 729, m19 1127 → 887 (r9-g) | **bit-identical** (md5) |
| L4 | `FAST_SRCPREP FAST_FB_BG FAST_ENC=ultrafast` | the source timing / landmarks / rigid motion are computed during LatentSync (CPU); the fallback scoring runs parallel to the layer SR; ultrafast x264 for the intermediate layers | 729 → 629 (r9-g) | PASS |
| CF-detail threads | `CFD_EXTRA=--threads=12` | r8_cf_detail per-frame work on 12 threads | 629 → 614 (r9-g) | **bit-identical** |
| MediaPipe reuse | `FAST_MPREUSE=1` | the composite's source pass = the audio stage's npz (identical); the generator's master pass indexed from it via the frame map (0.29 px mean diff) | −10 | PASS |
| mouth-only VAE decode | `LS_DEC_ROW0=15` | decode latent rows 15–63 only (the mask regenerates rows 23–60) | −7 (decode −31 %; max 24/255 on the mouth rows) | PASS alone; in the stack S together with MP reuse: stick +2 |
| fps_composite pre-threads | `FC_EXTRA=--pre-threads=12` | sanitize stats, C2 boundaries and the closed-loop sharpen samples on threads | fps_composite 61.8 → 47.2 s on m03 (≈ −22) | **bit-identical** |
| VAE compile | `LS_VAE_COMPILE=max-autotune-no-cudagraphs` | the VAE encoder / decoder torch.compile at one 16-image shape (padded) | −16 wall / −21 GPU (warm). Bench: decode −41 %, encode −36 %, diff ≤ 0.7/255 | PASS |
| half-rate generation | `LS_HALF_RATE=1` | LatentSync on every other master frame (12.5 fps, each frame keeps its own audio window); odd frames = bidirectional DIS interpolation inside the generated region | −67 to −83 wall, GPU −108 | see section 1 |
| stab threads | `STAB_EXTRA="--threads 12"` (+ `FAST_ENC` preset) | r9_hf_stabilise per-frame on threads | stab 42 → 18 s on m03 | **bit-identical** (for the "fx + stab" config; stab itself costs +61 s/vmin at 1 thread) |
| skip-silent | `LS_SKIP_SILENT=1` | no LatentSync where dub energy < −45 dB AND the source mouth is closed AND no source speech (±0.2 s); runs ≥ 16 frames, minus the crossfade | m03 estimate ~2 % of frames (most dub-silent frames have a talking source) | run queued at hand-over (section 4) |
| LS_ROI | `LS_ROI=1` | LatentSync face align / paste-back on a crop | 0 at 720p (m03: affine + restore 10 s); aimed at 1080p (m01 ~130 s) | PASS on m03; **not tested at 1080p** |
| bridge batching | `LS_BRIDGE_BATCH=4` | the seam bridges denoised together | 0 (the UNet is saturated: 12.3–12.7 ms/frame at every batch); batch 8 crashes (kernel launch limit) | dropped |

## 3. Where the time is (and why ≤ 300 wall is not reached for the face alone)

- GPU-busy is only ~25–40 % of the face wall time. Mostly CPU, single process:
  - the generator's MediaPipe prep (~50 s/vmin);
  - CodeFormer / affine / restore glue;
  - the layer passes (CF-detail, SR flows, repair);
  - fps_composite (sanitize, retime flows, encode).
- With half-rate the face needs 158 GPU-s/vmin. The CPU part can overlap with another video's GPU work (batch throughput), but not inside one video without restructuring the pipeline into one process with the composite streaming behind the generator.
- LatentSync's UNet is compute-saturated in fp16 (DeepCache interval 5 already makes 4 of 5 calls partial).
- UNet torch.compile: −20 % per full call, but a ~290 s compile per input shape (DeepCache partial calls, bridges, tail windows) → not usable as is.
- TensorRT (torch-tensorrt 2.7 / TRT 10.9, installed on r9-g only): the build failed (16 GB workspace OOM, then no kernel for a motion-module node). It would also need a second engine for the DeepCache partial calls. Estimated gain ~1.3x on ~100 s → −45 s/vmin, real engineering.

## 4. Not finished / untested at hand-over (servers destroyed)

- **m01 (1080p60): nothing measured on r9-a.** The baseline alone is ~1 h. The 1080p levers (LS_ROI, the threading levers, which scale with resolution) are untested there. From r9-f's old logs the m01 composite is 1192 s/vmin (CF-detail 385 s, SR 259, fps composite 843): the biggest remaining target.
- **m19:** b0 + S2 + half-rate were queued in qa2; see `/workspace/dub/r9/face/m19_*` if they finished (they may not have).
- **skip-silent:** m03 S2k queued in qa2.
- S2 = S without the mouth decode PASSES (stick within +1) except the whole-frame gate, unfinished: S2 + VAE compile (~626 expected) is the most conservative passing stack; S2 + half-rate (~560) the next to gate.
- The fused mouth-ROI composite (CF-detail + SR + stab + repair in one pass): designed, not built. The per-step profile on m03 is CF-detail 13 s, SR 28, repair 9, fps_composite 47 after pre-threads.
- The r9-g commits (eefc560) are lost unless r9-g returns. Everything was rebuilt from the kit and re-measured on r9-a; only the r9-g timings in section 2 come from there.

## 5. Outputs

`/workspace/dub/outbox/speed_face/` on r9-a:
- `m03_WATCH_r9fix_vs_fast.mp4` (r9 + fix | recommended stack, side by side, htdemucs bed);
- `m03_r9fix_demucs.mp4`;
- `m03_fast_demucs.mp4` / `m03_fast_spleeter.mp4` (both beds).

The rebuild kit is in the coordination notes folder: `r9-speed-face-kit/`.
