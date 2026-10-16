# Track A — Phase 1: prior work, diagnosis so far, plan, shortlist (draft, 2026-10-16)

Desk research ran on the CPU while the baseline videos were rendering. Nothing new is installed and no GPU time has been spent on any
candidate. **Waiting for the coordinator's go** before installing a model or starting the Phase 2 A/B.

## 1. Prior work (read: reports/quality/*, feedback_2026-10-13/*, restore_2026-10-12/*, reports/pilot/*, git log 968f33a..559d700)

| area | done | state |
|---|---|---|
| speech gate (VAD source ∪ dub, margin 0.24 s), RetinaFace router stride 3, in-memory LS segments, 4-frame LS<->original crossfade | yes | **default** (optimized backend) |
| gate merge gap 1.2 s (03: 22 -> 10 -> 8 switches) | yes | opt-in, **in cand5**; 2.0 s on 04 (14 -> 4 switches, +17 % LS frames) = option waiting for a human look |
| mouth-only paste-back E4 (feathered lower-face ellipse from LS's bbox) + mask bbox smoothing ±3 (no measurable effect) | yes | opt-in, **in cand5** |
| LatentSync landmark smoothing ±2 frames before the affine crop (03 lip jitter 0.95 -> 0.89) | yes | opt-in, **in cand5** |
| CodeFormer: optimized in-memory, w 1.0, fp16; scope mouth (eyes no longer re-rendered: upper-face flicker 1.46 -> 1.01 on 01) | yes | optimized = refresh_e2e default; scope mouth **in cand5** |
| E5 LatentSync sampler probe on 04 seg 1: DeepCache 5/3/off, steps 20/25/30, batch 1/2 = no quality change; guidance 2.0 SyncNet 4.45 -> 5.49 (2.5/3.0 queued, never reported) | yes | not adopted |
| `--ls-steps` 16 + DeepCache 7 (-13 % LS time, SyncNet same); 12 steps: a human saw lip twitches that SyncNet / face_temporal did not | yes | opt-in, not in cand5 |
| audio: acoustic phrase cuts, long-pause rule, adaptive fades, TTS gap QA, per-word QA, ordered translation, background bed, fit-down rule, sentence flow | yes | cand5 audio = acoustic + long-pause 1.2 + adaptive + gap QA; the rest opt-in. **Not touched by this track.** |
| Qwen on GPU / resident worker / video-prep parallel | speed only | rejected (user: Qwen stays CPU) / opt-in / measured harmful |

**Never measured or tried (the gap this track fills):** where inside or around LatentSync the twitch comes from; the 16-frame window
seams; the reference-frame leakage; the 30 -> 25 fps master judder; any lip-sync backend other than LatentSync 1.6.

## 2. Diagnosis on the reproduced 03 baseline (no GPU beyond the metric pass; `scripts/quality/lipsync_jitter.py`, new)

Metrics are head-stabilised (one similarity transform per frame from the master's eyes + nose, applied to master and output alike) and are
output / master ratios (1.0 = as calm as the source at the same frames). `mouth_warp` = warping error after DIS optical flow (Lai et al. 2018):
change that smooth motion does not explain, i.e. texture popping / twitch. JSON: `/workspace/dub/artifacts/baseline/metrics/lipsync_jitter/03_baseline.lipsync_jitter.json`.

| 03 baseline (cand5) | frames | mouth_warp | mouth_diff | lip_jitter | open_jitter | upper_warp | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 inside continuous dubbed speech | 1174 | **1.16** | 0.88 | 0.89 | 0.76 | 1.02 | 0.78 |
| S2 LS <-> original switch ±5 frames | 80 | 1.09 | 1.01 | 1.11 | 0.85 | 1.02 | 0.85 |
| S3 head turn / nod (LS frames) | 178 | **1.20** | 0.93 | 0.91 | 0.62 | 1.02 | 0.81 |
| S4 closed mouth in the source (LS frames) | 112 | **1.20** | 1.03 | 1.02 | 0.70 | 1.02 | 0.78 |
| S4 closed mouth in the output (LS frames) | 953 | 1.09 | 0.85 | 0.83 | 0.44 | 1.02 | 0.79 |
| control: pass-through core | 195 | 1.02 | 1.01 | 1.00 | 1.01 | 1.01 | 1.01 |

Findings:
1. **LatentSync's 16-frame windows leave a seam every 0.64 s.** Transitions into a window start vs all other LS transitions:
   output mouth_diff ×1.46, mouth_warp ×1.43, master at the same frames ×0.97 / ×0.98 -> window step factor **×1.51 / ×1.45**
   (n = 123 seams on 03; upper face ×1.00). Same on every face video: mouth_warp step ×1.42 (01), ×1.40 (04), ×1.60 (05) (`PHASE0.md`). Per-phase profile of mouth_warp: phase 0 = 2.59, phases 1-15 = 1.71-1.99, master flat 1.57-1.76.
   This is generator-internal (windows are denoised independently; the noise is identical, the content is not) and is periodic = "constant".
2. **The generated mouth articulates about half as much as the source (03; 0.65-0.70x on 01 / 04 / 05) and carries more unexplained texture change**: inner-lip gap median
   0.044 vs 0.087 inter-ocular units (output 95th pct 0.14 vs 0.20); 953 of 1985 LS frames are "closed or almost closed" in the output vs
   112 in the source. Overall motion (mouth_diff 0.86) is calmer than the source, but motion-unexplained change is higher (warp 1.13).
   Hypothesis: **reference leakage** — at inference LatentSync's reference image is the current frame (with the Russian mouth), while it was
   trained with a reference window from a different time (`latentsync/data/unet_dataset.py` lines 72-79). The model partly copies the source
   mouth and partly follows the English audio.
3. **Whole-video 5 Hz judder from the 30 -> 25 fps master** (`ffmpeg -r 25` drops every 6th source frame): on the 03 master, head
   displacement per frame by index mod 5 = 0.78 / **1.19** / 0.80 / 0.79 / 0.78 px (control mod 6: flat 0.81-0.89). It affects original and
   lip-synced frames alike, so it is not LatentSync, but it feeds the generator and is visible as stutter. Fixing it changes the output FPS
   (25 -> 30) -> **needs the coordinator's decision** (see §5 Q2).
4. Switches (S2) are not the dominant source on 03 with cand5 (8 switches, warp ratio 1.09, upper face 1.02) — consistent with the 2026-10-13
   fixes. Crop inputs: bbox acceleration 1.36 px, affine-point acceleration 0.88 px (face_temporal) after the ±2 smoothing.

## 3. Phase 2 plan — cheapest useful A/B on 03 40-58 s (18 s, ~370 LS frames)

The window contains continuous speech (40-44.5 s, 46-56 s), LS->original->LS switches at 44.56 / 46.04 s, a head turn 47.5-48.6 s (worst
head-turn jitter at 47.8 s), a closed source mouth 41.7-42.5 s, and a speech->silence switch at 56.24 s. Same dubbed audio, gate, crossfade,
mouth-only paste-back, CodeFormer-mouth settings, 25 fps, frame count as the baseline; only the generator (or the one wrapper part under
test) changes. The "current LatentSync" arm is re-run on the same window (window phases restart at the cut, so the full-run frames are not
the fair reference). ~90 s GPU per LatentSync variant.

| id | variant (one change each) | question it answers | why it is not a repeat |
|---|---|---|---|
| L0 | wrapper only: crop -> inverse warp -> mouth composite -> CodeFormer-mouth, the original crop instead of the UNet output | how much jitter the crop / affine / paste-back / restorer adds alone | never isolated |
| L1 | L0 + VAE encode/decode of the crop | VAE texture flicker | never isolated |
| L2 | LatentSync driven by the ORIGINAL Russian audio | generator jitter without the language mismatch | never run |
| L3 | LatentSync, dub audio (= baseline on the window) | reference | – |
| S1 | overlapping windows (stride 8, per-step latent fusion with triangular weights, MimicMotion-style; training-free, in our wrapper) | removes the ×1.5 seam every 16 frames? | windows never overlapped |
| S2 | reference window from a different time (training-consistent; a calm, frontal, closed-mouth 16-frame window of the same segment) | leakage -> twitch / weak articulation? | never tried |
| S3 | S1 + S2 | combined | – |
| T1 | **TBDub student** (Apache-2.0), own venv `/venv/tbdub` (Python 3.10), driven by our aligned crops (`--cropped-input`) + our mouth ellipse | a seam-free modern backend vs LatentSync | new backend |

Measured for every arm: `lipsync_jitter.py` (4 situations + window test), `face_temporal.py`, SyncNet / AV offset on the window (final mix
and dub-only), mouth / upper sharpness, identity (DINOv2 on the mouth-masked face crop, Apache-2.0 — new metric dependency, asks for the go
too), LS time, peak VRAM. Muxed audio: the video stage reuses `final_mix_aac.m4a` -> bit-identical by construction (sha256 check).
Side-by-side clips ORIGINAL | BASELINE | CANDIDATE at the worst timestamps into `/workspace/dub/artifacts/lipsync-temporal/sbs/`.
Optional, only if T1 is unclear: MuseTalk 1.5 as a negative control (per-frame by design). VideoReTalking is not run (§4).

Estimated GPU for Phase 2: L0-L3 + S1-S3 ≈ 15 min, TBDub install + weights (~16 GB student + VAE + HuBERT) + 2 runs ≈ 30 min, metrics
≈ 15 min -> **≤ 1 GPU-hour**.

## 4. Candidate shortlist (licences: `LICENSES.md`)

1. **LatentSync 1.6 stabilised in the wrapper** (S1/S2, later stronger offline affine smoothing if L0 shows crop jitter matters). Code
   Apache-2.0, weights openrail++. **Shipping it needs the InsightFace buffalo_l detector replaced or licensed** (it is NC and is in the
   production image today) -> a MediaPipe Face Landmarker crop A/B belongs to Phase 3 whatever wins.
2. **TBDub** (Alibaba 2026-09, Apache-2.0 code + weights, MediaPipe / Wan2.2 VAE / HuBERT all Apache-2.0, no InsightFace / face parser):
   77-frame clips with cross-clip motion conditioning, 2-step student, 15-28 GiB.
3. Not shortlisted: Lip Forcing 14B (seam-free, Apache-2.0, but ~37-41 GB VRAM: does not fit the 5090; NC-ND teacher data);
   MuseTalk 1.5 (per-frame, README admits jitter; NC parser); VideoReTalking (NC: BFM, dlib-68, LRS2, PIRenderer code — `RESEARCH_ONLY — cannot ship`);
   KeySync / HighSync (`RESEARCH_ONLY — cannot ship`); X-Dub (superseded by TBDub); InfiniteTalk (re-performs the whole frame).

## 5. Questions for the coordinator

* Q1 go for Phase 2 as in §3 (install TBDub into its own venv, download its student weights; DINOv2 for the identity metric)?
* Q2 the 30 -> 25 fps master judder (§2.3) is around the generator but changes the output FPS if fixed (keep the 30 fps source frames,
  retime only the generated mouth). In scope for track A as a separate experiment, or parked?
* Q3 licence: InsightFace buffalo_l (NC) is in the production image (`infra/serverless/handler.py`); CodeFormer (NC) is in cand5 but
  "OFF in production" per `infra/serverless/README.md`. Which one is the shipping reference?
