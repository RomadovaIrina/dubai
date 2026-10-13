# Human feedback 2026-10-13 — twitching mouth / facial expression, foreign sounds, noticeable pauses

> «Вижу этот дергающийся рот постоянно и мимику и от этого меня тригерит. Так же обнаружил посторонние звуки, которых раньше не было.
> Обратите на это внимание тоже. По поводу пауз, тоже заметно конечно это всё.»

Status: **iteration 1 — reproduction + diagnosis on 03 + first A/B** (in progress; the other videos follow).

## Environment (clean box)

```text
repo/branch  RomadovaIrina/dubai @ tts-intelligibility-fix, HEAD 73a30275b7e1dd4477e2472a200c6743fdcfcc8f (remote HEAD checked = expected)
GPU          NVIDIA GeForce RTX 5090, 32607 MiB, driver 595.91.07 (CUDA 13.2 driver); torch 2.7.1+cu128 (pins unchanged)
disk         50 GB overlay (13 GB free after venv 8.2 GB + weights 20 GB; uv cache cleaned)
setup        setup_dabai.sh 0-9 exit 0 (~3 min), download_models.sh exit 0, run_smoke.sh 11/11 PASS
             pip-freeze drift: patch bumps of unpinned packages only (aiohttp, lmdb, multidict, ...); lock file not changed in git
inputs       test_videos/01..05 from the README Drive link; HF_TOKEN in /workspace/dub/.env (not in git)
caveat       another process (/mem, ~8.8 GB VRAM, GPU compute) shared the GPU during these runs -> wall / LatentSync seconds below
             are NOT valid for the runtime budget (LatentSync ran ~2x slower); quality metrics are unaffected. Runtime is re-measured separately.
```

## What was reviewed (checked in code / manifests, not commit messages)

| flag | default at 73a3027 | reviewed candidate (cand4) |
|---|---|---|
| `--alignment burst --video-backend optimized --codeformer optimized` | passed by `refresh_e2e.sh` (the production command) | same |
| `--background on` (Demucs bed under the dub) | **default** since 2026-10-12 | on |
| `--phrase-cut` | `timing` (no fades on the cut parts) | **`acoustic`** (+ 5 ms declick fades) |
| `--gate-merge-gap-s` | 0.6 | **1.2** |
| `--mouth-only-pasteback` | off | **on** |
| `--qwen-gpu-layers` | 0 (CPU, user decision) | 0 |

The review report (`restore_review_2026-10-12.md`) ends with "candidate 4 in progress" = the right column; the human feedback above refers
to it. Both were reproduced on 03 (same TTS audio: Qwen / Chatterbox are deterministic on this box — identical texts and unit timings).

## A. Twitching face — the cause is CodeFormer on the eyes / brows, not the mouth

New tool `scripts/quality/face_temporal.py` (insightface 2d106 = LatentSync's own detector on master and output; positions in a face frame so
head motion cancels; regions: LS core / +-5 frames of a switch / original). Ratios output / master (1.0 = as calm as the source):

| 03 | defaults | reviewed (cand4) | **VA = cand4 + `--codeformer-scope mouth`** |
|---|---:|---:|---:|
| LS <-> original switches (per min) | 22 (14.6) | 10 (6.6) | 10 (6.6) |
| short original gaps < 1.2 s between LS | 6 | 0 | 0 |
| LS segments < 2 s | 1 | 0 | 0 |
| mouth landmark jitter, inside LS | 0.94 | 0.95 | 0.95 |
| mouth-opening jitter, inside LS | 0.74 | 0.76 | 0.74 |
| eyes / brows landmark jitter, inside LS | **1.37** | **1.32** | 1.23 |
| upper-face pixel flicker, inside LS | **1.16** | **1.15** | **1.00** |
| upper-face sharpness, inside LS | 0.86 | 0.86 | **0.98** |
| upper-face sharpness step at a switch | 10.1 % | 9.3 % | **3.7 %** |

* The mouth inside a continuous LS segment is NOT more jittery than the source (0.94-0.95) and the crop inputs are calm (LatentSync bbox
  acceleration 1.4 px/frame², affine points 0.9 px/frame² on the master) -> no LatentSync / crop problem to fix first.
* The eyes / brows are. `MouthOnlyRunner` composited the LatentSync mouth into the source frame, **then** CodeFormer restored the whole face of
  that composite (`process_video`: runner -> `cf_runner.restore_frames`). CodeFormer re-synthesises eyes, lids and brows per frame: in every
  LS frame they differ from the source (lids more open, iris sharper — `sheets/03_eyes_8.64s_reviewed_VA_master.jpg`, rows reviewed / VA /
  master) and they jump back at every LS -> original switch. That is the "мимика" that twitches; mouth-only paste-back alone could not fix it.
* Fix candidate **VA** `--codeformer-scope mouth`: CodeFormer runs on the raw LatentSync frame and the mouth-only composite is applied AFTER it,
  so outside the lower-face ellipse the output is the source. Upper-face flicker 1.15 -> 1.00, sharpness 0.86 -> 0.98. Same LatentSync frames,
  same CodeFormer work -> no runtime cost. (The remaining 1.23 eye-landmark ratio is detector coupling to the changed mouth; the eye pixels are
  the source pixels.)
* Speed lever, quality-identical: the mouth-only composite is now built on the ellipse ROI only (bit-identical to the full-frame version,
  10 -> 2.5 ms per frame, ≈ -15 s on 03).
* Queued: VB = VA + mask bbox smoothing (+-3 frames), VD = VB + LatentSync landmark smoothing (+-2 frames, `--ls-landmark-smooth`).

## B. Foreign sounds — localisation on 03

New tool `scripts/quality/audio_stages.py`: event detectors on the final timeline + trace through raw TTS (`tts_u<id>_raw.wav`, now saved by
the runner) -> cut part -> placed / atempo unit track -> dub voice -> final mix -> MP4.

| 03 | defaults (timing cut, no fade) | reviewed (acoustic cut + 5 ms fade) |
|---|---:|---:|
| impulsive clicks in the dub voice | 1 | 0 |
| inner cut tails within 12 dB of the speech level / max | 4 / -6.9 dB | 0 / -18.1 dB |
| hot cut edges (head + tail) | 7 | 1 |
| speech-like energy outside the CTC words carried into a part | 0 | 0 |
| two dub parts overlapping | 0 | 0 |
| clicks in the final mix only (background bed) | 3 | 1 |
| residual source speech in the bed (Silero on the bed) | 1.3 s | 1.3 s |

* The dub voice of the reviewed candidate is clean on 03: the acoustic cut + declick fades is the better cut (not the source of new sounds).
* Everything left comes from the **background bed** (Demucs drums+bass+other, `--background on`, new on 2026-10-12 = "звуки, которых раньше
  не было"): residual source voice in a dub pause (88.6 s: the original's harmonics under silence of the dub, -19 dB rel. the dub voice,
  `sheets/03_reviewed_ev_bed_voice_0088.60.png`) and a separation transient that is in neither the original nor the dub voice (46.2 s,
  `sheets/03_reviewed_ev_click_0046.31.png`). 02 (room-tone fill) / 01 / 05 (music) / 04 still to check.

## C. Pauses — source hesitations copied to the wrong place of the English sentence

New tool `scripts/quality/pause_rhythm.py` (pause distribution, intra-sentence pauses classified preserved / inserted and weak boundary =
part ends without punctuation on a function word / inside a fixed phrase; source-without-dub by cause; timeline PNG).

* 03 reviewed: dub pauses > 0.5 s 21 vs 17 in the original, > 1 s 12 vs 10; 18 of 20 intra-sentence pauses are the source's own pauses, but 9
  of them (10.5 s) land on a weak English boundary: "Probably more in | 1.9 s | general", "I'll send it to our | 1.6 s | chat", "Well, and |
  1.4 s | basically", "moving | 1.5 s | forward". The cut DP (acoustic mode) has no syntax term, the timing split puts the word boundary there.
* Candidate **C1** `--long-pause-func w`: a cut after a function word / wh-word / number / inside a fixed phrase costs up to `w` more, scaled by
  the SOURCE pause at that boundary (0 at 0.3 s, full at 0.8 s); short pauses keep the plain acoustic rule (the rejected syntax mode moved cuts
  at every boundary and lost coverage). New tool `scripts/quality/recut_offline.py` re-runs split -> cut -> span merge -> placement -> track on
  the SAME raw TTS (reproduces the reviewed placements exactly, 32/32), so the policy is compared without Qwen / TTS noise:

| 03 offline | reviewed (w = 0) | w = 0.8 | **w = 1.2** | w = 1.5 |
|---|---:|---:|---:|---:|
| weak-boundary pauses > 0.3 s (> 0.8 s) · seconds | 9 (7) · 10.5 | 4 (3) · 4.4 | **2 (1) · 1.5** | 1 (1) · 0.9 |
| source speech without dub s | 8.8 | 9.0 | 9.3 | 9.5 |
| VAD IoU | 0.837 | 0.830 | 0.825 | 0.818 |
| atempo max / > 1.15 | 1.15 / 0 | same | same | same |

  w = 1.2: "Probably more in general | what I want to say…", "…with you | and in general…", "Well, | and basically…", "…to our chat | so
  let's…", "…we desire, | moving forward." Cost +0.5 s of source speech without dub on 90 s (the old "speaker talks, dub is silent" class) ->
  must be checked on all videos before acceptance.

## Next

VB / VD video A/B, reviewed-config runs of 04 / 05 / 01 / 02 (face metrics, bed check, pauses), C1 offline on all videos, then one E2E
candidate = VA (+ best of VB/VD) + C1 (+ bed fix if localised), regression on 01-05, runtime on a free GPU.
