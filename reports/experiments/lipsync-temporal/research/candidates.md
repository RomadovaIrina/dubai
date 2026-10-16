# Lip-sync temporal stability: candidate survey + LatentSync stabilisation

This is desk research only: web, GitHub, Hugging Face and arXiv pages. Nothing was installed, cloned or run, no weights were downloaded, and `/workspace/dub/dubai` was not touched. Date: 2026-10-16.

**Sources.**
- I verified LatentSync, MuseTalk 1.5, VideoReTalking, Lip Forcing, HighSync, SyncEdit and the Task 2 material myself. I also re-checked TBDub's LICENSE and HF card and the Hallo3 dataset licence.
- Three parallel desk sub-surveys covered the rest: older GAN baselines, 2025 diffusion/DiT work, and 2026 work plus APIs. Their licence quotes were cross-checked where they overlapped: TBDub, X-Dub, Lip Forcing, KeySync, InfiniteTalk and MuseTalk were each checked independently by at least two of us, with consistent results.
- Where a finding is not mine it says "(sub-survey)".
- **UNVERIFIED** means nobody could read the primary source. No licence was guessed.

**Not legal advice.** "Commercial YES" means the published licence text allows it. Training-data provenance is flagged separately because it is a residual risk that no open model fully clears.

## 0. Key findings (TL;DR)

1. **The current pipeline already has a licence problem.** LatentSync's `latentsync/utils/face_detector.py` calls `FaceAnalysis(allowed_modules=["detection","landmark_2d_106"], root="checkpoints/auxiliary")` without a `name`. InsightFace's `constant.py` sets `DEFAULT_MP_NAME = 'buffalo_l'`, so the **buffalo_l pack is auto-downloaded**. InsightFace says: "The pretrained models provided with this library are for non-commercial research only, whether downloaded automatically or manually." This must be replaced (MediaPipe, below) whatever backend we choose.
2. **Probable causes of the "constant twitch" are visible in LatentSync's own code:**
   - hard-cut, **non-overlapping 16-frame windows** (a seam every 0.64 s);
   - an **unsmoothed per-frame affine crop** (only an EMA on translation);
   - the **reference = the same frame, unmasked**, so the original Russian mouth leaks in;
   - internal **25 fps** re-encode against our 30 fps source.

   All of these can be fixed without training (§3).
3. **Best new commercial-compatible backends:**
   - **TBDub** (Alibaba, 2026-09): Apache-2.0 code and weights on an Apache Wan2.2/X-Dub chain, no InsightFace/BiSeNet, 2-step student at about 7 FPS, fits a 32 GB card.
   - **Lip Forcing** (KAIST, 2026-07): Apache-2.0, LatentSync-style mouth inpainting with autoregressive streaming, so no window seams; best FVD. But it needs about 37 GB+ VRAM, ships with buffalo_l, and its teacher was trained partly on CC BY-NC-ND data.
4. **MuseTalk 1.5 and VideoReTalking are not stability upgrades.**
   - Both generate per frame. MuseTalk's README says "There exists some jitter as the current pipeline adopts single-frame generation".
   - Both pull in non-commercial sub-models: MuseTalk's BiSeNet/CelebAMask-HQ parser; VideoReTalking's BFM, dlib-68, GPEN and LRS2-trained L-Net.
5. **No LatentSync 1.7 or 2.x exists.** The last release is 1.6 (2025-06-11).

---

## 1. Summary table

Abbreviations: NC = non-commercial; TF = training-free; RO = research-only; FA = face_alignment.

| model | repo | code licence | weights licence | commercial | problematic deps / data | temporal mechanism | speed / VRAM (RTX 5090 32 GB?) | verdict for us |
|---|---|---|---|---|---|---|---|---|
| **LatentSync 1.5/1.6** (current) | github.com/bytedance/LatentSync | Apache-2.0 | `openrail++` (HF) | **NO as shipped** (YES with detector swapped) | **InsightFace buffalo_l (NC)**. Data: VoxCeleb2 (metadata CC BY-SA 4.0, video copyright unaddressed), HDTF (CC BY 4.0), unnamed Chinese set | SD-UNet + AnimateDiff motion module; **fixed non-overlapping 16-frame windows**; ref = same frame; identical noise for all frames | 1.6: 18 GB; ~41 s per output second at 20 steps (#365). **5090: yes** | Baseline; apply §3 fixes + detector swap |
| **TBDub** (Alibaba Taobao/Tmall, arXiv 2609.06144, 2026-09-05) | github.com/TaoLiveAIGC/TBDub | Apache-2.0 (LICENSE verified) | `apache-2.0` (HF verified) | **YES** | MediaPipe Face Landmarker: model card PDFs link Apache-2.0 (text not extractable, low risk). HuBERT-large (Apache-2.0). Wan2.2 VAE (Apache-2.0). Data: Alibaba in-house | Wan-style DiT, mask-free 512² face edit; 77-frame clips with cross-clip motion conditioning; teacher 30 steps, student 2 steps (DMD2) | Student 7.13 FPS on H20; 27.9 GiB default, 15.0 GiB with `--cpu-offload`. **5090: yes** (offload for long clips) | **#1 A/B candidate** |
| **Lip Forcing** (KAIST/AIPARK, arXiv 2606.11180) | github.com/cvlab-kaist/LipForcing | Apache-2.0 ("Copyright 2026 NVIDIA CORPORATION & AFFILIATES") | `apache-2.0`, base Wan2.1-T2V-14B (Apache-2.0) | **UNCLEAR**: licences YES, but buffalo_l + NC-ND training data | **InsightFace buffalo_l (NC, swappable)**. Teacher data includes Hallo3 data (HF `cc-by-nc-nd-4.0`) + VoxCeleb2 + HDTF | Causal autoregressive DiT, 12-frame chunks, rolling KV cache + sink frame: **no window seams**; 2 steps, no CFG; LatentSync mouth mask, 512² | 14B: 15.1 FPS on H100, ~37 GB peak (precomputed text emb), 40.6 GB in paper. **5090: NO out of the box** (no fp8/offload flags). 1.3B (8.8 GB) not released | **#2 A/B candidate** if the data-provenance risk is acceptable and a 48–80 GB GPU is available |
| **X-Dub** public 5B (Kling/Tsinghua, arXiv 2512.25066) | github.com/KlingAIResearch/X-Dub | Apache-2.0 | `apache-2.0` (HF KlingTeam/X-Dub) | **YES** (DWPose weights UNVERIFIED) | DWPose / RTMW cocktail14 (dataset mix UNVERIFIED). Data: 600 h unnamed internet video | Mask-free DiT face edit, 512²; segments with motion-frame carry-over | ~21 GB; 0.5–0.9 FPS. **5090: yes** | README admits "occasional flickering"; superseded by TBDub |
| **MuseTalk 1.5** | github.com/TMElyralab/MuseTalk | MIT | `mit`: "available for any purpose, even commercially" | **NO as shipped**; YES after replacing BiSeNet (+S3FD) — VFHQ training-data provenance remains UNCLEAR | **face-parse BiSeNet 79999_iter (CelebAMask-HQ, NC)** used at inference. DWPose (HF `apache-2.0`; UBody data terms UNVERIFIED). S3FD via FA (BSD-3 code; weights terms not stated). Training data: HDTF + **VFHQ ("non-commercial research purposes" only)** | **Single-step, per-frame** 256² latent inpainting; no temporal module | 30 FPS on V100. **5090: yes** | Not a stability upgrade |
| **VideoReTalking** | github.com/OpenTalker/video-retalking | Apache-2.0 | release v0.0.1 assets; no separate weights licence | **NO** | **PIRenderer-derived D-Net code (CC BY-NC 4.0)**; **BFM 2009 (NC)**; **dlib shape_predictor_68 (NC per dlib-models)**; GPEN-BFR-512 ("commercial issues"); GFPGAN v1.3 (FFHQ-trained); L-Net trained on **LRS2 (BBC, RO)** | GAN pipeline per frame + 3DMM expression neutralisation | 2.67 FPS (Lip Forcing table). **5090: likely** (old CUDA 11.1 stack, needs porting) | Reject |
| **KeySync** (Imperial, arXiv 2505.00497) | github.com/antonibigata/keysync | Apache-2.0 (sub-survey) | HF `apache-2.0`, but **SVD-derived** (Stability AI Community License: ≤$1M revenue) (sub-survey) | **UNCLEAR, lean NO** | SVD licence; data CelebV-HQ + CelebV-Text (**NC**) | Keyframes (12 apart) + interpolation model, overlap-1 averaging, 512² | ~24 GB, ~1.05 FPS on H20 (sub-survey). **5090: yes** | Interesting design; licence blocks it |
| **InfiniteTalk** (MeiGen) | github.com/MeiGen-AI/InfiniteTalk | Apache-2.0 | `apache-2.0`; Wan2.1-I2V-14B (Apache) (sub-survey) | **UNCLEAR, lean OK** | chinese-wav2vec2-base (MIT, but trained on WenetSpeech, NC). FusionX LoRA is NC | **Regenerates the whole frame**: 81-frame chunks, 9 motion frames, sparse keyframes | Slow 14B; 4090 OOMs at 720p (sub-survey). **5090: only quantized/offloaded** | Wrong tool: re-performs head/body; FVD much worse than LatentSync |
| **MultiTalk** | github.com/MeiGen-AI/MultiTalk | Apache-2.0 | `apache-2.0` | same as InfiniteTalk | same | image-to-video only | - | Not applicable (no V2V) |
| **LTX-2.3 Dub-It / "LipDub" IC-LoRA; JUST-DUB-IT** (Lightricks) | github.com/Lightricks/LTX-2 | LTX-2 licence (sub-survey) | `ltx-2-community-license` (gated) | **YES if revenue <$10M**, plus AUP consent and AI-disclosure duties | Gemma-3-12B text encoder (Gemma terms) | Full-frame joint audio+video DiT | ~66 GiB of downloads; FP8/offload options | Different paradigm: generates its own speech from text, so it can't take our TTS track (sub-survey). Not a drop-in |
| **HighSync** (arXiv 2605.16918) | github.com/saeed5959/high_sync | MIT ("Copyright (c) 2026 saeed firouzi") | `mit` | **NO / high risk** | Trained on **VFHQ (NC)** + **CelebV-HQ (NC)** + HDTF. Uses **facebook denoiser dns64 (CC BY-NC 4.0)** at inference. DeepFace yolov8n-face (derronqi repo **GPL-3.0**; ultralytics AGPL) | SD1.5 + ReferenceNet, 12-frame motion module with masked temporal attention (anti-leakage) | ~10 GB. **5090: yes** | Diagnostic only |
| **OmniSync** (Kling, arXiv 2505.21448) | none ("© 2025 OmniSync. All rights reserved") | - | - | API only (Kling) | - | mask-free DiT, noise init from source, DS-CFG | - | Its inference ideas port to LatentSync (§3) |
| **ComplexSync** (arXiv 2609.29225, 2026-09-24) | github.com/Playmate111/ComplexSync (README only, licence null) | none | none | N/A | data SpeakerVid-5M, TalkVid, in-house | 1-step SD1.5 + "pluggable temporal smoother" | 72.7 FPS claimed | Watch list (code "in the coming weeks") |
| StableDub (2509.21887), SayAnything (2502.11515), SyncAnyone, UniSync, EdiDub, FlashLips, EfficientSync, IP-Adapter-is-all-you-need (2605.30230) | no code | - | - | N/A | SayAnything would be SVD-licensed | - | - | No release |
| **SyncEdit** (2603.09084) | github.com/l1346792580123/SyncEdit | **no LICENSE** (404) | - | **NO** (no grant) | HuMo + LTX-2.5 | training-free FlowEdit | - | Exclude |
| **JoyGen** (JD) | github.com/JOY-MM/JoyGen | Apache-2.0 (sub-survey) | - | **NO** | BFM + Deep3DFaceRecon + BiSeNet | per frame, 256² | - | Reject |
| **Diff2Lip** | github.com/soumik-kanad/diff2lip | **CC BY-NC 4.0** | none stated (Google Drive) | **NO** | S3FD; trained on VoxCeleb2 | pixel diffusion, 128², per frame | 19.8 FPS (FlashLips) | Reject |
| **DINet** | github.com/MRzzm/DINet | **none** (API null, LICENSE 404) | none stated | **NO** | **OpenFace 2.0 ("NONCOMMERCIAL RESEARCH USE ONLY")**; DeepSpeech | GAN, per frame, frontal only | - | Reject |
| **IP_LAP** | github.com/Weizhi-Zhong/IP_LAP | Apache-2.0 (template copyright) | none stated; "trained on LRS2" | **NO** | LRS2; FA/S3FD | 2-stage GAN, 128², 5-frame centre window | ~4 FPS | Reject |
| **TalkLip** | github.com/Sxjdwang/TalkLip | **none** | none; LRS2 | **NO** | **AV-HuBERT ("solely for your non-commercial research purposes")** | Wav2Lip-style 96², per frame | 51 FPS | Reject |
| **StyleSync** (Baidu) | github.com/guanjz20/StyleSync(_PyTorch) | none | **weights withheld** | **NO** | LRW + VoxCeleb2 | StyleGAN2, 256², per frame | - | Not available |
| **Wav2Lip + HD/GAN variants** | github.com/Rudrabha/Wav2Lip etc. | Wav2Lip: "can only be used for personal/research/non-commercial purposes" | LRS2 weights: "any form of commercial use is strictly prohibited" | **NO** | LRS2; variants add Real-ESRGAN/GFPGAN/BiSeNet | 96² GAN per frame (+5-frame box smoothing) | fast | Reject (commercial only via Sync Labs) |
| **sync. lipsync-2 / -2-pro / sync-3** | API | - | - | paid API | - | sync-3: whole shot "at once", "no chunk-boundary flickering" (vendor claim) | lipsync-2 ~$2.4–3/min; -2-pro ~$4–5/min; sync-3 ~$6.4–8/min (sub-survey, re-check) | Commercial reference / upper bound |

---

## 2. Per-candidate notes

### 2.1 LatentSync 1.5 / 1.6 (reference; currently in production) — verified by me

**Release and licences**
- Repo: https://github.com/bytedance/LatentSync. Paper: https://arxiv.org/abs/2412.09262 (v1 2024-12-12, v2 2025-03-13).
- Releases:
  - 1.5 on 2025-03-14: "temporal layer" fixed, Chinese data added.
  - 1.6 on 2025-06-11: trained on 512².
  - No newer release. TensorRT-RTX proposal #365/#366 dated 2026-07-03 is community work.
- **Code:** LICENSE = "Apache License Version 2.0, January 2004".
- **Weights:** HF `ByteDance/LatentSync-1.6` and `-1.5` show `License: openrail++`. CreativeML Open RAIL++-M grants a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable" licence. Attachment A use restrictions flow down. "Licensor claims no rights in the Output You generate".

**Sub-models at inference**
- Whisper `tiny.pt`: MIT.
- `stabilityai/sd-vae-ft-mse`: HF `mit`.
- **InsightFace buffalo_l: NC** (see §0).
- The HF `auxiliary/` files are training/eval only: VideoMAE-v2 (TREPA), vgg16, HyperIQA, S3FD, syncnet_v2, i3d.

**Training data**
- VoxCeleb2: "The provided VoxCeleb2 metadata is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License". Video copyright is not addressed.
- HDTF: "available to download under a Creative Commons Attribution 4.0 International License".
- An unnamed Chinese set (added in 1.5).

**Inference behaviour (read from `lipsync_pipeline.py`, `image_processor.py`, `affine_transform.py`, `util.py`, `stage2_512.yaml`)**
- `num_inferences = ceil(len/num_frames)`, 16 frames: **no overlap, no blending**.
- Initial noise is drawn once with frame dimension 1, then `.repeat(...num_frames...)`, so **every frame and window starts from identical noise**.
- Reference = the same frame, unmasked. The paper says: "During training, these frames are randomly selected, while during inference, they are taken from the current frames."
- Affine crop is computed per frame from 3 landmark means (eyebrow centres, nose) out of 106 points. The only temporal smoothing is `bias = p_bias * 0.2 + bias * 0.8` on the translation.
- `read_video` runs `ffmpeg ... -r 25 -crf 18`.
- CFG uses zero audio as the unconditional input. Default is 1.5 (config); `scripts/inference.py` defaults to 1.0.
- DeepCache is optional (`cache_interval=3, cache_branch_id=0`).
- Motion module: `temporal_position_encoding_max_len: 24`, `trepa_loss_weight: 10`.

**Metrics (paper, HDTF)**

| Method | FID | SSIM | Sync | LMD | FVD |
|---|---|---|---|---|---|
| LatentSync | 7.22 | 0.79 | 8.9 | 0.30 | 162.74 |
| MuseTalk | – | – | – | – | 246.75 |
| VideoReTalking | – | – | – | – | 270.56 |
| Diff2Lip | – | – | – | – | 260.45 |
| Wav2Lip | – | – | – | – | 304.35 |

- TREPA ablation: FVD 176.35 → 162.74.

**VRAM and speed**
- README: 8 GB (1.5), 18 GB (1.6).
- Issue #365: PyTorch at 20 steps takes 397.5 s for 9.68 s of output, 15.8 GB peak. "24 frames is the reported maximum temporal position"; 25–31 frames fail.

**Known issues**
- **#277:** 1.6 shows "more flahes around of mouth" than 1.5. The maintainer suggests lowering guidance_scale to about 1.5.
- **#271:** the wrong UNet config (`stage2.yaml` instead of `stage2_512.yaml`) gives "awful results".
- **#78 (open):** blur and jitter every 30 s. #151 links this to Whisper truncation at 30 s.
- **#67:** "Seems to be from detection box and landmark jitter". A user's Kalman filter helped. GFPGAN post-processing makes the speaker "speak through teeth".
- #315: white mouth corners. #334: jitter after retraining.

### 2.2 TBDub (Taobao & Tmall Group, Alibaba) — sub-survey, key facts re-verified by me

**Release and licences**
- Paper: https://arxiv.org/abs/2609.06144, "TBDub: Production-Oriented Visual Dubbing", v1 2026-09-05. Its stated goal is to fix X-Dub's "temporal and motion stability", identity and speed.
- Repo: https://github.com/TaoLiveAIGC/TBDub (V1.1 2026-09-12, per sub-survey).
- **Code:** LICENSE = "Apache License Version 2.0, January 2004" (template copyright line). README: "The code and released model weights are available under the Apache License 2.0." and "Third-party dependencies and checkpoints remain subject to their respective licenses."
- **Weights:** HF `TaoLiveAIGC/TBDub` = `apache-2.0`. Intended use is "research and authorized applications involving visual dubbing". The user is responsible for consent.

**Sub-models**

| Sub-model | Licence |
|---|---|
| `Wan2.2_VAE.safetensors` (from KlingTeam/X-Dub) | Apache-2.0 |
| `facebook/hubert-large-ll60k` | `apache-2.0` (sub-survey) |
| MediaPipe `face_landmarker.task` | Google model-card PDFs (Face Mesh V2, BlazeFace) contain a link to apache.org/licenses/LICENSE-2.0; text layer not extractable, so treat as Apache-2.0, low risk |

- No InsightFace, BiSeNet or restorer.
- `--cropped-input` skips detection and paste-back. We could feed our own **smoothed** 512² crops and keep our own mouth-ellipse composite, which also removes the MediaPipe dependency.

**Training data:** studio green-screen + livestream footage (3:1), proprietary (sub-survey).

**Architecture**
- X-Dub lineage: Wan-style DiT, mask-free editing of a 512² aligned face.
- Fixed 77-frame clips with "cross-clip motion conditioning".
- Teacher: 30 steps. Student: 2 steps via DMD/DMD2.
- Paste-back with colour-statistics correction.

**Speed and VRAM**
- Student 7.13 FPS on H20, against LatentSync 1.31 FPS.
- Default 27.93 GiB GPU-resident; 15.01 GiB with `--cpu-offload` (student).
- "final VAE decoding spans the assembled sequence", so memory grows with length.
- **5090 (32 GB): fits.** Use `--cpu-offload` for long segments. Blackwell needs a cu128+ torch build; TBDub's validated environment is Python 3.10 (torch build not checked).

**Metrics (38 TalkVid clips; sub-survey)**

| Model | MOS lip-sync | MOS identity | MOS visual | Sync-C | CSIM |
|---|---|---|---|---|---|
| TBDub Student | 3.85 | 3.72 | 3.80 | 7.16 | 0.91 |
| LatentSync | 1.91 | 2.23 | 1.87 | 7.76 | 0.97 |
| X-Dub | 3.57 | 2.83 | 2.88 | – | – |
| KeySync | 2.75 | 2.40 | 2.33 | – | – |
| LipForcing | 2.18 | 2.88 | 2.52 | – | – |

- **No FVD or flicker metric.**

**Known issues**
- Very new; only one GitHub issue.
- Single person only. A missed-detection gap longer than 0.4 s raises an error.
- 25 fps internally (sub-survey). The README does not state fps; output for 15.04 s was 376 frames.

**Commercial: YES.** Mask-free editing means the whole face crop is regenerated, so we should keep our mouth-only ellipse paste-back.

### 2.3 Lip Forcing (KAIST AI + AIPARK) — verified by me

**Release and licences**
- Paper: https://arxiv.org/abs/2606.11180 (2026-06-09).
- Repo: https://github.com/cvlab-kaist/LipForcing (released 2026-07-07). The 1.3B student is "coming soon"; issues #6 and #8 are still asking for it.
- **Code:** LICENSE = "Apache License Version 2.0, January 2004", "Copyright 2026 NVIDIA CORPORATION & AFFILIATES" (derived from NVIDIA FastGen).
- `licenses/README.md` lists: Wan Apache-2.0, OmniAvatar Apache-2.0, LatentSync Apache-2.0 (face detection/alignment/compositing), SyncNet MIT, TAEHV MIT, detectron2 Apache-2.0, diffusers Apache-2.0.
- **Weights:** HF `JinhyukJang/lipforcing` = `apache-2.0`, finetune of `Wan-AI/Wan2.1-T2V-14B`. Wan's card says `apache-2.0`: "The models in this repository are licensed under the Apache 2.0 License."

**Sub-models**

| Sub-model | Licence |
|---|---|
| Wan VAE + UMT5-XXL | Apache-2.0 |
| `facebook/wav2vec2-base-960h` | `apache-2.0`; LibriSpeech |
| TAEHV `taew2_1.pth` | MIT |
| LatentSync `mask.png` | Apache-2.0 |
| **InsightFace buffalo_l, "auto-downloaded on first use"** | **NC; swap required** |

- `--skip_preprocessing` (batch script) accepts "pre-aligned 512×512 crops", so detection can be done with our own detector.

**Training data**
- The paper lists VoxCeleb2 + Hallo3 + HDTF.
- `fudan-generative-ai/hallo3_training_data` is `cc-by-nc-nd-4.0`. That this is the exact set used is inferred from the paper's wording, not confirmed.
- **Provenance risk: UNCLEAR.**

**Architecture**
- OmniAvatar-LS 14B V2V teacher (50 steps) → causal student (Self-Forcing DMD + SyncNet reward). 2 steps, no CFG.
- Chunks of 3 latent frames (12 px frames). KV cache: sink frame + 6 recent frames, dynamic RoPE.
- Inputs: lip mask, reference frame, masked reference video, and "a reference frame sequence from outside the input window", which reduces leakage.
- TREPA used in training. 512² crop, paste-back. Output is `floor(audio_seconds × 25)` frames, i.e. 25 fps.

**Speed and VRAM**
- 14B: 15.11 FPS on H100. Paper: 40.63 GB allocated. README: "~37 GB peak", "~50 GB" with runtime text encoding, "48 GB cards work with precomputed embeddings".
- No fp8, offload or quantisation flags exist (README checked).
- **RTX 5090 (32 GB): NO out of the box.** 14B bf16 weights alone are about 28 GB. Running it would need our own fp8 weight-only quantisation or block offload, which is untested and a quality risk on a 2-step distilled model.
- The 1.3B student (8.8 GB) would fit, but is unreleased.

**Metrics (HDTF, 33 clips)**

| Method | Steps | FPS | Sync-C | Sync-D | CSIM | FID | FVD |
|---|---|---|---|---|---|---|---|
| Lip Forcing 14B | 2 | 15.11 | 7.59 | 7.23 | 0.949 | 7.01 | **107.88** |
| LatentSync | 20 | 3.23 | 8.10 | 6.51 | 0.967 | 6.90 | 117.91 |
| MuseTalk | – | – | – | – | – | – | 127.44 |
| X-Dub | – | – | – | – | – | – | 183.99 |

- User-study MOS (sync / quality / ID / naturalness): Lip Forcing 14B 4.38 / 4.33 / 4.46 / 4.32; LatentSync 3.96 / 3.54 / 3.82 / 3.53.
- In TBDub's MOS study, LipForcing ranked below TBDub and X-Dub. The two studies conflict, so our own A/B decides.
- No flicker metric. The paper claims robustness to error accumulation on long videos.

**Known issues:** #7 "Dark spot artifacts appear on facial skin"; #5 Chinese weaker than LatentSync.

**Commercial: UNCLEAR.** Licences are YES, but buffalo_l must be swapped and the NC-ND training data is a provenance risk.

### 2.4 X-Dub (Kling + Tsinghua) — sub-survey (two independent checks agree)

- Paper: arXiv 2512.25066 (ICML 2026). Repo: https://github.com/KlingAIResearch/X-Dub (code and weights 2026-03-19).
- Licences: code LICENSE Apache-2.0; HF `KlingTeam/X-Dub` `apache-2.0`. The public model is Wan2.2-TI2V-5B based (Apache-2.0); the paper's internal 1B model is not released.
- Sub-models: umT5-XXL (Apache), Whisper large-v2 (MIT), wav2vec2-base-960h (Apache), DWPose YOLOX + RTMW cocktail14 (training-set terms UNVERIFIED).
- Architecture: mask-free DiT editing of the 512² crop, with segments carrying motion frames over.
- Speed and VRAM: about 21 GB; 0.5–0.9 FPS.
- README admits "occasional flickering" and "slightly weaker temporal stability" for the public model, identity/colour drift, and about 2% noisy frames. The online crop "may introduce visible jitter".
- **Commercial: YES.** Not preferable to TBDub.

### 2.5 MuseTalk 1.5 (Tencent Music Lyra Lab) — verified by me

**Release and licences**
- Repo: https://github.com/TMElyralab/MuseTalk. v1.5: "[03/28/2025]". Training code: "[04/05/2025]".
- Paper: https://arxiv.org/abs/2410.10122 (v1 2024-10-14, v3 2025-03-26).
- **Code:** LICENSE = "MIT License / Copyright (c) 2024 Tencent Music Entertainment Group".
- **Weights:** HF `TMElyralab/MuseTalk` = `mit`. README:
  - "The code of MuseTalk is released under the MIT License. There is no limitation for both academic and commercial usage."
  - "The trained model are available for any purpose, even commercially."
  - "Other open-source models used must comply with their license, such as `whisper`, `ft-mse-vae`, `dwpose`, `S3FD`, etc.."
  - "The testdata are collected from internet, which are available for non-commercial research purposes only."

**Sub-models (from `download_weights.sh` + code)**

| Sub-model | Licence / status |
|---|---|
| `stabilityai/sd-vae-ft-mse` | `mit` |
| `openai/whisper-tiny` | HF `apache-2.0` |
| DWPose `yzd-v/DWPose` `dw-ll_ucoco_384.pth` | HF `apache-2.0`; IDEA-Research/DWPose repo Apache-2.0. Trained on COCO-WholeBody + UBody; UBody terms UNVERIFIED (OSX repo shows only MIT for code) |
| S3FD via local `face_detection` | `https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth`; face-alignment code "BSD 3-Clause, Copyright (c) 2017, Adrian Bulat"; weight terms not stated (UNCLEAR, low risk) |
| **face-parse-bisent `79999_iter.pth`** (Google Drive) + torchvision `resnet18-5c106cde.pth` | Repo zllrunning/face-parsing.PyTorch is MIT, but it is trained on CelebAMask-HQ: "available for non-commercial research purposes only" and "The use of this software is RESTRICTED to non-commercial research and educational purposes." **Used at inference** in `blending.py` (`face_seg(...)` → `fp(image, mode=mode)`): **NC trap** |
| `latentsync_syncnet.pt` | training only |

**Training data:** HDTF + VFHQ (about 24 h after filtering). VFHQ: "only available to download for non-commercial research purposes". **Provenance risk.**

**Architecture**
- Single-step UNet in VAE latent space. Input = masked lower half + full-face reference; Whisper-tiny cross-attention.
- v1.5 adds perceptual, GAN and sync losses, and spatio-temporal sampling (Informative Frame Sampling for the reference in training).
- At inference "the reference image is extracted from the current frame", the same leakage pattern as LatentSync.
- **Per-frame**; no temporal module. 256² face.
- `preprocessing.py`: landmarks via DWPose, boxes via S3FD, `bbox_shift` offsets the box. **No smoothing across frames.**

**Speed and VRAM:** "30fps+ on an NVIDIA Tesla V100"; fp16 runs on 4 GB.

**Metrics**
- No FVD in its own paper. Paper Table 1 (HDTF): FID 6.52 vs LatentSync 8.41; LSE-C 6.53 vs 7.90.
- Its own user study: best visual quality (4.26), LatentSync better on lip-sync (4.07 vs 3.77).
- In LatentSync's paper: FVD 246.75 vs 162.74. In Lip Forcing's table: FVD 127.44 vs 117.91.

**More sub-survey detail**
- The LICENSE file appends third-party sections: MIT for sd-vae, whisper and face-parsing; Apache-2.0 for DWPose; BSD-3 for face-alignment; S3FD with no licence, only a link.
- S3FD is trained on WIDER FACE, which the HF dataset card lists as CC BY-NC-ND 4.0 (second-hand, so UNVERIFIED).
- v1.5 inference fixes `bbox_shift=0`, `extra_margin=10` and `parsing_mode='jaw'`, and ping-pongs the sequence.
- **Derivatives:** none adds a temporal module.
  - lipku/LiveTalking (Apache-2.0, real-time wrapper).
  - ComfyUI-MuseTalk variants (chaojie: MIT; kijai: no licence).
  - sieve-community/sievesync (MIT, but bundles CodeFormer (S-Lab NC) and LivePortrait with InsightFace (NC)).
  - MuseTalkPlus (no licence).
- SyncAnyone also reports MuseTalk FVD 245.8 vs LatentSync 228.3 on HDTF.

**Known jitter issues**
- README: "There exists some jitter as the current pipeline adopts single-frame generation."
- #426 (Oct 2026, RTX 4090): "mouth jitters regardless of resolution" (sub-survey).
- Paper limitations: "occasional jitter is observed due to the single-frame generation process".
- #190 "mouth keeps trembling": the contributor says the released model "was not trained with a sync loss, so no temporal information is used".
- #388 (2025-11) jitter in Gradio; #11, #98, #32 (mouth moving in silence); PR #423 (teeth-whitening fix, open).

**Commercial: NO as shipped** because of the BiSeNet parser. The parser can be removed, since our pipeline does its own ellipse paste-back, but VFHQ provenance remains. **Not a stability upgrade.**

### 2.6 VideoReTalking (OpenTalker / Tencent AI Lab, SIGGRAPH Asia 2022) — verified by me

**Release and licences**
- Repo: https://github.com/OpenTalker/video-retalking. Paper: https://arxiv.org/abs/2211.14758 (2022-11-27).
- Only release: v0.0.1 (March 2023).
- **Code:** LICENSE = "Apache License Version 2.0, January 2004" (template copyright). README disclaimer: "This is not an official product of Tencent."
- **Weights:** release assets `30_net_gen.pth`, `BFM.zip`, `DNet.pt`, `ENet.pth`, `expression.mat`, `face3d_pretrain_epoch_20.pth`, `GFPGANv1.3.pth`, `GPEN-BFR-512.pth`, `LNet.pth`, `ParseNet-latest.pth` (+2 more not displayed). No separate weights licence is stated.

**Sub-models**

| Sub-model | Licence / status |
|---|---|
| **Basel Face Model 2009** | "The license granted is for internal, non-commercial research, evaluation or testing purposes only." **NC** |
| Deep3DFaceRecon `face3d_pretrain_epoch_20.pth` | Repo MIT. Requires BFM09 registration; trained on CelebA, LFW, 300WLP, IJB-A, LS3D-W, FFHQ. Terms UNVERIFIED, NC-encumbered via BFM |
| **dlib `shape_predictor_68_face_landmarks.dat`** (in `Croper`) | dlib-models: "The license for this dataset excludes commercial use" … "the trained model therefore can't be used in a commercial product." **NC** |
| GPEN-BFR-512 (+ RetinaFace-R50, ParseNet) | GPEN README: "(not our best model due to commercial issues)"; no licence for the weights. **UNVERIFIED/risky**. ParseNet is face parsing (training set not verified) |
| GFPGAN v1.3 | Code Apache-2.0; weights FFHQ-trained (FFHQ CC BY-NC-SA trap). **UNCLEAR** |
| GANimation `30_net_gen.pth` (expression edit) | UNVERIFIED |
| L-Net | Trained on **LRS2** ("lip-reading dataset with large-scale 160p videos from BBC programs"; LRS2 is research-only). D-Net trained on VoxCeleb |

**Additional code-licence trap (sub-survey):** D-Net is PIRenderer-based, and PIRender's LICENSE.md is "Attribution-NonCommercial 4.0 International". The repo also derives from Wav2Lip (NC).

**Architecture**
- GAN pipeline: D-Net expression neutralisation → L-Net lip-sync inpainting at **96²** → E-Net enhancement to 384² → GPEN/GFPGAN → Laplacian/Poisson blending.
- **Per frame.** The crop is fixed from the first frame (#194: head motion breaks it). `inference.py` has no smoothing; `face_detect` in utils averages boxes over T=5 (sub-survey).
- The paper admits "small temporal jitter and flashing" (sub-survey).

**Speed and metrics**
- Speed: 2.67 FPS (Lip Forcing table).
- FVD 270.56 vs LatentSync 162.74 (LatentSync paper); 306.63 vs 117.91 (Lip Forcing paper).
- "DNet cannot handle extreme poses" (README).

**Commercial: NO** (BFM, dlib-68, LRS2, GPEN). **Reject.**

### 2.7 KeySync — sub-survey
- Repo https://github.com/antonibigata/keysync; Apache-2.0 `LICENCE` file. HF `toninio19/keysync` `apache-2.0`.
- **U-Net and VAE are initialised from Stable Video Diffusion**: "STABILITY AI COMMUNITY LICENSE AGREEMENT", which terminates above "USD $1,000,000 in annual revenue" and covers fine-tunes.
- Training data: HDTF + CelebV-HQ + CelebV-Text, both CelebV sets "non-commercial research purposes only". Face detection via face_alignment SFD.
- Architecture: 14 keyframes (12 apart) + interpolation model; 512², 10 EDM steps.
- Cross-sync results vs LatentSync ≤1.5: FVD 206 vs 362; LipLeak 0.16 vs 0.33.
- About 24 GB, about 1.05 FPS on H20.
- Issues: #35 identity and teeth inconsistency; #29 colour distortion.
- **Commercial: UNCLEAR, lean NO.**

### 2.8 InfiniteTalk / MultiTalk (MeiGen-AI) — sub-survey
- **InfiniteTalk:**
  - arXiv 2508.14033. Apache-2.0 code and weights; Wan2.1-I2V-14B-480P (Apache-2.0).
  - Audio: chinese-wav2vec2-base, HF `mit`, but "Pretrained on 10k hours WenetSpeech L subset", and WenetSpeech is non-commercial: **provenance risk**.
  - **Full-frame regeneration**: 81-frame chunks, 9 motion frames, sparse keyframes. Head and body are re-synthesised. "Mimics the original video's camera movement, though not identically". SDEdit mode "introduces color shift".
  - FVD on HDTF: 131.65 vs LatentSync 48.45 (their table). CSIM 0.775 vs 0.916.
  - FusionX LoRA is "research and non-commercial use only".
  - Verdict: incompatible with mouth-only segment edits.
- **MultiTalk:** image-to-video only. **Not applicable.**

### 2.9 LTX-2.3 Dub-It / "LipDub" IC-LoRA, JUST-DUB-IT (Lightricks) — sub-survey
- HF `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` / `-DubIt`: `ltx-2-community-license`, gated. Entities "with annual revenues of at least $10,000,000" need a paid licence.
- Attachment A bans impersonation "without their consent" and requires AI-generated disclosure.
- JUST-DUB-IT (arXiv 2601.22143).
- Mask-free, full frame. Generates new speech from a text prompt with the voice cloned from the source. **It has no driving-audio input** (sub-survey; one sub-survey marks this UNVERIFIED), so it would replace our TTS step, not our lip-sync step.
- "LipDub AI" (lipdub.ai) is a separate closed SaaS.

### 2.10 HighSync — verified by me
- Repo https://github.com/saeed5959/high_sync. LICENSE "MIT License / Copyright (c) 2026 saeed firouzi". HF `saeed-5959/high_sync` `mit`.
- Base `lambdalabs/sd-image-variations-diffusers` (`license: apache-2.0`), `sd-vae-ft-mse` (MIT), Whisper tiny (MIT).
- `inference.py` calls `denoiser.pretrained.dns64()`. facebookresearch/denoiser is "Attribution-NonCommercial 4.0 International": **NC**.
- Face detection: DeepFace `detector_backend='yolov8'`; yolov8n-face from derronqi/yolov8-face (**GPL-3.0**), loaded through ultralytics (AGPL-3.0).
- Training data:
  - VFHQ: "only available to download for non-commercial research purposes".
  - CelebV-HQ: "available for non-commercial research purposes only" and forbids exploiting "any portion of derived data" commercially.
  - HDTF.
- Paper value: it diagnoses LatentSync-style **leakage**. Per-frame crop height tracks mouth opening; it fixes this with the maximum box height per clip, masked temporal attention and PNG masks.
- Metrics: FID 7.36 vs LatentSync 8.98 on HDTF; user-study quality 4.28 vs 3.95; no FVD.
- About 10 GB, so a 5090 is fine.
- **Commercial: NO.** Diagnostic only.

### 2.11 Others
- **OmniSync** (arXiv 2505.21448, NeurIPS 2025): no code ("© 2025 OmniSync. All rights reserved."); Kling API only. FVD 199.6 vs LatentSync 216.9. Two inference tricks port to LatentSync (§3).
- **ComplexSync** (arXiv 2609.29225, 2026-09-24, WACV 2027): the repo has only a README ("inference code, model weights … in the coming weeks"); GitHub licence null. 1-step SD1.5 with a pluggable temporal smoother, 72.7 FPS claimed. Watch list.
- **StableDub** (2509.21887) and **SayAnything** (2502.11515): no code. SayAnything is SVD-based.
- **SyncEdit** (2603.09084): no LICENSE file, so excluded.
- **JoyGen**: BFM + Deep3DFaceRecon + BiSeNet: **NO**.
- **LatentSync 2.x: does not exist.** HF ByteDance has only `LatentSync`, `-1.5`, `-1.6`.
- **Image-to-video only (excluded):** Wan2.2-S2V, HuMo, OmniHuman, StableAvatar, Hallo3/4, Ovi, HunyuanVideo-Avatar (Tencent licence excludes the EU, UK and South Korea), EchoMimicV3, FantasyTalking(2), LongCat-Video-Avatar, Sonic, MoCha, OmniAvatar (its V2V adapter exists only as Lip Forcing's teacher).

### 2.12 Older GAN/diffusion baselines — sub-survey (licence quotes from LICENSE/README)

All of these generate per frame at 96–256 px. Their FVD is worse than LatentSync in every independent table:
- LatentSync paper: Wav2Lip 304.35, Diff2Lip 260.45.
- Lip Forcing: Wav2Lip 384.82, Diff2Lip 285.69.
- SyncAnyone (arXiv 2512.21736): IP-LAP 338.2 vs LatentSync 228.3.

FlashLips (arXiv 2512.20033) is the one partial exception. In cross-audio FVD it has IP-LAP at 43.13 vs LatentSync 46.08, but in reconstruction LatentSync wins (36.47 vs 39.89).

**None can be a stability upgrade, and none is commercially licensed.**

| Model | Links / dates | Licence | Data / sub-models | Architecture | Known issues |
|---|---|---|---|---|---|
| **Diff2Lip** | github.com/soumik-kanad/diff2lip (last commit 2024-08-16); WACV 2024, arXiv 2308.09716 | LICENSE.md "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)" | VoxCeleb2; S3FD | Pixel-space diffusion, lower half of a 128² face, 25 DDIM steps, per frame | Lip artifacts (#38) |
| **DINet** | github.com/MRzzm/DINet (last push 2023-09-25); AAAI 2023, arXiv 2303.03988 | **No licence** (GitHub API null) | Weights unlicensed (Google Drive). Needs **OpenFace 2.0**: "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY". DeepSpeech audio; HDTF | Per frame; 256² mouth region inside a 416×320 face; frontal poses only | Jitter / colour (#107); mouth moves in silence (#87); MuseTalk table LSE-C 5.96 |
| **IP_LAP** | github.com/Weizhi-Zhong/IP_LAP (last push 2024-01-06); CVPR 2023, arXiv 2305.08293 | Apache-2.0, copyright placeholder unfilled | Weights unlicensed; "Our models are trained on LRS2" (BBC, RO). MediaPipe + FA | 128² output; landmark transformer + renderer; 5-frame window; `windows_T=1`, i.e. no smoothing | Lip shaking (#6, #32, #33, #53, #57) |
| **TalkLip** | github.com/Sxjdwang/TalkLip (last push 2023-11-01); CVPR 2023, arXiv 2303.17480 | No licence | LRS2 weights. **AV-HuBERT at inference**: "solely for your non-commercial research purposes" | 96², per frame, no box smoothing | — |
| **StyleSync** | arXiv 2305.05445, CVPR 2023; github.com/guanjz20/StyleSync and _PyTorch | No LICENSE | Weights withheld: "The decision to suspend the release of the model weights has been made by our team". Data: LRW + VoxCeleb2 | StyleGAN2, 256², per frame | — |

**Wav2Lip family**
- **Wav2Lip** (github.com/Rudrabha/Wav2Lip; arXiv 2008.10010):
  - "This repository can only be used for personal/research/non-commercial purposes."
  - "As the models are trained on the LRS2 dataset, any form of commercial use is strictly prohibited."
  - "For commercial requests please contact us directly at rudrabha@synclabs.so or prajwal@synclabs.so."
  - A "HD model trained on a dataset allowing commercial usage" exists but is not open.
  - Architecture: 96² GAN with T=5 box smoothing.
  - Issues: lip shaking (#476, #568, #226).
- **Variants** (all inherit the NC Wav2Lip weights):
  - Wav2Lip-HD: MIT code; adds Real-ESRGAN and BiSeNet/CelebAMask-HQ.
  - Easy-Wav2Lip: no LICENSE; archived 2025-01-28.
  - wav2lip_288x288: MIT label; released weights UNVERIFIED; code derived from the NC repo.
  - Wav2Lip-GFPGAN: no licence; GFPGAN adds FFHQ/NVIDIA-op terms.
- **Commercial: NO** for all public weights.

### 2.13 Commercial APIs (reference only; sub-survey, prices to re-check)

| Vendor / model | Price | Notes |
|---|---|---|
| sync.so lipsync-2 | ~$2.4–3/min | 512² face |
| sync.so lipsync-2-pro | ~$4–5/min | 512² face + diffusion SR |
| sync.so sync-3 (2026-04-06) | ~$6.4–8/min | "no chunk-boundary flickering", 4K (vendor claim) |
| fal.ai LatentSync | $0.20 per request ≤40 s, then $0.005/s | |
| fal.ai sync v2 | $3/min | |
| fal.ai sync v2 pro | $5/min | |
| HeyGen video translate Precision | ~$4/min | rate in flux |
| Kling LipSync | – | 2–10 s clips only |
| ElevenLabs Dubbing | – | does **not** lip-sync natively |

---

## 3. Task 2: stabilising LatentSync (TF = training-free)

Ordered by expected payoff against "constant twitching". Every item needs an A/B on our footage. I found **no public LatentSync fork or ComfyUI wrapper implementing items 1–6**. I checked the ShmuelRonen and GeekyGhost wrappers, cog forks, and issue searches for jitter, flicker, shaking and 抖动.

1. **Smooth the alignment track (TF, cheap, highest payoff).**
   - LatentSync builds the crop affine per frame from 3 landmark means of InsightFace 2d106, with only `bias = p_bias*0.2 + bias*0.8` on translation. Landmark noise becomes crop jitter, so the inverse-warped mouth "swims" against the untouched background.
   - Fix: smooth landmarks or affine parameters **offline over the whole segment** using a non-causal Savitzky-Golay, One-Euro or Kalman RTS filter. Use the *same* smoothed matrices for crop and paste-back.
   - HighSync also shows per-frame box height leaking mouth opening into upper-face position; it fixes this with a constant box height per clip.
   - Evidence: issue #67 (Kalman helped on head motion).
   - Do this together with the detector swap to MediaPipe or similar.
2. **Overlapping windows with per-step latent fusion (TF).**
   - Today windows are hard-cut every 16 frames.
   - Fix: denoise windows with stride 8–12 and fuse overlapping latents at **every** denoising step.
     - MimicMotion "progressive latent fusion": position-weighted, λ=1/(C+1); the paper states it is training-free.
     - Or AnimateDiff-Evolved context options: `context_overlap` default 4, `pyramid` fuse.
   - Cost: 1.3–2× compute; about 30 lines in `lipsync_pipeline.py`.
3. **Longer windows (TF, small risk).** `temporal_position_encoding_max_len: 24`; issue #365 reports 22 and 24 frames work and 25–31 fail. `num_frames=24` gives 1.5× fewer seams but is outside the 16-frame training distribution.
4. **Reference-frame choice (TF).**
   - Inference uses the same unmasked frame as reference (paper: "taken from the current frames"), while training used random frames. The original Russian mouth therefore leaks in and fights the English audio. This is the leakage KeySync, HighSync and Lip Forcing all design against.
   - Fix: pick one reference per window or segment from a *different* time, choosing a closed or neutral mouth, a head pose near the window median, and a sharp frame.
   - Risk: pose mismatch on head turns, so choose per window by pose.
5. **Guidance (TF).**
   - README: higher guidance_scale "may cause the video distortion or jitter". The maintainer's fix for flashes is about 1.5 (#277).
   - Try 1.0–1.5, or OmniSync-style temporal decay ω(t)=ω_peak·(t/T)^1.5: strong early for lip shape, weak late. Optionally use a spatial Gaussian around the mouth.
6. **Source-anchored initial noise (TF, leakage risk).**
   - OmniSync starts from `(1-τ)·source+τ·ε`, τ=0.92, and reports "evident temporal inconsistencies" without it.
   - For LatentSync this means `scheduler.add_noise(VAE(original crop), noise, t_start)` at strength about 0.85–0.95.
   - Noise is already shared: LatentSync uses one identical noise frame for all frames and windows. FreeNoise-style rescheduled noise (ICLR 2024, TF) or FreeInit low-frequency re-init (TF) are open experiments.
7. **Frame rate (check first).**
   - LatentSync re-encodes to 25 fps (`-r 25`), and every candidate above also runs at 25 fps.
   - If our 30 fps footage goes 30→25→30 by frame duplication, the mouth repeats a frame about every 5 frames. That is a 6 Hz stutter that reads exactly as "twitching".
   - Fix: map generated 25 fps mouth frames to 30 fps with motion-compensated interpolation in the aligned crop, or drive the model at 30 fps with resampled audio features (untested).
8. **Whisper 30 s boundary (TF patch).** Issues #78 and #151 report blur and jitter at multiples of 30 s. Apply the overlapping `transcribe()` fix from #151, or keep calls under 30 s.
9. **Mask choice (TF, out of distribution).**
   - Measured from the PNGs (256²):

     | Mask | Region | Area |
     |---|---|---|
     | `mask.png` (default in all shipped configs) | lower face from row 95, nose notch | 40.7% |
     | `mask2` | straight cut at row 128 | 33.8% |
     | `mask3` | small mouth + chin semicircle | 19.3% |
     | `mask4` | from row 60 | 58.4% |

   - A smaller generated area means less of the face can jitter, but only `mask.png` was trained.
   - Our paste-back ellipse should be defined in the **smoothed aligned-crop space**.
10. **Checkpoint and config.** A/B 1.5 at 256 against 1.6 at 512 (#277 reports more mouth flashes on 1.6). Make sure 1.6 runs with `stage2_512.yaml` (#271).
11. **DeepCache:** no documented temporal side effect found (UNVERIFIED). A/B with it on and off.
12. **Segment edges:** crossfade generated↔original over 3–5 frames at speech-segment boundaries.
13. **Post-hoc deflicker on the aligned mouth crop (TF, last resort).** Options: FastBlend (Apache-2.0, "model-free"), All-In-One-Deflicker (Apache-2.0, per-video optimisation, slow), or a flow-guided EMA inside the mask. Keep strength low, because it dulls p/b/m closures.
14. **Training-based (not TF).** Fine-tune the motion module with TREPA (weight 10) on in-domain footage, using a different-frame reference and constant crop height (HighSync recipe). The 1.5 changelog says stage-2 training fits in 20 GB.

---

## 4. Ranked shortlist for an A/B against current LatentSync

All arms should use the **same** front end:
- a commercial detector/landmarker (MediaPipe Face Landmarker; model-card PDF links Apache-2.0) instead of buffalo_l;
- offline-smoothed crops;
- our mouth-ellipse paste-back;
- consistent 25↔30 fps handling.

1. **LatentSync 1.6 with §3 fixes 1, 2, 4, 5 and 7 (plus 3 and 10 as variants).**
   - Cheapest; keeps a known-good sync level (best Sync-C in most tables).
   - Directly targets the mechanisms that produce periodic twitch: window seams, crop jitter, reference leakage and frame-rate judder.
   - Commercial **YES** once buffalo_l is gone (Apache code, openrail++ weights).
   - Fits a 5090.
2. **TBDub Student (2-step), driven with `--cropped-input` from our smoothed crops.**
   - Cleanest licence chain found: Apache-2.0 code and weights, Apache Wan2.2 VAE and HuBERT, MediaPipe only if we use its own crop, proprietary training data.
   - Explicitly targets X-Dub's temporal and motion stability; best MOS in its paper (LatentSync last).
   - Fast (7 FPS on H20); 15–28 GiB, so it fits a 5090.
   - Caveats: no FVD or flicker metric; mask-free whole-face editing, so keep our mouth ellipse; very new.
3. **Lip Forcing 14B: only if a 48–80 GB GPU is available and the provenance risk is accepted.**
   - Architecturally the most seam-free: autoregressive with a KV-cache sink, LatentSync-style mouth inpainting.
   - Best FVD vs LatentSync (107.9 vs 117.9) and better MOS in its own paper; ranked below TBDub in TBDub's study.
   - Blockers: swap buffalo_l (use `--skip_preprocessing` + our crops); teacher data includes Hallo3 data (CC BY-NC-ND); **does not fit a 32 GB RTX 5090 without un-supported fp8 or offload work**. Revisit when the 1.3B student (8.8 GB) ships.

**Diagnostic upper bounds (do not ship):**
- **sync.so sync-3 / lipsync-2-pro**: commercial API reference for "how stable can it get". Paid; it is a service, not a model.
- **KeySync**: shows whether keyframe + interpolation removes the twitch. SVD Community License + NC data.
- **HighSync**: shows whether leakage removal alone fixes it. NC data and NC denoiser.

**Do not pursue:**
- MuseTalk 1.5 (per-frame jitter, BiSeNet NC);
- VideoReTalking (BFM, dlib-68, LRS2);
- Wav2Lip family (research-only weights);
- InfiniteTalk / MultiTalk (full-frame re-performance / image-to-video only);
- LTX Dub-It (generates its own speech);
- X-Dub (superseded by TBDub).
