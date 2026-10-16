# Track A licence audit — lip-sync generators and the face stack they lean on (draft, 2026-10-16)

Desk research only: no candidate was installed or run. Full evidence with quoted LICENSE / model-card lines:
`research/candidates.md` (generators) and `research/deps_licenses.md` (detectors, landmarkers, parsers, embeddings).
Re-verified by me from the primary pages on 2026-10-16: LatentSync (code + HF card), TBDub (GitHub README + HF card), MuseTalk (README),
the InsightFace buffalo_l files in this checkout and in `infra/serverless/handler.py`. Not legal advice: every NO / UNCLEAR needs counsel.

Legend: **YES** = code and weights licences both allow commercial use; **NO** = non-commercial / research-only / no licence;
**UNCLEAR** = permissive file licence but non-commercial training data or missing text. `RESEARCH_ONLY — cannot ship` marks anything
that may run here only as a diagnostic upper bound.


## 0. CLEAN-ONLY status (user decision 2026-10-16 ~21:00 UTC) — clean face stack, quarantine, eval-only tools

Rule: only components with verdict YES (code + weights + every pretrained sub-model) may run in a product-path run.

| component of the clean0 video stage (`--face-stack mediapipe --codeformer off`) | licence | verdict |
|---|---|---|
| MediaPipe 0.10.21 (code, `/venv/mediapipe`, subprocess) + `face_landmarker.task` float16 v1 (BlazeFace short-range + Face Mesh V2 + blendshapes; sha256 64184e22…c9ff) | Apache-2.0 code; model cards "LICENSED UNDER Apache License, Version 2.0"; trained on consented Google-collected images | **YES** — replaces the router (RetinaFace R50), LatentSync's crop / verify detector and the mouth-mask / smoothing landmarks (InsightFace buffalo_l) |
| venv deps of MediaPipe: opencv-contrib-python 4.11, jax / jaxlib, protobuf 4.25, numpy 1.26, matplotlib, sounddevice | Apache-2.0 / Apache-2.0 / BSD-3 / BSD / PSF-style / MIT | YES |
| LatentSync 1.6 UNet (`openrail++`), Whisper tiny (MIT), sd-vae-ft-mse (MIT), DeepCache / kornia (Apache-2.0), Silero VAD (MIT) | see §1 | YES (LatentSync: Open RAIL++-M use restrictions must flow into our ToS — false information to harm, defamation, harassment, impersonation without consent, …) |
| the 106-point index table + anchor correction in `scripts/pilot/clean_face_stack.py` (`MAP106`, `clean_face_stack_calib.json`) | our code; derived offline from cached evaluation landmarks of the baseline masters (no InsightFace model runs in the product path) | YES |
| CodeFormer / ParseNet / RetinaFace R50 / InsightFace buffalo_l | NC | **quarantined** in `/workspace/dub/quarantine_nc/` (README with sha256 + reason); `guard_nc()` makes their loaders raise in clean runs; `download_models.sh` fetches them only with `DUB_ALLOW_NC=1` |

**Training data of the generators (coordinator request 2026-10-16, for the generator decision):**
* **LatentSync 1.5 / 1.6 was TRAINED on VoxCeleb2 + HDTF** (paper arXiv 2412.09262: "a mixture of VoxCeleb2 and HDTF"), plus undisclosed
  "Chinese data" (1.5 changelog) and undisclosed 512² video (1.6 card). VoxCeleb2 / HDTF are research datasets of YouTube videos -> LatentSync
  is in the **YES\* (permissive weights, NC / research-only training data)** class, like pyannote. The weights licence itself (Open RAIL++-M) allows commercial use.
* **TBDub**: proprietary studio green-screen + livestream footage (3:1, per its paper via the desk sub-survey, not re-verified), and its DiT
  starts from X-Dub (Kling: "600 h" of unnamed internet video) on a Wan2.2 base -> **YES\*** (permissive weights, undisclosed / unnamed data provenance; no
  dataset is named as non-commercial).
* MediaPipe Face Landmarker: consented, Google-collected images (model cards) -> YES.
* Background bed: HDemucs MUSDB18-HQ -> **NO**, quarantined 2026-10-16 (`--background` now defaults to off, loader blocked by `guard_nc()`).

EVAL_ONLY (never imported by the product path): SyncNet `syncnet_v2.model` + S3FD `sfd_face.pth` (SyncNet scoring),
YuNet `face_detection_yunet_2023mar.onnx` (common metric `mouth_jitter_hr.py`; MIT file, WIDER FACE data), MediaPipe in IMAGE mode for
`lipsync_jitter.py` v3 (metric landmarks). `face_temporal.py` (InsightFace landmarks) is retired by this track.

## 1. What the cand5 video stage loaded (before the CLEAN-ONLY decision)

| model | repository | code licence | weights licence | commercial | problematic dependencies / data | replacement risk |
|---|---|---|---|---|---|---|
| LatentSync 1.6 UNet | [bytedance/LatentSync](https://github.com/bytedance/LatentSync), [HF ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) | Apache-2.0 | `openrail++` tag only (CreativeML Open RAIL++-M; no LICENSE file in the HF repo) | YES with conditions | RAIL use restrictions must flow into our ToS; training data VoxCeleb2 + HDTF + undisclosed "Chinese data" (1.5) / 512 px data (1.6) | core model; conditions are paperwork |
| **InsightFace buffalo_l `det_10g.onnx` + `2d106det.onnx`** (LatentSync `FaceDetector`, auto-downloaded; shipped by `infra/serverless/handler.py:55`) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | MIT | **non-commercial research only** ("whether downloaded automatically or manually") | **NO** | SCRFD on WIDER FACE (CC BY-NC-ND) | **production blocker today**. Options: buy the commercial licence (recognition-oss-pack@insightface.ai, terms unknown) or replace with MediaPipe Face Landmarker (Apache-2.0) and re-calibrate the 3-point affine (LatentSync was trained on InsightFace-aligned crops -> needs an A/B) |
| buffalo_l `1k3d68`, `genderage`, `w600k_r50` + 288 MB `buffalo_l.zip` | same | MIT | non-commercial | NO | – | not loaded; should be deleted from images / volumes |
| Whisper tiny (LatentSync audio encoder) | [openai/whisper](https://github.com/openai/whisper) | MIT | MIT (sha256 = OpenAI's file) | YES | – | – |
| sd-vae-ft-mse (LatentSync VAE) | [HF stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | – | MIT | YES | LAION subsets | low |
| RetinaFace R50 `detection_Resnet50_Final.pth` (face router) | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) via facexlib | MIT | none stated | UNCLEAR | WIDER FACE (CC BY-NC-ND) | MediaPipe BlazeFace full-range (Apache-2.0) |
| Silero VAD (speech gate) | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | MIT | YES | – | – |
| DeepCache, kornia | – | Apache-2.0 | no weights | YES | – | – |
| **CodeFormer + ParseNet** (cand5 runs `--codeformer optimized --codeformer-scope mouth`) | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | **S-Lab 1.0 (NC)** | NC | **NO** | FFHQ (CC BY-NC-SA), CelebAMask-HQ (NC) | `infra/serverless/README.md` says CodeFormer is OFF in production, but the human-reviewed cand5 baseline uses it -> the coordinator should decide which one is the shipping reference (track B owns the restorer) |
| SyncNet `syncnet_v2.model`, S3FD `sfd_face.pth` (evaluation only) | [joonson/syncnet_python](https://github.com/joonson/syncnet_python) | MIT | "research purposes" / no licence | UNCLEAR / NO | BBC data / WIDER FACE | internal metric only, never shipped |

## 2. Lip-sync generator candidates

| model | repository | code licence | weights licence | commercial | problematic dependencies (detectors, landmarkers, parsers, VAE, audio encoders, training data) | replacement risk / verdict |
|---|---|---|---|---|---|---|
| **TBDub** (Alibaba, [arXiv 2609.06144](https://arxiv.org/abs/2609.06144), V1.1 2026-09) | [TaoLiveAIGC/TBDub](https://github.com/TaoLiveAIGC/TBDub), [HF](https://huggingface.co/TaoLiveAIGC/TBDub) | Apache-2.0 | `apache-2.0` ("code and released model weights … Apache License 2.0") | **YES** | MediaPipe Face Landmarker (Apache-2.0; bypassable with `--cropped-input`), Wan2.2 VAE from KlingTeam/X-Dub (Apache-2.0), facebook/hubert-large-ll60k (Apache-2.0); in-house training data (undisclosed) | **shortlist #1 new backend**: 77-frame clips with cross-clip motion conditioning (no 16-frame seams), 2-step student, 15-28 GiB (fits 32 GB) |
| **MuseTalk 1.5** (2025-03-28) | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) | MIT | "available for any purpose, even commercially" | NO as shipped / YES after replacing the parser | BiSeNet face parser 79999_iter (CelebAMask-HQ, NC) used for blending; DWPose (Apache-2.0, UBody data UNVERIFIED); S3FD; training data HDTF + VFHQ (VFHQ "non-commercial research purposes") -> UNCLEAR provenance | mandatory evaluation: **per-frame single-step generation; README: "There exists some jitter as the current pipeline adopts single-frame generation"** -> not a stability upgrade; at most a cheap negative control with our own mouth-ellipse composite instead of BiSeNet |
| **VideoReTalking** | [OpenTalker/video-retalking](https://github.com/OpenTalker/video-retalking) | Apache-2.0 (+ PIRenderer-derived D-Net code CC BY-NC 4.0) | release assets, no separate licence | **NO** | Basel Face Model 2009 (NC), dlib shape_predictor_68 (NC), GPEN, GFPGAN (FFHQ), L-Net trained on LRS2 (BBC, research-only) | mandatory evaluation: `RESEARCH_ONLY — cannot ship`; per-frame GAN, FVD 270.6 vs LatentSync 162.7 in the LatentSync paper -> not worth GPU time even as an upper bound |
| Lip Forcing 14B (KAIST, [arXiv 2606.11180](https://arxiv.org/abs/2606.11180)) | [cvlab-kaist/LipForcing](https://github.com/cvlab-kaist/LipForcing) | Apache-2.0 | `apache-2.0` (base Wan2.1-T2V-14B Apache-2.0) | UNCLEAR | buffalo_l detector (NC, bypassable); teacher data includes Hallo3 data (CC BY-NC-ND) | autoregressive, seam-free, best FVD in its paper; **~37-41 GB VRAM, does not fit the 32 GB RTX 5090** (no fp8 / offload); revisit when the 1.3B student ships |
| X-Dub 5B (Kling) | [KlingAIResearch/X-Dub](https://github.com/KlingAIResearch/X-Dub) | Apache-2.0 | `apache-2.0` | YES (DWPose data UNVERIFIED) | DWPose | README admits "occasional flickering"; superseded by TBDub (same lineage) |
| KeySync | [antonibigata/keysync](https://github.com/antonibigata/keysync) | Apache-2.0 | apache-2.0 tag but SVD-derived (Stability Community Licence, < $1M revenue) | UNCLEAR, lean NO | CelebV-HQ / CelebV-Text (NC) | `RESEARCH_ONLY — cannot ship` |
| HighSync | [saeed5959/high_sync](https://github.com/saeed5959/high_sync) | MIT | MIT | NO (data + deps) | VFHQ + CelebV-HQ (NC), facebook denoiser dns64 (CC BY-NC), yolov8-face (GPL-3.0 / AGPL) | `RESEARCH_ONLY — cannot ship` (anti-leakage design, diagnostic only) |
| InfiniteTalk / MultiTalk | [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) | Apache-2.0 | apache-2.0 | UNCLEAR | chinese-wav2vec2 trained on WenetSpeech (NC) | regenerates the whole frame (head / body re-performed); MultiTalk is image-to-video only -> out of scope |
| Wav2Lip family, Diff2Lip, DINet, IP_LAP, TalkLip, StyleSync, JoyGen | see `research/candidates.md` §2.11-2.12 | various / none | research-only / none / withheld | **NO** | LRS2, OpenFace, AV-HuBERT, BFM, BiSeNet | rejected |
| OmniSync, StableDub, SayAnything, ComplexSync, SyncEdit | – | no code or no licence | – | N/A | – | not released; OmniSync's inference ideas (source-anchored noise, decaying guidance) are portable training-free tricks |
| sync. lipsync-2-pro / sync-3 | API | – | – | paid service | customer video leaves the box | commercial reference only |

## 3. Metric models used by this track

| model | use here | licence | status |
|---|---|---|---|
| InsightFace 2d106det (via LatentSync FaceDetector) | `face_temporal.py`, `lipsync_jitter.py` landmarks (same detector the shipped crop uses) | NC | internal measurement only; commercial use of NC weights for QA is still a risk -> MediaPipe Face Mesh cross-check planned |
| DINOv2 (planned, identity / appearance similarity on the mouth-masked face crop) | metric | Apache-2.0 code + weights | YES |
| facenet-pytorch VGGFace2 (optional face-specific identity) | metric | MIT code; VGGFace2 released CC BY-SA 4.0 (since withdrawn) | UNCLEAR, internal only |
| OpenCV DIS optical flow (warping error) | metric | Apache-2.0 / BSD (OpenCV) | YES |
