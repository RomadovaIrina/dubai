# Video-stage dependency and license audit (lip-sync stack)

Date: 2026-10-16. This is CPU-only desk research. Sources are the local checkout at `/workspace/dub/dubai` (read-only) plus LICENSE files, READMEs and model cards fetched on 2026-10-16. Nothing was installed, downloaded as a model, or run. **This is not legal advice.** Every NO and UNCLEAR below needs counsel sign-off before launch.

Legend for the "commercial" column:
- **YES**: the code license and the weights license both allow commercial use, and no training-data restriction is stated.
- **NO**: the licensor says non-commercial / research-only for the weights, or the code has no license at all.
- **UNCLEAR**: the weights carry a permissive license (or none), but the stated training data is non-commercial, or the terms are ambiguous or unavailable.

The usual caveat applies to most CNNs here and is not repeated per row: ImageNet-pretrained backbones and web-scraped training images carry their own (often research-only) terms.

---

## 0. What the production video stage actually loads

The production manifest is `infra/serverless/handler.py` `MANDATORY` (cross-checked with `scripts/download_models.sh` and `infra/serverless/README.md`).

| Weight | Where it is used | In production? |
|---|---|---|
| `latentsync/latentsync_unet.pt` (LatentSync 1.6) | LatentSync UNet | YES |
| `latentsync/whisper/tiny.pt` | LatentSync audio encoder | YES |
| `sd-vae-ft-mse/*` | LatentSync VAE | YES |
| `insightface/models/buffalo_l/det_10g.onnx` + `2d106det.onnx` | LatentSync `FaceDetector` (`third_party/latentsync/latentsync/utils/face_detector.py`, `FaceAnalysis(allowed_modules=["detection","landmark_2d_106"], root="checkpoints/auxiliary")`). The same landmarks feed `scripts/quality/face_temporal.py` and `scripts/optim/codeformer_accel.py::lm5_from_insightface106` | YES |
| `buffalo_l/1k3d68.onnx`, `genderage.onnx`, `w600k_r50.onnx` | Auto-downloaded as part of the `buffalo_l.zip` pack (`third_party/latentsync/checkpoints/auxiliary/models/buffalo_l/`, plus a 288 MB `buffalo_l.zip` left next to it). **Not loaded** because `allowed_modules` excludes them | On disk only |
| `facexlib/detection_Resnet50_Final.pth` (RetinaFace R50) | `scripts/optim/retinaface_router.py` (production `face_router="retinaface"`) and `scripts/check_retinaface.py` | YES |
| Silero VAD (`silero_vad/data/*.jit/.onnx`, bundled in the `silero-vad 6.2.1` wheel) | Speech gate (`face_aware_latentsync_accel.py::silero_speech_intervals`, `run_clean_pipeline_05.py`) | YES |
| DeepCache 0.1.1, kornia 0.8.0 | UNet feature caching; `affine_transform.py` warp/erode/blur | YES (code only, no weights) |
| CodeFormer `codeformer.pth` + `facelib/parsing_parsenet.pth` (+ the same RetinaFace R50 file) | `scripts/optim/codeformer_accel.py`, experiments | **NO**: "CodeFormer (stage 7, OFF in production — `basicsr`/`facelib` are never imported by the clean pipeline)" (`infra/serverless/README.md`) |
| `latentsync/auxiliary/syncnet_v2.model`, `sfd_face.pth` | SyncNet confidence/offset evaluation (`third_party/latentsync/eval/*`, `scripts/pilot/syncnet_common.py`) | NO (eval only; "SyncNet/eval weights" are excluded from the image) |

The local grep found no other auto-downloads in our own `scripts/`. These are present in third-party code but are **not reached** by our pipeline:
- In `third_party/CodeFormer`: `RealESRGAN_x2plus.pth`, `codeformer_inpainting.pth`, `codeformer_colorization.pth`, `yolov5{l,n}-face.pth`, `parsing_bisenet.pth`, dlib `mmod_human_face_detector`, `shape_predictor_5`, and `shape_predictor_68` (the last via `scripts/crop_align_face.py`).
- In `third_party/latentsync`: HyperIQA torchvision ResNets in `eval/hyper_iqa.py`, VideoMAEv2 `vit_g` in `trepa/`, and MediaPipe in `preprocess/` (training data pipeline only).
- `face-alignment 1.4.1` and `lpips 0.1.4` are installed in `/venv/dabai` but are only probed by `scripts/check_versions.py`.

`whisper/tiny.pt` sha256 is `65147644…22b9`. That is byte-identical to OpenAI's official `tiny.pt` URL hash in `third_party/latentsync/latentsync/whisper/whisper/__init__.py`, so it is OpenAI's MIT weight even though it ships inside the `openrail++`-tagged ByteDance repo.

---

## Task A: license audit of the current video stage

| Model | Repository / URL | Code license | Weights license | Commercial | Training data terms | Notes / replacement risk |
|---|---|---|---|---|---|---|
| **LatentSync 1.6 UNet** `latentsync_unet.pt` | https://huggingface.co/ByteDance/LatentSync-1.6 | Apache-2.0 (code repo) | HF metadata `license: openrail++` only. There is no LICENSE file in the HF repo (file list: `.gitattributes, .gitignore, README.md, auxiliary/*, config.json, latentsync_unet.pt, stable_syncnet.pt, whisper/tiny.pt`). HF maps `openrail++` to "Open Rail++-M License" = the CreativeML Open RAIL++-M text (SDXL) | **YES, with conditions** (UNCLEAR on the exact text, because ByteDance never attached one) | Paper (arXiv 2412.09262v2): "a mixture of VoxCeleb2 and HDTF". v1.5 changelog: "added Chinese data" (source not disclosed). v1.6 card: "trained … on 512 x 512 resolution videos" (source not disclosed). VoxCeleb2: "metadata is licensed under a Creative Commons Attribution-ShareAlike 4.0"; videos are YouTube content and downloads are withdrawn. HDTF: "CC BY 4.0" (YouTube-sourced) | Core model, no drop-in replacement. Open RAIL++-M obligations for SaaS: Attachment A use restrictions must be in our ToS and passed to users. Several restrictions are directly relevant to lip-sync (false information to harm, defamation, harassment). Training data is only partly disclosed. |
| LatentSync code | https://github.com/bytedance/LatentSync (local `third_party/latentsync/LICENSE`, pinned `a229c39`) | Apache-2.0 | n/a | YES | n/a | Keep NOTICE/attribution. |
| Whisper tiny (`whisper/tiny.pt`) | https://github.com/openai/whisper | MIT | MIT ("Whisper's code and model weights are released under the MIT License") | YES | 680k h of internet audio; no dataset license published | The ByteDance HF copy is byte-identical (sha256 verified). |
| `stabilityai/sd-vae-ft-mse` | https://huggingface.co/stabilityai/sd-vae-ft-mse | (diffusers, Apache-2.0) | `license: mit` (card metadata) | YES | Fine-tuned on "LAION-Aesthetics and LAION-Humans, an unreleased subset". Base kl-f8 "originally trained on OpenImages". LAION is web-scraped links whose image copyright stays with the owners | Low risk. |
| insightface python package 0.7.3 | https://github.com/deepinsight/insightface (PyPI METADATA `License: MIT`) | MIT | n/a | YES | n/a | The code is fine. The models are the problem (next rows). |
| insightface `buffalo_l/det_10g.onnx` (SCRFD-10GF) | https://github.com/deepinsight/insightface/tree/master/model_zoo | MIT | **Non-commercial research only** | **NO** | SCRFD trains on WIDER FACE (`detection/scrfd/README.md`). WIDER FACE: "Creative Common License (cc by-nc-nd)" | **Production blocker.** Commercial license: "For open-sourced face recognition models (e.g., buffalo_l package), please contact recognition-oss-pack@insightface.ai" (README update dated 2025-11-24). |
| insightface `buffalo_l/2d106det.onnx` | https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg | MIT | **Non-commercial research only** | **NO** | Not documented (README: "MobileNet-0.5, size 5MB … predict 2d-106 landmarks") | **Production blocker.** The LatentSync crop/affine and our jitter metrics depend on it (Task B). |
| insightface `buffalo_l/1k3d68`, `genderage`, `w600k_r50` | same | MIT | **Non-commercial research only** | **NO** | `w600k_r50` = "ResNet50@WebFace600K". WebFace260M terms page was unreachable (face-benchmark.org timed out): **UNVERIFIED**. insightface itself says the training data is NC | Not loaded, but they sit on disk and on the model volume. Delete them, or restrict the volume to the two loaded files (they are unlicensed for us either way). |
| CodeFormer code + `codeformer.pth` | https://github.com/sczhou/CodeFormer | **S-Lab License 1.0** (NC) | Same ("This project is licensed under NTU S-Lab License 1.0") | **NO** | FFHQ: "made available under Creative Commons BY-NC-SA 4.0" (dataset). Images under CC BY / BY-NC / PD | OFF in production. Keep it out of any shipped path. Commercial use requires "contact the contributor(s)". |
| facexlib 0.3.0 | https://github.com/xinntao/facexlib | MIT (LICENSE file); `setup.py` says "Apache License 2.0". Both permissive | n/a | YES | n/a | The README warns: "You need to refer to their original LICENCEs for your intended use." |
| `detection_Resnet50_Final.pth` (RetinaFace R50, biubug6 lineage; same file in facexlib and CodeFormer releases) | https://github.com/biubug6/Pytorch_Retinaface | MIT | No separate weight license (MIT repo; weights distributed via Google Drive / re-hosted in GitHub releases) | **UNCLEAR** | WIDER FACE (cc by-nc-nd). The R50 backbone is ImageNet-pretrained | Used in production as the face router. Same data-provenance issue as YuNet/SCRFD. |
| `parsing_parsenet.pth` (ParseNet from PSFRGAN) | https://github.com/chaofengc/PSFRGAN (local `facelib/parsing/parsenet.py`: "Modified from https://github.com/chaofengc/PSFRGAN") | CC BY-NC-SA 4.0 (PSFRGAN) | CC BY-NC-SA 4.0 (no separate statement) | **NO** | CelebAMask-HQ: "available for **non-commercial research purposes** only" | CodeFormer path only (OFF). |
| SyncNet `syncnet_v2.model` (eval only) | https://github.com/joonson/syncnet_python; weights from https://www.robots.ox.ac.uk/~vgg/software/lipsync/ | MIT (syncnet_python). LatentSync's eval port is Apache-2.0 | "The model can be used for research purposes under Creative Commons Attribution License." | **UNCLEAR** (research-only wording) | BBC broadcast data (page: "help in obtaining the dataset" from BBC Research) | Internal QA metric, never shipped. "Research purposes" is ambiguous for internal QA in a commercial company. |
| S3FD `sfd_face.pth` (eval only) | port from https://github.com/cs-giung/face-detection-pytorch (via syncnet_python `detectors/README.md`). Original: https://github.com/sfzhang15/SFD | **No license file** (GitHub API: none, for both cs-giung and sfzhang15) | None stated; hosted by VGG | **NO/UNCLEAR** | WIDER FACE (cc by-nc-nd) | Eval only. Could be swapped for BlazeFace inside our SyncNet wrapper if it matters. |
| DeepCache 0.1.1 | https://github.com/horseee/DeepCache | Apache-2.0 | none (training-free) | YES | n/a | — |
| kornia 0.8.0 | https://github.com/kornia/kornia | Apache-2.0 | none used (warp_affine / erosion / gaussian_blur2d only) | YES | n/a | — |
| Silero VAD (silero-vad 6.2.1 wheel, weights bundled) | https://github.com/snakers4/silero-vad | MIT | MIT ("Published under permissive license (MIT) Silero VAD has zero strings attached") | YES | "trained on huge corpora that include over 6000 languages"; no dataset terms published | The README badge alt text still says "CC BY-NC 4.0" but displays MIT. The LICENSE file and wheel metadata are MIT. (Other Silero STT/TTS models have different licenses; only the VAD is used.) |

**Task A bottom line.** In the shipped path, the only hard NO items are the two insightface ONNX files (`det_10g`, `2d106det`). The RetinaFace router weights are UNCLEAR (permissive weights, NC training data). LatentSync itself is commercially usable under Open RAIL++-M conditions. There is a documentation gap: ByteDance attached no license text, and 1.5/1.6 training data is undisclosed. Eval-only SyncNet/S3FD are research-licensed or unlicensed. CodeFormer/ParseNet are NC and must stay OFF.

---

## Task B: commercially usable face detection + landmarks + parsing

### What LatentSync needs from the detector (local code)

- `face_detector.py`: picks the largest face from `face.bbox`, filtered by `w>=50, h>=80, 0.2<w/h<1.5, det_score>=0.5`.
  - The crop bbox comes from `landmark_2d_106`: x-range and bottom from `LMK_ADAPT_ORIGIN_ORDER = [1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,43,48,49,51,50,102,103,104,105,101,73,74,86]`.
  - The top is `mean(lmk[73], lmk[74])` mirrored by the distance to the lowest point.
- `image_processor.py`: builds a 3-point affine from three averages:
  - `mean(lmk[[43,48,49,51,50]])` (code comment: "left eyebrow center")
  - `mean(lmk[101:106])` ("right eyebrow center")
  - `mean(lmk[[74,77,83,86]])` ("nose center")
- Training-distribution constraint: the LatentSync README data pipeline, step 5, reads "Affine transform the faces according to the landmarks detected by InsightFace". The UNet was therefore trained on InsightFace-aligned crops.
  - Any replacement must reproduce those three anchor points (and the crop extent) closely. Otherwise the crop distribution shifts.
  - A replacement needs an A/B on SyncNet confidence, mouth-region PSNR and landmark jitter, not just a license swap.

### Candidates

| Model | Repository / URL | Code license | Weights license | Commercial | Training data and terms | Landmarks | Speed (published, not measured here) | Notes / replacement risk |
|---|---|---|---|---|---|---|---|---|
| **MediaPipe Face Landmarker** (BlazeFace short-range + Face Mesh V2 + Blendshapes) | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker ; https://github.com/google-ai-edge/mediapipe | Apache-2.0 | Model cards: "LICENSED UNDER Apache License, Version 2.0" (Face Mesh V2, BlazeFace SR) | **YES** | Google-collected smartphone images via an AR app. BlazeFace card: "trained and evaluated on consented images". Face Mesh annotated by 11 in-house annotators | 478 3D (468 mesh + 10 iris) | BlazeFace SR "~275FPS on Pixel 2 single-core CPU"; Face Mesh "real time" on mobile | **Best license fit.** Domain risk: the card says it is not suitable for faces "looking away from the camera (more than 80°)" or "only partially visible (less than 50%)". The short-range detector needs face bbox sides ≥20% of the image. Use the full-range detector (next row) or another detector for wide shots. 478→106 is not 1:1, so the brow/nose/contour points must be remapped and calibrated against the insightface anchors (see the risk above). GPU delegate in Linux Python not verified; assume CPU TFLite. |
| MediaPipe BlazeFace full-range / sparse | https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector | Apache-2.0 | "Apache License, Version 2.0" (Full Range card) | **YES** | "consented images … mobile AR application", 100+K images | 6 keypoints + bbox | "~50FPS on Pixel 3 single-core CPU" | Face bbox ≥5% of the image; "further than 5 meters" is out of scope. A license-clean replacement for the RetinaFace router (accuracy on film/TV content untested). |
| OpenCV YuNet | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | Apache-2.0 (zoo), libfacedetection BSD-3 | Model dir LICENSE: MIT (Shiqi Yu) | **UNCLEAR** | WIDER FACE (README: AP on "WIDER Face validation set"; training repo libfacedetection.train). WIDER is cc by-nc-nd | 5 | 5.6 ms CPU (face_alignment README table, Apple M2, 450×450) | Permissive weight license but NC training data. Only 5 points, so not enough for the LatentSync affine/contour. |
| biubug6 RetinaFace (mobilenet0.25 / R50) | https://github.com/biubug6/Pytorch_Retinaface | MIT | No separate statement | **UNCLEAR** | WIDER FACE (cc by-nc-nd); ImageNet backbone | 5 | 25.2 ms CPU (same table) | = current router. 5 points only. |
| face_alignment (FAN) 1.4.1 | https://github.com/1adrianb/face-alignment | BSD-3-Clause | None stated (weights at adrianbulat.com) | **NO** (data) | Paper: "we used 300W-LP-2D and 300W-LP-3D to train 2D-FAN and 3D-FAN". 300W-LP is rendered from 300-W: "The data are provided for research purposes only. Commercial use (i.e., use in training commercial algorithms) is not allowed." | 68 (2D/3D) | SFD detector 138.8 ms CPU (default) | Training data of the current `2DFAN4_1.6` checkpoints is not documented: UNCLEAR at best. The default SFD detector has the same WIDER issue. |
| SPIGA | https://github.com/andresprados/SPIGA | BSD-3-Clause | None stated (HF `aprados/spiga`, no license tag) | **NO/UNCLEAR** | Per-dataset checkpoints: WFLW (WIDER-derived, no explicit license), 300W (NC), MERLRAV/AFLW, COFW | 98 (WFLW) / 68 / 29 | n/a | Also gives head pose. Data is NC. |
| PIPNet | https://github.com/jhb86253817/PIPNet | MIT | None stated | **NO/UNCLEAR** | 300W, COFW, WFLW, AFLW; GSSL variant "300W+CelebA" (CelebA: "non-commercial research purposes only") | 98 / 68 / 29 / 19 | README: "fast" (speed/accuracy plot on WFLW; numbers not re-checked) | Data is NC. |
| STAR | https://github.com/ZhenglinZhou/STAR | **No LICENSE file** (GitHub API: none; LICENSE 404) | None | **NO** | WFLW / 300W / COFW | 98 / 68 / 29 | n/a | With no license, all rights are reserved. |
| dlib `shape_predictor_68_face_landmarks` | https://github.com/davisking/dlib-models | dlib BSL-1.0 | Repo CC0, **but**: "the trained model therefore can't be used in a commercial product" | **NO** | iBUG 300-W (research only) | 68 | fast CPU | CodeFormer `scripts/crop_align_face.py` downloads it (not used by us). |
| dlib `shape_predictor_5_face_landmarks` | same | BSL-1.0 | Public domain (Davis King) | **YES** (UNCLEAR on image provenance) | "dlib 5-point face landmark dataset … downloading images from the internet and annotating them" | 5 | fast CPU | Too few points for LatentSync. |
| HRNet facial landmarks | https://github.com/HRNet/HRNet-Facial-Landmark-Detection | MIT (GitHub API) | None stated (OneDrive links) | **NO/UNCLEAR** | COFW, AFLW, WFLW, 300W | 29 / 19 / 98 / 68 | HRNetV2-W18 4.3 GFLOPs | Data is NC. |
| **InsightFace commercial license** (keep buffalo_l) | contact recognition-oss-pack@insightface.ai | MIT | Commercial license on request | **YES if purchased** (terms/price unknown) | — | 106 | already measured in our pipeline | **Zero-risk technically** (no crop-distribution shift). Worth asking before re-engineering. |

**Face parsing (for masks / metrics)**

| Model | URL | Code | Weights | Commercial | Training data | Notes |
|---|---|---|---|---|---|---|
| ParseNet (current, CodeFormer) | PSFRGAN | CC BY-NC-SA 4.0 | same | **NO** | CelebAMask-HQ (NC) | — |
| BiSeNet face-parsing.PyTorch (= facexlib `parsing_bisenet`) | https://github.com/zllrunning/face-parsing.PyTorch | MIT | none stated | **NO** (data) | CelebAMask-HQ: "not to … exploit for any commercial purposes, any portion of the images and any portion of derived data" | — |
| SegFormer face-parsing | https://huggingface.co/jonathandinu/face-parsing | — | card: "**License:** non-commercial research and educational purposes" | **NO** | CelebAMask-HQ; base `nvidia/mit-b5` | — |
| FaRL / facer parsers | https://github.com/FacePerceiver/facer , /FaRL | MIT | none stated | **NO** (data) | LAION-Face pretrain; fine-tuned on LaPa ("for non-commercial purposes") or CelebAMask-HQ | — |
| Anything trained on Microsoft FaceSynthetics | https://github.com/microsoft/FaceSynthetics | — | — | **NO** | R-UDA: "you may not use the Data or any Results in any commercial offering … Artificial intelligence models trained on Data … are Results" | Synthetic, but research-only. |
| **MediaPipe Selfie Multiclass segmentation** | https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter | Apache-2.0 | card: "Apache License, Version 2.0" | **YES** | Google data, "thousands of samples with diverse identities" (source not further specified) | Classes: background, hair, body-skin, **face-skin**, clothes, others. No lips/teeth/eyes classes. |
| **Face Mesh landmark polygons** (lips outer/inner, face oval) | MediaPipe (above) | Apache-2.0 | Apache-2.0 | **YES** | as Face Mesh | Geometric mouth/face masks. No semantic handling of occluders (hands, mics). Combine with Selfie Multiclass face-skin for occlusion. |
| SAM 2 (prompted with landmark points/boxes) | https://github.com/facebookresearch/sam2 | Apache-2.0 | "The SAM 2 model checkpoints … are licensed under Apache 2.0" | **YES** (SA-V dataset terms not checked) | — | Heavy. Option for offline metric masks, not per-frame production. |

**Task B bottom line.** The only fully clean stack found (Apache-2.0 code + weights + consented training data) is MediaPipe: BlazeFace full/short-range, Face Mesh V2 (478 points) and Selfie Multiclass. Every academic landmark model (FAN, SPIGA, PIPNet, STAR, HRNet, dlib-68) is trained on 300-W / WFLW / AFLW / COFW / CelebA. Those datasets are research-only, so the models are NO/UNCLEAR whatever their code license says. Every public face parser is trained on CelebAMask-HQ / LaPa / FaceSynthetics and is NO. The alternative to re-engineering is buying the InsightFace commercial license for buffalo_l.

---

## Task C: face-identity embedding for an "identity preservation" metric

| Model | Repository / URL | Code license | Weights license | Commercial | Training data and terms | Notes / risk |
|---|---|---|---|---|---|---|
| ArcFace via insightface (`buffalo_l/w600k_r50`, antelopev2 etc.) | https://github.com/deepinsight/insightface | MIT | **NC research only** ("ALL models are available for non-commercial research purposes only") | **NO** (unless licensed) | WebFace600K / Glint360K / MS1M (NC per insightface) | Already on disk, but not licensed for us. Commercial licensing via recognition-oss-pack@insightface.ai. |
| FaceNet, facenet-pytorch `InceptionResnetV1(pretrained='vggface2')` | https://github.com/timesler/facenet-pytorch | MIT | No separate statement. "initialized using parameters ported from David Sandberg's tensorflow facenet repo" (MIT) | **UNCLEAR, leaning usable** | VGGFace2 (2019 page): "available to download for commercial/research purposes under a Creative Commons Attribution-ShareAlike 4.0 International License. The copyright remains with the original owners of the image." Downloads are now withdrawn and the license line is gone from the current page. The `casia-webface` variant is trained on CASIA-WebFace (terms page 404, not verified; treat as research-only) | The only face-specific option whose training-data license allowed commercial use at release. Open questions: whether ShareAlike reaches a trained model, image copyright, dataset withdrawal. Use the VGGFace2 weights only. |
| AdaFace | https://github.com/mk-minchul/AdaFace | MIT | None stated (Google Drive) | **NO/UNCLEAR** | Checkpoints on CASIA-WebFace, VGGFace2 (R18 only), MS1MV2/V3, WebFace4M/12M. MS1M was withdrawn by Microsoft in 2019 (widely reported; not re-verified here); WebFace is NC per insightface | The R18-VGGFace2 checkpoint has the same profile as FaceNet-VGGFace2. The others are NC. |
| SFace (OpenCV zoo) | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface | Apache-2.0 | Apache-2.0 (model dir LICENSE) | **UNCLEAR** | Not stated for the shipped ONNX. The paper trained on "CASIA-WebFace, VGGFace2, and MS-Celeb-1M". The upstream repo zhongyy/SFace has **no license** | Permissive file license, undocumented data provenance. |
| dlib `dlib_face_recognition_resnet_model_v1` (extra candidate) | https://github.com/davisking/dlib-models | BSL-1.0 | Public domain ("anyone can do whatever they want") | **UNCLEAR** | "about half the images are from VGG and face scrub". VGG-Face: "Creative Commons Attribution-NonCommercial 4.0". FaceScrub: CC BY-NC-ND 3.0 | Clean author grant, NC source data. |
| **DINOv2** (ViT-S/B/L/g; generic fallback) | https://github.com/facebookresearch/dinov2 | Apache-2.0 | "DINOv2 code and model weights are released under the Apache License 2.0" | **YES** | LVD-142M (curated web images; model card) | Not identity-specific. On a lip-sync output (same frame, regenerated mouth) it measures appearance/texture change, not identity. Do **not** use the XRay-DINO / Cell-DINO variants in the same repo; those are "FAIR Noncommercial Research License". |

### Internal metric vs shipping

- Running a model only as an offline/internal QA metric means nothing is distributed. That removes the redistribution and SaaS obligations (for example the Open RAIL++ pass-through).
- It does **not** turn commercial use into non-commercial use. CC BY-NC defines NonCommercial as "not primarily intended for or directed towards commercial advantage or monetary compensation". QA that tunes a paid product is directed towards commercial advantage.
- Microsoft's R-UDA spells this out: no use of Data or Results "as part of a product or service (or to improve any product or service)".
- So "metric only, not shipped" lowers exposure (nothing distributed, nothing customer-facing) but does not cure an NC or research-only term. Treat NC weights as NO for internal QA too, unless counsel decides otherwise.
- Separately, a face embedding is a biometric template (GDPR Art. 9 / BIPA-type rules) once it is computed on customer videos. Compute it transiently and do not store embeddings.

**Safest choice for the identity metric:**
1. **DINOv2 (Apache-2.0) as the license-clean primary.** Compute cosine on the face crop, and also on the crop with the mouth region masked so it is not dominated by the edited area.
2. If a face-recognition-specific signal is required, **facenet-pytorch VGGFace2** as a secondary internal-only metric, with the provenance caveat documented.
3. Avoid ArcFace/insightface, AdaFace (MS1M/WebFace), and dlib-ResNet for anything not research-only. Avoid SFace until its training data is documented.

---

## Evidence (quotes and links, retrieved 2026-10-16)

**LatentSync 1.6 weights**
- https://huggingface.co/ByteDance/LatentSync-1.6/raw/main/README.md front matter: `license: openrail++`, `library_name: torchgeo`.
- Card: "we trained LatentSync 1.6 on 512 x 512 resolution videos … the only modification was upgrading the training dataset to 512 x 512 videos."
- HF API file list contains no LICENSE file.
- HF license table (https://github.com/huggingface/hub-docs/blob/main/docs/hub/repositories-licenses.md): "Open Rail++-M License | `openrail++` | license: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md".

**CreativeML Open RAIL++-M** (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- Grant: "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute … the Model".
- "'Distribution' means … including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access."
- "You may host for Third Party remote access purposes (e.g. software-as-a-service) … provided that … Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement … and You shall give notice to subsequent users".
- "You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5)."
- "Licensor claims no rights in the Output You generate".
- Attachment A includes: "To generate or disseminate verifiably false information and/or content with the purpose of harming others"; "To defame, disparage or otherwise harass others"; "To generate or disseminate personal identifiable information that can be used to harm an individual"; "For the purpose of exploiting, harming or attempting to exploit or harm minors in any way".

**LatentSync training data**
- arXiv 2412.09262v2 §5.1: "We used a mixture of VoxCeleb2 [10] and HDTF [51] datasets as our training set."
- `docs/changelog_v1.5.md`: "we added Chinese data to the training of the new model version."
- README data pipeline step 5: "Affine transform the faces according to the landmarks detected by InsightFace".
- VoxCeleb2 (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html): "The provided VoxCeleb2 metadata is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License … The video files for the VoxCeleb 2 dataset are no longer available".
- HDTF (https://github.com/MRzzm/HDTF): "The HDTF dataset is available to download under a Creative Commons Attribution 4.0 International License."

**Whisper** (https://github.com/openai/whisper)
- README: "Whisper's code and model weights are released under the MIT License."
- Model card: "trained on 680,000 hours of audio and the corresponding transcripts collected from the internet."

**sd-vae-ft-mse** (https://huggingface.co/stabilityai/sd-vae-ft-mse)
- `license: mit`.
- "finetuned from the original kl-f8 autoencoder on a 1:1 ratio of LAION-Aesthetics and LAION-Humans, an unreleased subset containing only SFW images of humans … (the autoencoder was originally trained on OpenImages)".

**InsightFace**
- Main README (https://github.com/deepinsight/insightface#license): "The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage."
- Same section: "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only."
- Same section: "Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy(which is for non-commercial research purposes only)."
- Same section: "2025-11-24 Update: … For open-sourced face recognition models (e.g., buffalo_l package), please contact recognition-oss-pack@insightface.ai for licensing."
- Installed 0.7.3 METADATA: "**The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models.**"
- model_zoo/README.md: ":bell: **ALL models are available for non-commercial research purposes only.**"; table row: "buffalo_l | SCRFD-10GF | ResNet50@WebFace600K | 2d106 & 3d68 | Gender&Age | 326MB".

**WIDER FACE** (http://shuoyang1213.me/WIDERFACE/)
- "License Creative Common License (cc by-nc-nd)".
- The HF mirror README says "CC BY-NC-ND 4.0".

**CodeFormer** (https://github.com/sczhou/CodeFormer)
- LICENSE "S-Lab License 1.0 … Redistribution and use for non-commercial purpose … In the event that redistribution and/or use for commercial purpose in source or binary forms … is required, please contact the contributor(s) of the work."
- README: "This project is licensed under NTU S-Lab License 1.0. Redistribution and use should follow this license."

**FFHQ** (https://github.com/NVlabs/ffhq-dataset)
- "The dataset itself … is made available under Creative Commons BY-NC-SA 4.0 license by NVIDIA Corporation."

**facexlib** (https://github.com/xinntao/facexlib)
- "This project is released under the MIT license."
- "Note that we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use."
- Table: "Detection | Pytorch_Retinaface | MIT"; "Parsing | face-parsing.PyTorch | MIT".
- `setup.py`: `license='Apache License 2.0'`.

**Pytorch_Retinaface** (https://github.com/biubug6/Pytorch_Retinaface)
- LICENSE.MIT "MIT License Copyright (c) 2019". README trains on WIDERFACE; weights on Google Drive / Baidu.

**PSFRGAN / ParseNet** (https://github.com/chaofengc/PSFRGAN)
- "This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License."
- Training: "Download CelebAHQ-Mask dataset".

**CelebAMask-HQ** (https://github.com/switchablenorms/CelebAMask-HQ)
- "The CelebAMask-HQ dataset is available for **non-commercial research purposes** only."
- "You agree **not to** reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data."

**SyncNet**
- https://www.robots.ox.ac.uk/~vgg/software/lipsync/: "The model can be used for research purposes under Creative Commons Attribution License."
- syncnet_python LICENSE.md: MIT, "Copyright (c) 2016-present Joon Son Chung".
- `download_model.sh` fetches `syncnet_v2.model` and `sfd_face.pth` from robots.ox.ac.uk.
- `detectors/README.md`: "This face detector is adapted from https://github.com/cs-giung/face-detection-pytorch" (no LICENSE, GitHub API `license: null`).

**DeepCache / kornia**
- Both LICENSE files are Apache License 2.0 (GitHub API `Apache-2.0`).

**Silero VAD** (https://github.com/snakers4/silero-vad)
- LICENSE "MIT License Copyright (c) 2020-present Silero Team".
- README: "Published under permissive license (MIT) Silero VAD has zero strings attached - no telemetry, no keys, no registration".
- Wheel `silero_vad-6.2.1.dist-info/licenses/LICENSE` = MIT. Weights live in `silero_vad/data/`.

**MediaPipe**
- Face Mesh V2 card (https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf): "LICENSED UNDER Apache License, Version 2.0"; "Facial surface represented as 478 3D landmarks".
- Same card, input limits: "not suitable for detecting faces: looking away from the camera (more than 80°), inclined from the vertical orientation (more than 8°), only partially visible (less than 50% of the face)".
- Same card, intended use: "The primary intended application is AR entertainment."
- BlazeFace short-range card: "LICENSED UNDER Apache License, Version 2.0"; "trained and evaluated on consented images of people using a mobile AR application"; "Face bounding box sides should be at least 20% of the corresponding image sides".
- BlazeFace full-range card: "Apache License, Version 2.0"; "at least 5% of the corresponding image sides"; "The dataset consists of 100+K images."
- Multiclass segmentation card: "LICENSED UNDER Apache License, Version 2.0"; classes "background, hair, body-skin, face-skin, clothes, others (accessories)".
- Out of scope in all cards: "Any form of surveillance or identity recognition".

**OpenCV zoo**
- Root README: "OpenCV Zoo is licensed under the Apache 2.0 license. Please refer to licenses of different models."
- `face_detection_yunet/LICENSE`: "MIT License Copyright (c) 2020 Shiqi Yu"; YuNet evaluated on "WIDER Face validation set".
- `face_recognition_sface/LICENSE`: Apache 2.0; "Model files encode MobileFaceNet instances trained on the SFace loss function".
- zhongyy/SFace README: "models trained on CASIA-WebFace, VGGFace2, and MS-Celeb-1M"; the repo has no LICENSE.

**300-W** (https://ibug.doc.ic.ac.uk/resources/300-W/)
- "The data are provided for research purposes only. Commercial use (i.e., use in training commercial algorithms) is not allowed."

**dlib-models** (https://github.com/davisking/dlib-models)
- "anyone can do whatever they want with these model files as I've released them into the public domain".
- For the 68-point model: "The license for this dataset excludes commercial use and Stefanos Zafeiriou, one of the creators of the dataset, asked me to include a note here saying that the trained model therefore can't be used in a commercial product."
- ResNet face model: "about half the images are from VGG and face scrub".

**FAN** (arXiv 1703.07332)
- "we used 300W-LP-2D and 300W-LP-3D to train 2D-FAN and 3D-FAN".
- 300W-LP is "a synthetically generated dataset obtained by rendering the faces of 300-W into larger poses".

**WFLW** (https://github.com/wywu/LAB)
- "contains 10000 faces … with 98 fully manual annotated landmarks". The repo license is Caffe/"Other". No dataset license was found; images come from WIDER FACE. **UNVERIFIED** beyond that (WFLW project page returned 404).

**CelebA** (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- "The CelebA dataset is available for non-commercial research purposes only."

**LaPa** (https://github.com/jd-opensource/lapa-dataset)
- "made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation."

**jonathandinu/face-parsing** (https://huggingface.co/jonathandinu/face-parsing)
- "**License:** non-commercial research and educational purposes".

**FaceSynthetics** (https://github.com/microsoft/FaceSynthetics)
- R-UDA 2.1: "you may not use the Data or any Results in any commercial offering, including as part of a product or service (or to improve any product or service) you use or provide to others."
- R-UDA 5.5: "Artificial intelligence models trained on Data … are Results."

**SAM 2** (https://github.com/facebookresearch/sam2)
- "The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under Apache 2.0".

**facenet-pytorch** (https://github.com/timesler/facenet-pytorch)
- MIT; "pretrained on VGGFace2 and CASIA-Webface"; "initialized using parameters ported from David Sandberg's tensorflow facenet repo".
- VGGFace2, archived page http://web.archive.org/web/20191216191335/http://www.robots.ox.ac.uk:80/~vgg/data/vgg_face2/: "The VGGFace2 dataset is available to download for commercial/research purposes under a Creative Commons Attribution-ShareAlike 4.0 International License. The copyright remains with the original owners of the image."
- The current page says "The download links for the VGGFace2 dataset are no longer available from this website."

**VGG-Face** (https://www.robots.ox.ac.uk/~vgg/data/vgg_face/licence.txt)
- "The images are covered under a Creative Commons Attribution-NonCommercial 4.0 International license".

**FaceScrub** (http://vintage.winklerbros.net/facescrub.html)
- "released under a creative commons license" (link target: CC BY-NC-ND 3.0).

**AdaFace** (https://github.com/mk-minchul/AdaFace)
- MIT; checkpoint table lists CASIA-WebFace, VGGFace2, WebFace4M/12M, MS1MV2/V3.

**DINOv2** (https://github.com/facebookresearch/dinov2)
- "DINOv2 code and model weights are released under the Apache License 2.0."
- MODEL_CARD: "Training data: LVD-142M".
- The same README says XRay-DINO / Cell-DINO weights are under the "FAIR Noncommercial Research License".

**CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
- "NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation."

### Explicit uncertainties / not verified
- **LatentSync weights:** the exact license text is not attached by ByteDance; HF's `openrail++` mapping is assumed. 1.5/1.6 training data beyond VoxCeleb2/HDTF is undisclosed.
- **WebFace260M / WebFace600K terms:** face-benchmark.org was unreachable; only insightface's own NC statement is available.
- **CASIA-WebFace, AFLW, COFW, MS-Celeb-1M terms:** the official pages returned 404 or were not re-checked. They are marked NC/withdrawn from general knowledge, not from quotes.
- **2d106det training data:** undocumented by insightface.
- **SFace ONNX training dataset:** undocumented.
- **face_alignment `2DFAN4_1.6` checkpoints:** whether they were retrained on other data is undocumented.
- **ShareAlike on trained models:** whether CC BY-SA (VGGFace2) attaches to trained models is legally unsettled.
- **Speeds:** none were measured (no model runs allowed). The figures are the vendors' or READMEs' own.
