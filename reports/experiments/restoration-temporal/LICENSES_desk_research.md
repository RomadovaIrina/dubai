# Restoration and temporal stabilization: license audit and candidate shortlist

*Desk research only: web pages, LICENSE files and model cards. Nothing was installed, downloaded or run. Date: 2026-10-16.*
*Pipeline under review: LatentSync 1.6 lip-sync → CodeFormer on the mouth region → feathered lower-face composite.*

**This is not legal advice.** Each claim below links to the primary source it came from. Where a source could not be read, the entry says **UNCLEAR** and explains why. Key verdicts were spot-checked against the raw LICENSE file or the HF API on 2026-10-16: CodeFormer, KEEP, GFPGAN, OSDFace, GPEN, SeedVR, FlashVSR, CoTracker, TAPNet, SAM 2, DOVER/FAST-VQA, pyiqa, MuseTalk, PMRF, YuNet/SFace, dlib-models, FlyingChairs and all HF license fields cited.

## How to read the verdicts

Three separate questions are kept apart throughout:
- **Code:** the license on the source code.
- **Weights:** the license or terms on the released checkpoint. If no separate weights terms exist, the entry says "falls under the repo license (inferred)".
- **Data:** restrictions on the training dataset. Whether non-commercial (NC) dataset terms carry over to trained weights is legally unsettled, so it is flagged as a **risk**, not treated as a license.

| Commercial verdict | Meaning |
|---|---|
| **YES** | Code and weights are permissive, and no known NC component is in the inference path |
| **YES\*** | Code and weights are permissive, but the training data is NC or unclear (FFHQ, WIDER FACE, FlyingChairs/Things, …). Needs legal sign-off |
| **UNCLEAR** | The license is missing, contradictory, or an NC sub-component may be embedded |
| **NO** | An explicit non-commercial or research-only license applies to the code or the weights |

## TL;DR

1. **The current stack is not commercially clean.**
   - **CodeFormer:** code and weights are NC (S-Lab 1.0; CC BY-NC-SA 4.0 before 2022-09-14).
   - **CodeFormer's parser** (`parsing_parsenet.pth`) comes from PSFRGAN, which is CC BY-NC-SA, and was trained on CelebAMask-HQ (NC).
   - **LatentSync's face detector** (InsightFace `buffalo_l`: det_10g + 2d106det) is NC. InsightFace now sells a license (recognition-oss-pack@insightface.ai).
   - **LatentSync 1.6 itself:** Apache-2.0 code, OpenRAIL++ weights. Commercial use is OK with use restrictions.
2. **Commercial-safe frame restorers** (permissive code and weights; all trained on FFHQ, so they are YES\*): **GFPGAN v1.3/v1.4**, **RestoreFormer++**, **VQFR**. They must be fed our own aligned crops and composited with our own mask. Do **not** use facexlib's default RetinaFace and ParseNet path.
3. **No face-specific temporal restorer is commercially licensed.** KEEP, PGTFormer, DicFace, SVFR, Upscale-A-Video and MGLD-VSR are all NC, and VFHQ is NC.
   - Permissive temporal options are generic VSR models: **BasicVSR++ / RealBasicVSR** (Apache; REDS is CC BY 4.0), **SeedVR2** (Apache code and weights; heavy) and **FlashVSR** (Apache; Wan2.1 base).
   - Alternatively, build our own flow-guided smoothing.
4. **Flow and tracking:**
   - **OpenCV DIS** (Apache, no learned weights) carries zero weights risk.
   - **SEA-RAFT** (BSD-3) and **NeuFlow v2** (Apache) are YES\*: almost all learned flow is trained on FlyingChairs/Things, whose terms say "Any commercial use is prohibited".
   - **TAPIR / BootsTAPIR / TAPNext** have Apache code **and** checkpoints. **CoTracker v1–v3 is CC BY-NC 4.0** (NO).
5. **Landmarks and masks:** **MediaPipe Face Landmarker** has Apache weights, trained on Google's own consented data. **SAM 2.1** is Apache. **No face parser with a mouth class has clean weights**: they are all CelebAMask-HQ or LaPa (NC). Build mouth masks from MediaPipe lip landmarks, optionally refined with SAM 2.1.
6. **Identity metric:** no clean option. OpenCV **SFace** has Apache weights but unclear training data (YES\*). Otherwise, buy the InsightFace license.

## Part A — License audit of the current stack


### A1. CodeFormer (sczhou/CodeFormer)

| Item | Finding | Source |
|---|---|---|
| Code license | **S-Lab License 1.0** (Copyright 2022 S-Lab). "Redistribution and use for **non-commercial purpose** in source and binary forms ... are permitted". Last paragraph: "In the event that redistribution and/or use for commercial purpose ... is required, please contact the contributor(s) of the work." Local copy `dubai/third_party/CodeFormer/LICENSE` is identical. | https://github.com/sczhou/CodeFormer/blob/master/LICENSE |
| License history | Added 2022-09-14 (commit 0fb6cf1 "change license."). **Before that the README said CC BY-NC-SA 4.0.** Either way non-commercial. | https://github.com/sczhou/CodeFormer/commit/0fb6cf1693 |
| Weights (`codeformer.pth` and all v0.1.0 assets) | Release v0.1.0 published **2022-08-10**; release note only says "This release is mainly for storing pre-trained models". **No separate weights license** → weights are covered by the repo license (CC BY-NC-SA 4.0 at upload time, S-Lab 1.0 NC now). README "Ecosystem" section states third-party commercial sites "use our models **without any license or authorization**" → authors treat commercial use of the weights as unlicensed. | https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0 ; https://github.com/sczhou/CodeFormer#-ecosystem-applications--deployments |
| `detection_Resnet50_Final.pth` (in CodeFormer v0.1.0) | Byte size 109,497,761 = **identical size to facexlib v0.1.0 asset** of the same name, i.e. a re-host of the RetinaFace-R50 weights of biubug6/Pytorch_Retinaface (MIT code), trained on **WIDER FACE (CC BY-NC-ND)**. See A2. | GitHub API asset listing of both releases |
| `parsing_parsenet.pth` (in CodeFormer v0.1.0) | Byte size 85,331,193 = **identical to facexlib v0.2.2** `parsing_parsenet.pth`; facexlib's `parsenet.py` header: "Modified from https://github.com/chaofengc/PSFRGAN". Origin PSFRGAN (CC BY-NC-SA 4.0), trained on **CelebAMask-HQ** (non-commercial). See A2. | https://github.com/xinntao/facexlib/releases/tag/v0.2.2 |
| Other bundled code | `facelib/detection/yolov5face` copied from deepcam-cn/yolov5-face = **GPL-3.0** (only used if `--detection_model YOLOv5l/YOLOv5n`). dlib `shape_predictor_68_face_landmarks` also re-hosted (dlib-models README: trained on iBUG 300-W, not for commercial use — see Part B landmarks table). | https://github.com/deepcam-cn/yolov5-face (GPL-3.0) |
| Training data | FFHQ (CodeFormer paper/README) — FFHQ compilation CC BY-NC-SA 4.0 (see Part B note). | |
| Commercial route | **No published commercial-licensing program.** Only route: the license's "contact the contributor(s)" clause; README contact = shangchenzhou@gmail.com. S-Lab is NTU's lab; NTU guidance says code with commercial value is commercialised via **NTUitive** ("commercial use requires a commercial license issued by NTU"; contact https://www.ntuitive.sg/contact-us). Precedent: GFPGAN *removed* CodeFormer integration (2022-09-19, xinntao: "Due to the license of CodeFormer and the author's request, we remove codeformer in this repo"). | https://libguides.ntu.edu.sg/rdm/licensing_code ; https://github.com/TencentARC/GFPGAN/issues/266#issuecomment-1250520390 |
| **Verdict** | **Commercial: NO** (code + weights NC; parser weights NC; detector training data NC). Usable only as a research upper bound unless a license is negotiated with the authors/NTUitive. | |

### A2. basicsr and facexlib (and the weights they pull)

| Component | Code license | Weights origin & license | Training data | Commercial | Source |
|---|---|---|---|---|---|
| **BasicSR** (XPixelGroup/BasicSR) | **Apache-2.0** (`LICENSE.txt`). **But** `LICENSE/` folder carries third-party licenses: **LICENSE-NVIDIA = "Nvidia Source Code License-NC"** ("may be used ... non-commercially", i.e. research/evaluation only) covering the StyleGAN2 CUDA kernels `basicsr/ops/fused_act` and `basicsr/ops/upfirdn2d` (file headers: "This work is made available under the Nvidia Source Code License-NC"); also stylegan2-pytorch (MIT), SwinIR (Apache-2.0), timm (Apache-2.0). Last push 2024-07-21; PyPI 1.4.2. | n/a | n/a | **YES for Apache parts**; avoid/strip `ops/fused_act` + `ops/upfirdn2d` (CodeFormer's archs do not import them — checked locally) | https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE.txt ; https://github.com/XPixelGroup/BasicSR/tree/master/LICENSE |
| **facexlib** (xinntao/facexlib) | **MIT** (LICENSE file; note setup.py/PyPI metadata wrongly says "Apache License 2.0"). README: "we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use." Last push 2024-02-29; PyPI 0.3.0. | see rows below | | code YES; weights per row | https://github.com/xinntao/facexlib/blob/master/LICENSE ; https://github.com/xinntao/facexlib#readme |
| `detection_Resnet50_Final.pth` (facexlib v0.1.0 / CodeFormer v0.1.0) | biubug6/Pytorch_Retinaface: **MIT** (`LICENSE.MIT`, "Copyright (c) 2019") | Re-host of biubug6 `Resnet50_Final.pth` (facexlib README credits biubug6 for detection; original on Google Drive/Baidu). **No separate weights license** (falls under MIT by default — inferred). | **WIDER FACE — "Creative Common License (cc by-nc-nd)"** per official site; ImageNet-pretrained backbone. | **UNCLEAR** (MIT weights, NC-ND training data) — legally grey, commonly used commercially, but flag | https://github.com/biubug6/Pytorch_Retinaface ; http://shuoyang1213.me/WIDERFACE/ |
| `parsing_parsenet.pth` (facexlib v0.2.2) | facexlib `parsenet.py`: "Modified from https://github.com/chaofengc/PSFRGAN". **PSFRGAN = CC BY-NC-SA 4.0** (LICENSE: "PSFR-GAN is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License") | Most likely PSFRGAN's `parse_multi_iter_90000.pth` (v0.1.0 asset, 85,331,669 B vs facexlib 85,331,193 B — 476-byte difference consistent with re-serialisation; **not hash-verified**). Weights inherit CC BY-NC-SA 4.0. | **CelebAMask-HQ** (PSFRGAN README trains its parser with `--dataset celebahqmask`) → non-commercial research only | **NO** | https://github.com/chaofengc/PSFRGAN/blob/master/LICENSE ; https://github.com/chaofengc/PSFRGAN#train-the-model |
| `parsing_bisenet.pth` (facexlib v0.2.0; also in CodeFormer v0.1.0, 53,289,463 B) | zllrunning/face-parsing.PyTorch: **MIT** | facexlib README credits zllrunning; original `79999_iter.pth` on Google Drive. No separate weights license. | **CelebAMask-HQ** (zllrunning README: "Train the model using CelebAMask-HQ dataset") | **NO / high risk** (data is NC-research-only and forbids commercial exploitation of "any portion"); MIT label on weights does not cure the data restriction | https://github.com/zllrunning/face-parsing.PyTorch |
| `recognition_arcface_ir_se50.pth` (facexlib v0.1.0) | TreB1eN/InsightFace_Pytorch (MIT per facexlib table) | trained on MS1M-family (MS-Celeb-1M withdrawn by Microsoft) | MS1M | **NO / high risk** | facexlib README table |

**Dataset terms verified:**
- **CelebAMask-HQ**: "available for **non-commercial research purposes only**"; "You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes"; repo code: "use of this software is RESTRICTED to non-commercial research and educational purposes." No LICENSE file in repo. — https://github.com/switchablenorms/CelebAMask-HQ
- **CelebA** (parent of CelebAMask-HQ): "available for non-commercial research purposes only ... You agree not to ... exploit for any commercial purposes, any portion of the images and **any portion of derived data**." — https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **WIDER FACE**: "License: Creative Common License (cc by-nc-nd)". — http://shuoyang1213.me/WIDERFACE/

### A3. InsightFace `buffalo_l` (used by LatentSync `FaceDetector`)

LatentSync `latentsync/utils/face_detector.py` builds `FaceAnalysis(allowed_modules=["detection","landmark_2d_106"], root="checkpoints/auxiliary")` → loads `buffalo_l/det_10g.onnx` (SCRFD-10GF) + `buffalo_l/2d106det.onnx`. `requirements.txt` pins `insightface==0.7.3`.

| Item | Finding | Source |
|---|---|---|
| Library code | **MIT**: "The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage." | https://github.com/deepinsight/insightface#license |
| Pretrained models (incl. buffalo_l) | "The training data containing the annotation (and the models trained with these data) are available for **non-commercial research purposes only**. Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy". model_zoo README: "**ALL models are available for non-commercial research purposes only.**" python-package (insightface 2.x, 2026) docs: "The pretrained models provided with this library are available for non-commercial research purposes only, including both automatically and manually downloaded models"; new packages ship a signed `MODEL.LICENSE` — "public packages carry a non-commercial grant". | https://github.com/deepinsight/insightface#license ; https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md ; https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md |
| buffalo_l contents | SCRFD-10GF detector, ResNet50@WebFace600K recogniser, 2d106 & 3d68 landmarks, gender/age; 326 MB. | python-package/docs/model_zoo.md (Legacy model packs table) |
| Commercial route | **Exists**: README "2025-11-24 Update: ... For open-sourced face recognition models (e.g., buffalo_l package), please contact **recognition-oss-pack@insightface.ai** for licensing." (contact@insightface.ai for SDK/InspireFace). | https://github.com/deepinsight/insightface#license |
| **Verdict** | **Commercial: NO as-is** (det_10g + 2d106det are NC). Either buy a license from InsightFace or swap LatentSync's FaceDetector to a commercial-safe detector+landmarker (e.g. YuNet / MediaPipe; see Part B) — note LatentSync was trained with InsightFace-aligned crops, so a swap needs alignment-parity checks. | |

### A4. LatentSync 1.6 (recorded only)

| Item | Finding | Source |
|---|---|---|
| Code | **Apache-2.0** (GitHub license API + local `LICENSE`). Last commit 2025-06-20. | https://github.com/bytedance/LatentSync/blob/main/LICENSE |
| Weights `ByteDance/LatentSync-1.6` | HF card metadata `license: openrail++` (CreativeML Open RAIL++-M family: commercial use allowed subject to use-based restrictions in its Attachment A). No LICENSE file in the HF repo; same tag on LatentSync-1.5 and LatentSync (1.0). lastModified 2025-06-12. | https://huggingface.co/ByteDance/LatentSync-1.6 |
| Inference deps | `stabilityai/sd-vae-ft-mse` (HF license: **MIT**); Whisper `tiny.pt` (openai/whisper, MIT); **InsightFace buffalo_l (NC — see A3)**. HF repo also ships eval/aux files (`sfd_face.pth`, `syncnet_v2.model`, `i3d_torchscript.pt`, `koniq_pretrained.pkl`, `vgg16`, VideoMAE `vit_g_hybrid...ssv2`) — not audited individually. | https://huggingface.co/stabilityai/sd-vae-ft-mse |
| Training data | README mentions VoxCeleb2 and HDTF (SyncNet accuracy); not audited. | https://github.com/bytedance/LatentSync |

## Part B: candidate shortlist

"Last activity" is the last commit or push from the GitHub API or commits page, or the HF `lastModified` date.

### B1. Frame-level face restorers (CodeFormer replacements)

| Candidate | Code license | Weights (host and license) | Commercial | License-risk dependencies | Last activity | Type, speed and VRAM |
|---|---|---|---|---|---|---|
| **GFPGAN v1.3 / v1.4**: [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0 "except for the third-party components listed" ([LICENSE](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)). Lists basicsr (Apache), stylegan2-pytorch (MIT) plus the NVIDIA Source Code License-NC for the official StyleGAN2, and a DFDNet note (CC BY-NC-SA). The DFDNet note sits under the basicsr entry and most likely refers to `basicsr/archs/dfdnet_arch.py`, which GFPGAN inference does not use (inferred). | GitHub release v1.3.0 (`GFPGANv1.3.pth`, `GFPGANv1.4.pth`); no separate terms, so falls under the repo license (inferred) | **YES\*** for the "clean" v1.2–1.4 models, which per the README need no CUDA extensions. **UNCLEAR** for the original v1 paper model, whose StyleGAN2 CUDA ops are NVIDIA-NC | FFHQ (NC data). `GFPGANer` hardcodes facexlib `retinaface_resnet50` and `use_parse=True`, which loads `parsenet` (PSFRGAN, NC). **Bypass both**: pass aligned crops (`has_aligned`) and use our own paste mask | 2024-04-02 (v1.3.8, Sep 2022) | Per-frame, 512 px. No speed stated. Lightweight GAN |
| **RestoreFormer++**: [wzhouxiff/RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus) | Apache-2.0 (plain) | Google Drive; repo license (inferred) | **YES\*** | FFHQ; basicsr / realesrgan; facexlib helper | 2024-09-01 | Per-frame. No speed stated |
| **RestoreFormer**: [wzhouxiff/RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) | Apache-2.0 | Repo license (inferred); inference also ships in GFPGAN ≥1.3.7 | **YES\*** | FFHQ (inferred) | ~2023-09 | Per-frame |
| **VQFR**: [TencentARC/VQFR](https://github.com/TencentARC/VQFR) | Apache-2.0 with listed third-party components (opencv, ffmpeg, numpy; no NC clause found) | Google Drive; repo license (inferred) | **YES\*** | FFHQ; facexlib detection and parsing | 2022-12-15 | Per-frame |
| **FaceFusion ONNX hub** ([facefusion/models-3.0.0](https://huggingface.co/facefusion)) | FaceFusion itself is OpenRAIL-AS | FaceFusion's `face_enhancer/core.py` labels: gfpgan_1.2/1.3/1.4 = Apache-2.0, restoreformer_plus_plus = Apache-2.0, codeformer = S-Lab-1.0, gpen_bfr_* = Non-Commercial. HF repo has no license tag | Follows the upstream model (GFPGAN/RF++ YES\*) | Re-exported weights; check hashes against upstream | — | Convenient ONNX exports of the YES\* models |
| **InstantIR**: [instantX-research/InstantIR](https://github.com/instantX-research/InstantIR) | Apache-2.0 | [HF InstantX/InstantIR](https://huggingface.co/InstantX/InstantIR): apache-2.0 | **YES\*** with OpenRAIL++ use restrictions | SDXL base 1.0 (CreativeML Open RAIL++-M); DINOv2-large (Apache); trained on DIV2K, Flickr2K, LSDIR and FFHQ | 2024-11-14 | Per-frame, general (not face-specific) image restoration. Heavy SDXL; no VRAM stated |
| **PMRF**: [ohayonguy/PMRF](https://github.com/ohayonguy/PMRF) | MIT | [HF ohayonguy/PMRF_blind_face_image_restoration](https://huggingface.co/ohayonguy/PMRF_blind_face_image_restoration): mit | **UNCLEAR**. The README says "the MMSE model is also in the PMRF checkpoint", and that MMSE model is DifFace's `swinir_restoration512_L1.pth` (S-Lab NC). So the MIT checkpoint likely embeds NC-licensed weights | FFHQ; HF demo uses facexlib and RealESRGAN | 2025-02-05 | Per-frame; 25 flow steps, ~0.2B params |
| **DiffBIR**: [XPixelGroup/DiffBIR](https://github.com/XPixelGroup/DiffBIR) | Apache-2.0 | [HF lxq007/DiffBIR-v2](https://huggingface.co/lxq007/DiffBIR-v2): apache-2.0 | **UNCLEAR** | SD 2.1 base (OpenRAIL++; the Stability HF repo now returns 401, which hurts reproducibility). The face stage-1 `swinir_face` comes from DifFace (S-Lab NC). FFHQ; facexlib | 2025-07-29 | Per-frame (no video mode); 50 steps; slow |
| **DAEFR**: [LIAGM/DAEFR](https://github.com/LIAGM/DAEFR) | **No license** (all rights reserved by default) | [HF LIAGM/DAEFR_pretrain_model](https://huggingface.co/LIAGM/DAEFR_pretrain_model): apache-2.0 | **UNCLEAR** (Apache weights, but no code grant) | FFHQ; basicsr; taming | 2024-10-09 | Per-frame |
| OSEDiff: [cswry/OSEDiff](https://github.com/cswry/OSEDiff) | Apache-2.0 | Face checkpoint `osediff_face.pkl` (FFHQ); weights terms not stated | UNCLEAR → candidate | SD 2.1; RAM/DAPE taggers | — | One-step; ~0.10 s/img on A100 |
| DR2: [Kaldwin0106/DR2](https://github.com/Kaldwin0106/DR2) | MIT | Baidu; no license | UNCLEAR | FFHQ diffusion | — | Per-frame, diffusion |
| GPEN: [yangxy/GPEN](https://github.com/yangxy/GPEN) | **No LICENSE** (raw `/LICENSE` returns 404). The README line "For academic and non-commercial use only" was removed on 2026-03-10. README still says models were taken down "due to commercial issues" | Aliyun / ModelScope; no license | **NO** (no grant; previously NC) | RetinaFace-R50, ParseNet; FFHQ | 2026-03-10 | Per-frame |
| DifFace / ResShift: [zsyOAOA/DifFace](https://github.com/zsyOAOA/DifFace), [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) | S-Lab License 1.0 | GitHub releases (S-Lab) | **NO** | FFHQ | 2025-01-25 / 2026-07-08 | Per-frame; ResShift uses 4 or 15 steps |
| OSDFace: [jkwang28/OSDFace](https://github.com/jkwang28/OSDFace) | "OSDFace Non-Commercial Academic Research License v1.0" (added 2026-10-14). Explicitly covers "Pretrained model weights and checkpoints" | Google Drive / SharePoint (NC) | **NO** | SD 2.1; FFHQ | 2026-10-14 | One-step |
| PGDiff (S-Lab), InterLCM (CC BY-NC 4.0), DMDNet (CC BY-NC-SA), PSFRGAN (CC BY-NC-SA), DT-BFR (CC BY-NC-SA), HiFaceGAN ("academic and educational use only"), HYPIR ("Non-Commercial Use Only"), RestorerID (Apache code, but needs IP-Adapter-FaceID and InsightFace, both NC) | — | — | **NO** | — | — | — |
| FaithDiff (MIT; SDXL/RealVisXL), TSD-SR (Apache; SD3-Medium base) | — | HF | YES\* subject to base-model license | General super-resolution, not face-specific | — | — |

### B2. Video and temporal restoration

| Candidate | Code license | Weights | Commercial | License-risk dependencies and training data | Last activity | Temporal mechanism, speed and VRAM |
|---|---|---|---|---|---|---|
| **KEEP**: [jnjaby/KEEP](https://github.com/jnjaby/KEEP) | **S-Lab License 1.0** ("Copyright 2024 S-Lab", NC). Confirmed from [LICENSE](https://github.com/jnjaby/KEEP/blob/main/LICENSE) | GitHub release V1.0.0 (S-Lab) | **NO** | CodeFormer-based; `facelib/yolov5face` (GPL-3.0); VFHQ | 2025-05-14 | Kalman-inspired recurrent feature propagation. No VRAM stated |
| **PGTFormer**: [kepengxu/PGTFormer](https://github.com/kepengxu/PGTFormer) | BSD-style with an NC clause; commercial use via Xidian University | HF kepeng/pgtformer-base (no license field), GDrive, Baidu | **NO** | VFHQ; dlib | 2025-03-21 | Parsing-guided temporal transformer; no pre-alignment needed |
| **DicFace**: [fudan-generative-vision/DicFace](https://github.com/fudan-generative-vision/DicFace) | README says NTU S-Lab 1.0; there is no LICENSE file | HF fudan-generative-ai/DicFace_model says "mit", which conflicts with the README | **NO** (treat as S-Lab; it bundles codeformer.pth) | VFHQ | 2025-07-25 | Dirichlet-constrained codebook plus a spatio-temporal transformer |
| **SVFR**: wangzhiyaoo/SVFR | MIT code, but "pretrained models … non-commercial research purposes only" | HF (401 when fetched) | **NO** | SVD-xt (Stability community license); insightface_glint360k | — | Video diffusion; ≥16 GB |
| **Upscale-A-Video**: [sczhou/Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video) | S-Lab License 1.0 ("For the commercial use of the code, please consult Dr. Shangchen Zhou") | GDrive (S-Lab) | **NO** | SD-x4-upscaler-style latent diffusion model (inferred); optional LLaVA | 2024-09-27 | Temporal layers plus flow-guided recurrent latent propagation |
| MGLD-VSR (S-Lab 1.0), VRT / RVRT (CC BY-NC 4.0) | — | — | **NO** | — | — | — |
| **BasicVSR++**: [ckkelvinchan/BasicVSR_PlusPlus](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) / [open-mmlab/mmagic](https://github.com/open-mmlab/mmagic) | Apache-2.0 (both) | download.openmmlab.com; no separate terms, so Apache (inferred) | **YES** (REDS is CC BY 4.0; Vimeo-90K has no stated license) | SPyNet flow inside (trained on Sintel/Chairs: data risk); mmcv | 2023-12-27 / mmagic 2024-08-06 | Bidirectional recurrent propagation plus flow-guided deformable alignment. Generic VSR, not face-specific |
| **RealBasicVSR**: [ckkelvinchan/RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) | Apache-2.0 | Dropbox/GDrive; Apache (inferred) | **YES** (REDS) | mmcv-full, mmedit | 2023-06-05 | Recurrent propagation plus an image-cleaning module (real-world degradations) |
| **SeedVR2**: [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) | Apache-2.0 (LICENSE verified) | [HF SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) and [-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B): apache-2.0 | **YES** (training data not disclosed) | — | 2026-01-27 | One-step diffusion transformer with window attention. 1×H100-80GB handles ~100 frames at 720p. Heavy |
| **FlashVSR**: [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) | Apache-2.0 (LICENSE verified) | [HF JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1): apache-2.0 | **YES** | Wan2.1-T2V-1.3B base (Apache-2.0); VSR-120K dataset not released | 2026-10-16 | Streaming, block-sparse attention; ~17 FPS at 768×1408 on A100 |
| IART, RealViformer | MIT | GDrive; MIT (inferred) | **YES** (REDS) | — | — | Generic VSR |
| UltraVSR / MIA-VSR | Apache-2.0 | Weights not checked | YES\* | UltraVSR uses SD 2.1-base | — | — |
| StableVSR: [claudiom4sir/StableVSR](https://github.com/claudiom4sir/StableVSR) | MIT | HF claudiom4sir/StableVSR: **no license field** | **UNCLEAR** | SD-x4-upscaler (OpenRAIL++; card says "intended for research purposes only") | 2025-02-07 | Temporal conditioning module; ~14.5 GB |
| Vivid-VR: [csbhr/Vivid-VR](https://github.com/csbhr/Vivid-VR) | **No LICENSE** | HF csbhr/Vivid-VR: apache-2.0 | **UNCLEAR** | **Requires CogVideoX1.5-5B** (CogVideoX License: registration for a basic commercial license, <1M visits per month) plus CogVLM2/Llama3 | 2026-02-04 | DiT, 121-frame windows; ~43 GB (~25 GB with offload) |
| DOVE (Apache; CogVideoX base), STAR (no LICENSE; I2VGen-XL is NC), VEnhancer (no license), DiffVSR (no license) | — | — | **UNCLEAR / NO** | — | — | — |
| DVFace: [zhengchen1999/DVFace](https://github.com/zhengchen1999/DVFace) (2026) | MIT | Not released yet | Watch | Wan2.1 base | 2026-04-17 | One-step video face restoration with a spatio-temporal dual codebook |
| "TDM" | No face-specific TDM code found. arXiv 2501.02269 is an all-in-one video restoration method (SD plus ControlNet) | — | UNCLEAR | — | — | — |

**Lip-sync-specific post-processing.** None of the tools checked ships a dedicated temporal enhancer for the mouth or teeth.
- **VideoRetalking:** code is Apache-2.0, weights are unlicensed. Its enhancement step relies on GPEN and GFPGAN. **UNCLEAR**.
- **MuseTalk:** code is MIT, and the README says "The trained model are available for any purpose, even commercially". Its blending uses a BiSeNet parser, which was trained on CelebAMask-HQ (NC). Relevant as a commercially licensed lip-sync reference.
- **Wav2Lip:** non-commercial and trained on LRS2. **NO**.
- **Diff2Lip:** CC BY-NC 4.0. **NO**.
- **DINet:** no license. **NO**.

### B3. Optical flow

Training-data caveat for every learned flow model:
- **FlyingChairs:** "The dataset is provided for research purposes only … Any commercial use is prohibited" ([page](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)). FlyingThings3D has the same terms, and its FAQ says commercial licenses are "not possible".
- **KITTI:** CC BY-NC-SA 3.0.
- **Sintel and HD1K:** terms not found (UNCLEAR).

| Candidate | Code | Weights | Commercial | Last activity | Speed |
|---|---|---|---|---|---|
| **OpenCV DIS / Farneback** | Apache-2.0 (opencv 4.x) | none | **YES** (no weights risk) | — | Real-time on CPU |
| **SEA-RAFT**: [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | BSD-3 (verified) | [HF MemorySlices/*](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M): bsd-3-clause | **YES\*** (Chairs/Things/KITTI/TartanAir/Spring) | 2026-03-26 | "≥2.3× faster" than prior methods |
| **NeuFlow v2**: [neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2) | Apache-2.0 (verified) | In repo; Apache (inferred) | **YES\*** | 2024-09-02 | >20 FPS at 512×384 on Jetson Orin Nano |
| RAFT: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD-3 | No separate license | **YES\*** | 2025-08-24 | — |
| torchvision `raft_large` / `raft_small` | BSD-3 | download.pytorch.org. The docs say weights "may have their own licenses or terms … derived from the dataset … your responsibility". `C_T_*` = Chairs+Things only; the default `C_T_SKHT_V2` adds Sintel, KITTI and HD1K | **YES\*** | — | large: 5.3M params |
| MemFlow (Apache-2.0), GMFlow (Apache-2.0), UniMatch (MIT), FlowFormer / FlowFormer++ / VideoFlow (Apache) | permissive | Releases, GDrive or S3; no separate terms | **YES\*** | 2025-10 / 2024-02 / 2025-01 | GMFlow ~26 ms on A100 |
| ptlflow (RAPIDFlow, DPFlow weights) | Apache-2.0 | Weights are CC BY-NC-SA 4.0 / "academic and research purposes only" | **NO** (weights) | 2026-10-11 | — |

### B4. Point tracking (mouth-ROI tracking)

| Candidate | Code | Weights | Commercial | Last activity |
|---|---|---|---|---|
| **TAPIR / BootsTAPIR / TAPNext**: [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) | Apache-2.0 | README: "All pre-trained model checkpoints released in this repository … (including TAP-Net, TAPIR, BootsTAPIR, TAPNext, TAPNext++, and TRAJAN weights) are also licensed under Apache 2.0" | **YES** (training data is Kubric synthetic; BootsTAPIR's real-video self-training data was not verified) | 2026-09-15 |
| **LocoTrack**: [cvlab-kaist/locotrack](https://github.com/cvlab-kaist/locotrack) | Apache-2.0 (verified) | HF hamacojr/LocoTrack-pytorch-weights: apache-2.0 | **YES** (Kubric) | 2025-04-16; "6× faster" |
| AllTracker: aharley/alltracker | MIT | HF: MIT | YES\* (trained on flow datasets too) | 2025 |
| **CoTracker v1 / v2 / v3**: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | **CC BY-NC 4.0** ([LICENSE.md](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md)). README: "The majority of CoTracker is licensed under CC-BY-NC". Only the Particle Video (MIT) and TAP-Vid/LocoTrack (Apache) portions are exempt | [HF facebook/cotracker3](https://huggingface.co/facebook/cotracker3): cc-by-nc-4.0 (same for facebook/cotracker) | **NO** (all versions) | 2025-01-21 |
| TAPTR (IDEA License, NC), SpatialTracker (CC BY-NC 4.0), DELTA (Snap NC) | — | — | **NO** | — |
| Track-On | MIT | DINOv3 backbone has a separate gated license (not read) | UNCLEAR | — |

### B5. Face detection and landmarks

| Candidate | Code | Weights | Commercial | Data and notes |
|---|---|---|---|---|
| **MediaPipe Face Detector (BlazeFace) / Face Landmarker (Face Mesh V2, 478 pts)**: [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | Model card PDFs: "LICENSED UNDER Apache License, Version 2.0" | **YES** | Trained on Google's own consented smartphone images. Out of scope per the card: surveillance and identity recognition. BlazeFace: 2.9 ms CPU (Pixel 6) |
| **YuNet**: [opencv_zoo face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) / libfacedetection | MIT ("All files in this directory are licensed under MIT License"); libfacedetection is BSD-3 | MIT (.onnx in the folder) | **YES\*** (WIDER FACE = CC BY-NC-ND) | Detects faces of roughly **10–300 px**, so downscale close-ups. ~6.5 ms at 640×480 on CPU |
| biubug6/Pytorch_Retinaface | MIT | No separate license | **YES\* / UNCLEAR** | WIDER FACE |
| SCRFD (insightface) | Apache-2.0 / MIT | InsightFace NC | **NO** | — |
| face-alignment ([1adrianb](https://github.com/1adrianb/face-alignment)) | BSD-3 | Not stated | **NO / UNCLEAR** | 300W-LP / LS3D-W. [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/): "Commercial use (i.e., use in training commercial algorithms) is not allowed." Default SFD detector is slow on CPU |
| SPIGA | BSD-3 | Not stated | **NO** | WFLW / 300W / COFW / MERLRAV |
| dlib 68-point (`shape_predictor_68`, also re-hosted in CodeFormer's release) | Boost / CC0 | [dlib-models](https://github.com/davisking/dlib-models): "The license for this dataset excludes commercial use … the trained model therefore can't be used in a commercial product" | **NO** | iBUG 300-W. The 5-point model is OK |
| YOLOv5-face / YOLOv8-face | GPL-3.0 (AGPL for Ultralytics) | — | **NO** for closed source | yolov5face code is inside CodeFormer's and KEEP's `facelib` |
| uniface | MIT | Per model (marks SCRFD, 2d106det and ArcFace as NC) | Per model | — |

### B6. Face parsing and mouth masks

| Candidate | Code | Weights | Commercial | Data |
|---|---|---|---|---|
| **SAM 2 / 2.1**: [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Apache-2.0 (plus BSD-3 `cc_torch`) | "SAM 2 model checkpoints … licensed under Apache 2.0" | **YES** | SA-V (the dataset license governs the dataset only). Tiny model ~91 FPS on A100; video propagation built in |
| SAM v1 | Apache-2.0 | Apache-2.0 | **YES** | SA-1B research license (dataset only) |
| SAM 3 / 3.1 | Custom "SAM License" | Same (gated on HF) | **YES with conditions** (no military/ITAR use; ship the license text) | — |
| MobileSAM / EfficientSAM | Apache-2.0 | Not stated separately | Probably YES / UNCLEAR | Distilled on SA-1B |
| **MediaPipe Face Mesh lip landmarks → polygon** | Apache-2.0 | Apache-2.0 | **YES** | Recommended mouth/teeth mask source |
| MediaPipe SelfieMulticlass | Apache-2.0 | Apache-2.0 | YES | No lip or mouth class (only hair, body-skin, face-skin, clothes, other) |
| BiSeNet `79999_iter` (zllrunning) / facexlib `parsing_bisenet` | MIT | Not stated | **NO** | CelebAMask-HQ (NC) |
| facexlib / PSFRGAN `parsing_parsenet` | CC BY-NC-SA 4.0 | — | **NO** | CelebAMask-HQ |
| yakhyo/face-parsing; jonathandinu/face-parsing (HF card: "non-commercial research and educational purposes"); SegFace (HF tag "mit", but trained on CelebAMask-HQ, LaPa and Helen); facer / FaRL (LaPa, CelebAMask-HQ); DML-CSR (InsightFace NC) | — | — | **NO** | LaPa: "freely available … for non-commercial purposes" |

### B7. Identity metric

| Candidate | Code | Weights | Commercial | Data |
|---|---|---|---|---|
| ArcFace via InsightFace (`buffalo_l` w600k_r50, `antelopev2` glintr100) | MIT | NC (see A3); paid license available | **NO** without a license | WebFace600K / Glint360K |
| **OpenCV SFace** ([opencv_zoo face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)) | — | "All files in this directory are licensed under Apache 2.0 License" | **YES\*** | Zoo does not state the training set; the SFace paper uses CASIA-WebFace, VGGFace2 and MS-Celeb-1M (risk) |
| dlib ResNet face recognition | Boost | CC0 | YES\* | FaceScrub, VGG, scraped images |
| facenet-pytorch (VGGFace2 / CASIA weights) | MIT | Not stated | **UNCLEAR** | The VGGFace2 site lists no terms ("download links … no longer available"). CASIA terms not found |
| AdaFace, CVLFace, GhostFaceNets, TransFace | MIT / none | "Follow the license of the training dataset" | **NO / UNCLEAR** | MS1M (withdrawn 2019) / WebFace (academic only) |
| EdgeFace | BSD-3 on GitHub | HF Idiap card: **CC BY-NC-SA 4.0** | **NO** | WebFace260M |
| DeepFace (serengil) | MIT | "License types will be inherited" per model: VGG-Face is CC BY-NC (NO); ArcFace follows InsightFace (NO) | Per model | — |

**Note:** an identity metric used only internally for QA is lower risk than shipping a model. However, NC licenses (S-Lab, CC-NC, PolyForm-NC) generally treat R&D inside a for-profit company as commercial use, so get legal sign-off even for metrics.

### B8. Temporal and perceptual metrics

| Metric | Code | Model deps | Commercial |
|---|---|---|---|
| **Warping error** (reimplement: warp t-1→t with DIS or SEA-RAFT flow, add an occlusion mask, take masked L1, restricted to the mouth ROI) | own code | flow model (see B3) | **YES** with DIS; YES\* with a learned flow |
| Reference implementation [fast_blind_video_consistency](https://github.com/phoenix104104/fast_blind_video_consistency) | MIT, but the LICENSE also contains a CC BY-NC 4.0 EdgeConnect section | Uses FlowNet2 (license unread) | UNCLEAR, so reimplement instead |
| **LPIPS**: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | BSD-2 | Linear heads ship in the repo; the backbone uses torchvision ImageNet AlexNet/VGG (torchvision disclaimer) | **YES** (low risk) |
| **tLPIPS / tOF** (TecoGAN) | Apache-2.0 | LPIPS | **YES** |
| **Flicker** (VBench `temporal_flickering` = frame MAE, no model) | Apache-2.0 | none | **YES** |
| VBench `motion_smoothness` | Apache-2.0 | AMT `amt-s.pth` (CC BY-NC 4.0) | **NO** |
| FVD (google-research) | Apache-2.0 | I3D from TF-Hub (license unread) | Code YES; model UNCLEAR. Metric only |
| FVD (stylegan-v) | No license; README says "likely restricted by the NVidia license" | — | NO / UNCLEAR |
| **DOVER, FAST-VQA / FasterVQA** | **S-Lab License 1.0** (verified; changed from MIT in Aug 2024: "Sadly we have to change to non-commercial license") | LSVQ | **NO** |
| **pyiqa / IQA-PyTorch** (MUSIQ, CLIP-IQA, …) | **PolyForm Noncommercial 1.0.0** (verified) plus S-Lab parts | HF | **NO** |
| SyncNet / LSE-C/D (lip-sync quality; LatentSync ships `syncnet_v2.model`) | not audited | not audited | UNCLEAR (not audited) |

### B9. Training-dataset terms referenced above

| Dataset | Terms (quoted or summarized) | Source |
|---|---|---|
| FFHQ | Images are CC BY 2.0, BY-NC 2.0, PD or CC0 ("free use … for non-commercial purposes"); the dataset compilation is **CC BY-NC-SA 4.0**; "not intended for … facial recognition" | https://github.com/NVlabs/ffhq-dataset |
| CelebA / CelebAMask-HQ | "non-commercial research purposes only … not … exploit for any commercial purposes, any portion of the images and any portion of derived data" | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html ; https://github.com/switchablenorms/CelebAMask-HQ |
| WIDER FACE | "Creative Common License (cc by-nc-nd)" | http://shuoyang1213.me/WIDERFACE/ |
| VFHQ | "only available to download for non-commercial research purposes" | https://liangbinxie.github.io/projects/vfhq/ |
| CelebV-HQ | "non-commercial research purposes only" | https://github.com/CelebV-HQ/CelebV-HQ |
| REDS | CC BY 4.0 | https://seungjunnah.github.io/Datasets/reds.html |
| FlyingChairs / FlyingThings3D | "research purposes only … Any commercial use is prohibited" | https://lmb.informatik.uni-freiburg.de/resources/datasets/ |
| KITTI | CC BY-NC-SA 3.0 | https://www.cvlibs.net/datasets/kitti/ |
| 300-W | "Commercial use … is not allowed" | https://ibug.doc.ic.ac.uk/resources/300-W/ |
| LaPa | "non-commercial purposes" | https://github.com/JDAI-CV/lapa-dataset |
| WebFace260M | academic research only (seen only via a search snippet; TLS error on direct fetch: UNCLEAR) | — |
| MS-Celeb-1M | Withdrawn by Microsoft in 2019 | press coverage |

## Recommended to test first (commercial-safe)

Test in this order. All need legal sign-off where marked \*.

1. **Remove facexlib and InsightFace from our own post-processing path.**
   - Use **MediaPipe Face Landmarker** (Apache weights, consented data) for per-frame mouth/jaw landmarks, ROI tracking and the **lip/teeth composite mask**. Optionally refine the mask with **SAM 2.1** (Apache), prompted by lip landmarks; it also propagates masks through video.
   - Use **YuNet** (MIT\*) as a fast backup detector.
   - For LatentSync's own detector: either license `buffalo_l` from InsightFace, or swap in MediaPipe/YuNet. A swap needs alignment-parity validation, because LatentSync was trained on InsightFace-aligned crops.
2. **Frame restorer A/B against CodeFormer.**
   - Candidates: **GFPGAN v1.4\***, **RestoreFormer++\*** and **VQFR\*** (all Apache code and weights; FFHQ data).
   - Feed them our own 512 aligned crops with `has_aligned=True`, and paste with our own feathered mouth mask. Do **not** call facexlib's `retinaface_resnet50` or `parsenet`.
   - FaceFusion's ONNX exports of GFPGAN 1.4 and RestoreFormer++ are a convenient runtime.
3. **Temporal stabilization, cheap path.**
   - Compute flow on the mouth ROI with **OpenCV DIS** (Apache, no weights).
   - Apply flow-guided temporal blending/EMA of the restored mouth, with an occlusion mask. Smooth the landmarks with a One-Euro or Kalman filter (own code; KEEP's Kalman idea is reimplementable from the paper, not its code).
   - Upgrade the flow to **SEA-RAFT\*** (BSD-3 weights) or **NeuFlow v2\*** (Apache) if DIS is too noisy.
   - For tracking, use **TAPIR / BootsTAPIR / TAPNext** or **LocoTrack** (Apache code and weights), **not CoTracker**.
4. **Temporal stabilization, learned VSR path on the lower-face crop.**
   - **BasicVSR++ / RealBasicVSR** (Apache; REDS CC BY 4.0: cleanest data provenance in this list; light).
   - **FlashVSR** (Apache; Wan2.1 base; ~17 FPS on A100).
   - **SeedVR2-3B** (Apache; heavy, quality upper bound among permissive options).
   - These are generic VSR models, not face priors. Expect less hallucinated detail than CodeFormer but better temporal consistency.
5. **Metrics** (all reimplementable with permissive parts):
   - Masked warping error (DIS or SEA-RAFT\*), LPIPS (BSD-2), tLPIPS/tOF (TecoGAN, Apache) and frame-MAE flicker (VBench formula).
   - Identity: **OpenCV SFace\*** (Apache weights), or InsightFace with a purchased license.
   - Avoid DOVER, FAST-VQA, pyiqa, VBench motion_smoothness and CoTracker-based metrics in anything shipped.

## Research-only: upper-bound comparisons only (do not ship)

- **Face restorers:** CodeFormer (incl. its RetinaFace/ParseNet/dlib/YOLOv5-face bundle), DifFace, ResShift, OSDFace, PGDiff, InterLCM, DMDNet, PSFRGAN, DT-BFR, HYPIR, GPEN (no license; previously NC), RestorerID (via IP-Adapter-FaceID).
- **Video face restoration:** KEEP, PGTFormer, DicFace, SVFR, Upscale-A-Video, MGLD-VSR, VRT/RVRT.
- **Tracking and flow:** CoTracker v1/v2/v3, TAPTR, SpatialTracker, DELTA, ptlflow RAPIDFlow/DPFlow weights.
- **Landmarks, parsing and identity:** InsightFace buffalo_l / antelopev2 / SCRFD / 2d106det / DML-CSR (unless licensed), face-alignment (300W-LP), SPIGA, dlib 68-point, every CelebAMask-HQ or LaPa face parser (BiSeNet, ParseNet, SegFace, jonathandinu, facer), EdgeFace, AdaFace/CVLFace, VGG-Face.
- **Metrics:** DOVER, FAST-VQA, pyiqa, VBench motion_smoothness (AMT).
- **Grey zone (UNCLEAR, needs a license read or author contact before any use):** PMRF (MIT checkpoint, but it embeds DifFace's S-Lab SwinIR), DiffBIR face mode (DifFace SwinIR-face, SD 2.1), DAEFR (no code license), StableVSR, Vivid-VR / DOVE / STAR (CogVideoX or I2VGen-XL bases), VEnhancer, DR2, OSEDiff (weights terms), facenet-pytorch.

## Could not verify (UNCLEAR items)

- **Hash identity of re-hosted weights.** CodeFormer's `parsing_parsenet.pth` vs PSFRGAN's `parse_multi_iter_90000.pth` was compared by byte size only (85,331,193 vs 85,331,669). The facexlib re-hosts of the RetinaFace and BiSeNet weights are matched to the originals by name and size only.
- **Licenses not read:** the SD 2.1 license text (Stability HF repo returns 401; GitHub repo 404), the SVD and SD3 licenses, Glint360K and WebFace260M primary terms, CASIA-WebFace terms, and the LS3D-W terms (behind a form).
- **Unreachable pages:** the WFLW terms page (404), Sintel and HD1K terms, the HF pages for SVFR and DicFace (401), and the TF-Hub I3D license.
- **Gaps in this audit:** last-commit dates for some repos (GitHub API rate limit), and the full file scope of GFPGAN's "DFDNet" CC BY-NC-SA note (inferred to be BasicSR's `dfdnet_arch.py`).
- **Commercial licensing routes.** CodeFormer has no public commercial program; the only route is the license's "contact the contributor(s)" clause. KEEP, Upscale-A-Video and DOVER use the same S-Lab terms. NTU's route is NTUitive.
