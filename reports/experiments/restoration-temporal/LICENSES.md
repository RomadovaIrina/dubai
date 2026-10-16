# License audit — track B (restoration / temporal stabilization)

Full evidence with a primary-source link for every claim: `LICENSES_desk_research.md` (desk research 2026-10-16, nothing installed).
Verdicts: **YES** = code + weights permissive, no NC component in the inference path; **YES\*** = code + weights permissive but the
training data is NC / unclear (FFHQ, WIDER FACE, FlyingChairs) — needs legal sign-off; **UNCLEAR**; **NO** = NC / research-only code or
weights. Not legal advice.

## Baseline dependencies this track evaluates or leans on

| model | repository | code license | weights license | commercial | problematic dependencies | replacement risk |
|---|---|---|---|---|---|---|
| **CodeFormer** (cand5 restoration, mouth scope) | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) @ b33cc7d | [NTU S-Lab 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE): non-commercial (CC BY-NC-SA 4.0 before 2022-09-14) | release v0.1.0, no separate terms -> repo license; README: commercial sites "use our models without any license or authorization" | **NO** | `parsing_parsenet.pth` = PSFRGAN (CC BY-NC-SA, CelebAMask-HQ NC); `detection_Resnet50_Final.pth` (MIT, WIDER FACE NC-ND); bundled yolov5-face (GPL-3.0), dlib 68 (NC); training data FFHQ (NC-SA) | HIGH — the baseline ships an NC model; route = authors / NTUitive, no public program |
| **InsightFace buffalo_l** (det_10g + 2d106det) | [deepinsight/insightface](https://github.com/deepinsight/insightface#license) | MIT | "ALL models ... non-commercial research purposes only" (auto-downloaded too); paid license via recognition-oss-pack@insightface.ai | **NO** without the paid license | used by LatentSync's detector (track A) AND by this track's CodeFormer 5-point alignment, paste-back mask bbox, `--ls-landmark-smooth` | HIGH — replace with MediaPipe / YuNet in the paste-back path, or buy the license |
| basicsr (vendored in CodeFormer) | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE.txt) | Apache-2.0, except `ops/fused_act`, `ops/upfirdn2d` (NVIDIA Source Code License-NC; not imported by CodeFormer / GFPGAN-clean) | – | YES (Apache parts) | – | low |
| facexlib | [xinntao/facexlib](https://github.com/xinntao/facexlib/blob/master/LICENSE) | MIT | per model: RetinaFace R50 UNCLEAR (WIDER FACE NC-ND), ParseNet NO, BiSeNet NO (CelebAMask-HQ) | code YES | the RetinaFace router of the baseline uses `detection_Resnet50_Final.pth` | medium |
| LatentSync 1.6 (recorded, track A) | [bytedance/LatentSync](https://github.com/bytedance/LatentSync) | Apache-2.0 | HF `ByteDance/LatentSync-1.6`: openrail++ (commercial with use restrictions) | YES with OpenRAIL use restrictions; its detector is NO (buffalo_l) | sd-vae-ft-mse (MIT), Whisper tiny (MIT) | not this track |

## Metric dependencies (this track's tools)

| component | use | code | weights | commercial | note |
|---|---|---|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` (sha256 8f2383e4…52fa4) | `mouth_jitter_hr.py` stabilisation landmarks | MIT ([opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)) | MIT | **YES\*** | WIDER FACE training data (CC BY-NC-ND) |
| OpenCV DIS optical flow | `mouth_jitter_hr.py` motion / residual | Apache-2.0 (opencv 4.9) | none | **YES** | no learned weights |
| InsightFace 2d106 (via LatentSync FaceDetector) | `face_temporal.py` (legacy metric) | MIT | NC | **NO** | diagnostic only; superseded by `mouth_jitter_hr.py` for this track |

## Candidates (Phase 2+)

| model | repository | code license | weights license | commercial | problematic dependencies | replacement risk / plan |
|---|---|---|---|---|---|---|
| **no restorer** (LatentSync raw in the mouth composite) | – | – | – | **YES** | – | softer mouth (sharpness 0.61 of the source) |
| **own temporal stabilisation** (DIS-flow motion-compensated filtering of the mouth ROI, window-boundary smoothing) | this repo | repo | none | **YES** | none (OpenCV DIS) | – |
| **GFPGAN v1.4** (clean arch) | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) | Apache-2.0 (+ listed third-party; NVIDIA-NC only for the v1 paper model's CUDA ops, not the clean v1.3/1.4) | release v1.3.0 `GFPGANv1.4.pth`, repo license | **YES\*** | FFHQ (NC-SA); `GFPGANer` would load facexlib RetinaFace + ParseNet (NO) -> bypassed: own aligned crops (`has_aligned`), own paste mask | install allowed (code + weights YES); FFHQ needs legal sign-off |
| **RestoreFormer++** | [wzhouxiff/RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus) | Apache-2.0 | repo license (inferred) | **YES\*** | FFHQ; facexlib helper (bypassed) | same |
| **RestoreFormer** | [wzhouxiff/RestoreFormer](https://github.com/wzhouxiff/RestoreFormer); weights also in GFPGAN release v1.3.4 | Apache-2.0 | repo license (inferred) | **YES\*** | FFHQ | same |
| **VQFR** | [TencentARC/VQFR](https://github.com/TencentARC/VQFR) | Apache-2.0 (+ opencv / ffmpeg / numpy notices) | Google Drive, repo license (inferred) | **YES\*** | FFHQ; facexlib detection / parsing (bypass) | same |
| MediaPipe Face Landmarker (478 pts) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | model card: Apache-2.0 | **YES** | trained on Google's consented data | candidate for the paste-back mask / alignment landmarks (own venv: its opencv-contrib dependency would shadow the pinned cv2) |
| BasicVSR++ / RealBasicVSR | [ckkelvinchan/BasicVSR_PlusPlus](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) | Apache-2.0 | Apache (inferred), REDS CC BY 4.0 | **YES** (SPyNet flow inside: Chairs/Things data risk) | mmcv | fallback temporal restorer on the lower-face crop |
| SEA-RAFT / NeuFlow v2 | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2) | BSD-3 / Apache-2.0 | BSD-3 / Apache | **YES\*** | FlyingChairs/Things ("any commercial use is prohibited") | only if DIS is too noisy |
| TAPIR / BootsTAPIR / TAPNext, LocoTrack | [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet), [cvlab-kaist/locotrack](https://github.com/cvlab-kaist/locotrack) | Apache-2.0 | Apache-2.0 | **YES** | Kubric synthetic data | point tracking if needed |
| KEEP, PGTFormer, DicFace, SVFR, Upscale-A-Video, MGLD-VSR | see appendix | NC | NC | **NO** — `RESEARCH_ONLY — cannot ship` | VFHQ (NC) | not planned |
| CoTracker v1-v3 | [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md) | CC BY-NC 4.0 | CC BY-NC 4.0 | **NO** | – | not used |
| PMRF, DiffBIR face, DAEFR, StableVSR | see appendix | – | – | **UNCLEAR** | PMRF embeds DifFace S-Lab weights; DAEFR no code license | needs the coordinator's go |
| GPEN, DifFace, ResShift, OSDFace | see appendix | none / S-Lab / NC | NC | **NO** | – | not planned |

Action needed (coordinator / user): (1) cand5's CodeFormer is non-commercial; (2) InsightFace buffalo_l is non-commercial and sits in
both the generator input path and this track's paste-back; (3) every permissive face restorer is trained on FFHQ (CC BY-NC-SA) -> legal
sign-off before shipping any of them.
