# Models and weights of the current clean pipeline (measured on r13-t, 03.11)

| Model | Size | Licence | Source (HF revision) | Used by | Status |
|---|---:|---|---|---|---|
| faster-whisper large-v3 | 3.09 GB | MIT | hf:Systran/faster-whisper-large-v3 @edaa852ec7 | ASR (6.1), TTS QA, subtitle checks | required |
| Qwen2.5-14B-Instruct GGUF Q5_K_M | 10.51 GB | Apache-2.0 | hf:Qwen/Qwen2.5-14B-Instruct-GGUF (*q5_k_m*) @b466e1f8c0 | translation / duration-aware rewrites / judge (6.4) | required |
| Qwen2.5-7B-Instruct GGUF Q8_0 | 8.1 GB | Apache-2.0 | hf:Qwen/Qwen2.5-7B-Instruct-GGUF @bb5d59e06d | NOT used any more (14B on the GPU since 29.10) | drop from the image (DUB_SKIP_QWEN7B=1) |
| Chatterbox Multilingual v3 | 3.21 GB | MIT | hf:ResembleAI/chatterbox (ve.pt, s3gen.pt, conds.pt, t3_mtl23ls_v3.safetensors, tokenizers) @5bb1f6ee58 | TTS (7.2), VC (7.4) | required |
| LatentSync 1.6 | 5.29 GB | OpenRAIL++ (user-approved) | hf:ByteDance/LatentSync-1.6 (latentsync_unet.pt, whisper/tiny.pt, auxiliary/syncnet_v2.model, auxiliary/sfd_face.pth) @c42c7e6c8e | lip-sync (8.1), SyncNet metric (12.1) | required |
| sd-vae-ft-mse | 0.33 GB | MIT | hf:stabilityai/sd-vae-ft-mse @31f26fdeee | LatentSync VAE | required |
| wav2vec2-base-960h | 0.38 GB | Apache-2.0 | hf:facebook/wav2vec2-base-960h @22aad52d43 | TTS QA (CTC), subtitle word timing (9.2), tempo audit | required |
| MediaPipe Face Landmarker | 0.0 GB | Apache-2.0 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task (sha256 pinned) | faces, head pose (5.4), crops | required |
| Spleeter 2stems | 0.16 GB | MIT | https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz (-> torch npz) | the Spleeter background bed | required |
| Real-ESRGAN x2plus | 0.07 GB | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth (sha256 pinned) | composite layer SR | required |
| CodeFormer + ParseNet | 0.46 GB | S-Lab License 1.0 = NON-COMMERCIAL | https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/{codeformer,parsing_parsenet}.pth | mouth detail (8.3), DUB_ALLOW_NC=1 | LICENCE DEVIATION, user-accepted 29.10 |
| htdemucs (demucs 4.0.1) | 0.08 GB | weights: not cleared for commercial use (treated as NC) | pip demucs==4.0.1 (code MIT) + its pretrained htdemucs checkpoint (dl.fbaipublicfiles.com) | the htdemucs (main) background bed, DUB_ALLOW_NC_BED=1 | LICENCE DEVIATION, user-accepted |
| pyannote speaker-diarization-3.1 | 0.0 GB | MIT | hf:pyannote/speaker-diarization-3.1 + segmentation-3.0 + wespeaker-voxceleb-resnet34-LM (GATED: HF token + accepted terms) | diarization (6.3) | required: pre-fetch into HF_HOME at build time (5.1: no download at start) |
| Silero VAD | 0.0 GB | MIT | pip silero-vad | VAD (5.2, 6.2 guard), speech gate | required |
| KeySync | 0.0 GB | Stability AI Community License (< $1M revenue, registration) | git https://github.com/antonibigata/keysync @5e2503b + ks_sm120.patch; weights ~13 GB (scripts/keysync/ on experiment/r13-keysync) | the alternative lip-sync generator (KeySync agent) | OPTIONAL; LICENCE DEVIATION, user-accepted |

Total of the required weights present here: **23.6 GB** (without the dropped Qwen 7B and the optional KeySync).
Per-file sha256 of the small directories: `models_manifest.json`.
