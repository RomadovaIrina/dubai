# Licences of the CURRENT production path (r15 T, 05.11.2026)

This covers everything a quality-mode job loads today on the research servers: the eye-approved q2 + F3 + c8 face, the user-accepted audio chain and the mux. It builds on two earlier audits:
- `coordination/LICENSES_audio.md` (17.10, audio);
- `r8f_restorer_licences.md` (29.10, restorers).

The licence texts were checked there. This file maps the result onto what the pipeline actually loads now, with the r15 checks:
- which weight files are loaded;
- a pip-metadata scan of all 225 packages.

**Verdicts:**

| Verdict | Meaning |
|---|---|
| OK | A permissive licence. |
| OK\*(U) | A permissive licence; the training data is undisclosed. |
| GREY | A permissive licence or none, but some training data is research-only / non-commercial. |
| NC | The licence itself forbids commercial use. |
| NO | No weights licence, and the training data is non-commercial. |

The **Profile** column says which profile loads the component: `current` (the servers today), `clean` (the image option), or both.

## Non-commercial or unclear: the items to decide
| # | Component (where) | Licence | Verdict | Source | Profile |
|---|---|---|---|---|---|
| 1 | **CodeFormer**: `codeformer.pth` + code (face_gen `--codeformer optimized`, composite CF detail layer `CFD=self`) | S-Lab License 1.0, non-commercial (code AND weights) | **NC** (user accepted the risk 29.10) | https://github.com/sczhou/CodeFormer (release v0.1.0) | current |
| 2 | **htdemucs** weights `955717e8-8726e21a.th` (demucs 4.0.1): the MAIN background bed (the "demucs" output) | Code MIT. The weights state no licence. Trained on MUSDB18-HQ ("Other (Non-Commercial)") + Meta-internal songs | **NO** (audio audit 17.10; user-accepted as a deviation) | https://github.com/facebookresearch/demucs ; https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th | current |
| 3 | ParseNet `parsing_parsenet.pth` (CodeFormer face parser) | CodeFormer release (NC lineage) | **NC**, but **NOT loaded**: `--codeformer-parse off` in production. On the servers' disk only | CodeFormer release v0.1.0 | – |
| 4 | pyannote speaker-diarization-3.1 + segmentation-3.0 (MIT) + wespeaker-voxceleb-resnet34-LM (CC BY 4.0) | MIT / CC BY 4.0 (attribution). Some segmentation training sets are research-only (MSDWild, …). Gated (accept terms) | **GREY** (audit: "YES\*(NC)") | https://huggingface.co/pyannote/speaker-diarization-3.1 | both |
| 5 | Spleeter 2stems: the second bed in current, the ONLY bed in clean | Code MIT. The README states no weights licence or training data | **GREY** (the cleanest separator we have; audit: "confirm") | https://github.com/deezer/spleeter (release v1.4.0) | both |
| 6 | Real-ESRGAN x2plus: composite layer SR | BSD-3 code + weights. DF2K + OST training data; DIV2K is "academic research purpose only" | **GREY** | https://github.com/xinntao/Real-ESRGAN (release v0.2.1) | both |
| 7 | LatentSync 1.6 (the lip-sync generator; also the student LoRA of mode `fast`, a derivative) | CreativeML OpenRAIL++-M: commercial use allowed, but the use restrictions must be passed on to users (ToS). Training set not named | **OK with use restrictions** (user-approved) | https://huggingface.co/ByteDance/LatentSync-1.6 | both |
| 8 | ffmpeg 6.1.1 (Ubuntu build, `--enable-gpl` with libx264) | GPL | Calling a separate CLI in a SaaS is fine. **Shipping the image to third parties** = GPL source obligations. H.264 encoding at scale: a patent-pool (Via LA) question for counsel | Ubuntu 24.04 `ffmpeg` package | both |
| 9 | pykakasi 2.3.0 (Chatterbox, Japanese text only) | GPL-3.0-or-later | Same as ffmpeg: fine server-side; obligations only when the image is distributed | PyPI | both |

## Commercially fine (for the record)
| Component | Licence | Verdict | Source |
|---|---|---|---|
| faster-whisper large-v3 (ASR) | MIT | OK\*(U) | https://huggingface.co/Systran/faster-whisper-large-v3 |
| Silero VAD | MIT | OK\*(U) | https://github.com/snakers4/silero-vad |
| Qwen2.5-14B-Instruct GGUF Q5_K_M (translation) + llama-cpp-python | Apache-2.0 / MIT | OK\*(U) | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF |
| Chatterbox Multilingual v3 (TTS; its S3 tokenizer comes from CosyVoice2, Apache-2.0) | MIT | OK\*(U). Every output carries the Perth watermark (MIT, always on): keep it (provenance; EU AI Act Art. 50 question for counsel) | https://huggingface.co/ResembleAI/chatterbox |
| wav2vec2-base-960h (CTC word timing, TTS QA) | Apache-2.0 (LibriSpeech CC BY 4.0) | OK | https://huggingface.co/facebook/wav2vec2-base-960h |
| noisereduce (gap clean-up) | MIT | OK | PyPI |
| sd-vae-ft-mse (LatentSync VAE), whisper-tiny (in LatentSync) | MIT | OK\*(U) | HF stabilityai/sd-vae-ft-mse, openai/whisper-tiny |
| MediaPipe Face Landmarker (faces, head pose, crops) | Apache-2.0 | OK | https://storage.googleapis.com/mediapipe-models/face_landmarker/ |
| insightface (Python package, MIT) | code only | OK: its NC weights (buffalo_l) are never fetched; tripwires block them | PyPI |
| libass, fonts (DejaVu: Bitstream Vera licence; Noto: OFL-1.1) | free | OK | Ubuntu |
| The other 220 Python packages | MIT / BSD / Apache / ISC / PSF / LGPL (easydict, soxr) | OK (pip-metadata scan, r15) | `infra/docker/requirements.dabai.lock.txt` |

**Not in the production path:**
- LatentSync's `sfd_face.pth` / `syncnet_v2.model`: the SyncNet metric (12.1) only.
- `stable_syncnet.pt`: distillation training only.
- InsightFace / RetinaFace weights: banned, never fetched.

## Replacements checked for the clean profile (r15)
- **CodeFormer.** No face restorer with a clean data chain exists:
  - GFPGAN, RestoreFormer(++), GPEN and VQFR are all FFHQ-trained (CC BY-NC-SA 4.0 dataset), so GREY.
  - DifFace, PGDiff and DMDNet are NC.
  - GFPGAN v1.4 (the best grey candidate) needs its own alignment / paste path that bypasses facexlib's NC detector and parser. That is about half a day of wiring plus an eye loop, and it would still be GREY. So it was **not wired today**.
  - The clean option with a clean data chain is a generic video SR on the mouth: RealBasicVSR / BasicVSR++, REDS CC BY 4.0, Apache-2.0. It needs mmcv on torch 2.7 / sm_120 and "does not hallucinate teeth or lip detail".
  - Until the eye says the clean profile is clearly worse, the clean arm keeps only the Real-ESRGAN layer SR (GREY, as in current).
- **htdemucs.** Every strong open separator is MUSDB-trained:
  - Demucs v4 / mdx_extra: NO;
  - Open-Unmix umxhq: MUSDB; umxl: CC BY-NC-SA;
  - MDX-Net / BS-RoFormer community weights: MUSDB plus unclear data.
  - **Spleeter stays the only clean-ish bed.** The alternatives are a commercial separation API / licence, or training a separator on licensed stems.

## Decision needed (the user / legal; not settled by the ML side)
1. Profile of the product image: `DUB_PROFILE=clean` (no CodeFormer, Spleeter bed) or `current` (as on the servers, with the NC deviation items 1–2).
   - It is an explicit, REQUIRED build arg with no default (INTERFACE.md section 8).
   - The look of each: the A/B clips in `outbox/r15_t/` (old03 30 s, m03 20 s).
2. If `clean`: is GREY acceptable (items 4–6), or should legal clear those too?
3. If the image is handed to third parties (not just run by us): the GPL items 8–9.
