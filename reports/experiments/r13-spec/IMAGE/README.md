# DabAI working image: hand-off for infra (5.1)

The ML side does not build or deploy the image; this folder is what infra needs to build it. Everything was measured on r13-t (03.11):
- RTX 5090, driver 590.48;
- Ubuntu 24.04;
- CUDA 12.8;
- ffmpeg 6.1.1 with libass.

| File | What |
|---|---|
| `setup_from_scratch.sh` | One script, in order: the pipeline venv → models → MediaPipe venv → CUDA llama.cpp → Qwen 14B and the rest → runtime caches → NVML → a check. Idempotent. |
| `MODELS.md`, `models_manifest.json` | Every model / weight: size, licence, source + HF revision, what uses it, status; sha256 of the small files. |
| `requirements.lock.txt` | `pip freeze` of `/venv/dabai` (the pipeline): torch 2.7.1+cu128; Chatterbox pinned to git `5de7a54`; resemble-perth `ff1c8ac`. |
| `requirements.mediapipe.lock.txt`, `requirements.main.lock.txt` | The two small side venvs. |

## Layout the code expects
- **Repo:** `/workspace/dub/dubai`. Models are in `models/`, CodeFormer and the CUDA llama.cpp in `third_party/`. Override the repo path with `DUB_ROOT`.
- **Venvs:**
  - `/venv/dabai`: the pipeline;
  - `/venv/mediapipe`: `DUB_MEDIAPIPE_PY`;
  - `/venv/main`: only `nvidia-ml-py` for `scripts/r10/gpu_sampler.py`.
- `/workspace/dub/sep_pylib`: the demucs code. `/workspace/dub/quarantine_nc/torch`: the htdemucs weights (TORCH_HOME of the sep stage only).
- **Secrets:** `/workspace/dub/.env` with `HF_TOKEN` (mode 600). `scripts/env.sh` reads it.
- **Caches that must ALSO be on the volume** (otherwise they are downloaded at the first job, which 5.1 forbids):
  - `~/.cache/torch/pyannote`: pyannote 3.1, 32 MB, gated;
  - `~/.pkuseg`: 91 MB, Chatterbox's Chinese segmenter.

  Either keep `HOME=/root` on the volume, or copy both after step 6.

## Sizes
- Required weights: **23.6 GB**.
- Venvs: `/venv/dabai` 8.4 GB (mostly torch + CUDA libs); the other two are small.
- Not needed any more, and to be dropped from the old image:
  - **Qwen 7B Q8_0 (8.1 GB):** the pipeline uses the 14B on the GPU since 29.10;
  - **InsightFace buffalo_l / RetinaFace R50:** banned and never fetched.
- Optional: KeySync, ~13 GB of weights in its own venv; see `scripts/keysync/` on `experiment/r13-keysync`.

## Licence deviations (user-accepted; they must stay visible in the image)
| Component | Licence | Why it is in | Switch |
|---|---|---|---|
| CodeFormer + ParseNet | S-Lab 1.0, **non-commercial** | Mouth detail in the composite (quality); the user decided on 29.10. Ask the authors for a commercial licence. | `DUB_ALLOW_NC=1` |
| htdemucs weights | not cleared for commercial use | The main ("demucs") background bed; the Spleeter (MIT) bed is always produced too | `DUB_ALLOW_NC_BED=1` |
| KeySync | Stability AI Community License (operator revenue < $1M, registration) | The alternative lip-sync generator | separate venv |
| LatentSync 1.6 | OpenRAIL++ (use restrictions) | The main lip-sync generator; approved | – |

The clean guard (`scripts/pilot/clean_face_stack.py`) refuses the NC weights unless the switch is set. The image should set them explicitly in the job environment, so the deviation is a visible decision.

## Smoke test after the build
```bash
ID=t03 bash scripts/r9/e2e.sh /path/to/03.mp4 /tmp/t03     # ~16 min on a 5090; /tmp/t03/manifest.json status SUCCESS
python scripts/r13/dabai_job.py --input /path/to/03.mp4 --out /tmp/j03     # the backend interface: /tmp/j03/status.json
```
