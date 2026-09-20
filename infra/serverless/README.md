# DabAI Serverless worker (Pilot 0.10)

Production-like RunPod Serverless worker: **multi-stage image with all software, zero weights**; every model comes from
the `dabai-models` network volume (`/runpod-volume/models`, layout in `MODEL_VOLUME_MANIFEST.json` on the volume).

| file | purpose |
|---|---|
| `Dockerfile` | multi-stage build (uv → base Ubuntu 24.04 → builder → runtime). Build context = repo root (`.dockerignore` at the root). |
| `entrypoint.sh` | fail-fast on a missing volume, idempotent symlinks, offline env, then `handler.py` |
| `handler.py` | RunPod handler: `ready` (13-point readiness) and `info`; fitness checks; `--check/--info/--import-smoke/--mock-volume` CLI |
| `build.sh` | `docker build` + size / largest layers / weight-leak check (`PUSH=1` to push) |
| `smoke.sh` | local smoke of a built image with a mock volume (no GPU needed) |
| `../../scripts/pilot/infra_10/` | endpoint control scripts for `benchmark_10_cold_start.py` |

## What is inside the image (and what is not)

* Ubuntu 24.04 (glibc 2.39, **ffmpeg 6.1.1** — the pilot host), tini, libGL/glib/libsndfile/libgomp.
* CPython **3.11.16** (uv-managed, pinned) in `/venv/dabai`, built exactly like `scripts/setup_dabai.sh` stages 0–6 from
  `scripts/requirements/*.txt` with **`reports/pip-freeze.lock.txt` as an additional constraints file** → every
  transitive version equals the frozen pilot environment (`torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`,
  `torchaudio 2.7.1+cu128`, `pyannote.audio 3.4.0`, `faster-whisper 1.2.1`, `llama-cpp-python 0.3.35` CPU wheel,
  Chatterbox `5de7a54`, `insightface 0.7.3`, `onnxruntime-gpu 1.26.0`, `opencv-python 4.9.0.80`, …), plus `runpod==1.12.0`.
* LatentSync source at `a229c39` under `/opt/dabai/third_party/latentsync` (`checkpoints/` filled by symlinks at start).
* DabAI code: `scripts/` → `/opt/dabai/scripts` (`DUB_ROOT=/opt/dabai`, `DUB_ENV=/venv/dabai`).
* **Not included**: CodeFormer (stage 7, OFF in production — `basicsr`/`facelib` are never imported by the clean pipeline),
  Jupyter (stage 8), SyncNet/eval weights, any model weight, any HF download. The build fails if a `*.gguf`,
  `latentsync_unet.pt`, `model.bin` or `*.safetensors` > 20 MB ends up in a layer.
* CUDA runtime = the `nvidia-*-cu12` pip packages shipped with the torch cu128 wheels (same as the pilot venv); they are on
  `LD_LIBRARY_PATH`, so ctranslate2 / onnxruntime-gpu find cuBLAS/cuDNN regardless of import order. The base image can be
  switched to `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` with `--build-arg BASE_IMAGE=...` (nothing else changes).

## Volume mapping (done by `entrypoint.sh`, verified by `handler.py`)

| what | value |
|---|---|
| volume mount | `/runpod-volume` (RunPod Serverless), model root `/runpod-volume/models` |
| `DUB_ROOT/models` | symlink → `/runpod-volume/models` (`pilot_common.MODELS`) |
| LatentSync | `third_party/latentsync/checkpoints/latentsync_unet.pt` → `models/latentsync/latentsync_unet.pt`; `checkpoints/whisper` → `models/latentsync/whisper`; `checkpoints/auxiliary` → `models/insightface` (LatentSync's `FaceDetector` uses `FaceAnalysis(root="checkpoints/auxiliary")` → `models/buffalo_l/*.onnx`; `~/.insightface` is not used) |
| `HF_HOME` | `/runpod-volume/models/hf` |
| `HF_HUB_OFFLINE` / `HF_HUB_DISABLE_TELEMETRY` | `1` / `1` (baked into the image) |
| `PYANNOTE_CACHE` | `/runpod-volume/models/hf/hub` — **mandatory**: pyannote.audio 3.4.0 passes its own `CACHE_DIR = os.getenv("PYANNOTE_CACHE", "~/.cache/torch/pyannote")` to `hf_hub_download`, it does not read `HF_HOME` |
| `PKUSEG_HOME` | `/runpod-volume/models/pkuseg` — Chatterbox's tokenizer calls `spacy_pkuseg.pkuseg()` for every language; without `$PKUSEG_HOME/spacy_ontonotes.zip` it would download |
| Chatterbox Cangjie | `hf_hub_download("ResembleAI/chatterbox", "Cangjie5_TC.json", cache_dir=<ckpt dir>)` → pre-populated HF cache layout inside `models/chatterbox/` |
| `DABAI_REQUIRE_SM` | `sm_120` (RTX 5090 = Pilot 0.10 target). Any other GPU → `ready:false`. Set `any` on non-benchmark endpoints only. |

## `ready` action

```json
{"input": {"action": "ready"}}
```
Validates: torch import · `torch.cuda.is_available()` · GPU visible + a tiny CUDA matmul · compute capability (`sm`) ·
`/runpod-volume/models` · every mandatory weight file (non-zero size) · all 3 Qwen Q8_0 shards (`-of-N` set complete) ·
pyannote repos resolvable from the HF cache (`refs/main` → snapshot files) · Chatterbox Cangjie cache · `HF_HOME`,
`PYANNOTE_CACHE`, `PKUSEG_HOME`, `HF_HUB_OFFLINE=1` · the four symlinks · `sm == DABAI_REQUIRE_SM`.
Returns `{"ready": true|false, "gpu": ..., "sm": ..., "models_root": ..., "missing": [...], "warnings": [...], ...}`.
READY does **not** load models into VRAM and runs no inference (0.10 measures start-to-ready, not the first job).
`{"input": {"action": "info"}}` returns image build info / versions / whitelisted env (never secrets).

The same checks run as RunPod **fitness checks** at worker start (SDK ≥ 1.9): a worker with a missing volume, missing
weights or no CUDA exits before taking any job.

## Build / test / push

```bash
infra/serverless/build.sh                         # -> ghcr.io/romadovairina/dabai-worker:sha-<12>
infra/serverless/smoke.sh ghcr.io/romadovairina/dabai-worker:sha-<12>          # mock volume, no GPU
infra/serverless/smoke.sh ghcr.io/romadovairina/dabai-worker:sha-<12> --gpu    # + CUDA checks on this host
docker login ghcr.io -u <github-user>             # PAT with write:packages
PUSH=1 infra/serverless/build.sh v0.10.0          # or: docker push ghcr.io/romadovairina/dabai-worker:v0.10.0
```
GHCR repository names are lowercase (`romadovairina`). `.github/workflows/build-worker-image.yml` is a manual
(`workflow_dispatch`) alternative that builds on GitHub runners and pushes with `GITHUB_TOKEN`.

In-container checks: `docker run --rm -v <vol>:/runpod-volume --gpus all IMAGE check` (readiness JSON, exit 0/1),
`... IMAGE info`, `... IMAGE serve --rp_serve_api` (local FastAPI on :8000).

## sm_120 acceptance (deferred to the real RTX 5090 worker)

The preparation Pod (RTX 2000 Ada, sm_89) cannot validate Blackwell. On the first RTX 5090 worker run, in order:
`{"action":"ready"}` → `sm_120` expected; then `python scripts/check_torch.py`, `python scripts/check_onnxruntime.py`
(CUDA EP on sm_120; the pilot kept `onnxruntime-gpu`), `python scripts/check_whisper.py` (int8_float16 → float16 on
sm_120). Only then run Pilot 0.10 with `scripts/pilot/infra_10/`.

## Endpoint settings for Pilot 0.10 (created manually in the console / REST — not by these scripts)

| setting | value | why |
|---|---|---|
| image | `ghcr.io/romadovairina/dabai-worker:<tag>` | multi-stage image, no weights |
| GPU | RTX 5090 (sm_120), 1 GPU/worker | target topology |
| network volume | `dabai-models` (mounted at `/runpod-volume`) | weights; pins the endpoint to the volume's data center |
| container disk | ≥ 20 GB | runtime media in `/tmp/dabai_pilot_05` + RunPod disk fitness check (10 % free) |
| active (min) workers | 0 | scale-to-zero → every start is a cold start |
| max workers | 1 | one worker under test |
| idle timeout | 5 s (minimum) | fast teardown between runs |
| FlashBoot | **off** | FlashBoot revives retained state → not a cold start (`ensure_stopped.sh` refuses it) |
| scaler | `QUEUE_DELAY`, 1 s (or `REQUEST_COUNT` 1) | a queued job must start a worker immediately |
| execution timeout | ≥ 600 000 ms | dubbing jobs later; `ready` itself takes seconds |
| CUDA versions | 12.8 and newer | torch cu128 |
| env | nothing required (all baked). Optional: `DABAI_REQUIRE_SM`, `DUB_THREADS` (llama.cpp / torch CPU threads = worker vCPU quota), `RUNPOD_*` fitness-check knobs (leave default for the benchmark) |
