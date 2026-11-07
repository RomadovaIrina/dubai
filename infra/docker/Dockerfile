# syntax=docker/dockerfile:1.7
# =====================================================================================================================================
# DabAI ML worker (r15 T): the whole dubbing pipeline (scripts/r9/e2e.sh) + the job interface (scripts/job: CLI + FastAPI) for ONE
# RTX 5090 (sm_120, 32 GB). Contract: scripts/job/INTERFACE.md. Build: infra/docker/build.sh (context = the repo root).
#
#   * NO model weights inside: they live on a volume mounted at /workspace/dub/dubai/models (TZ 5.1: weights on a network volume, no
#     download at start), filled ONCE with `docker run ... fetch-models` (infra/docker/fetch_models.sh, 23.6 GB + the student LoRA).
#   * NO non-commercial component inside: CodeFormer (code + weights, S-Lab NC), htdemucs weights, InsightFace / RetinaFace weights are
#     never installed in the image.
#   * PROFILE = an explicit, REQUIRED build arg (a product / licence decision; there is no default): DUB_PROFILE=clean | current.
#       clean    no CodeFormer detail layer, Spleeter background bed (commercially clean)
#       current  = the research servers' production path: + CodeFormer detail + the htdemucs bed (NC deviation, user-accepted 29.10); the NC
#                parts come from a SEPARATE volume at /opt/dabai-nc (fetch-models fills it), never from the image or the models volume
#     The baked value is the run-time default; `-e DUB_PROFILE=...` overrides it. The entrypoint refuses to start without a valid profile.
#   * Software = EXACTLY the validated r14-d environments: pinned locks installed with --no-deps (no resolver), CPython 3.11.17 (uv),
#     torch 2.7.1+cu128, ffmpeg 6.1.1 (Ubuntu 24.04, libass + libx264), llama-cpp-python 0.3.35 built for sm_120.
# =====================================================================================================================================
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04
ARG UV_VERSION=0.13.4
ARG PYTHON_VERSION=3.11.17

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ------------------------------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-${UBUNTU} AS builder
ARG PYTHON_VERSION
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_PYTHON_PREFERENCE=only-managed UV_NO_CACHE=1 UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 DUB_ROOT=/workspace/dub/dubai VENV_ROOT=/venv DABAI_PYTHON=${PYTHON_VERSION}
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates curl pkg-config \
      ffmpeg libgl1 libglib2.0-0 libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install "${PYTHON_VERSION}"
WORKDIR /workspace/dub/dubai
COPY infra/docker/install_env.sh infra/docker/requirements.*.lock.txt infra/docker/
RUN bash infra/docker/install_env.sh venvs api
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
RUN LATENTSYNC_COMMIT=${LATENTSYNC_COMMIT} bash infra/docker/install_env.sh latentsync
ARG LLAMA_JOBS=16
# no driver at build time: libcuda comes from the toolkit's stubs for the link step only (the runtime gets the real one from the host)
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs JOBS=${LLAMA_JOBS} CUDACXX=/usr/local/cuda/bin/nvcc bash infra/docker/install_env.sh llama_cuda
RUN find /venv -depth -type d -name __pycache__ -exec rm -rf {} + ; rm -rf /root/.cache /tmp/*

# ------------------------------------------------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-${UBUNTU} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG GIT_SHA=unknown
ARG IMAGE_TAG=dabai-worker:dev
ARG BUILD_DATE=unknown
ARG DUB_PROFILE
RUN case "${DUB_PROFILE:-}" in clean|current) echo "profile ${DUB_PROFILE}";; \
      *) echo "build arg DUB_PROFILE=clean|current is REQUIRED (a product / licence decision, see scripts/job/INTERFACE.md section 8)" >&2; exit 1;; esac
LABEL org.opencontainers.image.title="dabai-worker" org.opencontainers.image.revision="${GIT_SHA}" org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.description="DabAI dubbing ML worker (one RTX 5090). Weights on a mounted volume; profile ${DUB_PROFILE}." \
      ai.dabai.interface="dabai-job/2" ai.dabai.profile="${DUB_PROFILE}"
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg 6.1.1 (libass, libx264), fonts for the subtitles (DejaVu = the 4 styles; Noto / Noto CJK = glyph fallback for the other scripts),
# GL / glib for OpenCV, libsndfile, OpenMP, tini as PID 1, procps (pkill / pgrep used by the stages)
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 libgomp1 tini procps ca-certificates curl \
      fonts-dejavu-core fonts-noto-core fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/uv/python /opt/uv/python
COPY --from=builder /venv /venv
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_PYTHON_PREFERENCE=only-managed UV_NO_CACHE=1 UV_LINK_MODE=copy
COPY . /workspace/dub/dubai
COPY --from=builder /workspace/dub/dubai/third_party /workspace/dub/dubai/third_party
RUN echo "${GIT_SHA}" > /workspace/dub/dubai/VERSION \
 && mkdir -p /workspace/dub/artifacts /workspace/dub/r8a /data/jobs /workspace/dub/dubai/models /workspace/dub/dubai/logs \
 && touch /workspace/dub/gpu.lock
ENV DUB_ROOT=/workspace/dub/dubai DUB_ENV=/venv/dabai VIRTUAL_ENV=/venv/dabai DUB_MEDIAPIPE_PY=/venv/mediapipe/bin/python \
    HF_HOME=/workspace/dub/dubai/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    DUB_PROFILE=${DUB_PROFILE} DABAI_NC_DIR=/opt/dabai-nc DUB_THREADS=16 \
    DABAI_JOBS_DIR=/data/jobs DABAI_IMAGE=${IMAGE_TAG} DABAI_GIT_SHA=${GIT_SHA} \
    DABAI_STUDENT_LORA=/workspace/dub/dubai/models/student/latentsync_student_r2_s4_step01500_ema.pt \
    PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
VOLUME ["/workspace/dub/dubai/models", "/data"]
WORKDIR /workspace/dub/dubai
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD [ "$(cat /tmp/dabai_mode 2>/dev/null)" != api ] || curl -fsS "http://127.0.0.1:${PORT:-8000}/health" > /dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/workspace/dub/dubai/infra/docker/entrypoint.sh"]
CMD ["api"]
