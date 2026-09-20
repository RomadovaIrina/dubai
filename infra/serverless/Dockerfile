# syntax=docker/dockerfile:1.7
# =====================================================================================================================
# DabAI — RunPod Serverless worker image for Pilot 0.10 (cold-start readiness on the target topology).
#
#   build context = repository root:
#     docker build -f infra/serverless/Dockerfile -t ghcr.io/romadovairina/dabai-worker:<tag> \
#         --build-arg GIT_SHA=$(git rev-parse HEAD) --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) .
#
# Multi-stage:
#   uv       -> pinned uv binary (installs the exact CPython 3.11.16 the pilot env used)
#   base     -> Ubuntu 24.04 (glibc 2.39 / ffmpeg 6.1.1, the pilot host) + OS runtime libs + managed Python
#   builder  -> compilers/git; /venv/dabai built exactly like scripts/setup_dabai.sh stages 0-6 (no CodeFormer,
#               no Jupyter) with scripts/requirements/* AND reports/pip-freeze.lock.txt as constraints -> every
#               transitive version equals the frozen, smoke-tested pilot environment; LatentSync clone at the
#               pinned commit; RunPod SDK
#   runtime  -> base + /venv/dabai + DabAI code + LatentSync source. NO model weights, NO downloads at start.
#
# All weights live on the RunPod network volume (dabai-models), mounted at /runpod-volume on Serverless:
#   /runpod-volume/models/MODEL_VOLUME_MANIFEST.json describes the layout; infra/serverless/entrypoint.sh maps it.
#
# The CUDA runtime comes from the torch cu128 wheels (nvidia-*-cu12 pip packages, same as the pilot venv); the host
# driver is injected by the NVIDIA container toolkit. BASE_IMAGE can be switched to
# nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 without any other change if a system CUDA runtime is ever wanted.
# =====================================================================================================================
ARG BASE_IMAGE=ubuntu:24.04
ARG UV_VERSION=0.12.17
ARG PYTHON_VERSION=3.11.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------------------------------------------------
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS base
ARG PYTHON_VERSION
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
# ffmpeg 6.1.1 (atempo/libx264/aac used by the pipeline), libGL/glib for opencv-python, libsndfile for soundfile,
# libgomp for llama.cpp / ctranslate2 / numba OpenMP, tini as PID 1.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl tini ffmpeg libgl1 libglib2.0-0 libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install "${PYTHON_VERSION}" && uv python list --only-installed

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS builder
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG PYTHON_VERSION
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
ARG CHATTERBOX_COMMIT=5de7a54aa4e5e2baadb0182dde554908b48b85c2
ARG PERTH_COMMIT=ff1c8ac55a976971245cdd53c18d6131ca00d993
ARG RUNPOD_SDK_VERSION=1.12.0
ENV DUB_ENV=/venv/dabai \
    PY=/venv/dabai/bin/python \
    UV_PYTHON_DOWNLOADS=never
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*
RUN uv venv --python "${PYTHON_VERSION}" --seed "${DUB_ENV}" && "${PY}" --version

WORKDIR /tmp/req
COPY scripts/requirements/ /tmp/req/
COPY reports/pip-freeze.lock.txt /tmp/req/pip-freeze.lock.txt
# Constraint set = the repo's constraints.txt + the frozen pilot environment (direct-URL lines cannot be constraints;
# those two packages are installed by commit below). Jupyter-only packages are not part of the worker.
RUN grep -v ' @ ' pip-freeze.lock.txt > freeze.constraints.txt \
 && sed -E '/^(ipykernel|nbconvert|nbformat)\b/d' 10-core.txt > 10-core.worker.txt \
 && sed -E "s#^resemble-perth @ .*#resemble-perth @ git+https://github.com/resemble-ai/Perth.git@${PERTH_COMMIT}#" \
      50-chatterbox-deps.txt > 50-chatterbox-deps.worker.txt \
 && cat 50-chatterbox-deps.worker.txt
ENV PIPC="-c /tmp/req/constraints.txt -c /tmp/req/freeze.constraints.txt"

# stage 0 — immutable torch baseline (cu128 wheels; they carry the CUDA runtime libraries)
RUN uv pip install --python "${PY}" -r 00-torch.txt \
 && "${PY}" -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda)"
# stage 1-3 — core / audio / faces
# hadolint ignore=SC2086,DL3059
RUN uv pip install --python "${PY}" ${PIPC} -r 10-core.worker.txt -r 20-audio.txt -r 30-faces.txt
# stage 4 — llama.cpp CPU wheel (AVX2, as in the pilot); fallback: portable source build (GGML_NATIVE=OFF, never -march=native)
# hadolint ignore=SC2086
RUN uv pip install --python "${PY}" ${PIPC} -r 40-llama.txt \
 || CMAKE_ARGS="-DGGML_NATIVE=OFF" CMAKE_BUILD_PARALLEL_LEVEL=8 uv pip install --python "${PY}" ${PIPC} --no-binary llama-cpp-python llama-cpp-python==0.3.35
# stage 5 — Chatterbox Multilingual v3 (git, --no-deps: it hard-pins torch 2.6) + its deps
# hadolint ignore=SC2086
RUN uv pip install --python "${PY}" --no-deps "git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_COMMIT}" \
 && uv pip install --python "${PY}" --no-deps s3tokenizer==0.3.0 \
 && uv pip install --python "${PY}" ${PIPC} onnx ml_dtypes protobuf \
 && uv pip install --python "${PY}" ${PIPC} -r 50-chatterbox-deps.worker.txt \
 && "${PY}" -c "import chatterbox, importlib.metadata as m; print('chatterbox', m.version('chatterbox-tts'))"
# stage 6 — LatentSync 1.6 deps, insightface 0.7.3 (sdist, no build isolation), onnxruntime-gpu as the only ORT dist,
#           opencv-python 4.9.0.80 as the only cv2 (insightface->albumentations drags opencv-python-headless)
# hadolint ignore=SC2086
RUN uv pip install --python "${PY}" ${PIPC} -r 60-latentsync.txt \
 && uv pip install --python "${PY}" ${PIPC} cython setuptools wheel \
 && uv pip install --python "${PY}" ${PIPC} --no-build-isolation insightface==0.7.3 \
 && (uv pip uninstall --python "${PY}" onnxruntime onnxruntime-gpu || true) \
 && uv pip install --python "${PY}" --no-deps onnxruntime-gpu==1.26.0 \
 && (uv pip uninstall --python "${PY}" opencv-python-headless || true) \
 && uv pip install --python "${PY}" --reinstall --no-deps opencv-python==4.9.0.80 \
 && "${PY}" -c "import cv2, onnxruntime, insightface; print('cv2', cv2.__version__, '| onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers(), '| insightface', insightface.__version__)"
# RunPod Serverless SDK (>=1.9 for fitness checks); constrained so nothing in the frozen set moves
# hadolint ignore=SC2086
RUN uv pip install --python "${PY}" ${PIPC} "runpod==${RUNPOD_SDK_VERSION}" \
 && "${PY}" -c "import runpod; print('runpod', runpod.__version__)"

# LatentSync source at the pinned commit (reports/third_party-commits.txt). checkpoints/ is filled by symlinks at start.
WORKDIR /opt/dabai/third_party/latentsync
RUN git init -q . \
 && git remote add origin https://github.com/bytedance/LatentSync \
 && git fetch -q --depth 1 origin "${LATENTSYNC_COMMIT}" \
 && git checkout -q FETCH_HEAD \
 && git rev-parse HEAD > COMMIT \
 && mkdir -p checkpoints

# Provenance of what was actually installed (compared against the pilot freeze in the build log).
WORKDIR /opt/dabai/reports
# hadolint ignore=SC3001
RUN uv pip freeze --python "${PY}" > serverless-pip-freeze.txt \
 && echo "== packages whose version differs from reports/pip-freeze.lock.txt (informational) ==" \
 && (diff <(grep -v ' @ ' /tmp/req/pip-freeze.lock.txt | sort) <(grep -v ' @ ' serverless-pip-freeze.txt | sort) || true) \
 && find "${DUB_ENV}" -name '*.pyc' -path '*__pycache__*' -prune -o -type d -name 'tests' -path '*site-packages*' -prune -print0 | xargs -0 rm -rf \
 && rm -rf /root/.cache /tmp/req

# ---------------------------------------------------------------------------------------------------------------------
FROM base AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG IMAGE_REF=ghcr.io/romadovairina/dabai-worker:dev
LABEL org.opencontainers.image.title="dabai-worker" \
      org.opencontainers.image.description="DabAI dubbing pipeline worker for RunPod Serverless (Pilot 0.10). Weights on network volume." \
      org.opencontainers.image.source="https://github.com/RomadovaIrina/dubai" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Same variables scripts/env.sh exports on the pilot box, plus the network-volume mapping. HF_HUB_OFFLINE=1 is baked:
# a worker that needs the internet for weights is a bug.
ENV DUB_ROOT=/opt/dabai \
    DUB_ENV=/venv/dabai \
    VIRTUAL_ENV=/venv/dabai \
    PATH=/venv/dabai/bin:/usr/local/bin:/usr/bin:/bin \
    MODELS_ROOT=/runpod-volume/models \
    HF_HOME=/runpod-volume/models/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_CACHE=/runpod-volume/models/hf/hub \
    PKUSEG_HOME=/runpod-volume/models/pkuseg \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    DUB_THREADS=16 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    DABAI_REQUIRE_SM=sm_120 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    LD_LIBRARY_PATH=/venv/dabai/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cublas/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cudnn/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cufft/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/curand/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cusolver/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cusparse/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cusparselt/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/nccl/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/nvtx/lib:/venv/dabai/lib/python3.11/site-packages/nvidia/cufile/lib

COPY --from=builder /venv/dabai /venv/dabai
COPY --from=builder /opt/dabai/third_party /opt/dabai/third_party
COPY --from=builder /opt/dabai/reports/serverless-pip-freeze.txt /opt/dabai/reports/serverless-pip-freeze.txt
COPY scripts/ /opt/dabai/scripts/
COPY reports/pip-freeze.lock.txt reports/third_party-commits.txt /opt/dabai/reports/
COPY infra/serverless/handler.py infra/serverless/entrypoint.sh /opt/dabai/infra/serverless/

WORKDIR /opt/dabai
# hadolint ignore=SC1091
RUN chmod +x infra/serverless/entrypoint.sh scripts/*.sh scripts/pilot/*.sh 2>/dev/null; \
    mkdir -p reports/pilot logs third_party/latentsync/checkpoints \
 && printf '{"image_ref":"%s","git_sha":"%s","build_date":"%s","python":"%s","base_image":"%s"}\n' \
      "${IMAGE_REF}" "${GIT_SHA}" "${BUILD_DATE}" "$(python --version 2>&1 | cut -d' ' -f2)" "$(. /etc/os-release && echo "$PRETTY_NAME")" \
      > infra/serverless/BUILD_INFO.json \
 && python -m compileall -q scripts third_party/latentsync/latentsync infra/serverless \
 && cat infra/serverless/BUILD_INFO.json

# Build-time proof (no GPU, no volume, no network): every production module imports from this image ...
RUN python infra/serverless/handler.py --import-smoke
# ... and no production weight is baked into a layer (whisper model.bin, Qwen gguf, LatentSync unet, VAE/T3 safetensors).
RUN if find / -xdev -type f \( -name '*.gguf' -o -name 'latentsync_unet.pt' -o -name 'model.bin' -o -name '*.safetensors' \) -size +20M -print | grep .; then \
      echo 'ERROR: model weights found inside the image' >&2; exit 1; fi

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/dabai/infra/serverless/entrypoint.sh"]
CMD ["serve"]
