# syntax=docker/dockerfile:1.7
# (no COPY --exclude: the venv split into < 10 GB layers is done by moving the big trees to /split in the builder, see below)
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
    UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8 DUB_ROOT=/workspace/dub/dubai VENV_ROOT=/venv DABAI_PYTHON=${PYTHON_VERSION}
# r16 TZ (11.11): the uv steps below use a BuildKit CACHE MOUNT (UV_NO_CACHE=0 there): a build that dies on a dropped connection (flaky proxies)
# resumes with the wheels it already has. The cache is never part of an image layer (same image, faster / restartable builds).
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates curl pkg-config \
      ffmpeg libgl1 libglib2.0-0 libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked UV_NO_CACHE=0 uv python install "${PYTHON_VERSION}"
WORKDIR /workspace/dub/dubai
COPY infra/docker/install_env.sh infra/docker/requirements.*.lock.txt infra/docker/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked UV_NO_CACHE=0 bash infra/docker/install_env.sh venvs api
# the scripts call bare python3 / python (ks_install.sh -> ks_refidx.py, ks_face.sh, e2e.sh): the CUDA base has none -> the pipeline venv first on PATH
ENV PATH=/venv/dabai/bin:${PATH}
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
RUN LATENTSYNC_COMMIT=${LATENTSYNC_COMMIT} bash infra/docker/install_env.sh latentsync
ARG LLAMA_JOBS=16
# no driver at build time: libcuda comes from the toolkit's stubs for the link step only (the runtime gets the real one from the host)
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked UV_NO_CACHE=0 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs JOBS=${LLAMA_JOBS} CUDACXX=/usr/local/cuda/bin/nvcc bash infra/docker/install_env.sh llama_cuda
# r16 T: the KeySync-fast face (DUB_FACE=keysync, the r16 default): keysync 5e2503b + the sm120 r15 patch in its OWN venv (torch 2.7.1 cu128 +
# TensorRT 10.16, ~6 GB) at /opt/keysync. Weights / the student / the TRT capture on the models volume (fetch-models 8/8); the TensorRT plans and
# the inductor cache are built on the first job into the volume (KS_TRT_CACHE, TORCHINDUCTOR_CACHE_DIR: plans are tied to GPU + driver + TRT).
COPY scripts/keysync scripts/keysync
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked UV_NO_CACHE=0 \
    KS_ROOT=/opt/keysync KS_NO_WEIGHTS=1 bash scripts/keysync/r16/ks_install.sh && rm -rf /opt/keysync/code/.git /opt/keysync/code/pretrained_models
RUN find /venv /opt/keysync -depth -type d -name __pycache__ -exec rm -rf {} + ; rm -rf /root/.cache /tmp/*
# the big CUDA / torch / TensorRT package trees moved OUT of the venvs into /split/<n> (each becomes its own runtime layer: GHCR <= 10 GB per
# layer, <= 10 min per upload); the runtime stage puts them back at the same paths. Sizes logged for the record.
ARG SP=lib/python3.11/site-packages
RUN set -e; i=0; for d in /venv/dabai/${SP}/nvidia /venv/dabai/${SP}/torch /opt/keysync/venv/${SP}/nvidia /opt/keysync/venv/${SP}/torch \
      /opt/keysync/venv/${SP}/tensorrt_libs; do i=$((i + 1)); mkdir -p /split/$i; [ -d $d ] && mv $d /split/$i/tree || mkdir /split/$i/tree; echo "$d" > /split/$i/path; done; \
    du -sh /venv /opt/keysync /split/* | sed 's/^/SIZE /'

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
      ai.dabai.interface="dabai-job/2" ai.dabai.profile="${DUB_PROFILE}" \
      org.opencontainers.image.source="https://github.com/RomadovaIrina/dubai"
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg 6.1.1 (libass, libx264), fonts for the subtitles (DejaVu = the 4 styles; Noto / Noto CJK = glyph fallback for the other scripts),
# GL / glib for OpenCV, libsndfile, OpenMP, tini as PID 1, procps (pkill / pgrep used by the stages)
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 libgomp1 tini procps ca-certificates curl \
      fonts-dejavu-core fonts-noto-core fonts-noto-cjk gcc libc6-dev && rm -rf /var/lib/apt/lists/*   # gcc + libc6-dev: Triton (torch.compile / inductor: the KeySync VAE) compiles its launcher at run time (11.11 VM smoke: "Failed to find C compiler")
COPY --from=builder /opt/uv/python /opt/uv/python
# the venvs in several layers (GHCR: <= 10 GB per layer, <= 10 min per upload): the big CUDA / torch / TensorRT trees each in their own layer
#   (the builder moved them to /split/1..5; the order below = the builder's list)
ARG SP=lib/python3.11/site-packages
COPY --from=builder /venv /venv
COPY --from=builder /opt/keysync /opt/keysync
COPY --from=builder /split/1/tree /venv/dabai/${SP}/nvidia
COPY --from=builder /split/2/tree /venv/dabai/${SP}/torch
COPY --from=builder /split/3/tree /opt/keysync/venv/${SP}/nvidia
COPY --from=builder /split/4/tree /opt/keysync/venv/${SP}/torch
COPY --from=builder /split/5/tree /opt/keysync/venv/${SP}/tensorrt_libs
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
    KS_ROOT=/opt/keysync KS_LORA=/workspace/dub/dubai/models/keysync/student/lora_pick.pt KS_TRT_CAP=/workspace/dub/dubai/models/keysync/trt_cap \
    KS_TRT_CACHE=/workspace/dub/dubai/models/keysync/trt_cache TORCHINDUCTOR_CACHE_DIR=/workspace/dub/dubai/models/keysync/inductor \
    KS_LMK_CACHE=/data/ks_lmk KS_TORCH_HOME=/workspace/dub/dubai/models/keysync/torch DABAI_FACE_DEFAULT=auto \
    PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# r16 TZ (Irina 11.11): the image's KeySync QUALITY DEFAULTS, pinned here so a code-default change never alters the image silently: ONE encode
# after the paste. The post-paste stages KS_TAIL_SRC (code default ON since 55df400) and KS_OCCL re-encode the WHOLE frame without colour
# management (paler frame, colr tag lost: C:\Users\irina\Videos\DabAI_AB\r16\АНАЛИЗ_0811_vs_1011.md 3.3-3.4), so they are OFF; cap / pause /
# CMS stay off, the paste mask stays the oval. Change a default = edit this line and rebuild (or `-e KS_...` per container).
# (KS_CAP must be EMPTY to be off: ks_face.sh runs the cap warp whenever it is non-empty, "0" included.)
# Rebuild with other defaults: --build-arg KS_TAIL_SRC=on ... (build.sh: FACE_ARGS="--build-arg K=V ..."). KS_CM_SHADOW="" = no cm-shadow even
# when a container sets KS_MASK=tight.
# r18 (Irina's blind pick 13.11, long5 / old03 / m03 grids): KS_MASK=hybrid = the 08.11 oval above the mouth corners + the tight chin below (the chin / neck
# seam; on old03 the oval pulled background into the stubble). Still ONE encode after the paste. KS_MASK=oval = the 08.11 paste exactly.
# KS_TAIL_FADE=on (Irina's blind pick 13.11, the m03 end face): after a face's last strong dub phrase it fades to the source INSIDE the paste (ks_plain
# --tail-src; not KS_TAIL_SRC, the re-encoding pass, which stays off). KS_HYBRID_UP: 0 = the hybrid of 13.11; 0.60 = the tight boundary up to the nose
# level (old03: no background pulled into the stubble) - set it here once Irina's blind "в | в-new | б" verdict is in. KS_OCCL=mask (the cup / turn gate
# as a mask factor, q05) is off until her blind pick.
ARG KS_TAIL_SRC=off KS_OCCL=off KS_PAUSE=off KS_CAP="" KS_CMS=0 KS_MASK=hybrid KS_CM_SHADOW="" KS_TAIL_FADE=on KS_HYBRID_UP=0
ENV KS_TAIL_SRC=${KS_TAIL_SRC} KS_OCCL=${KS_OCCL} KS_PAUSE=${KS_PAUSE} KS_CAP=${KS_CAP} KS_CMS=${KS_CMS} KS_MASK=${KS_MASK} KS_CM_SHADOW=${KS_CM_SHADOW} KS_TAIL_FADE=${KS_TAIL_FADE} KS_HYBRID_UP=${KS_HYBRID_UP} \
    PATH=/venv/dabai/bin:${PATH}
# r17 (12.11-12.11, VM speed_guard on a 3-shot clip): the wall-time speedups, ON in the image. Verified within the run-to-run floor
# (flags off vs flags off PSNR 44.04 / 38.55 dB; off vs on 44.16 / 38.61; dubs + route identical): 896 -> 668 s on the clip (KeySync -33 %).
# The resident worker (models loaded once, parked between stages; bit-identical to cold, VM 11.11) is on in API mode.
# r17 13.11: KS_SERVE_AUDIO_EXIT=1 = the KeySync daemon off the GPU during audio (VM: peak 31.6 -> 30.4 GB, output within the floor)
ARG KS_SHOT_PREP_JOBS=8 KS_SHOT_PIPE=1 KS_SYNC_SERVE=1 E2E_ROUTE_PRE=1 DABAI_RESIDENT=1 KS_SERVE_AUDIO_EXIT=1
ENV KS_SHOT_PREP_JOBS=${KS_SHOT_PREP_JOBS} KS_SHOT_PIPE=${KS_SHOT_PIPE} KS_SYNC_SERVE=${KS_SYNC_SERVE} E2E_ROUTE_PRE=${E2E_ROUTE_PRE} DABAI_RESIDENT=${DABAI_RESIDENT} KS_SERVE_AUDIO_EXIT=${KS_SERVE_AUDIO_EXIT}
VOLUME ["/workspace/dub/dubai/models", "/data"]
WORKDIR /workspace/dub/dubai
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD [ "$(cat /tmp/dabai_mode 2>/dev/null)" != api ] || curl -fsS "http://127.0.0.1:${PORT:-8000}/health" > /dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/workspace/dub/dubai/infra/docker/entrypoint.sh"]
CMD ["api"]
