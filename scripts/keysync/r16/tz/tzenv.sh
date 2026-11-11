#!/bin/bash
# r16 TZ runs on r16-q = the clean image's env (infra/docker/Dockerfile + entrypoint.sh DUB_PROFILE=clean): no NC (CodeFormer / htdemucs), 16 threads
export DUB_ROOT=/workspace/dub/dubai-tz DUB_ENV=/venv/dabai DUB_MEDIAPIPE_PY=/venv/mediapipe/bin/python DUB_PROFILE=clean DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0 \
       DUB_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 DABAI_FACE_DEFAULT=auto KS_ROOT=/workspace/dub/keysync
cd /workspace/dub/dubai-tz && exec "$@"
