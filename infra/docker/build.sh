#!/bin/bash
# DabAI image (r15 T): build the ML worker image (context = the repo root). Builds locally only; PUSHES NOTHING.
#     bash infra/docker/build.sh [tag]            env: IMAGE (default dabai-worker), LLAMA_JOBS (16), PLATFORM (linux/amd64)
# Needs: Docker with BuildKit (docker buildx), ~60 GB free disk for the build, network (~6 GB of wheels + the CUDA base images).
# The result is ~18-20 GB (CUDA runtime + cuDNN + torch + three venvs), WITHOUT weights. No GPU is needed to build, one is needed to run.
# Do NOT run on a laptop without checking the disk: it pulls ~10 GB of base layers.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE=${IMAGE:-dabai-worker}; SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown); TAG=${1:-$SHA}
[ -z "$(git status --porcelain -- scripts infra reports 2>/dev/null)" ] || echo "WARNING: uncommitted changes in scripts/ infra/ reports/: the image will not match $SHA"
DOCKER_BUILDKIT=1 docker build -f infra/docker/Dockerfile --platform ${PLATFORM:-linux/amd64} \
  --build-arg GIT_SHA=$SHA --build-arg IMAGE_TAG=$IMAGE:$TAG --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) --build-arg LLAMA_JOBS=${LLAMA_JOBS:-16} \
  -t $IMAGE:$TAG .
docker image inspect $IMAGE:$TAG --format 'built {{.RepoTags}} {{.Size}} bytes'
cat <<EOF

Built $IMAGE:$TAG (pipeline $SHA). Nothing was pushed. Next steps (yours, with your registry credentials):
  # 1. the weights volume, once (needs HF_TOKEN with the pyannote 3.1 terms accepted; ~24 GB download):
  docker volume create dabai-models
  docker run --rm --gpus all -e HF_TOKEN -v dabai-models:/workspace/dub/dubai/models \\
      -v /path/to/student:/in:ro -e STUDENT_LORA_FILE=/in/latentsync_student_r2_s4_step01500_ema.pt $IMAGE:$TAG fetch-models
  # 2. self-test + a smoke job:
  docker run --rm --gpus all -v dabai-models:/workspace/dub/dubai/models $IMAGE:$TAG check
  docker run --rm --gpus all -v dabai-models:/workspace/dub/dubai/models -v \$PWD/jobs:/data/jobs -v /path/to/clip.mp4:/in.mp4:ro $IMAGE:$TAG smoke /in.mp4
  # 3. the API (one container per GPU):
  docker run -d --gpus '"device=0"' --shm-size 16g -p 8000:8000 -e DABAI_API_TOKEN=<choose one> \\
      -v dabai-models:/workspace/dub/dubai/models -v /srv/dabai/jobs:/data/jobs $IMAGE:$TAG
  # 4. push (registry login is yours; we hold no credentials):
  docker tag $IMAGE:$TAG <registry>/<namespace>/$IMAGE:$TAG
  docker push <registry>/<namespace>/$IMAGE:$TAG
EOF
