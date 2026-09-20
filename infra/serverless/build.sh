#!/usr/bin/env bash
# Build the DabAI Serverless worker image (context = repo root) and report size / largest layers / weight-leak check.
# Usage: infra/serverless/build.sh [tag]        (default tag: sha-<short git sha>)
#   IMAGE_REPO   registry path              (default ghcr.io/romadovairina/dabai-worker; GHCR names must be lowercase)
#   BASE_IMAGE   override the runtime base  (default ubuntu:24.04; alt: nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04)
#   PUSH=1       push after a successful build (requires a prior `docker login ghcr.io`)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v docker >/dev/null || { echo "docker is not installed on this machine; build on a Docker-capable host (see infra/serverless/README.md)" >&2; exit 2; }
GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
TAG="${1:-sha-${GIT_SHA:0:12}}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/romadovairina/dabai-worker}"
IMAGE="${IMAGE_REPO}:${TAG}"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== building ${IMAGE} (git ${GIT_SHA}, base ${BASE_IMAGE:-ubuntu:24.04})"
DOCKER_BUILDKIT=1 docker build -f "${ROOT}/infra/serverless/Dockerfile" -t "${IMAGE}" \
  --build-arg GIT_SHA="${GIT_SHA}" --build-arg BUILD_DATE="${BUILD_DATE}" --build-arg IMAGE_REF="${IMAGE}" \
  ${BASE_IMAGE:+--build-arg BASE_IMAGE="${BASE_IMAGE}"} "${ROOT}"
echo "== image size"; docker image inspect "${IMAGE}" --format '{{.Size}}' | awk '{printf "%.2f GB\n", $1/1e9}'
echo "== largest layers"; docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "${IMAGE}" | sort -h -r | head -12 | cut -c1-160
echo "== weight-leak check (must print nothing)"
docker run --rm --entrypoint sh "${IMAGE}" -c "find / -xdev -type f \( -name '*.gguf' -o -name 'latentsync_unet.pt' -o -name 'model.bin' -o -name '*.safetensors' \) -size +20M -print"
if [ "${PUSH:-0}" = "1" ]; then echo "== pushing ${IMAGE}"; docker push "${IMAGE}"; fi
echo "== done: ${IMAGE}"
