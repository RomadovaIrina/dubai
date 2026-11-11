#!/bin/bash
# DabAI image (r16 TZ, Irina 11.11): tag + push the worker image to her PUBLIC GitHub Container Registry, next to the repo RomadovaIrina/dubai.
# Run on the GPU VM AFTER build_smoke.sh passed (datacentre bandwidth; never from the laptop through the proxy).
#     GHCR_TOKEN=<GitHub token with write:packages> bash infra/docker/push_ghcr.sh <local image:tag> [remote tag ...]
#   env: GHCR_USER (romadovairina), GHCR_REPO (ghcr.io/romadovairina/dubai), DRY=1 = every check, no login / push
# The image is PUBLIC, so before anything leaves the machine it must pass (or nothing is pushed):
#   1. profile clean (label ai.dabai.profile + ENV DUB_PROFILE), the source label = the public repo
#   2. no layer > 9 GB uncompressed (GHCR: 10 GB per layer, 10 min per upload)
#   3. docker history: no build arg / ENV that looks like a secret (token, key, password, HF_TOKEN, proxy credentials)
#   4. a file scan of the whole image filesystem: no weights (*.pt *.pth *.safetensors *.gguf *.onnx *.ckpt *.bin > 1 MB), no .env, no HF / torch
#      cache, no git credentials / ssh keys, no file containing a token pattern (hf_..., ghp_..., github_pat_..., sk-...) in code / config
# The token is read from the env only, passed to docker login on stdin, never echoed.
set -euo pipefail
SRC=${1:?usage: push_ghcr.sh <local image:tag> [remote tag ...]}; shift || true
REPO=${GHCR_REPO:-ghcr.io/romadovairina/dubai}; USER_=${GHCR_USER:-romadovairina}
TAGS=("$@"); [ ${#TAGS[@]} -gt 0 ] || TAGS=("${SRC##*:}")
fail(){ echo "PUSH REFUSED: $*" >&2; exit 1; }
docker image inspect "$SRC" > /dev/null || fail "no local image $SRC"

# ---- 1. profile + labels
P=$(docker image inspect "$SRC" --format '{{index .Config.Labels "ai.dabai.profile"}}')
S=$(docker image inspect "$SRC" --format '{{index .Config.Labels "org.opencontainers.image.source"}}')
[ "$P" = clean ] || fail "profile label is '$P', only clean goes public"
docker image inspect "$SRC" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -qx 'DUB_PROFILE=clean' || fail "ENV DUB_PROFILE is not clean"
[ "$S" = https://github.com/RomadovaIrina/dubai ] || fail "label org.opencontainers.image.source is '$S'"
echo "OK  profile clean, source $S"

# ---- 2. layer sizes
BIG=$(docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$SRC" | awk -F'\t' '
  { s=$1; v=s+0; if (s ~ /GB$/) v*=1e9; else if (s ~ /MB$/) v*=1e6; else if (s ~ /kB$/) v*=1e3; if (v > 9e9) print s " " substr($2, 1, 120) }')
[ -z "$BIG" ] || fail "layers over 9 GB (GHCR max 10 GB): $BIG"
echo "OK  every layer < 9 GB ($(docker history --format '{{.Size}}' "$SRC" | wc -l) layers)"

# ---- 3. history / env / labels: nothing secret-looking
PAT='(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|HF_TOKEN=.|GHCR_TOKEN=.|OPENAI_API_KEY=.|DABAI_API_TOKEN=.|PASSWORD=.|://[^/@ ]+:[^/@ ]+@)'
H=$( { docker history --no-trunc --format '{{.CreatedBy}}' "$SRC"; docker image inspect "$SRC" --format '{{json .Config}}'; } | grep -Eo "$PAT" | sort -u || true)
[ -z "$H" ] || fail "secret-looking strings in the image history / config: $(echo "$H" | sed -E 's/(.{6}).*/\1…/' | tr '\n' ' ')"
echo "OK  history + config: no secrets"

# ---- 4. file scan of the image filesystem (a throwaway container, no network, the image's own tools)
SCAN=$(docker run --rm --network none --entrypoint bash "$SRC" -c '
  cd /; X="-path /proc -prune -o -path /sys -prune -o -path /dev -prune -o"
  find / $X -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.gguf" -o -name "*.onnx" -o -name "*.ckpt" -o -name "*.th" -o -name "*.tflite" -o -name "*.bin" -o -name "*.model" \) -size +1M -printf "WEIGHT %s %p\n" 2>/dev/null
  find / $X \( -name ".env" -o -name "*.env" -o -name ".netrc" -o -name ".git-credentials" -o -name "id_rsa*" -o -name "id_ed25519*" -o -name "token" -o -path "*/huggingface/token" \) -printf "SECRETFILE %p\n" 2>/dev/null
  for d in /root/.cache/huggingface /root/.cache/torch /workspace/dub/dubai/models /data; do [ -n "$(ls -A $d 2>/dev/null)" ] && echo "NONEMPTY $d"; done
  grep -rIlE "(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{32,})" /workspace /root /etc /opt/keysync/code /opt/keysync/*.py 2>/dev/null | sed "s/^/TOKENPAT /"
  true')
# the venvs ship a few small *.bin / *.model data files of libraries (tokenizer tables, ...): only > 1 MB are listed; weights are tens of MB+
W=$(grep "^WEIGHT" <<< "$SCAN" | grep -vE "(/(venv|venv/dabai|venv/mediapipe|venv/main|venv/api)/lib/python3\.11/site-packages/| /opt/uv/python/| /usr/(lib|share)/)" || true)
OTH=$(grep -E "^(SECRETFILE|NONEMPTY|TOKENPAT)" <<< "$SCAN" | grep -vE "SECRETFILE /(usr|opt/uv)/" || true)
echo "$SCAN" | grep "^WEIGHT" | sort -k2 -n -r | head -15 | sed 's/^/    (site-packages data) /' || true
[ -z "$W$OTH" ] || fail "file scan: $W $OTH"
echo "OK  file scan: no weights / caches / secrets outside the library site-packages"

[ "${DRY:-0}" = 1 ] && { echo "DRY=1: checks passed, nothing pushed (would push ${TAGS[*]/#/$REPO:})"; exit 0; }
[ -n "${GHCR_TOKEN:-}" ] || fail "set GHCR_TOKEN (GitHub token with write:packages)"
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$USER_" --password-stdin > /dev/null
for t in "${TAGS[@]}"; do docker tag "$SRC" "$REPO:$t"; docker push "$REPO:$t"; echo "PUSHED $REPO:$t"; done
docker logout ghcr.io > /dev/null
echo "Done. Make the package public once (GitHub -> Packages -> dubai -> Package settings -> Change visibility) if it is not already."
