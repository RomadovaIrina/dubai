#!/bin/bash
# r17 on the Vast VM: clone the release branch, cut the smoke clip, run build_smoke.sh (build -> fetch-models -> check -> cold / warm smoke -> API).
# Usage: copy the KeySync kit (lora_pick.pt lora_meta.json kf.pt interp.pt) to /root/vmkit/kit and a source clip to /root/vmkit/m03_src.MOV,
#   HF_TOKEN=... in /workspace/.env (600), then: systemd-run --unit dabai-bs bash vm_run_vast.sh. Prune the build cache after "built" (VM_REQUIREMENTS: disk).
# Detached (systemd-run unit dabai-bs); log /root/bs_out/build_smoke.log + /root/bs_out/vm_run.log. HF_TOKEN from the box env, never printed.
set -uo pipefail
B=${BRANCH:-experiment/r16-docker}; O=/root/bs_out; mkdir -p $O; exec >> $O/vm_run.log 2>&1
echo "== $(date -u +%FT%TZ) vm_run start"
set -a; . /workspace/.env 2>/dev/null || . /etc/environment 2>/dev/null; set +a
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN present (${#HF_TOKEN} chars)" || { echo "NO HF_TOKEN in /workspace/.env"; exit 2; }
command -v docker >/dev/null || { echo "installing docker"; apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io docker-buildx; }
docker info 2>/dev/null | grep -qi nvidia || {
  echo "installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker && systemctl restart docker; }
docker buildx version || { DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker-buildx || DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker-buildx-plugin; }
command -v ffmpeg >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q ffmpeg
command -v python3 >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "GPU in docker FAILED"; exit 3; }
cd /root; [ -d dubai/.git ] && git -C dubai fetch -q origin $B && git -C dubai checkout -q -B $B FETCH_HEAD || git clone -q -b $B https://github.com/RomadovaIrina/dubai dubai
cd dubai; echo "repo $(git rev-parse --short=12 HEAD)"
[ -s /root/m03_20s.mp4 ] || ffmpeg -hide_banner -loglevel error -ss 0 -t 20 -i /root/vmkit/m03_src.MOV -c:v libx264 -crf 16 -preset fast -c:a aac -b:a 192k /root/m03_20s.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 /root/m03_20s.mp4
export DUB_PROFILE=clean KS_KIT_DIR=/root/vmkit/kit
bash infra/docker/build_smoke.sh /root/m03_20s.mp4 $O
RC=$?; echo "== $(date -u +%FT%TZ) build_smoke rc=$RC"; cat $O/report.json
