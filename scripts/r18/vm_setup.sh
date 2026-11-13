#!/bin/bash
# r18: one-time setup of a Vast KVM VM (ubuntu 22.04, RTX 5090): docker + the NVIDIA toolkit, the published image, the models volume, check,
# Q's detectors (branch experiment/r16-qa) beside this checkout. Run detached:  systemd-run --unit r18-setup bash /root/r18/code/scripts/r18/vm_setup.sh
# The HF token is read from HF_TOKEN, /root/.hf_token or /workspace/.env and is never printed. Log: /root/r18/log/setup.log; the last line is R18_SETUP_OK.
set -uo pipefail
B=${R18_ROOT:-/root/r18}; CODE=${R18_CODE:-$B/code}; IMG=${R18_IMG:-ghcr.io/romadovairina/dubai:latest}; VOL=${R18_VOL:-dabai-models}
mkdir -p $B/log $B/in $B/ref $B/dubs $B/data $B/out; exec >> $B/log/setup.log 2>&1
echo "== $(date -u +%FT%TZ) r18 setup start"
[ -n "${HF_TOKEN:-}" ] || { [ -s /root/.hf_token ] && HF_TOKEN=$(tr -d ' \r\n' < /root/.hf_token); }
[ -n "${HF_TOKEN:-}" ] || { set -a; . /workspace/.env 2>/dev/null; set +a; }
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN present (${#HF_TOKEN} chars)" || { echo "R18_SETUP_FAIL no HF_TOKEN (HF_TOKEN env, /root/.hf_token or /workspace/.env)"; exit 2; }
export HF_TOKEN
command -v docker > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io; }
docker info 2> /dev/null | grep -qi nvidia || {
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker && systemctl restart docker; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h /var/lib/docker / | tail -2; nproc; free -g | sed -n 2p
for i in 1 2 3 4 5; do docker pull -q $IMG && break; echo "pull retry $i"; sleep 20; done
docker image inspect $IMG --format 'image {{.Id}} {{index .Config.Labels "org.opencontainers.image.revision"}}' || { echo "R18_SETUP_FAIL image pull"; exit 3; }
docker run --rm --gpus all --entrypoint nvidia-smi $IMG --query-gpu=name --format=csv,noheader || { echo "R18_SETUP_FAIL GPU in docker"; exit 4; }
t0=$(date +%s)   # the models volume (~38 GB; idempotent) + our KeySync kit from the private HF repo
for i in 1 2 3; do docker run --rm --gpus all -e HF_TOKEN -e KS_KIT_REPO=${KS_KIT_REPO:-RomadovaIrina/dabai-keysync-kit} -v $VOL:/workspace/dub/dubai/models $IMG fetch-models > $B/log/fetch_$i.log 2>&1 && break; echo "fetch-models retry $i: $(tail -1 $B/log/fetch_$i.log | cut -c1-200)"; done
echo "fetch-models $(( $(date +%s) - t0 )) s: $(tail -1 $B/log/fetch_*.log | tail -1 | cut -c1-200)"
docker run --rm --gpus all -v $VOL:/workspace/dub/dubai/models $IMG check > $B/log/check.log 2>&1; echo "check: $(tail -1 $B/log/check.log | cut -c1-200)"
grep -q CHECK_OK $B/log/check.log || { echo "R18_SETUP_FAIL check (log/check.log)"; exit 5; }
mkdir -p $CODE/scripts/qa16   # Q's detectors, untouched, beside this checkout (their model path = <repo>/models/mediapipe)
git -C $CODE fetch -q origin experiment/r16-qa 2> /dev/null
for f in scripts/quality/r16q_detect.py scripts/quality/r16q_defects.py scripts/r16q/det_diff.py; do git -C $CODE show FETCH_HEAD:$f > $CODE/scripts/qa16/$(basename $f) || echo "WARN no $f"; done
echo "== $(date -u +%FT%TZ) R18_SETUP_OK code $(git -C $CODE rev-parse --short HEAD) qa $(git -C $CODE rev-parse --short FETCH_HEAD 2> /dev/null)"
