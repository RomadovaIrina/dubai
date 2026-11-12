#!/bin/bash
# r17 Saturday 14.11 (ОБРАЗ_DOCKER.md §11): ONE run on a fresh Docker GPU VM, in this order. Pushes NOTHING (the GHCR push is a separate,
# approved step: PUSH=1 + GHCR_TOKEN, or by hand). Progress -> /root/sat/progress.log; every stage keeps its own dir under /root/sat.
# Prerequisites (upload first): /workspace/.env (HF_TOKEN), /root/vmkit/kit (the 4 KeySync kit files) + /root/vmkit/m03_src.MOV,
#   /root/mamakris-otzyvy (the stories), /root/raw (01.MP4 02.mp4 03.mp4 04.mp4 05.mp4 = D:\Downloads\05), /root/qclips (q02_vera.mp4 q03_priest.mp4)
#   bash infra/docker/saturday_run.sh [branch]
set -uo pipefail
B=${1:-experiment/r16-docker}; S=/root/sat; mkdir -p $S; P=$S/progress.log; log(){ echo "[$(date -u +%T)] $*" | tee -a $P; }
set -a; . /workspace/.env; set +a; [ -n "${HF_TOKEN:-}" ] || { log "NO HF_TOKEN"; exit 2; }
cd /root; [ -d dubai/.git ] && git -C dubai fetch -q origin $B && git -C dubai checkout -q -B $B FETCH_HEAD || git clone -q -b $B https://github.com/RomadovaIrina/dubai dubai
cd /root/dubai; SHA=$(git rev-parse --short=12 HEAD); IMG=dabai-worker:$SHA-clean; log "repo $SHA"
# ---- 0. clips (20-30 s cuts; the image's ffmpeg once it exists, the host's before)
mkdir -p $S/clips
cut(){ [ -s $S/clips/$2.mp4 ] && return 0; [ -f "$1" ] || { log "missing source $1"; return 1; }
  ffmpeg -v error -y -ss ${4:-0} -t ${3:-20} -i "$1" -c:v libx264 -crf 16 -preset fast -c:a aac -b:a 192k $S/clips/$2.mp4; }
command -v ffmpeg > /dev/null || { apt-get update -q > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -q ffmpeg > /dev/null; }
cut /root/vmkit/m03_src.MOV m03; cut /root/mamakris-otzyvy/story-1774250817106-616905094.mp4 m18; cut /root/raw/03.mp4 old03
cut /root/qclips/q02_vera.mp4 q02 30; cut /root/qclips/q03_priest.mp4 q03 30; cp $S/clips/m03.mp4 /root/m03_20s.mp4
log "clips: $(ls $S/clips | tr '\n' ' ')"
# ---- 1. build + fetch-models + check + cold / warm smoke + API (the build cache pruned right after the build: disk)
( until grep -qE "\] (built |BUILD FAILED)" $S/bs/build_smoke.log 2>/dev/null; do sleep 10; done; grep -q "\] built " $S/bs/build_smoke.log && docker builder prune -af > /dev/null 2>&1 ) &
DUB_PROFILE=clean KS_KIT_DIR=/root/vmkit/kit bash infra/docker/build_smoke.sh /root/m03_20s.mp4 $S/bs > $S/bs.log 2>&1
log "1 build_smoke: $(tr -d '\n ' < $S/bs/report.json)"; grep -q '"smoke_warm_status":"SUCCESS"' <(tr -d '\n ' < $S/bs/report.json) || { log "STOP: the smoke did not pass"; exit 4; }
# ---- 2. the VRAM margin fix (KS_SERVE_AUDIO_EXIT=1): the resident API, 3 short jobs; the peak must drop below ~29.5 GB; the output = the cold one
O=$S/vram; mkdir -p $O/data; H=http://127.0.0.1:18800; AU=(-H "Authorization: Bearer sat")
CID=$(docker run -d --gpus all --shm-size 16g -p 18800:8000 -e DABAI_API_TOKEN=sat -e KS_SERVE_AUDIO_EXIT=1 -v dabai-models:/workspace/dub/dubai/models -v $O/data:/data -v $S/clips:/in:ro $IMG)
for i in $(seq 600); do curl -sf $H/health > /dev/null && break; sleep 0.5; done; for i in $(seq 900); do curl -s $H/health | grep -q '"state": *"ready"' && break; sleep 1; done
( mx=0; while [ ! -f $O/stop ]; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ $v -gt $mx ] && { mx=$v; echo $mx > $O/vram_max.txt; }; sleep 1; done ) &
for j in v1 v2 v3; do curl -s -o /dev/null "${AU[@]}" -H "content-type: application/json" -d "{\"job_id\": \"$j\", \"input\": \"/in/m03.mp4\", \"target_lang\": \"en\"}" $H/jobs; done
for i in $(seq 3600); do s=$(curl -s "${AU[@]}" $H/jobs/v3 | python3 -c "import json, sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null); case $s in SUCCESS|PARTIAL_SUCCESS|ERROR) break;; esac; sleep 3; done
touch $O/stop; docker stop $CID > /dev/null
for j in v1 v2 v3; do log "2 $j $(python3 -c "import json; s=json.load(open('$O/data/jobs/$j/status.json')); print(s['status'], 'wall', s.get('wall_s'))") $(grep -h 'KS_SERVE waited\|KS_SERVE resident' $O/data/jobs/$j/e2e/logs/face_keysync.log 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-160)"; done
log "2 VRAM max with KS_SERVE_AUDIO_EXIT=1: $(cat $O/vram_max.txt) MiB (12.11 without: 31614)"
docker run --rm -v /root:/root --entrypoint /venv/dabai/bin/python $IMG /root/dubai/scripts/keysync/r16/tz/out_compare.py --a $S/bs/jobs/smoke/smoke.mp4 --b $O/data/jobs/v2/v2.mp4 --src $S/clips/m03.mp4 2>/dev/null | grep OUT_COMPARE | sed 's/^/2 cold smoke vs resident v2 (same clip, fresh dubs): /' | tee -a $P
# ---- 3. the probe-by-check proof (5 clips x 2 jobs)
bash infra/docker/probe_proof.sh $IMG $S/clips $S/probe > $S/probe.log 2>&1; log "3 $(grep PROBE_PROOF $S/probe/progress.log)"
# ---- 4. the edge set (16 cases, long_10min skipped: done 12.11)
bash infra/docker/edge_run.sh $IMG /root/raw /root/mamakris-otzyvy $S/edge > $S/edge.log 2>&1; log "4 $(grep EDGE_SUITE $S/edge/suite.log)"
# ---- 5. the push: only with an explicit PUSH=1 (needs Irina's yes in the session that runs this)
if [ "${PUSH:-0}" = 1 ]; then bash infra/docker/push_ghcr.sh $IMG $SHA-clean latest > $S/push.log 2>&1; log "5 push: $(grep -E 'PUSHED|REFUSED' $S/push.log | tr '\n' ' ')"; else log "5 push: skipped (PUSH=1 to push)"; fi
log "SATURDAY_DONE"
