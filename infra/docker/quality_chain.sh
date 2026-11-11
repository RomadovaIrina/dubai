#!/bin/bash
# r17 on the VM: A2 / A3 (the image as built, A's fixed dub) -> B (resident API, 3 jobs at once = 12.4 burst, then a kill / restart) -> compares.
# progress lines -> /root/q/progress.log
set -uo pipefail
T=dabai-worker:6b500430b423-clean; C=${C:?code snapshot}; Q=/root/q; P=$Q/progress.log; CLIP=/root/clip/m03_20s.mp4
log(){ echo "[$(date -u +%T)] $*" >> $P; }
DUBDIR=${DUBDIR:?A dub dir (dubbed_16k.wav + final_mix_aac.m4a)}
run_image(){   # run_image <name>: the image as built (no mounts), A's fixed dub, keep_intermediate
  local N=$1; mkdir -p $Q/$N
  printf '{"job_id": "%s", "input": "/in.mp4", "out_dir": "/data/jobs/%s", "target_lang": "en", "output": {"keep_intermediate": true}, "debug_env": {"DUB_AUDIO_FROM": "/dub"}}\n' $N $N > $Q/$N/job.json
  local t0=$(date +%s)
  docker run --rm --gpus all --shm-size 16g -v dabai-models:/workspace/dub/dubai/models -v $Q/$N:/data -v $Q/$N/job.json:/job.json:ro -v $CLIP:/in.mp4:ro \
    -v $DUBDIR:/dub:ro $T job /job.json > $Q/$N/run.log 2>&1
  log "${N}_DONE rc=$? $(( $(date +%s) - t0 )) s status $(python3 -c "import json; print(json.load(open('$Q/$N/jobs/$N/status.json'))['status'])" 2>/dev/null)"
}
cmp(){   # cmp <name> <a.mp4> <b.mp4>
  docker run --rm -v /root:/root --entrypoint /venv/dabai/bin/python $T /root/q_compare.py --a $2 --b $3 --src $CLIP --json $Q/cmp_$1.json > $Q/cmp_$1.log 2>&1
  log "CMP $1 $(tail -1 $Q/cmp_$1.log | cut -c13-400)"
}
out(){ python3 -c "import json; print(json.load(open('$1'))['output'])" 2>/dev/null | sed "s#^/data#$(dirname $(dirname $(dirname $1)))#"; }
run_new(){   # run_new <name>: the NEW code (mounted snapshot), the same fixed dub
  local N=$1; mkdir -p $Q/$N
  printf '{"job_id": "%s", "input": "/in.mp4", "out_dir": "/data/jobs/%s", "target_lang": "en", "output": {"keep_intermediate": true}, "debug_env": {"DUB_AUDIO_FROM": "/dub"}}
' $N $N > $Q/$N/job.json
  local t0=$(date +%s)
  docker run --rm --gpus all --shm-size 16g -v $C/scripts:/workspace/dub/dubai/scripts:ro -v $C/infra:/workspace/dub/dubai/infra:ro -v dabai-models:/workspace/dub/dubai/models     -v $Q/$N:/data -v $Q/$N/job.json:/job.json:ro -v $CLIP:/in.mp4:ro -v $DUBDIR:/dub:ro $T job /job.json > $Q/$N/run.log 2>&1
  log "${N}_DONE rc=$? $(( $(date +%s) - t0 )) s status $(python3 -c "import json; print(json.load(open('$Q/$N/jobs/$N/status.json'))['status'])" 2>/dev/null)"
}
run_image A2; run_image A3; run_new A4
A=$(out $Q/A/jobs/qa_cold/status.json); A2=$(out $Q/A2/jobs/A2/status.json); A3=$(out $Q/A3/jobs/A3/status.json); A4=$(out $Q/A4/jobs/A4/status.json)
cmp A2_vs_A3_floor "$A2" "$A3"; cmp A2_vs_A4_newcode "$A2" "$A4"
for n in A2 A3 A4; do log "PASTE $n: $(grep -h 'plain paste:' $(find $Q/$n -name face_keysync.log | head -1) | grep -o '[0-9]* workers[^)]*')"; done
# ---- B: the resident API, 3 jobs posted at once (12.4 burst), then the restart test
log "B start"
VOL=dabai-models; O=$Q/B; mkdir -p $O/data; t0=$(date +%s.%N); el(){ python3 -c "import time; print(round(time.time() - $t0, 1))"; }
CID=$(docker run -d --gpus all --shm-size 16g -p 18300:8000 -e DABAI_RESIDENT=1 -e DABAI_API_TOKEN=bench -v $C/scripts:/workspace/dub/dubai/scripts:ro \
  -v $C/infra:/workspace/dub/dubai/infra:ro -v $VOL:/workspace/dub/dubai/models -v $O/data:/data -v $CLIP:/in/clip.mp4:ro $T)
H=http://127.0.0.1:18300; AU=(-H "Authorization: Bearer bench")
for i in $(seq 600); do curl -sf $H/health > $O/health0.json && break; sleep 0.5; done; log "B (a) /health 200 at $(el) s"
for i in $(seq 1800); do curl -s $H/health > $O/health1.json; grep -q '"state": *"ready"' $O/health1.json && break; sleep 1; done
log "B (b) resident ready at $(el) s: $(python3 -c "import json; r=json.load(open('$O/health1.json'))['resident']; print({k: r.get(k) for k in ('state', 'ks_ready_s', 'audio_up_s', 'total_s')})")"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_idle.txt; log "B VRAM idle (models parked) $(cat $O/vram_idle.txt)"
for j in 1 2 3; do curl -s "${AU[@]}" -H "content-type: application/json" -d "{\"job_id\": \"rb$j\", \"input\": \"/in/clip.mp4\", \"target_lang\": \"en\", \"output\": {\"keep_intermediate\": true}}" $H/jobs > $O/post_rb$j.json; done
log "B posted 3: $(cat $O/post_rb1.json | cut -c1-80) ..."
mx=0; for i in $(seq 7200); do
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ $v -gt $mx ] && mx=$v
  s3=$(curl -s "${AU[@]}" $H/jobs/rb3 | python3 -c "import json, sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null)
  s2=$(curl -s "${AU[@]}" $H/jobs/rb2 | python3 -c "import json, sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null)
  [ "$s2" = SUCCESS ] || [ "$s2" = PARTIAL_SUCCESS ] || [ "$s2" = ERROR ] && [ "$s3" = RUNNING ] && break; sleep 2; done
log "B rb1 / rb2 done, rb3 running; VRAM max so far $mx MiB"
for j in 1 2; do log "B rb$j $(python3 -c "
import json; s=json.load(open('$O/data/jobs/rb$j/status.json')); print(s['status'], 'wall', s.get('timing', {}).get('wall_s') if isinstance(s.get('timing'), dict) else '', 'gpu_busy_s', s.get('gpu_busy_s'), 'started', s.get('started_at'), 'finished', s.get('finished_at'))")"; done
sleep 60; log "B KILL the container during rb3 (12.4 restart)"; docker kill $CID > /dev/null
CID=$(docker run -d --gpus all --shm-size 16g -p 18300:8000 -e DABAI_API_TOKEN=bench -v $C/scripts:/workspace/dub/dubai/scripts:ro -v $C/infra:/workspace/dub/dubai/infra:ro \
  -v $VOL:/workspace/dub/dubai/models -v $O/data:/data -v $CLIP:/in/clip.mp4:ro $T)
for i in $(seq 600); do curl -sf $H/health > /dev/null && break; sleep 0.5; done
log "B after restart: rb3 = $(curl -s "${AU[@]}" $H/jobs/rb3 | python3 -c "import json, sys; s=json.load(sys.stdin); print(s.get('status'), (s.get('error') or {}).get('code'), (s.get('error') or {}).get('retryable'))")"
docker logs $CID > $O/container_after_restart.log 2>&1; docker stop $CID > /dev/null
for j in 1 2; do cp $O/data/jobs/rb$j/job.log $O/rb$j.job.log 2>/dev/null; done
RB1=$(out $O/data/jobs/rb1/status.json); RB2=$(out $O/data/jobs/rb2/status.json)
cmp A_vs_rb1_resident_first "$A" "$RB1"; cmp A_vs_rb2_resident_warm "$A" "$RB2"
for n in A/jobs/qa_cold B/data/jobs/rb1 B/data/jobs/rb2; do log "DUB md5 $n: dub_f0 $(find $Q/$n -name dub_f0.wav | head -1 | xargs -r md5sum | cut -c1-32) mp_landmarks $(find $Q/$n -path '*ks/master_mp.npz' | head -1 | xargs -r md5sum | cut -c1-32)"; done
log "CHAIN_DONE"
