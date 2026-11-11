#!/bin/bash
# r17 resident worker benchmark (TZ 0.10 / 0.4 / Irina 11.11), on a Docker GPU host with the image + a filled models volume:
#   bash infra/docker/resident_bench.sh <image> <clip.mp4> [out dir] [jobs (2)]
# One container in API mode with DABAI_RESIDENT=1 -> container start -> /health 200 (a) -> resident ready (b: KeySync loaded + parked, audio worker
# up) -> N jobs of the same clip one after the other through the API; per job: wall, stage walls, GPU s, the KeySync daemon's unpark / park lines
# and the audio worker's park lines, the GPU memory in use between the jobs. -> <out>/resident_bench.json. Pushes nothing.
set -uo pipefail
IMG=${1:?image}; CLIP=$(readlink -f "${2:?clip}"); O=$(mkdir -p "${3:-resident_bench_out}" && readlink -f "${3:-resident_bench_out}"); N=${4:-2}
VOL=${VOL:-dabai-models}; P=${PORT:-18200}; H=http://127.0.0.1:$P; TOK=bench; AU=(-H "Authorization: Bearer $TOK")
[ -f "$CLIP" ] || { echo "no clip $CLIP"; exit 2; }
mkdir -p $O/data; t0=$(date +%s.%N); el() { python3 -c "import time; print(round(time.time() - $t0, 1))"; }
CID=$(docker run -d --gpus all --shm-size 16g -p $P:8000 -e DABAI_RESIDENT=1 -e DABAI_API_TOKEN=$TOK -v $VOL:/workspace/dub/dubai/models \
      -v $O/data:/data -v $CLIP:/in/clip.mp4:ro "$IMG")
echo "container $CID"; trap 'docker logs $CID > $O/container.log 2>&1; docker stop $CID > /dev/null 2>&1' EXIT
for i in $(seq 600); do curl -sf $H/health > $O/health0.json && break; sleep 0.5; done; A=$(el); echo "(a) /health 200 at $A s"
for i in $(seq 3600); do curl -s $H/health > $O/health1.json; grep -q '"state": *"ready"' $O/health1.json && break
  grep -q '"state": *"starting"' $O/health1.json || { echo "resident not starting: $(cat $O/health1.json)"; break; }; sleep 1; done
B=$(el); echo "(b) resident ready at $B s: $(python3 -c "import json; print(json.load(open('$O/health1.json')).get('resident'))")"
nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_idle.txt
for j in $(seq 1 $N); do
  id=rb$j; tj=$(date +%s.%N)
  curl -s "${AU[@]}" -H "content-type: application/json" -d "{\"job_id\": \"$id\", \"input\": \"/in/clip.mp4\", \"target_lang\": \"en\"}" $H/jobs > $O/post_$id.json
  for i in $(seq 7200); do curl -s "${AU[@]}" $H/jobs/$id > $O/status_$id.json; grep -q -E '"status": *"(SUCCESS|PARTIAL_SUCCESS|ERROR)"' $O/status_$id.json && break; sleep 1; done
  echo "job $id: $(python3 -c "import json; s=json.load(open('$O/status_$id.json')); print(s['status'], 'gpu_busy_s', s.get('gpu_busy_s'))") wall $(python3 -c "import time; print(round(time.time() - $tj, 1))") s"
  echo "$(python3 -c "import time; print(round(time.time() - $tj, 1))")" > $O/wall_$id.txt
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_after_$id.txt
done
docker cp $CID:/data/resident $O/resident 2>/dev/null
python3 - $O $N $A $B <<'PY'
import glob, json, os, re, sys
o, n, a, b = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
r = {"a_health_s": a, "b_resident_ready_s": b, "vram_idle": open(f"{o}/vram_idle.txt").read().strip(), "jobs": []}
for j in range(1, n + 1):
    s = json.load(open(f"{o}/status_rb{j}.json")); jl = f"{o}/data/jobs/rb{j}/job.log"; txt = open(jl).read() if os.path.exists(jl) else ""
    st = re.findall(r"== rb\d+ stage (\w+) (\d\d:\d\d:\d\d)", txt)
    ksl = sorted(glob.glob(f"{o}/data/jobs/rb{j}/e2e/logs/face_keysync.log"))
    kl = open(ksl[0]).read() if ksl else ""
    r["jobs"].append({"id": f"rb{j}", "status": s.get("status"), "wall_s": float(open(f"{o}/wall_rb{j}.txt").read()), "gpu_busy_s": s.get("gpu_busy_s"),
                      "stages": st, "keysync_served": "KS_SERVE resident" in kl, "ks_serve": re.findall(r"KS_SERVE[^\n]*", kl)[:3],
                      "vram_after": open(f"{o}/vram_after_rb{j}.txt").read().strip()})
aw = f"{o}/resident/audio/worker.log"
r["audio_worker"] = re.findall(r"\[worker\][^\n]*", open(aw).read()) if os.path.exists(aw) else []
json.dump(r, open(f"{o}/resident_bench.json", "w"), indent=1); print("RESIDENT_BENCH " + json.dumps(r))
PY
