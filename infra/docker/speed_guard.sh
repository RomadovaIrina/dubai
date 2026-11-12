#!/bin/bash
# r17 quality guard for the wall-time speedups (Irina: identical output or Irina's eye), on a Docker GPU VM with the image + the models volume.
#   bash infra/docker/speed_guard.sh <image> <stories dir> [out dir]
# Clip: 3 x 20 s of 3 different stories = 3 shots (the multi-shot path the speedups act on), 720x1280 30 fps.
# base = every speedup OFF; fast = KS_SHOT_PREP_JOBS=8 KS_SHOT_PIPE=1 KS_SYNC_SERVE=1 E2E_ROUTE_PRE=1 (+ the ks_pre / paste workers already on).
# Each a fresh job (fresh dub: the dub is deterministic, md5-equal run to run, VM 11.11). Compared: every frame (out_compare.py), the decoded mix,
# the per-shot dub md5, route.json, the sync probe / check json, the stage walls. Not identical -> base2 (the floor) is run and the verdict
# says which speedup to bisect. -> <out>/speed_guard.json. Pushes nothing.
set -uo pipefail
IMG=${1:?image}; ST=$(readlink -f "${2:?stories dir}"); O=$(mkdir -p "${3:-speed_guard_out}" && readlink -f "${3:-speed_guard_out}"); VOL=${VOL:-dabai-models}
log(){ echo "[$(date -u +%T)] $*" | tee -a $O/progress.log; }
mapfile -t F < <(ls -S $ST/* | head -3)   # the 3 largest stories (long enough for 20 s)
mkdir -p $O/clip; : > $O/clip/list.txt
for k in 0 1 2; do docker run --rm -v $ST:/s:ro -v $O/clip:/o --entrypoint ffmpeg $IMG -v error -y -ss 2 -t 20 -i "/s/$(basename "${F[$k]}")" \
  -vf "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p" -c:v libx264 -crf 16 -preset fast -c:a aac -b:a 192k -ar 48000 -ac 2 /o/p$k.mp4
  echo "file /o/p$k.mp4" >> $O/clip/list.txt; done
docker run --rm -v $O/clip:/o --entrypoint ffmpeg $IMG -v error -y -f concat -safe 0 -i /o/list.txt -c copy /o/guard3.mp4; CLIP=$O/clip/guard3.mp4
log "clip $(docker run --rm -v $O/clip:/o --entrypoint ffprobe $IMG -v error -show_entries format=duration -of csv=p=0 /o/guard3.mp4) s from ${F[*]##*/}"
run(){   # run <name> <debug_env json>
  local N=$1; mkdir -p $O/$N
  printf '{"job_id": "%s", "input": "/in.mp4", "out_dir": "/data/jobs/%s", "target_lang": "en", "output": {"keep_intermediate": true}, "debug_env": %s}\n' $N $N "$2" > $O/$N/job.json
  local t0=$(date +%s)
  docker run --rm --gpus all --shm-size 16g -v $VOL:/workspace/dub/dubai/models -v $O/$N:/data -v $O/$N/job.json:/job.json:ro -v $CLIP:/in.mp4:ro $IMG job /job.json > $O/$N/run.log 2>&1
  log "$N rc=$? $(( $(date +%s) - t0 )) s: $(grep -E "^== $N (SUCCESS|PARTIAL|ERROR) flags" $O/$N/jobs/$N/job.log | grep -o 'total.*' | cut -c1-250)"
}
OFF='{"KS_SHOT_PREP_JOBS": "1", "KS_SHOT_PIPE": "0", "KS_SYNC_SERVE": "0", "E2E_ROUTE_PRE": "0"}'
ON='{"KS_SHOT_PREP_JOBS": "8", "KS_SHOT_PIPE": "1", "KS_SYNC_SERVE": "1", "E2E_ROUTE_PRE": "1"}'
run base "$OFF"; run fast "$ON"
cmp(){ docker run --rm -v $O:$O --entrypoint /venv/dabai/bin/python $IMG $O/out_compare.py --a $O/$1/jobs/$1/$1.mp4 --b $O/$2/jobs/$2/$2.mp4 --src $CLIP --json $O/cmp_$1_$2.json > $O/cmp_$1_$2.log 2>&1; grep OUT_COMPARE $O/cmp_$1_$2.log | cut -c13-400; }
cp "$(dirname "$0")/../../scripts/keysync/r16/tz/out_compare.py" $O/out_compare.py 2>/dev/null || cp /root/dubai/scripts/keysync/r16/tz/out_compare.py $O/out_compare.py
log "CMP base vs fast: $(cmp base fast)"
md5s(){ (cd $O/$1/jobs/$1/e2e/work && find . \( -name 'dub_f*.wav' -o -name route.json -o -name 'sync_*.json' \) -type f | sort | while read f; do echo "$f $(md5sum < $f | cut -c1-12)"; done); }
md5s base > $O/md5_base.txt; md5s fast > $O/md5_fast.txt
log "per-file md5 (dubs / route / sync json) differ: $(diff $O/md5_base.txt $O/md5_fast.txt | grep -c '^[<>]') lines"
if ! python3 -c "import json, sys; sys.exit(0 if json.load(open('$O/cmp_base_fast.json'))['all_identical'] else 1)"; then
  log "NOT identical: running base2 for the floor"; run base2 "$OFF"; log "CMP base vs base2 (floor): $(cmp base base2)"; fi
python3 - $O <<'PY'
import json, os, re, sys
o = sys.argv[1]; r = {}
for n in ("base", "fast", "base2"):
    f = f"{o}/{n}/jobs/{n}/e2e/timing.json"
    if os.path.exists(f): t = json.load(open(f)); r[n] = {"total_s": t["total_wall_s"], "s_per_vmin": t["total_s_per_video_min"], "stages": t["stage_wall_s"]}
for c in ("base_fast", "base_base2"):
    f = f"{o}/cmp_{c}.json"
    if os.path.exists(f): r["cmp_" + c] = json.load(open(f))
r["verdict"] = "IDENTICAL: the speedups can be default on" if r.get("cmp_base_fast", {}).get("all_identical") else "NOT identical: compare with the floor, bisect the flags"
json.dump(r, open(f"{o}/speed_guard.json", "w"), indent=1); print("SPEED_GUARD " + json.dumps(r))
PY
