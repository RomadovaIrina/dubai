#!/bin/bash
# r15 W: the production baseline q2 + F3 (one-command E2E via dabai_job, as scripts/r14/eval_r14.sh) on old03 / m03 / m18.
# Its dub audio (artifacts/r4a/r9_q2_<id>) is THE audio of every r15 W arm (scripts/r15/arm.sh: face stages only).
# The E2E audio stage (Qwen 14B + Chatterbox on the GPU, ~27 GB peak) takes no gpu.lock, so it OOMs next to another agent's locked GPU job:
# a side process holds gpu.lock from the job start until the e2e reaches its "sep" stage (nothing in precheck / audio waits on the lock:
# the backgrounded demucs remix's gap ASR queues on it); then up to 3 tries.
#   bash scripts/r15/base_q2.sh [old03 m03 m18]
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; mkdir -p $X
COMMON='"LS_CMS_READ": "1", "FC_CMS": "1", "FC_EXTRA": "--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45"'
declare -A SRC=([old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [m03]=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV [m18]=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4)
for id in ${*:-old03 m03 m18}; do
  in=${SRC[$id]}; n=q2_$id; O=$X/$n
  for i in $(seq 360); do [ -s "$in" ] && s1=$(stat -c %s "$in") && sleep 10 && [ "$s1" = "$(stat -c %s "$in")" ] && break; sleep 10; done   # the upload may still be running
  [ -s "$O/$n.mp4" ] && { echo "have $n"; continue; }
  python3 -c "import json; json.dump({'input': '$in', 'out_dir': '$O', 'job_id': '$n', 'target_lang': 'en', 'debug_env': json.loads('{$COMMON}')}, open('$X/$n.job.json', 'w'))"
  for try in 1 2 3; do
    rm -rf $O $X/$n.locked $X/$n.ended; mkdir -p $O
    ( exec 9>/workspace/dub/gpu.lock; flock 9; touch $X/$n.locked
      until grep -qE "^(sep|polyglot|pace_fix|face_gen|mux|end)	" $O/e2e/work/stages.tsv 2>/dev/null || [ -f $X/$n.ended ]; do sleep 1; done ) &
    HP=$!; until [ -f $X/$n.locked ]; do sleep 1; done
    t0=$(date +%s); python scripts/r13/dabai_job.py $X/$n.job.json > $X/$n.log 2>&1; touch $X/$n.ended; wait $HP
    echo "$n try $try $(tail -1 $X/$n.log | cut -c1-160) $(( $(date +%s) - t0 )) s"
    [ -s $O/$n.mp4 ] && break
  done
done
echo BASE_DONE
