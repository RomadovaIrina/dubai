#!/bin/bash
# r14 student eval on the held-out videos: the one-command E2E (dabai_job) with the production default (k5) and with the student merged into the
# generator; SyncNet (sync_speech, the gate's), flicker (scripts/r14/flicker_metric.py, the user's old03 timestamps), 2-panels k5 | student
# (full + 0.5x mouth crop) -> outbox/r14_d/.
#   LORA=<ckpt.pt> STEPS=4 TAG=s4 bash scripts/r14/eval_r14.sh [old03 m03 m18]      (BASE=1 also runs the k5 baseline if missing)
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r14d/eval; OB=/workspace/dub/outbox/r14_d; mkdir -p $X $OB; A=/workspace/dub/artifacts/r4a
TAG=${TAG:-s4}; STEPS=${STEPS:-4}; LORA=${LORA:?LORA=<ckpt>}
declare -A SRC=([old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [m03]=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV [m18]=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4)
job(){   # job <name> <input> [debug_env json]
  local n=$1 in=$2 env=${3:-}; local O=$X/$n
  [ -s "$O/$n.mp4" ] && { echo "have $n"; return; }
  python3 -c "import json,sys; d={'input': '$in', 'out_dir': '$O', 'job_id': '$n', 'target_lang': 'en'}; e='''$env'''; d.update({'debug_env': json.loads(e)} if e else {}); json.dump(d, open('$X/$n.job.json', 'w'))"
  local t0=$(date +%s); python scripts/r13/dabai_job.py $X/$n.job.json > $X/$n.log 2>&1; echo "$n $(tail -1 $X/$n.log | cut -c1-160) $(( $(date +%s) - t0 )) s"
}
for id in ${*:-old03 m03 m18}; do
  in=${SRC[$id]}
  [ -n "${BASE:-1}" ] && job k5_$id $in
  job ${TAG}_$id $in "{\"R10_GEN_PY\": \"/workspace/dub/dubai/scripts/r13/distill/student_gen.py\", \"R13_STUDENT_LORA\": \"$LORA\", \"R13_STUDENT_STEPS\": \"$STEPS\"}"
  for n in k5_$id ${TAG}_$id; do
    v=$(ls $X/$n/$n.mp4 2>/dev/null); w=$(ls $A/r9_$n/$n/dubbed_16k.wav 2>/dev/null | head -1)
    [ -n "$v" ] && [ -n "$w" ] && [ ! -s $X/$n.sync.txt ] && flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $v --audio $w --label $n 2>&1 | grep "^== sync" > $X/$n.sync.txt
    echo "$(cat $X/$n.sync.txt 2>/dev/null)"
  done
  lm=$(find /workspace/dub/artifacts -path "*k5_$id*" -name "source_lm*.npz" 2>/dev/null | head -1)
  if [ -n "$lm" ]; then
    AT=""; [ $id = old03 ] && AT="--at 3.5 5.5 20.9 24 27.5"
    python scripts/r14/flicker_metric.py --lm $lm --video src=$in k5=$X/k5_$id/k5_$id.mp4 $TAG=$X/${TAG}_$id/${TAG}_$id.mp4 $AT --json $X/${TAG}_$id.flicker.json
    python scripts/r13/distill/mouth_2panel.py --left $X/k5_$id/k5_$id.mp4 --right $X/${TAG}_$id/${TAG}_$id.mp4 --lm $lm --out $OB/${id}_mouth_k5_vs_${TAG}.mp4 --labels "k5 (production)" "student $TAG"
    ffmpeg -v error -y -i $OB/${id}_mouth_k5_vs_${TAG}.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast $OB/${id}_mouth_k5_vs_${TAG}_half_speed.mp4
    ffmpeg -v error -y -i $X/k5_$id/k5_$id.mp4 -i $X/${TAG}_$id/${TAG}_$id.mp4 -filter_complex "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=inputs=2[v]" -map "[v]" -map 0:a -c:v libx264 -crf 20 -preset fast -c:a aac $OB/${id}_full_k5_vs_${TAG}.mp4
  else echo "no landmarks for $id (flicker / 2-panel skipped)"; fi
done
echo "EVAL_DONE $TAG"
