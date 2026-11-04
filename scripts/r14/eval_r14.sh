#!/bin/bash
# r14 student eval on the held-out videos: the one-command E2E (dabai_job) with the production LatentSync stack q2 + agent F's F3 composite
# flags, and the same with the student merged into the generator; SyncNet (sync_speech, the gate's), flicker (scripts/r14/flicker_metric.py,
# the user's old03 timestamps), 2-panels q2 | student (full + 0.5x mouth crop) -> outbox/r14_d/.
#   LORA=<ckpt.pt> STEPS=4 TAG=s4 [XENV='"LS_STRIDE": "8", "LS_OVERLAP": "fuse"'] bash scripts/r14/eval_r14.sh [old03 m03 m18]      (BASE=1: the q2 + F3 baseline too, when missing)
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r14d/eval; OB=/workspace/dub/outbox/r14_d; mkdir -p $X $OB; A=/workspace/dub/artifacts/r4a
TAG=${TAG:-s4}; STEPS=${STEPS:-4}; LORA=${LORA:?LORA=<ckpt>}
[ -f $X/skip_$TAG ] && { echo "skip $TAG ($X/skip_$TAG)"; exit 0; }   # arms dropped while the chain runs
# both arms: q2 (reports/experiments/r13-ls/REPORT.md: BT.709 colour management + luma-only capped lip sharpen) + F3 (r14 agent F: --reg-smooth 4 --mouth-region 0.45)
COMMON='"LS_CMS_READ": "1", "FC_CMS": "1", "FC_EXTRA": "--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45"'
declare -A SRC=([old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [m03]=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV [m18]=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4)
job(){   # job <name> <input> [debug_env json]
  local n=$1 in=$2 env=${3:-}; local O=$X/$n
  [ -s "$O/$n.mp4" ] && { echo "have $n"; return; }
  python3 -c "import json,sys; d={'input': '$in', 'out_dir': '$O', 'job_id': '$n', 'target_lang': 'en'}; e='''$env'''; d.update({'debug_env': json.loads(e)} if e else {}); json.dump(d, open('$X/$n.job.json', 'w'))"
  local t0=$(date +%s); python scripts/r13/dabai_job.py $X/$n.job.json > $X/$n.log 2>&1; echo "$n $(tail -1 $X/$n.log | cut -c1-160) $(( $(date +%s) - t0 )) s"
}
for id in ${*:-old03 m03 m18}; do
  in=${SRC[$id]}
  [ -n "${BASE:-1}" ] && job q2_$id $in "{$COMMON}"
  job ${TAG}_$id $in "{$COMMON, \"R10_GEN_PY\": \"/workspace/dub/dubai/scripts/r13/distill/student_gen.py\", \"R13_STUDENT_LORA\": \"$LORA\", \"R13_STUDENT_STEPS\": \"$STEPS\"${XENV:+, $XENV}}"
  for n in q2_$id ${TAG}_$id; do
    v=$(ls $X/$n/$n.mp4 2>/dev/null); w=$(ls $A/r9_$n/$n/dubbed_16k.wav 2>/dev/null | head -1)
    [ -n "$v" ] && [ -n "$w" ] && [ ! -s $X/$n.sync.txt ] && flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $v --audio $w --label $n 2>&1 | grep "^== sync" > $X/$n.sync.txt
    echo "$(cat $X/$n.sync.txt 2>/dev/null)"
  done
  lm=$(find /workspace/dub/artifacts -path "*q2_$id*" -name "source_lm*.npz" 2>/dev/null | head -1)
  if [ -n "$lm" ]; then
    AT=""; [ $id = old03 ] && AT="--at 3.5 5.5 20.9 24 27.5"
    python scripts/r14/flicker_metric.py --lm $lm --video src=$in q2=$X/q2_$id/q2_$id.mp4 $TAG=$X/${TAG}_$id/${TAG}_$id.mp4 $AT --json $X/${TAG}_$id.flicker.json
    python scripts/r13/distill/mouth_2panel.py --left $X/q2_$id/q2_$id.mp4 --right $X/${TAG}_$id/${TAG}_$id.mp4 --lm $lm --out $OB/${id}_mouth_q2_vs_${TAG}.mp4 --labels "q2 + F3 (production)" "student $TAG"
    ffmpeg -v error -y -i $OB/${id}_mouth_q2_vs_${TAG}.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast $OB/${id}_mouth_q2_vs_${TAG}_half_speed.mp4
    ffmpeg -v error -y -i $X/q2_$id/q2_$id.mp4 -i $X/${TAG}_$id/${TAG}_$id.mp4 -filter_complex "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=inputs=2[v]" -map "[v]" -map 0:a -c:v libx264 -crf 20 -preset fast -c:a aac $OB/${id}_full_q2_vs_${TAG}.mp4
    if [ $id = old03 ]; then   # the user's flicker timestamps: each span (+-0.25 s) at 0.5x with its time burnt in, concatenated
      parts=""; k=0
      for span in 2.75:4.25 4.75:6.25 20.55:21.25 23.5:24.5 26.75:28.25; do
        s=${span%:*}; e=${span#*:}; k=$((k+1))
        ffmpeg -v error -y -ss $s -to $e -i $OB/${id}_mouth_q2_vs_${TAG}.mp4 -filter_complex "[0:v]setpts=2*PTS,drawtext=text='t=$s-$e s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
          -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac -ar 48000 $X/ts_$k.mp4 && parts="$parts -i $X/ts_$k.mp4"
      done
      n=$(echo $parts | grep -o "\-i" | wc -l)
      ffmpeg -v error -y $parts -filter_complex "$(for i in $(seq 0 $((n-1))); do echo -n "[$i:v][$i:a]"; done)concat=n=$n:v=1:a=1[v][a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac $OB/${id}_mouth_q2_vs_${TAG}_user_timestamps_half_speed.mp4
    fi
  else echo "no landmarks for $id (flicker / 2-panel skipped)"; fi
done
echo "EVAL_DONE $TAG"
