#!/bin/bash
# r6 @1250 (the final; sync rises with the steps): m18 E2E (layers kept for the lip recipe), then the eye clips q2u | r6@1250
# (mouth 2-panel, 0.5x, time burnt in) at the user's spots: old03 3-7 / 23-25 s, m03 17-21 s, m18 2-6 s; table; commit.
#   setsid nohup bash scripts/r15/r6_more.sh > /workspace/dub/r15x/r6_more.log 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null
X=${R5_ROOT:-/workspace/dub/r15x}; E=$X/eval; N=$X/R6_NOTES.txt; C=$X/ckpt/r6_s4; OB=/workspace/dub/outbox/r15x; export EVAL_DIR=$E
note(){ echo "$(date +%T) $*" | tee -a $N; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 2000 ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done; note "GPU not idle after 60 min, going anyway"; }
FU='"LS_FACE_UNITS": "1"'; C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'; TAG=r6s1250c8u
if [ ! -s $E/${TAG}_m18/${TAG}_m18.mp4 ]; then idle
  Q2=q2u Q2ENV="$FU" OB=$OB LORA=$C/step_01250_ema.pt STEPS=4 TAG=$TAG XENV="$C8, $FU" bash scripts/r14/eval_r14.sh m18 >> $X/eval_r6m.log 2>&1
  [ -s $E/${TAG}_m18/${TAG}_m18.mp4 ] || note "$TAG m18: no output video ($(tail -1 $E/${TAG}_m18.log 2>/dev/null | cut -c1-200))"; fi
note "m18: $(cat $E/q2u_m18.sync.txt 2>/dev/null) | $(cat $E/${TAG}_m18.sync.txt 2>/dev/null)"
for spec in old03:3:7 old03:23:25 m03:17:21 m18:2:6; do IFS=: read v s e <<< "$spec"
  src=$OB/${v}_mouth_q2u_vs_$TAG.mp4; [ -s $src ] || { note "no 2-panel $src"; continue; }
  ffmpeg -v error -y -ss $s -to $e -i $src -filter_complex "[0:v]setpts=2*PTS,drawtext=text='q2u | r6@1250   t=$s-$e s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac $OB/${v}_mouth_q2u_vs_${TAG}_t$s-${e}_half_speed.mp4 && echo "clip ${v} $s-$e"
done
python scripts/r15/r5_table.py --root $X --run r6_s4 --base q2u --arms s4p1500c8u r5fc8u r6c8u r6s500c8u r6s1250c8u --ids old03 m03 m18 > $X/R6_TABLE.md 2>&1
D=reports/experiments/r14-distill; cp $X/R6_TABLE.md $X/R6_NOTES.txt $D/; git add $D/R6_TABLE.md $D/R6_NOTES.txt scripts/r15/r6_more.sh
git commit -q -m "r6 @1250 on m18 + the eye-spot clips q2u | r6@1250 (old03 3-7 / 23-25, m03 17-21, m18 2-6 s)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && note "commit $(git rev-parse --short HEAD)"
note "R6_MORE_DONE"
