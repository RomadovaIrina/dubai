#!/bin/bash
# r6 follow-ups (coordinator, after the lip-recipe eval): (1) r6c8u on m18 (+ q2u m18 on this server), (2) the pick's robustness:
# steps 500 and 1250 on old03 + m03, (3) m03 mouth "explosions" (agent F's scripts/r15/explode.py, f74f9b4) q2u vs r6c8u (+ the r14
# student and the recipe finals); then the tables, the report files committed, the r6 bundle, R6_DONE.
#   setsid nohup bash scripts/r15/r6_follow.sh > /workspace/dub/r15x/r6_follow.log 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null
X=${R5_ROOT:-/workspace/dub/r15x}; E=$X/eval; N=$X/R6_NOTES.txt; C=$X/ckpt/r6_s4; OB=/workspace/dub/outbox/r15x; B=/workspace/dub/artifacts/r4m; export EVAL_DIR=$E
note(){ echo "$(date +%T) $*" | tee -a $N; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 2000 ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done; note "GPU not idle after 60 min, going anyway"; }
FU='"LS_FACE_UNITS": "1"'; C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'
ev(){ local tag=$1 ck=$2 id=$3; [ -s $E/${tag}_$id/${tag}_$id.mp4 ] && return 0
  local free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9); [ $free -ge ${MIN_FREE_G:-6} ] || { note "$tag $id skipped: only ${free} GB free"; return 0; }; idle
  Q2=q2u Q2ENV="$FU" OB=$OB LORA=$ck STEPS=4 TAG=$tag XENV="$C8, $FU" bash scripts/r14/eval_r14.sh $id >> $X/eval_r6f.log 2>&1
  [ -s $E/${tag}_$id/${tag}_$id.mp4 ] || note "$tag $id: no output video ($(tail -1 $E/${tag}_$id.log 2>/dev/null | cut -c1-200))"
  [ "${KEEP_LAYERS:-0}" = 1 ] || rm -rf -- $B/r9g_${tag}_$id; }   # the generator-stage layers (~2.5 GB / arm): the final is in $E (r15-w ran out of disk at 19:30)
until grep -q "R6 chain done" $N || ! pgrep -f "scripts/r15/r6_chain.sh" >/dev/null; do sleep 30; done
note "follow-ups (re)start: $(df -h --output=avail /workspace | tail -1 | tr -d " ") free"
PK=$C/$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick'])"); note "follow-ups start (pick $(basename $PK))"
KEEP_LAYERS=1 ev r6c8u $PK m18
STEPRUNS(){ for id in m03 old03; do ev r6s500c8u $C/step_00500_ema.pt $id; ev r6s1250c8u $C/step_01250_ema.pt $id; done; }
STEPRUNS
python scripts/r15/r5_table.py --root $X --run r6_s4 --base q2u --arms s4p1500c8u r5fc8u r6c8u r6s500c8u r6s1250c8u --ids old03 m03 m18 > $X/R6_TABLE.md 2>&1
tail -14 $X/R6_TABLE.md | tee -a $N
# (3) explosions on m03 (CPU): each primary's events + every final's frame count over the threshold
v=m03; W=$B/r9g_q2u_$v/work_q2u_${v}_r9g_q2u_$v; T=$B/r9g_q2u_$v/timing; SRC=$(ls $E/q2u_$v/e2e/work/in/q2u_$v.* | head -1)
FIN="--final q2u=$E/q2u_$v/q2u_$v.mp4 --final r6c8u=$E/r6c8u_$v/r6c8u_$v.mp4 --final s4p1500c8u=$E/s4p1500c8u_$v/s4p1500c8u_$v.mp4"
for a in q2u r6c8u; do f=$B/r9f_${a}_${v}rec/${a}_${v}_r9f_${a}_${v}rec.mp4; [ -s $f ] && FIN="$FIN --final ${a}rec=$f"; done
for P in q2u r6c8u; do   # the primary first: its events and sheet
  F1="--final $P=$E/${P}_$v/${P}_$v.mp4 $(echo "$FIN" | sed "s|--final $P=[^ ]*||")"
  python scripts/r15/explode.py --work $W --timing $T/q2u_${v}_master_timing_F.json --src $SRC --src-lm $T/q2u_${v}_source_lm_F.npz $F1 \
    --json $X/explode_${v}_$P.json --sheet $OB/${v}_explode_${P}.jpg > $X/explode_${v}_$P.log 2>&1 || note "explode $P failed: $(tail -1 $X/explode_${v}_$P.log | cut -c1-200)"
  note "explode m03 primary $P: $(grep -E "pop mean|== explode" $X/explode_${v}_$P.log | tr '\n' ' ' | cut -c1-600)"
done
D=reports/experiments/r14-distill; for f in R6_TABLE.md R6_RECIPE_TABLE.md R6_NOTES.txt; do [ -s $X/$f ] && cp $X/$f $D/$f; done
cp $C/ckpt_val.json $D/ckpt_val_r6_s4.json; for f in $X/explode_m03_*.json; do [ -s $f ] && cp $f $D/; done
for p in $D/R6_TABLE.md $D/R6_RECIPE_TABLE.md $D/R6_NOTES.txt $D/ckpt_val_r6_s4.json $D/explode_m03_q2u.json $D/explode_m03_r6c8u.json scripts/r15/r6_follow.sh scripts/r15/explode.py; do [ -e $p ] && git add $p; done
git commit -q -m "r6 (r15 lever b) results: the CFG-distilled 4-step student vs q2u / the r14 student (E2E old03 / m03 / m18, steps 250 / 500 / 1250), F's lip recipe on all three, m03 explosions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && note "commit $(git rev-parse --short HEAD)"
git bundle create $X/r6-distill.bundle experiment/r14-distill ^origin/experiment/r13-spec ^origin/experiment/r13 && git bundle verify $X/r6-distill.bundle >/dev/null 2>&1 \
  && note "bundle ok $X/r6-distill.bundle ($(du -h $X/r6-distill.bundle | cut -f1))" || note "BUNDLE FAILED"
touch $X/R6_DONE; note "R6_DONE"
