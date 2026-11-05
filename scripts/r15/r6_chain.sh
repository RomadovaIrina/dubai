#!/bin/bash
# r15 lever b = r6: the r5 recipe (4-step student, q2-matched teacher targets of train_r5) + a CFG-distillation term (distill2 --cfg-w):
# the student's eps matches the frozen teacher's GUIDED eps (g 2.0) at every sampled x_t, t. 1250 steps, ckpt every 250, pick as r5;
# E2E old03 + m03 (c8 + LS_FACE_UNITS), then agent F's clean lip recipe on the control + r6 (+ q2u) via scripts/r15/r6_recipe.sh.
#   setsid nohup bash scripts/r15/r6_chain.sh > /workspace/dub/r15x/r6_chain.log 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null
X=${R5_ROOT:-/workspace/dub/r15x}; E=$X/eval; N=$X/R6_NOTES.txt; T=$X/train_r6; C=$X/ckpt/r6_s4; OB=/workspace/dub/outbox/r15x; export EVAL_DIR=$E
note(){ echo "$(date +%T) $*" | tee -a $N; }; done_(){ [ -f $X/done6_$1 ]; }; mark(){ date +%T > $X/done6_$1; }; die(){ note "STOP: $*"; exit 1; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 2000 ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done; note "GPU not idle after 60 min, going anyway"; }
STEPS=${R6_STEPS:-1250}; CW=${R6_CFG_W:-5}
ARGS="--student-steps 4 --sync-w 0.15 --pix-w 0.5 --lpips-w 0.2 --temp-w 0.5 --temp-nolips --lip-box 0.25,0.538,0.747,0.787 --ema 0.999 --lr 1e-4 --p16 0.7 --cfg-w $CW --cfg-g 2.0"
if ! done_ train; then
  mkdir -p $T; ln -sfn $X/train_r5/t $T/t; idle; note "train r6_s4: $STEPS steps, cfg_w $CW (smoke: cfg term 0.0015 vs mse 0.008, 2.86 s/step)"
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $T $ARGS --steps $STEPS --val-every 250 --val-max 24 --run r6_s4 --ckpt-root $X/ckpt > $X/run_r6.log 2>&1
  grep -q "\[train\] done" $X/run_r6.log || die "training failed: $(tail -1 $X/run_r6.log | cut -c1-200)"
  mark train; note "r6_s4 trained: $(tail -1 $C/latest.txt 2>/dev/null)"
fi
if ! done_ pick; then
  flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $T --ckpt $C --student-steps 4 --mse-tol 0.2 > $X/ckpt_val_r6.log 2>&1 || die "ckpt_val failed"
  mark pick; note "pick: $(python3 -c "import json; d=json.load(open('$C/ckpt_val.json')); print(d['pick'], '-', d['pick_reason'])")"
fi
PK=$C/$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick'])")
FU='"LS_FACE_UNITS": "1"'; C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'
for id in old03 m03; do
  [ -s $E/r6c8u_$id/r6c8u_$id.mp4 ] && continue; idle
  Q2=q2u Q2ENV="$FU" OB=$OB LORA=$PK STEPS=4 TAG=r6c8u XENV="$C8, $FU" bash scripts/r14/eval_r14.sh $id >> $X/eval_r6.log 2>&1
  [ -s $E/r6c8u_$id/r6c8u_$id.mp4 ] || note "r6c8u $id: no output video ($(tail -1 $E/r6c8u_$id.log 2>/dev/null | cut -c1-200))"
done
python scripts/r15/r5_table.py --root $X --run r6_s4 --base q2u --arms s4p1500c8u r5fc8u r6c8u --ids old03 m03 > $X/R6_TABLE.md 2>&1; tail -9 $X/R6_TABLE.md | tee -a $N
note "E2E done"; touch $X/R6_E2E_DONE
[ -f scripts/r15/r6_recipe.sh ] && bash scripts/r15/r6_recipe.sh >> $X/r6_recipe.log 2>&1 || note "r6_recipe.sh missing or failed (tail: $(tail -1 $X/r6_recipe.log 2>/dev/null | cut -c1-200))"
note "R6 chain done"
