#!/bin/bash
# r15 lever 1 = r5: the 4-step student distilled from teacher targets made with the EXACT production q2 generator settings
# (scripts/r9/e2e.sh + scripts/r4/r4_video.sh: DDIM --steps 10 on the speaking windows, CFG 2.0, DeepCache interval 5) instead of the
# r14 teacher (20 steps, CFG 2.0, no DeepCache); then the r14 recipe, the checkpoint pick, E2E on old03 / m03 / m18 (c8 + LS_FACE_UNITS)
# against q2 + F3 and the r14 student r2@1500 (the user's eye on m03, 05.11: the student "правое лучше" over production).
# Runbook + the file list: reports/experiments/r14-distill/R5_RUNBOOK.md. Resumable: every step leaves $X/done_<step>; rerun = continue.
#   setsid nohup bash scripts/r15/r5_chain.sh > /workspace/dub/r15x/r5_chain.log 2>&1 < /dev/null &
R=$(cd "$(dirname "$0")/../.." && pwd); cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null   # the checkout (or worktree) this script is in
X=${R5_ROOT:-/workspace/dub/r15x}; E=$X/eval; N=$X/R5_NOTES.txt; T=$X/train_r5; C=$X/ckpt/r5_s4; OB=/workspace/dub/outbox/r15x; M=$X/markers
mkdir -p $X $E $OB $M $X/ckpt; export EVAL_DIR=$E
note(){ echo "$(date +%T) $*" | tee -a $N; }
done_(){ [ -f $X/done_$1 ]; }; mark(){ date +%T > $X/done_$1; }
die(){ note "STOP: $*"; exit 1; }
idle(){ local ok=0 u; for i in $(seq 1 240); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)   # shared-GPU safety (r15-D 09:29 audio_failed)
  if [ "$u" -lt 2000 ] && flock -w 5 /workspace/dub/gpu.lock true; then ok=$((ok + 1)); [ $ok -ge 2 ] && return 0; else ok=0; fi; sleep 15; done
  note "GPU not idle after 60 min, going anyway"; }
FU='"LS_FACE_UNITS": "1"'; C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'
P2=${R5_CONTROL:-$X/ckpt/r2_s4/step_01500_ema.pt}   # the r14 student (control arm s4p1500c8u)
STEPS=${R5_STEPS:-2000}; IDS2=${R5_IDS:-m03 m18}   # budget knobs: train steps, the videos after old03
ARGS="--student-steps 4 --sync-w 0.15 --pix-w 0.5 --lpips-w 0.2 --temp-w 0.5 --temp-nolips --lip-box 0.25,0.538,0.747,0.787 --ema 0.999 --lr 1e-4 --p16 0.7"

# 0. setup check (no GPU work): the stack, the data, the inputs; nothing is printed from /workspace/.env
if ! done_ check; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $N
  [ $(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9) -ge 40 ] || die "less than 40 GB free on /workspace"
  for p in /venv/dabai/bin/python /venv/mediapipe/bin/python third_party/latentsync/checkpoints/latentsync_unet.pt /workspace/.env $P2 \
           /workspace/dub/inputs/test_videos_raw/03.mp4 /workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; do
    [ -e "$p" ] || die "missing $p"; done
  python -c "import DeepCache, lpips, diffusers" || die "python deps (DeepCache / lpips / diffusers)"
  python scripts/r13/distill/distill2.py --help | grep -q teacher-deepcache || die "distill2.py without --teacher-deepcache (repo not at the r5 commit?)"
  git log --oneline -1 | tee -a $N; grep -q LS_FACE_UNITS scripts/pilot/clean_face_stack.py || die "no LS_FACE_UNITS (F's d494c67) in this checkout"
  mark check; note "check ok"
fi

# 1. the 1817 selected r14 windows: own (distill_cache.tar -> $X/own/cap) + external (cap_ext -> $X/cap_ext), as symlinks in $X/sel
if ! done_ sel; then
  mkdir -p $X/sel; miss=0
  while read rel; do s=$X/$rel; [ -f $s ] || { miss=$((miss + 1)); continue; }; ln -sfn $s $X/sel/$(basename $rel); done < scripts/r15/r5_sel_list.txt
  [ $miss = 0 ] && [ $(ls $X/sel | wc -l) = 1817 ] || die "sel: $miss of 1817 windows missing (own/cap from distill_cache.tar, cap_ext/ from the laptop)"
  mark sel; note "sel: 1817 windows linked"
fi

# 2. teacher targets with the production q2 generator settings (~1 h; resumable: finished windows are skipped)
if ! done_ teacher; then
  mkdir -p $T
  # an optional GPU window for another agent first: $X/gate.env sets GATE_FILE (its done marker) and GATE_UNTIL (epoch s, the cap)
  if [ -f $X/gate.env ]; then . $X/gate.env; note "teacher waits for ${GATE_FILE} or $(date -d @${GATE_UNTIL} +%T)"
    until [ -f "${GATE_FILE}" ] || [ $(date +%s) -ge ${GATE_UNTIL} ]; do sleep 30; done; note "gate open ($([ -f "${GATE_FILE}" ] && echo marker || echo cap))"; fi
  idle; note "teacher q2-matched (10 steps, CFG 2.0, DeepCache 5) start"
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py teacher --cap $X/sel --out $T --tbatch 4 --teacher-steps 10 --teacher-cfg 2.0 --teacher-deepcache 5 > $X/teacher_r5.log 2>&1
  n=$(ls $T/t | wc -l); [ $n = 1817 ] || die "teacher: $n of 1817 targets ($(tail -1 $X/teacher_r5.log | cut -c1-200))"
  mark teacher; note "teacher: $(grep '\[teacher\] done' $X/teacher_r5.log)"
fi

# 3. smoke (30 steps), 4. train (2000 steps, val + ckpt every 250)
if ! done_ train; then
  S=$X/smoke_r5; mkdir -p $S; ln -sfn $T/t $S/t
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $S $ARGS --steps 30 --val-every 30 --val-max 4 --run smoke_r5 --ckpt-root $S/ckpt > $X/smoke_r5.log 2>&1
  grep -q "\[train\] done" $X/smoke_r5.log || die "smoke failed: $(tail -1 $X/smoke_r5.log | cut -c1-200)"
  note "SMOKE_OK: $(grep '\[val\]' $X/smoke_r5.log | tail -1 | cut -c1-250)"
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $T $ARGS --steps $STEPS --val-every 250 --val-max 24 --run r5_s4 --ckpt-root $X/ckpt > $X/run_r5.log 2>&1
  grep -q "\[train\] done" $X/run_r5.log || die "training failed: $(tail -1 $X/run_r5.log | cut -c1-200)"
  mark train; note "r5_s4 trained: $(tail -1 $C/latest.txt 2>/dev/null)"
fi

# 5. checkpoint pick: opening >= 1.03, then within 20 % of the lowest MSE the highest val sync (ckpt_val.py --mse-tol)
if ! done_ pick; then
  flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $T --ckpt $C --student-steps 4 --mse-tol 0.2 > $X/ckpt_val_r5.log 2>&1 || die "ckpt_val failed"
  mark pick; note "pick: $(python3 -c "import json; d=json.load(open('$C/ckpt_val.json')); print(d['pick'], '-', d['pick_reason'])")"
fi
PK=$C/$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick'])"); FN=$C/step_$(printf %05d $STEPS)_ema.pt

ev(){ local tag=$1 ck=$2 id=$3; [ -s $E/${tag}_$id/${tag}_$id.mp4 ] && return 0; idle; echo "$tag $id $(date +%T)" > $M/E2E_RUNNING
  Q2=q2u Q2ENV="$FU" OB=$OB LORA=$ck STEPS=4 TAG=$tag XENV="$C8, $FU" bash scripts/r14/eval_r14.sh $id >> $X/eval_r5.log 2>&1; rm -f $M/E2E_RUNNING
  [ -s $E/${tag}_$id/${tag}_$id.mp4 ] || note "$tag $id: no output video ($(tail -1 $E/${tag}_$id.log 2>/dev/null | cut -c1-200))"; }
sy(){ sed -n 's/.*: \([-0-9.]*\) over.*/\1/p' $E/$1.sync.txt 2>/dev/null; }

# 6. old03: q2u (baseline, first), the r14 student, the r5 pick and (if different) the r5 final; the better r5 by old03 sync goes on
ev s4p1500c8u $P2 old03; ev r5c8u $PK old03
[ "$PK" != "$FN" ] && ev r5fc8u $FN old03
B=r5c8u; BC=$PK
if [ "$PK" != "$FN" ] && python3 -c "import sys; sys.exit(0 if float('$(sy r5fc8u_old03)' or 0) > float('$(sy r5c8u_old03)' or 0) else 1)"; then B=r5fc8u; BC=$FN; fi
note "old03 sync: q2u $(sy q2u_old03), r14 student $(sy s4p1500c8u_old03), r5 pick $(sy r5c8u_old03), r5 final $(sy r5fc8u_old03) -> $B ($(basename $BC)) on $IDS2"

# 7. m03 / m18: the r14 student + the better r5
for id in $IDS2; do ev s4p1500c8u $P2 $id; ev $B $BC $id; done

# 8. clips (the r5 arm vs q2u and vs the r14 student; m03 17-21 s and m18 2-6 s at 0.5x), 9. tables
OB=$OB Q2=q2u bash scripts/r14/r15_clips.sh $B s4p1500c8u > $X/r5_clips.log 2>&1 || note "clips failed: $(tail -1 $X/r5_clips.log)"
python scripts/r15/r5_table.py --root $X --run r5_s4 --base q2u --arms s4p1500c8u r5c8u r5fc8u --ids old03 $IDS2 > $X/R5_TABLE.md 2>&1
cat $X/R5_TABLE.md | tee -a $N

# 10. commit (the run's report files) + the incremental bundle; R5_DONE last
D=reports/experiments/r14-distill; mkdir -p $D; cp $X/R5_TABLE.md $D/R5_TABLE.md; cp $X/R5_NOTES.txt $D/R5_NOTES.txt; cp $C/ckpt_val.json $D/ckpt_val_r5_s4.json
git config user.name >/dev/null || git config user.name "DabAI agent D"; git config user.email >/dev/null || git config user.email "dabai-agent-D@users.noreply.github.com"
for p in $D/R5_TABLE.md $D/R5_NOTES.txt $D/ckpt_val_r5_s4.json; do git add "$p"; done
git commit -q -m "r5 (r15 lever 1): the 4-step student from q2-matched teacher targets (10 steps, CFG 2.0, DeepCache 5); val + E2E tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && note "commit $(git rev-parse --short HEAD)"
git bundle create $X/r5-distill.bundle experiment/r14-distill ^origin/experiment/r13-spec ^origin/experiment/r13 && git bundle verify $X/r5-distill.bundle >/dev/null 2>&1 \
  && note "bundle ok $X/r5-distill.bundle ($(du -h $X/r5-distill.bundle | cut -f1))" || note "BUNDLE FAILED"
touch $X/R5_DONE; note "R5_DONE"
