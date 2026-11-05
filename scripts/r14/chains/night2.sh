cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r14 night, re-planned after r3_s4 overshot (its one-sided opening hinge was gamed: opening 1.3-1.4x the teacher's, val MSE 2x untrained).
# Coordinator's priorities, deadline ~05:00 server time, then scripts/r14/night_finalize.sh:
#   1. r3 @ its pick (step 500) on old03 only; its m03 / m18 + r3c8 + r3c8rf only if it beats r2@1500 on old03 (sync > 6.657) AND looks
#      sane in the decode check (the session writes $X/r3_visual_ok)
#   2. s4p1500c8 = r2 step 1500 + LS_STRIDE 8 continue on old03 / m03 / m18 (the main deliverable)
#   3. time left: r3b = symmetric opening term |open_s - open_t|, open_w 0.5, 1500 steps; its old03 eval if there is time
#   4. time left: s4p1500c8rf = 2. + agent F's smart reference, old03 only
X=/workspace/dub/r14d; E=$X/eval; N=$X/NIGHT_NOTES.txt; C2=$X/ckpt/r2_s4; C3=$X/ckpt/r3_s4; P2=$C2/step_01500_ema.pt
note(){ echo "$(date +%T) $*" | tee -a $N; }
late(){ local t=$((10#$(date +%H%M))); [ $t -ge $1 ] && [ $t -lt 1200 ]; }   # past HHMM tonight (after midnight)
C8='"LS_STRIDE": "8", "LS_OVERLAP": "continue"'; RF='"R13_STUDENT_EXTRA": "--ref-mode fixed", "LS_REF_SMART": "1"'
ev(){ local tag=$1 lora=$2 xe=$3; shift 3; echo "eval $tag ($(basename $lora)) on $* $(date +%T)"; LORA=$lora STEPS=4 TAG=$tag XENV="$xe" bash scripts/r14/eval_r14.sh "$@"
  for id in "$@"; do [ -s $E/${tag}_$id/${tag}_$id.mp4 ] || note "$tag $id: no output video ($(tail -1 $E/${tag}_$id.log 2>/dev/null | cut -c1-200))"; done; }
sy(){ sed -n 's/.*: \([-0-9.]*\) over.*/\1/p' $E/$1.sync.txt 2>/dev/null; }
while pgrep -f "ckpt_val.py --out $X/train_r3" >/dev/null; do sleep 20; done
P3=$(python3 -c "import json; print(json.load(open('$C3/ckpt_val.json'))['pick'])" 2>/dev/null); P3=$C3/${P3:-step_00500_ema.pt}
note "night re-plan (r3 overshot); r3 eval checkpoint $(basename $P3): $(python3 -c "import json; print(json.load(open('$C3/ckpt_val.json'))['pick_reason'])" 2>/dev/null)"
ev r3 $P3 "" old03
R3GOOD=$(python3 -c "print(1 if float('$(sy r3_old03)' or 0) > 6.657 else 0)"); [ -f $X/r3_visual_ok ] || R3GOOD=0
note "r3 old03 sync $(sy r3_old03) (r2@1500 6.657), visual ok: $([ -f $X/r3_visual_ok ] && echo yes || echo no) -> r3 extended: $R3GOOD"
ev s4p1500c8 $P2 "$C8" old03 m03 m18
if [ $R3GOOD = 1 ]; then
  late 400 || ev r3 $P3 "" m03 m18
  late 400 || ev r3c8 $P3 "$C8" old03 m03 m18
  late 430 || ev r3c8rf $P3 "$C8, $RF" old03
fi
if ! late 300; then
  T=$X/train_r3b; mkdir -p $T; ln -sfn $X/train_s4/t $T/t; echo "train r3b $(date +%T)"
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $T --steps 1500 --student-steps 4 --sync-w 0.2 --pix-w 0.5 --lpips-w 0.2 \
    --temp-w 0.25 --temp-nolips --open-w 0.5 --open-sym --lip-box 0.25,0.538,0.747,0.787 --open-box 0.35,0.56,0.65,0.72 --ema 0.999 --lr 1e-4 --p16 0.7 \
    --run r3b_s4 --ckpt-root $X/ckpt > $X/run_r3b.log 2>&1 && note "r3b trained: $(tail -1 $X/ckpt/r3b_s4/latest.txt 2>/dev/null)" || note "r3b training failed: $(tail -1 $X/run_r3b.log | cut -c1-200)"
  [ -s $X/ckpt/r3b_s4/step_01500_ema.pt ] && ! late 430 && ev r3b $X/ckpt/r3b_s4/step_01500_ema.pt "" old03
else note "r3b skipped: past 03:00"; fi
late 430 || ev s4p1500c8rf $P2 "$C8, $RF" old03
echo "NIGHT2_DONE $(date +%T)"
bash scripts/r14/night_finalize.sh
