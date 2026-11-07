cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r14 student round 3 (the night run): after the r2 evals the coordinator kept (s4 old03 / m03 / m18 + s4p1500 old03), train r3_s4 on the
# same teacher targets (temp 0.25 off the lips, mouth-opening hinge, sync 0.2), 30-step smoke first, then the evals r3 -> r3c8 -> r3f8, then
# scripts/r14/night_finalize.sh (commit, bundle, NIGHT_REPORT.md, NIGHT_DONE). A failed step is noted in NOTES and skipped, never a stall.
X=/workspace/dub/r14d; E=$X/eval; T=$X/train_r3; S=$X/smoke_r3; B=0.25,0.538,0.747,0.787; N=$X/NIGHT_NOTES.txt
note(){ echo "$(date +%T) $*" | tee -a $N; }
# s4p1500 old03: done when its mouth json lands, or its job failed (log ends without SUCCESS and no process left)
until [ -s $E/s4p1500_old03.mouth.json ] || grep -q "EVAL_DONE s4p1500\|AFTER_S4_DONE" $X/after_s4b.log; do sleep 30; done
[ -f $E/skip_s4p1500 ] || note "s4p1500 old03 done"; touch $E/skip_s4p1500; echo "s4p1500 old03 done ($(cat $E/s4p1500_old03.sync.txt 2>/dev/null)); its m03 / m18 skipped"
until grep -q AFTER_S4_DONE $X/after_s4b.log || ! pgrep -f "after_s4b.sh" >/dev/null; do sleep 30; done
ARGS="--steps 3000 --student-steps 4 --sync-w 0.2 --pix-w 0.5 --lpips-w 0.2 --temp-w 0.25 --temp-nolips --open-w ${OPEN_W:-2.0} --lip-box $B --open-box 0.35,0.56,0.65,0.72 --ema 0.999 --lr 1e-4 --p16 0.7"
mkdir -p $T $S; ln -sfn $X/train_s4/t $T/t; ln -sfn $X/train_s4/t $S/t
# r2 again with the opening ratio per checkpoint: does the proxy see the damping E2E saw (old03: step 1500 x0.93, 3000 x0.80 vs q2)?
echo "ckpt_val r2 (+opening) $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $X/train_s4 --ckpt $X/ckpt/r2_s4 > $X/ckpt_val_r2b.log 2>&1 \
  && note "r2 val opening: $(grep -o '"ckpt": "[^"]*"\|"open_ratio": [0-9.]*' $X/ckpt_val_r2b.log | paste -sd' ' | sed 's/"ckpt": //g; s/"open_ratio": //g')" || note "ckpt_val r2 (+opening) failed"
echo "smoke $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $S $ARGS --steps 30 --val-every 30 --val-max 4 --run smoke_r3 --ckpt-root $S/ckpt > $X/smoke_r3.log 2>&1
if grep -q '"step": 30' $S/train_log.jsonl 2>/dev/null && ! grep -qi nan $S/train_log.jsonl; then
  note "SMOKE_OK"; echo "train r3_s4 $(date +%T)"
  # agent F's night smart-reference patch (working-tree diff; default off): applied now, while no generator job runs, committed by night_finalize
  RF_OK=""; git apply -3 $X/f_patches/0008-night-smart-ref.patch && RF_OK=1 && note "smart-ref patch applied" || note "smart-ref patch FAILED to apply"
  flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $T $ARGS --run r3_s4 --ckpt-root $X/ckpt > $X/run_r3.log 2>&1 || note "r3_s4 training exited non-zero (tail: $(tail -1 $X/run_r3.log | cut -c1-200))"
  echo "RUN_R3_DONE $(date +%T)"
  C=$X/ckpt/r3_s4; L=$C/step_03000_ema.pt
  [ -s $L ] || { L=$(ls $C/step_*_ema.pt 2>/dev/null | tail -1); note "no step_03000_ema.pt; evaluating the last saved EMA: ${L:-none}"; }
  if [ -n "$L" ]; then
    flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $T --ckpt $C > $X/ckpt_val_r3.log 2>&1 || note "ckpt_val r3 failed"; grep "best_by" $X/ckpt_val_r3.log
    # the eval checkpoint (coordinator): val opening ratio >= 0.95 first, then the lowest MSE (none that wide: the widest); else the final
    FIN=$L; P=$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick'])" 2>/dev/null)
    [ -n "$P" ] && [ -s $C/$P ] && L=$C/$P
    note "r3 eval checkpoint: $(basename $L) ($(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick_reason'])" 2>/dev/null || echo 'ckpt_val missing: the final'))"
    for arm in r3 r3c8 r3c8rf r3fin r3f8; do
      IDS="old03 m03 m18"; LA=$L
      case $arm in r3) XE="";; r3c8) XE='"LS_STRIDE": "8", "LS_OVERLAP": "continue"';; r3f8) XE='"LS_STRIDE": "8", "LS_OVERLAP": "fuse"';;
        r3c8rf) XE='"LS_STRIDE": "8", "LS_OVERLAP": "continue", "R13_STUDENT_EXTRA": "--ref-mode fixed", "LS_REF_SMART": "1"'   # agent F's night smart reference
          [ -n "$RF_OK" ] || { note "r3c8rf skipped: the smart-ref patch did not apply"; continue; };;
        r3fin) XE=""; IDS=old03; LA=$FIN   # the final EMA for comparison, old03 only, when the pick is another step
          [ "$L" = "$FIN" ] && continue;; esac
      { [ $arm = r3f8 ] || [ $arm = r3fin ]; } && [ $(date +%H) -ge 5 ] && [ $(date +%H) -lt 12 ] && { note "$arm skipped: past 05:00"; continue; }
      echo "eval $arm ($(basename $LA)) $(date +%T)"; LORA=$LA STEPS=4 TAG=$arm XENV="$XE" bash scripts/r14/eval_r14.sh $IDS
      for id in $IDS; do [ -s $E/${arm}_$id/${arm}_$id.mp4 ] || note "$arm $id: no output video (job failed: $(tail -1 $E/${arm}_$id.log 2>/dev/null | cut -c1-200))"; done
    done
  else note "r3_s4: no checkpoint at all, evals skipped"; fi
else note "SMOKE_FAILED (tail: $(grep -m1 -i "error\|Traceback" $X/smoke_r3.log | cut -c1-200)); r3 training + evals skipped"; fi
echo "R3_CHAIN_DONE $(date +%T)"
bash scripts/r14/night_finalize.sh
