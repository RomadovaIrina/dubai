cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r15-D: 6-step student r4_s6 = r2 recipe (sync 0.15, pix 0.5, lpips 0.2) + the temporal term off the lips (temp_w 0.5), no opening term,
# 2000 steps, val + ckpt every 250; pick by ckpt_val (opening >= 1.03, then MSE); E2E pick + c8 (no smart ref) on old03 / m03 / m18
X=/workspace/dub/r14d; N=$X/R15_NOTES.txt; S=$X/smoke_r4; T=$X/train_r4; C=$X/ckpt/r4_s6
note(){ echo "$(date +%T) $*" | tee -a $N; }
ARGS="--student-steps 6 --sync-w 0.15 --pix-w 0.5 --lpips-w 0.2 --temp-w 0.5 --temp-nolips --lip-box 0.25,0.538,0.747,0.787 --ema 0.999 --lr 1e-4 --p16 0.7"
mkdir -p $S $T; ln -sfn $X/train_s4/t $S/t; ln -sfn $X/train_s4/t $T/t
echo "smoke $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $S $ARGS --steps 30 --val-every 30 --val-max 4 --run smoke_r4 --ckpt-root $S/ckpt > $X/smoke_r4.log 2>&1
grep -q "\[train\] done" $X/smoke_r4.log || { note "smoke failed: $(tail -1 $X/smoke_r4.log | cut -c1-200)"; exit 1; }
note "SMOKE_OK r4: $(grep "\[val\]" $X/smoke_r4.log | tail -1 | cut -c1-300)"
echo "train $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r13/distill/distill2.py train --out $T $ARGS --steps 2000 --val-every 250 --val-max 24 --run r4_s6 --ckpt-root $X/ckpt > $X/run_r4.log 2>&1 \
  && note "r4_s6 trained: $(tail -1 $C/latest.txt 2>/dev/null)" || { note "r4_s6 training failed: $(tail -1 $X/run_r4.log | cut -c1-200)"; exit 1; }
flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $T --ckpt $C --student-steps 6 > $X/ckpt_val_r4.log 2>&1
P=$(python3 -c "import json; print(json.load(open(\"$C/ckpt_val.json\"))[\"pick\"])" 2>/dev/null); P=$C/${P:-step_02000_ema.pt}
note "r4 eval checkpoint $(basename $P): $(python3 -c "import json; print(json.load(open(\"$C/ckpt_val.json\"))[\"pick_reason\"])" 2>/dev/null)"
echo R4_PICKED > $X/R4_PICKED
OB=/workspace/dub/outbox/r15_d LORA=$P STEPS=6 TAG=s6c8 XENV="\"LS_STRIDE\": \"8\", \"LS_OVERLAP\": \"continue\"" bash scripts/r14/eval_r14.sh old03 m03 m18
for id in old03 m03 m18; do [ -s $X/eval/s6c8_$id/s6c8_$id.mp4 ] || note "s6c8 $id: no output video"; done
bash scripts/r14/r15_clips.sh s6c8 > $X/r15_clips.log 2>&1 || note "r15 clips failed: $(tail -1 $X/r15_clips.log)"
echo "R15_DONE $(date +%T)"
