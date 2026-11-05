cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
# r15-D, the rest of r15.sh after r4_s6 training (r15.sh was stopped mid-training to add the GPU yield; the training process kept running):
# yield the GPU to agent F (until /workspace/dub/r15f/GPU_DONE or 25 min), then ckpt_val pick, E2E s6c8 on old03 / m03 / m18, clips
X=/workspace/dub/r14d; N=$X/R15_NOTES.txt; T=$X/train_r4; C=$X/ckpt/r4_s6
note(){ echo "$(date +%T) $*" | tee -a $N; }
while pgrep -f "distill2.py train --out $T" >/dev/null; do sleep 30; done
grep -q "\[train\] done" $X/run_r4.log && note "r4_s6 trained: $(tail -1 $C/latest.txt 2>/dev/null)" || { note "r4_s6 training failed: $(tail -1 $X/run_r4.log | cut -c1-200)"; exit 1; }
w=0; until [ -f /workspace/dub/r15f/GPU_DONE ] || [ $w -ge 50 ]; do sleep 30; w=$((w + 1)); done
note "GPU yield to F over after $((w / 2)) min ($([ -f /workspace/dub/r15f/GPU_DONE ] && echo "F's GPU_DONE" || echo "25 min cap"))"
flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $T --ckpt $C --student-steps 6 > $X/ckpt_val_r4.log 2>&1
P=$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick'])" 2>/dev/null); P=$C/${P:-step_02000_ema.pt}
note "r4 eval checkpoint $(basename $P): $(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['pick_reason'])" 2>/dev/null)"
echo R4_PICKED > $X/R4_PICKED
OB=/workspace/dub/outbox/r15_d LORA=$P STEPS=6 TAG=s6c8 XENV='"LS_STRIDE": "8", "LS_OVERLAP": "continue"' bash scripts/r14/eval_r14.sh old03 m03 m18
for id in old03 m03 m18; do [ -s $X/eval/s6c8_$id/s6c8_$id.mp4 ] || note "s6c8 $id: no output video"; done
bash scripts/r14/r15_clips.sh s6c8 > $X/r15_clips.log 2>&1 || note "r15 clips failed: $(tail -1 $X/r15_clips.log)"
echo "R15_DONE $(date +%T)"
