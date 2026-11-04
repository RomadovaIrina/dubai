cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r14d; C=$X/ckpt/r2_s4; L=$C/step_03000_ema.pt
until grep -q RUN_S4_DONE $X/run_s4.log; do sleep 60; done
echo "ckpt_val $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r14/ckpt_val.py --out $X/train_s4 --ckpt $C > $X/ckpt_val.log 2>&1; grep "^{\|best_by" $X/ckpt_val.log | cut -c1-220
echo "eval s4 (final EMA) $(date +%T)"; LORA=$L STEPS=4 TAG=s4 bash scripts/r14/eval_r14.sh old03 m03 m18
echo "eval s4f8 $(date +%T)"; LORA=$L STEPS=4 TAG=s4f8 XENV='"LS_STRIDE": "8", "LS_OVERLAP": "fuse"' bash scripts/r14/eval_r14.sh old03 m03 m18
echo "eval s4f8r32 $(date +%T)"; LORA=$L STEPS=4 TAG=s4f8r32 XENV='"LS_STRIDE": "8", "LS_OVERLAP": "fuse", "R13_STUDENT_EXTRA": "--ref-mode shift32"' bash scripts/r14/eval_r14.sh old03 m03 m18
BP=$(python3 -c "import json; print(json.load(open('$C/ckpt_val.json'))['best_by_pix'])" 2>/dev/null)
if [ -n "$BP" ] && [ "$BP" != step_03000_ema.pt ]; then t=s4p$(echo $BP | grep -o "[0-9]\{5\}" | sed "s/^0*//"); echo "eval best-by-pix $BP as $t $(date +%T)"; LORA=$C/$BP STEPS=4 TAG=$t bash scripts/r14/eval_r14.sh old03 m03 m18; fi
echo "AFTER_S4_DONE $(date +%T)"
