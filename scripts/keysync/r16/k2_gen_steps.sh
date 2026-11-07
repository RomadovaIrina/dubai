cd /workspace/dub/keysync/code; . ../venv/bin/activate; mkdir -p /workspace/dub/keysync/lmk
for st in 2 4 8; do
 flock /workspace/dub/gpu.lock env KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk python ../ks_fast.py --jobs /workspace/dub/r16k2/gen/jobs.txt --out /workspace/dub/r16k2/gen/s$st \
  --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps $st --vae-batch 8 --vae-compile > /workspace/dub/r16k2/gen/s$st.log 2>&1
 grep KS_JOB /workspace/dub/r16k2/gen/s$st.log | sed "s/^/S$st /"
done; echo GEN_DONE
