cd /workspace/dub/keysync/code; . ../venv/bin/activate; R=/workspace/dub/r16k2; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs
for x in "m18 242" "m03 46" "old03 2002"; do set -- $x; v=$1; ref=$2; grep "^$v " $R/gen/jobs.txt > $R/gen/jobs_ref_$v.txt
 flock /workspace/dub/gpu.lock env KS_REF_IDX=$ref KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2" KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk python ../ks_fast.py --jobs $R/gen/jobs_ref_$v.txt --out $R/gen/rg8 --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile > $R/gen/rg8_$v.log 2>&1
 grep -E "KS_JOB|KS_GATE hi" $R/gen/rg8_$v.log | cut -c1-140
 K=$R/gen/rg8/$v/${v}_crop512_$v.mp4; /venv/mediapipe/bin/python /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $K --out $K.mp.npz > /dev/null 2>&1
 echo "CAV rg8 $v $(/venv/mediapipe/bin/python $R/k_cav.py $K $K.mp.npz)  (g8 ref0: $(/venv/mediapipe/bin/python $R/k_cav.py $R/gen/g8/$v/${v}_crop512_$v.mp4 $R/gen/g8/$v/${v}_crop512_$v.mp4.mp.npz))"
done; echo RG_DONE
