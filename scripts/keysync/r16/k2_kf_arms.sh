cd /workspace/dub/keysync/code; . ../venv/bin/activate; W=/workspace/dub/r16k2/win
arm(){ name=$1; refm03=$2; refm18=$3; shift 3
 printf "m03 $W/m03_r$refm03.mp4 $W/m03_dub.wav\nm18 $W/m18_r$refm18.mp4 $W/m18_dub.wav\n" > $W/jobs_$name.txt
 flock /workspace/dub/gpu.lock env KS_REF_IDX=0 KS_LMK_CACHE=/workspace/dub/keysync/lmk "$@" python ../ks_fast.py --jobs $W/jobs_$name.txt --out $W/out_$name --cfg ${CFG:-_s8} --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile > $W/$name.log 2>&1
 grep -E "KS_JOB|KS_KF_GUIDE|Error" $W/$name.log | sed "s/^/$name /" | cut -c1-260; }
arm base 0 0 env KS_DC=2:0
arm refA 84 238 env KS_DC=2:0
arm refB 25 76 env KS_DC=2:0
arm refC 887 7 env KS_DC=2:0
arm ar3 0 0 env KS_DC=2:0 KS_KF_AR=3
arm ar7 0 0 env KS_DC=2:0 KS_KF_AR=7
arm rr1 0 0 env KS_DC=2:0 KS_KF_RR=1
arm rr3 0 0 env KS_DC=2:0 KS_KF_RR=3
arm kfnodc 0 0 env KS_DC=
CFG=_s12 arm kf12 0 0 env KS_DC=2:0
CFG=_s16 arm kf16 0 0 env KS_DC=2:0
echo ARMS_DONE
