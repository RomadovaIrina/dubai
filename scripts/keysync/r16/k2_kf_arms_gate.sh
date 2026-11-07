cd /workspace/dub/keysync/code; . ../venv/bin/activate; W=/workspace/dub/r16k2/win
arm(){ name=$1; refm03=$2; refm18=$3; shift 3
 printf "m18w $W/m18_r$refm18.mp4 $W/m18_dub.wav\nm03 $W/m03_r$refm03.mp4 $W/m03_dub.wav\nm18 $W/m18_r$refm18.mp4 $W/m18_dub.wav\n" > $W/jobs_$name.txt
 flock /workspace/dub/gpu.lock env KS_REF_IDX=0 KS_DC=2:0 KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2" KS_LMK_CACHE=/workspace/dub/keysync/lmk "$@" python ../ks_fast.py --jobs $W/jobs_$name.txt --out $W/out_$name --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile > $W/$name.log 2>&1
 grep -E "KS_JOB|KS_GATE hi" $W/$name.log | sed "s/^/$name /" | cut -c1-120; }
arm gbase 0 0 env
arm grefA 84 238 env
arm grefArr3 84 238 env KS_KF_RR=3
arm grr3 0 0 env KS_KF_RR=3
MP=/venv/mediapipe/bin/python; cd $W
for a in gbase grefA grefArr3 grr3; do for v in m03 m18; do f=$(ls out_$a/$v/*.mp4 | head -1); ffmpeg -v error -y -i $f -vf "select=gte(n\,1),setpts=N/25/TB" -r 25 -c:v libx264 -crf 10 -pix_fmt yuv420p ${a}_$v.mp4; $MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video ${a}_$v.mp4 --out ${a}_$v.npz > /dev/null 2>&1; echo "CAVW $a $v $($MP ../k_cav.py ${a}_$v.mp4 ${a}_$v.npz)"; done; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
for a in gbase grefA grefArr3 grr3; do for v in m03 m18; do ffmpeg -v error -y -i $W/${a}_$v.mp4 -i $W/${v}_dub.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest $W/${a}_${v}_av.mp4; flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $W/${a}_${v}_av.mp4 --audio $W/${v}_dub.wav --out $W/${a}_$v.sync.json > /dev/null 2>&1; python3 -c "
import json,statistics as st; s=json.load(open(\"$W/${a}_$v.sync.json\")); sp=[p[\"conf_at0\"] for p in s[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]; print(\"SYNCW $a $v\", round(st.mean(sp),3) if sp else None, s[\"global\"])"; done; done; echo ARMS2_DONE
