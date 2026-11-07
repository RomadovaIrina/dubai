cd /workspace/dub/keysync/code; . ../venv/bin/activate
flock /workspace/dub/gpu.lock env KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2" KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk python ../ks_fast.py --jobs /workspace/dub/r16k2/gen/jobs_g.txt --out /workspace/dub/r16k2/gen/g8 --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt --interp-steps 2 --vae-batch 8 --vae-compile > /workspace/dub/r16k2/gen/g8.log 2>&1
grep -E "KS_JOB|KS_GATE hi" /workspace/dub/r16k2/gen/g8.log | cut -c1-200
R=/workspace/dub/r16k2; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; MP=/venv/mediapipe/bin/python
for v in m03 old03 m18; do K=$R/gen/g8/$v/${v}_crop512_$v.mp4; $MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $K --out $K.mp.npz > /dev/null 2>&1; echo "CAV g8 $v $($MP $R/k_cav.py $K $K.mp.npz)"; cd $R; KSLM=$K.mp.npz bash k2_var.sh $v g8 $K; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
for v in m03 old03 m18; do for m in g8 s2own; do F=$R/$v/$m/final.mp4; [ -s $F ] || continue; flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $F --audio $P/$v/dubbed_16k.wav --out $R/$v/$m/sync.json > $R/$v/$m/sync.log 2>&1; python3 -c "
import json,statistics as st; s=json.load(open(\"$R/$v/$m/sync.json\")); sp=[p[\"conf_at0\"] for p in s[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]; print(\"SYNC $v $m\", round(st.mean(sp),3))"; done; done; echo G_DONE
