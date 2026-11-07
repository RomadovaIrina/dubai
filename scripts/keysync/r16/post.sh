cd /workspace/dub/r16k2; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; MP=/venv/mediapipe/bin/python

cat gen/run.log | grep KS_JOB | python3 -c "
import sys,json
for l in sys.stdin:
    st=l.split()[0]; j=json.loads(l.split(\"KS_JOB \",1)[1]); s=j.get(\"stages_s\",{}); lm=s.get(\"extract_video_landmarks\",0)
    print(\"COST\",st,j[\"tag\"],round((j[\"wall_s\"]-lm)/j[\"video_s\"]*60,1))"


for v in m03 old03 m18; do for st in 4 8; do B=gen/s2/$v/${v}_crop512_$v.mp4; mkdir -p $v/b$st
 $MP k2_splice.py --base $B --base-lm $B.mp.npz --hi gen/s$st/$v/${v}_crop512_$v.mp4 --out $v/b$st/in.mp4 | sed "s/^/$v b$st /"
 KSLM=$B.mp.npz bash k2_var.sh $v b$st $v/b$st/in.mp4; done; KSLM=gen/s2/$v/${v}_crop512_$v.mp4.mp.npz bash k2_var.sh $v s2own gen/s2/$v/${v}_crop512_$v.mp4; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
for v in m03 old03 m18; do for m in bank60; do F=/workspace/dub/r16k2/$v/$m/final.mp4; [ -s $F ] && [ ! -s /workspace/dub/r16k2/$v/$m/sync.json ] || continue
 flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $F --audio $P/$v/dubbed_16k.wav --out /workspace/dub/r16k2/$v/$m/sync.json > /workspace/dub/r16k2/$v/$m/sync.log 2>&1
 python3 -c "
import json,statistics as st; s=json.load(open(\"/workspace/dub/r16k2/$v/$m/sync.json\")); sp=[p[\"conf_at0\"] for p in s[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]
print(\"SYNC $v $m\", round(st.mean(sp),3), s[\"global\"])"; done; done; echo POST_DONE
