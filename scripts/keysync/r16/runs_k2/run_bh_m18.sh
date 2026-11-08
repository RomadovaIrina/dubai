#!/bin/bash
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0
B=/workspace/dub/r16k2/bh; F=/workspace/dub/k2wt/scripts/keysync/r16/ks_face.sh; MP=/venv/mediapipe/bin/python; S=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; v=m18
while pgrep -f "[r]un_bh.sh" > /dev/null; do sleep 15; done
mkdir -p $B/$v; cp /workspace/dub/r16k2/m18/src25.mp4 /workspace/dub/r16k2/m18/src25.npz $B/$v/
for arm in plain k2; do O=$B/$v/$arm.mp4; if [ $arm = plain ]; then E="KS_REF=0 KS_GATE="; else E="KS_REF=auto"; fi
  [ -s $O ] || env $E bash $F ${v}_$arm $S $B/m18ad $B/$v/w_$arm $O > $B/$v/$arm.log 2>&1
  GLOG_minloglevel=2 $MP /workspace/dub/k2wt/scripts/quality/mp_landmarks.py --video $O --out $B/$v/$arm.npz > /dev/null 2>&1
  cd /workspace/dub/k2wt && /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $O --audio $B/m18ad/dubbed_16k.wav --out $B/$v/$arm.sync.json > /dev/null 2>&1
  python3 -c "
import json; s=json.load(open(\"$B/$v/$arm.sync.json\")); sp=[p[\"conf_at0\"] for p in s[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]
print(\"$v $arm ref $(grep -ho "ref [0-9]*" $B/$v/$arm.log | head -1) speech@0 %.2f n %d off %s\" % (sum(sp)/max(len(sp),1), len(sp), s[\"global\"][\"av_offset\"]))"
done
$MP $B/k_cavity.py --pairs source=$B/$v/src25.mp4:$B/$v/src25.npz golden=/workspace/dub/r16in/golden/m18_keysync_clean5.mp4:/workspace/dub/r16k2/m18/golden.npz plain=$B/$v/plain.mp4:$B/$v/plain.npz k2=$B/$v/k2.mp4:$B/$v/k2.npz > $B/$v/cavity.json 2>/dev/null
python3 -c "
import json; d=json.load(open(\"$B/$v/cavity.json\")); print(\"$v CAVITY\", {k: (x[\"open_frames\"], x[\"black_hole_share\"], x[\"cavity_L_mean\"]) for k, x in d.items()})"
echo BH_M18_DONE
