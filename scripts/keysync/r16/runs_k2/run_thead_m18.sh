#!/bin/bash
# T head (b372dae) ks_face on m18 with K2 inputs: delay 0, shift 0, ref 0, gate off -> offset 0 (= K2 chain) or -1 (= T e2e)?
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0 KS_REF=0 KS_GATE=
B=/workspace/dub/r16k2/thead
bash /workspace/dub/dubai/scripts/keysync/r16/ks_face.sh m18_thead /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 /workspace/dub/r16k2/gate/m18/ad $B/w $B/m18_thead_off.mp4 > $B/face.log 2>&1; echo "rc $?"
cd /workspace/dub/k2wt && /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $B/m18_thead_off.mp4 --audio /workspace/dub/r16k2/gate/m18/ad/dubbed_16k.wav --out $B/sync.json > /dev/null 2>&1
python3 -c "
import json; s=json.load(open(\"$B/sync.json\")); sp=[p[\"conf_at0\"] for p in s[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]
print(\"T-head m18 off delay0: speech@0 %.2f n %d off %s\" % (sum(sp)/max(len(sp),1), len(sp), s[\"global\"][\"av_offset\"]))"
