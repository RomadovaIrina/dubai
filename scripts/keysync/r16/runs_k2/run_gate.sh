#!/bin/bash
# r16b K2: cheaper-sync gate arms on m03 + m18 (ref 0, dub delay 40 ms as accepted, paste shift 0); full-length speech@0 + GPU s per video-minute
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0 KS_REF=0
B=/workspace/dub/r16k2/gate; F=/workspace/dub/k2wt/scripts/keysync/r16/ks_face.sh; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; MAM=/workspace/dub/inputs/mamakris
:
declare -A SRC=([m03]=$MAM/story-1774249919482-367803614.MOV [m18]=$MAM/story-1774250817106-616905094.mp4)
declare -A ARM=([off]="" [g8]="thr=0.12,cap=0.25,hi=8,lo=2" [g4]="thr=0.12,cap=0.25,hi=4,lo=2" [g3]="thr=0.12,cap=0.25,hi=3,lo=2" [g8t]="thr=0.18,cap=0.15,hi=8,lo=2")
for v in m18 m03; do mkdir -p $B/$v/ad; cp $P/$v/dubbed_16k.wav $P/$v/final_mix_aac.m4a $B/$v/ad/
  RT=""; [ $v = m03 ] && RT=/workspace/dub/r16in/day1/HANDOFF_Q/routes/repo_routing/m03.json
  for arm in off g8 g4 g3 g8t; do O=$B/$v/$arm.mp4
    [ -s $O ] || env KS_GATE="${ARM[$arm]}" bash $F ${v}_$arm ${SRC[$v]} $B/$v/ad $B/$v/w_$arm $O $RT > $B/$v/$arm.log 2>&1
    [ -s $O ] || { echo "$v $arm FAILED $(grep -E "Error|FAIL" $B/$v/$arm.log | tail -1 | cut -c1-120)"; continue; }
    ( cd /workspace/dub/k2wt && /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $O --audio $B/$v/ad/dubbed_16k.wav --out $B/$v/$arm.sync.json > /dev/null 2>&1 )
    J=$(grep -h "^KS_JOB" $B/$v/w_$arm/ks/ks_fast.log | tail -1 | grep -o "\"s_per_vmin\": [0-9.]*")
    G=$(grep -h "KS_GATE hi segments" $B/$v/w_$arm/ks/ks_fast.log | tail -1 | cut -c1-80 | tr -d %)
    python3 -c "
import json; s=json.load(open('$B/$v/$arm.sync.json')); sp=[p['conf_at0'] for p in s['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]
print('$v $arm speech@0 %.2f n %d off %s | $J | $G' % (sum(sp)/max(len(sp),1), len(sp), s['global']['av_offset']))"
  done
done
echo GATE_DONE
