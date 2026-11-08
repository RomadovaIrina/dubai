#!/bin/bash
# r16b K2: third black-hole arm r0g = KS_REF=0 + the b8 gate (the new e2e default); cavity over source / plain (ref0 no gate) / r0g / k2 (refpick + gate)
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0 KS_REF=0
H=/workspace/dub/r16in/day1/HANDOFF_Q; B=/workspace/dub/r16k2/bh; F=/workspace/dub/k2wt/scripts/keysync/r16/ks_face.sh; MP=/venv/mediapipe/bin/python
for v in q01_amir q02_vera q03_priest q04_farikh m18; do
  if [ $v = m18 ]; then S=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; AD=$B/m18ad; RT=""; else S=/workspace/dub/r16k2/src/$v.mp4; AD=$H/mixes/$v; RT=$H/routes/repo_routing/$v.json; fi
  O=$B/$v/r0g.mp4
  [ -s $O ] || bash $F ${v}_r0g $S $AD $B/$v/w_r0g $O $RT > $B/$v/r0g.log 2>&1
  [ -s $O ] || { echo "$v r0g FAILED"; continue; }
  GLOG_minloglevel=2 $MP /workspace/dub/k2wt/scripts/quality/mp_landmarks.py --video $O --out $B/$v/r0g.npz > /dev/null 2>&1
  ( cd /workspace/dub/k2wt && /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $O --audio $AD/dubbed_16k.wav --out $B/$v/r0g.sync.json > /dev/null 2>&1 )
  python3 -c "
import json; s=json.load(open('$B/$v/r0g.sync.json')); sp=[p['conf_at0'] for p in s['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]
print('$v r0g speech@0 %.2f n %d off %s' % (sum(sp)/max(len(sp),1), len(sp), s['global']['av_offset']))"
  $MP $B/k_cavity.py --pairs source=$B/$v/src25.mp4:$B/$v/src25.npz plain=$B/$v/plain.mp4:$B/$v/plain.npz r0g=$O:$B/$v/r0g.npz k2=$B/$v/k2.mp4:$B/$v/k2.npz > $B/$v/cavity3.json 2>/dev/null
  python3 -c "
import json; d=json.load(open('$B/$v/cavity3.json')); print('$v CAVITY3', {k: (x['open_frames'], x['black_hole_share'], x['cavity_L_mean']) for k, x in d.items()})"
done
echo R0G_DONE
