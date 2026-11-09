#!/bin/bash
# r16b K2 black hole on q01-q06: arm plain (ref 0, no gate) vs arm k2 (refpick auto + b8 gate); black-hole share vs the source + SyncNet speech@0
export KS_LORA=/workspace/dub/r16in/lora/lora_pick.pt KS_DELAY=0 KS_PASTE_SHIFT=0
H=/workspace/dub/r16in/day1/HANDOFF_Q; B=/workspace/dub/r16k2/bh; F=/workspace/dub/k2wt/scripts/keysync/r16/ks_face.sh; MP=/venv/mediapipe/bin/python
while pgrep -f "[r]un_q03.sh|[r]un_dissolve.sh" > /dev/null; do sleep 15; done
lmk() { [ -s $2 ] || GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MP /workspace/dub/k2wt/scripts/quality/mp_landmarks.py --video $1 --out $2 > /dev/null 2>&1; }
sync() { cd /workspace/dub/k2wt && /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $1 --audio $2 --out $3 > $3.log 2>&1; python3 - $3 <<"PY"
import json, sys
try: s = json.load(open(sys.argv[1]))
except Exception: print("sync_fail"); sys.exit()
sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]
print("speech@0 %.2f n %d off %s" % (sum(sp) / max(len(sp), 1), len(sp), s["global"]["av_offset"]))
PY
}
for v in q01_amir q02_vera q03_priest q04_farikh q05_rozhkov q06_suvorov; do
  S=/workspace/dub/r16k2/src/$v.mp4; mkdir -p $B/$v
  ffmpeg -nostdin -v error -y -i $S -an -vf fps=25 -c:v libx264 -crf 12 -pix_fmt yuv420p $B/$v/src25.mp4; lmk $B/$v/src25.mp4 $B/$v/src25.npz
  for arm in plain k2; do O=$B/$v/$arm.mp4
    if [ $arm = plain ]; then E="KS_REF=0 KS_GATE="; else E="KS_REF=auto"; fi
    [ -s $O ] || env $E bash $F ${v}_$arm $S $H/mixes/$v $B/$v/w_$arm $O $H/routes/repo_routing/$v.json > $B/$v/$arm.log 2>&1
    rc=$?; [ -s $O ] || { echo "$v $arm FAILED rc $rc: $(grep -E "Error|FAIL|KEEP" $B/$v/$arm.log | tail -1 | cut -c1-150)"; continue; }
    lmk $O $B/$v/$arm.npz
    echo "$v $arm ref $(grep -ho "ref [0-9]*" $B/$v/$arm.log | tr "\n" " ") | $(sync $O $H/mixes/$v/dubbed_16k.wav $B/$v/$arm.sync.json)"
  done
  $MP $B/k_cavity.py --pairs source=$B/$v/src25.mp4:$B/$v/src25.npz plain=$B/$v/plain.mp4:$B/$v/plain.npz k2=$B/$v/k2.mp4:$B/$v/k2.npz > $B/$v/cavity.json 2> $B/$v/cavity.err
  python3 -c "
import json; d=json.load(open(\"$B/$v/cavity.json\"))
print(\"$v CAVITY\", {k: (x[\"open_frames\"], x[\"black_hole_share\"], x[\"cavity_L_mean\"]) for k, x in d.items()})" 2>/dev/null || echo "$v cavity fail: $(tail -1 $B/$v/cavity.err)"
done
echo BH_DONE
