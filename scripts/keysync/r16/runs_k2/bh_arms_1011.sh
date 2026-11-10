#!/bin/bash
# r16d K2 black holes: 2 KeySync renders per video (ref 0 = now, KS_REF=auto = teeth ref per shot), then CPU post arms on the same raw layers:
#   C = opening cap, P = pause closure, CP = both, TCP = teeth ref + both. Metrics: black share vs source (k2_mouth_atlas), speech@0.
set -o pipefail
B=/workspace/dub/k2/bh; R=/workspace/dub/dubai; F=$R/scripts/keysync/r16/ks_face.sh; MP=/venv/mediapipe/bin/python; PY=/venv/dabai/bin/python
export KS_LORA=/workspace/dub/k2/lora/lora_pick.pt KS_PASTE_SHIFT=0 KS_ROOT=/workspace/dub/keysync
CAP=${CAP:-1.0}
declare -A SRC=( [old03]=/workspace/dub/inputs/test_videos_raw/03.mp4 [m03]=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV [m18]=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 [q02]=/workspace/dub/k2/src/q02_vera.mp4 [q04]=/workspace/dub/k2/src/q04_farikh.mp4 )
declare -A AD=( [old03]=/workspace/dub/k2/pi/old03 [m03]=/workspace/dub/k2/pi/m03 [m18]=/workspace/dub/k2/pi/m18 [q02]=$B/ad_q02 [q04]=$B/ad_q04 )
for v in q02 q04; do mkdir -p $B/ad_$v; [ -s $B/ad_$v/dubbed_16k.wav ] || { ffmpeg -nostdin -v error -y -i ${SRC[$v]} -vn -ac 1 -ar 16000 $B/ad_$v/dubbed_16k.wav; ffmpeg -nostdin -v error -y -i ${SRC[$v]} -vn -c:a aac -b:a 192k $B/ad_$v/final_mix_aac.m4a; }; done
VIDS=${VIDS:-old03 m18 m03 q04 q02}
for v in $VIDS; do mkdir -p $B/$v
  for g in now T; do [ -s $B/$v/$g.mp4 ] && continue
    R_=0; [ $g = T ] && R_=auto
    env KS_REF=$R_ bash $F ${v}_$g ${SRC[$v]} ${AD[$v]} $B/$v/w_$g $B/$v/$g.mp4 > $B/$v/$g.log 2>&1; echo "$v $g rc $? ref $(cat $B/$v/w_$g/ks/ref_f0.txt 2>/dev/null) $(grep -h "^KS_PROBE" $B/$v/$g.log | head -1 | cut -c1-60)"
  done
  for arm in C P CP TCP; do [ -s $B/$v/$arm.mp4 ] && continue
    g=now; [ $arm = TCP ] && g=T; KW=$B/$v/w_$g/ks; P=$KW/post_f0; [ -s $P/guard.mp4 ] || { echo "$v $arm: no layer"; continue; }
    PA=1; CA=$CAP; [ $arm = C ] && PA=0; [ $arm = P ] && CA=0
    $MP $R/scripts/keysync/r16/k2_pause.py --mode warp --ks $P/guard.mp4 --ks-lm $P/raw_lm.npz --wav $KW/dub_f0.wav --shift 0 --pause $PA --cap $CA --cap-src-lm $KW/crop_lm_f0.npz --out $B/$v/${arm}_layer.mp4 > $B/$v/${arm}_warp.log 2>&1 || { echo "$v $arm warp FAIL"; tail -2 $B/$v/${arm}_warp.log; continue; }
    OTH=""; [ -s $KW/others_f0.npz ] && OTH="--others $KW/others_f0.npz --others-grow 1.3"
    ( cd $KW/kit && KS_CMS=1 $PY ks_plain_par.py --jobs 8 --out $B/$v/$arm.mp4 -- --ks $B/$v/${arm}_layer.mp4 --ks-lm $P/raw_lm.npz --master $KW/master_25fps.mp4 --box $KW/data/box_f0.json --lm $KW/lm_f0.npz --audio ${AD[$v]}/final_mix_aac.m4a --shift 0 --absent skip $OTH --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $P/ecc.npz --align-scale 0 --tex 2 ) > $B/$v/${arm}_paste.log 2>&1 || echo "$v $arm paste FAIL"
    echo "$v $arm $(grep -h "K2_PAUSE" $B/$v/${arm}_warp.log | tr "\n" " " | cut -c1-220)"
  done
  # metrics: black share vs source + speech@0
  KW=$B/$v/w_now/ks; [ -s $B/$v/src.npz ] || GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MP $R/scripts/quality/mp_landmarks.py --video $KW/master_25fps.mp4 --out $B/$v/src.npz > /dev/null 2>&1
  for arm in now T C P CP TCP; do f=$B/$v/$arm.mp4; [ -s $f ] || continue
    [ -s $B/$v/$arm.npz ] || GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MP $R/scripts/quality/mp_landmarks.py --video $f --out $B/$v/$arm.npz > /dev/null 2>&1
    OMP_NUM_THREADS=4 $PY $R/scripts/keysync/r16/k2_mouth_atlas.py --name ${v}_$arm --ours $f --src $KW/master_25fps.mp4 --lm-ours $B/$v/$arm.npz --lm-src $B/$v/src.npz --out $B/$v/atlas_$arm > $B/$v/atlas_$arm.log 2>&1
    ( cd $R && $PY scripts/quality/sync_perframe.py --video $f --audio ${AD[$v]}/dubbed_16k.wav --out $B/$v/$arm.sync.json > /dev/null 2>&1 )
    python3 - $B/$v/atlas_$arm.json $B/$v/$arm.sync.json $v $arm <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["summary"]
try:
    s = json.load(open(sys.argv[2])); sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]; sync = "%.2f" % (sum(sp) / max(len(sp), 1))
except Exception: sync = "x"
print(f"METRIC {sys.argv[3]} {sys.argv[4]}: black ours {a['ours_black_share_open']:.3f} src {a['src_black_share_open']:.3f} | open ours {a['open_mean_ours']:.3f} src {a['open_mean_src']:.3f} | speech@0 {sync}", flush=True)
PY
  done
done
echo BH_DONE
