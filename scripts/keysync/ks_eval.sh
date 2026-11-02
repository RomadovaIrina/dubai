#!/bin/bash
# ks_eval.sh <arm> <tag>: quality of a KeySync speed arm vs the KeySync baseline, on its raw paste:
#   SyncNet AV offset of the raw output -> the lag fix (ks_tsmooth alpha KS_TSMOOTH, default 0.8, 0 = off) -> paste with that shift
#   -> SyncNet on dub-speech seconds at offset 0 (gate metric, 25 fps rendition) + strong lip flashes (r3_flash) + the lag spike rate (ks_lag)
A=$1; id=$2; K=/workspace/dub/keysync; O=$K/fast/$A; KS=$O/out/$id/${id}_$id.mp4; TS=${KS_TSMOOTH:-0.8}; E=$O/eval_$id; [ "$TS" != 0 ] && E=$O/eval_ts_$id; mkdir -p $E
case $id in q18) SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4;; qo3) SRC=/workspace/dub/inputs/test_videos_raw/03.mp4;; m03) SRC=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV;; esac
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
DUB=${KS_DUB:-/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav}; MIX=${KS_MIX:-/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/final_mix_aac.m4a}
[ -s $E/raw_sync.json ] || flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio $DUB --out $E/raw_sync.json > /dev/null 2>&1
SH=$(python3 -c "import json; print(-int(json.load(open('$E/raw_sync.json'))['global']['av_offset']))")
V=$KS
if [ "$TS" != 0 ]; then
  /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $KS --out $E/ks.mediapipe.npz > /dev/null 2>&1
  python $K/ks_tsmooth.py --layer $KS --lm $E/ks.mediapipe.npz --out $E/ks_ts.mp4 --alpha $TS > /dev/null; V=$E/ks_ts.mp4
fi
KS_SHIFT=$SH python $K/ks_paste.py $V $SRC $K/data/box_$id.json $MIX $E/paste.mp4 > /dev/null
ffmpeg -v error -y -i $E/paste.mp4 -vf fps=25 -c:v libx264 -crf 12 -an $E/p25.mp4
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $E/p25.mp4 --audio $DUB --out $E/sync.json > /dev/null 2>&1
python scripts/quality/r3_flash.py --video $E/paste.mp4 --src $SRC --lm ${KS_LM:-/workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz} --out $E/flash.json > /dev/null 2>&1
/venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $E/paste.mp4 --out $E/paste.mediapipe.npz > /dev/null 2>&1
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $SRC | python3 -c "import sys; n, d = sys.stdin.read().strip().split('/'); print(float(n) / float(d))")
/venv/dabai/bin/python $K/ks_lag.py --lm $E/paste.mediapipe.npz --src-lm $K/final/$id/src.mediapipe.npz --fps $FPS --shift $SH --out $E/lag > /dev/null 2>&1
python3 - $E $SH $A $id $TS <<'PY'
import json, sys, os, statistics as st
E, sh, A, id_, ts = sys.argv[1:6]
r = json.load(open(f"{E}/raw_sync.json")); s = json.load(open(f"{E}/sync.json")); f = json.load(open(f"{E}/flash.json"))
lag = json.load(open(f"{E}/lag.json")) if os.path.exists(f"{E}/lag.json") else {}
sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]
fl = sum(1 for e in f["flash_list"] if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0)
out = {"arm": A, "id": id_, "tsmooth": float(ts), "raw_conf": r["global"]["conf"], "raw_av_offset": r["global"]["av_offset"], "shift": int(sh), "sync_speech_at0": round(st.mean(sp), 3), "flash_strong": fl,
       "jerk_rate": lag.get("spike_rate"), "jerk_rate_src": lag.get("source_rate")}
json.dump(out, open(f"{E}/eval.json", "w")); print("KS_EVAL " + json.dumps(out))
PY
