#!/bin/bash
# ks_eval.sh <arm> <tag>: quality of a KeySync speed arm vs the KeySync baseline, on its raw paste:
#   SyncNet AV offset of the raw output -> paste with that shift -> SyncNet on dub-speech seconds at offset 0 (gate metric, 25 fps rendition) + strong lip flashes (r3_flash)
A=$1; id=$2; K=/workspace/dub/keysync; O=$K/fast/$A; KS=$O/out/$id/${id}_$id.mp4; E=$O/eval_$id; mkdir -p $E
case $id in q18) SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4;; qo3) SRC=/workspace/dub/inputs/test_videos_raw/03.mp4;; m03) SRC=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV;; esac
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
DUB=${KS_DUB:-/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav}; MIX=${KS_MIX:-/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/final_mix_aac.m4a}
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio $DUB --out $E/raw_sync.json > /dev/null 2>&1
SH=$(python3 -c "import json; print(-int(json.load(open('$E/raw_sync.json'))['global']['av_offset']))")
KS_SHIFT=$SH python $K/ks_paste.py $KS $SRC $K/data/box_$id.json $MIX $E/paste.mp4 > /dev/null
ffmpeg -v error -y -i $E/paste.mp4 -vf fps=25 -c:v libx264 -crf 12 -an $E/p25.mp4
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $E/p25.mp4 --audio $DUB --out $E/sync.json > /dev/null 2>&1
python scripts/quality/r3_flash.py --video $E/paste.mp4 --src $SRC --lm ${KS_LM:-/workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz} --out $E/flash.json > /dev/null 2>&1
python3 - $E $SH $A $id <<'PY'
import json, sys, statistics as st
E, sh, A, id_ = sys.argv[1:5]
r = json.load(open(f"{E}/raw_sync.json")); s = json.load(open(f"{E}/sync.json")); f = json.load(open(f"{E}/flash.json"))
sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]
fl = sum(1 for e in f["flash_list"] if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0)
out = {"arm": A, "id": id_, "raw_conf": r["global"]["conf"], "raw_av_offset": r["global"]["av_offset"], "shift": int(sh), "sync_speech_at0": round(st.mean(sp), 3), "flash_strong": fl}
json.dump(out, open(f"{E}/eval.json", "w")); print("KS_EVAL " + json.dumps(out))
PY
