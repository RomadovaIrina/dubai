#!/bin/bash
# r11 FACE KeySync-in-composite variants: ksv.sh <id> <gen tag> <layer suffix> <prefix> "name|extra flags" ...  -> SyncNet (gate number) + face_metrics of the final
R=${KSV_ROOT:-/workspace/dub/dubai}; cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
id=$1; ST=$2; LSX=$3; P=$4; shift 4
B=/workspace/dub/artifacts/r4m; W=$B/$ST/work_${id}_$ST; M=$W/${id}_${ST}_25fps.manifest.json; LS=$W/${id}_${ST}_25fps_$LSX.mp4
A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SLM=$B/$ST/timing/${id}_source_lm_F.npz
FB=$W/${id}_${ST}_25fps_fb_fb7.npz; mkdir -p /workspace/dub/r11/ksv
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5 --tex-tau 1000 --tex-sigma 2.5"
for spec in "$@"; do
  v=${spec%%|*}; X=${spec#*|}; X=${X//@FB/$FB}; T=${P}_$v
  FIX30_LS=$LS FIX30_MASTER_FPS=25 FIX30_TD=$B/$ST/timing FIX30_MODE=r3fix FIX30_THREADS=16 FIX30_ARGS="$V1 $X" bash scripts/quality/r3_fix30.sh $M $T > /tmp/ksv_$T.log 2>&1
  O=$W/${id}_${ST}_25fps_$T.mp4; [ -s $O ] || { echo "== $v FAILED"; tail -3 $W/${id}_${ST}_25fps_$T.log; continue; }
  s=$(flock /workspace/dub/gpu.lock python /workspace/dub/dubai/scripts/r11/sync_speech.py --video $O --audio $A --label "$id $v")
  python /workspace/dub/dubai/scripts/r10/face_metrics.py --video $O --src $SRC --lm-src $SLM --out /workspace/dub/r11/ksv/${id}_$T.json > /tmp/fm_$T.log 2>&1
  f=$(python3 -c "
import json; j=json.load(open(\"/workspace/dub/r11/ksv/${id}_$T.json\")); r=j.get(\"flicker\",{}); c=j.get(\"chin_flow\",{}) or {}
print(\"ring\", round(r.get(\"ring_bright_ratio\") or 0,2), \"ringS\", r.get(\"ring_bright_strong\"), \"| chin swim\", round(c.get(\"swim_rms\") or 0,4), \"vel\", round(c.get(\"vel_rms\") or 0,4), \"ev\", c.get(\"events\"), \"S\", c.get(\"strong\"))" 2>&1 | tail -1)
  echo "$s | $f"
done
echo KSV_DONE $P
