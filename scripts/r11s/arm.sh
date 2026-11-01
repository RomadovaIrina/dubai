#!/bin/bash
# r11 SYNC: one composite arm on the r9 + fix layers of <id> with this worktree's fps_composite + its gate SyncNet (sync_speech).
#   bash scripts/r11s/arm.sh <id> <tag> "<extra fps_composite flags>"   env: LS=<layer mp4> FB=<fallback npz> (default: the r9 + fix rep2 pair), BASEARGS (override V1)
# -> artifacts/r4m/r9g_<id>/work_<id>_r9g_<id>/<id>_r9g_<id>_25fps_<tag>.mp4, log in /workspace/dub/r11s/logs/<id>_<tag>.log
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
id=$1; T=$2; X=${3:-}; ST=r9g_$id; B=/workspace/dub/artifacts/r4m; W=$B/$ST/work_${id}_$ST; M=$W/${id}_${ST}_25fps.manifest.json; P=$W/${id}_${ST}_25fps
A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav; [ -s $A ] || A=$(ls /workspace/dub/r10/runs/$id/work/*/r9_${id}_htd/$id/dubbed_16k.wav 2>/dev/null | head -1)
LS=${LS:-${P}_ls_cfd_ia04_srt1_rep2.mp4}; FB=${FB-${P}_fb_fb7_r9f_${id}_rep2.npz}
V1=${BASEARGS:-"--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5 --tex-tau 1000 --tex-sigma 2.5"}
FBX=""; [ -n "$FB" ] && FBX="--fallback $FB"
L=/workspace/dub/r11s/logs/${id}_$T.log; mkdir -p $(dirname $L)
O=${P}_$T.mp4; rm -f $O
t0=$(date +%s); FIX30_LS=$LS FIX30_MASTER_FPS=25 FIX30_TD=$B/$ST/timing FIX30_MODE=r3fix FIX30_THREADS=${THR:-24} FIX30_ARGS="$V1 $FBX $X" bash scripts/quality/r3_fix30.sh $M $T > $L 2>&1
[ -s $O ] || { echo "== $id $T FAILED"; tail -5 ${P}_$T.log; exit 1; }
echo "== $id $T composite $(( $(date +%s) - t0 )) s" >> $L
V25=/tmp/r11s_${id}_$T.v25.mp4; ffmpeg -loglevel error -y -i $O -vf fps=25 -c:v libx264 -crf 12 -an $V25; flock /workspace/dub/gpu.lock python /workspace/dub/dubai/scripts/r11/sync_speech.py --native --video $V25 --audio $A --label "$id $T" | tee -a $L | grep "^=="
rm -f $V25
