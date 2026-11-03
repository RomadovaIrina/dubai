#!/bin/bash
# r13 colour + detail fix (cfhf) on a management final: v2 (alpha 0.8, the old flags) and cfhf (alpha 0.8 + KS_CMS=1, --end-fade 12,
# --skin-cm 12 --cm-robust 1 --guard-fade 10, KS_HF=3 KS_SHARP=0.3), the same KeySync 10-step output; gate numbers (SyncNet speech at0, strong flashes,
# jerk, colour check, sharpness) for both; a v2 | cfhf 2-panel + a 0.5x box crop at v2's worst upper-face colour jump.
#   r13_cfhf.sh <id> <source video> <name>   (m03: KS_BOX_K / KS_BOX_M were applied at prep; data/box_<id>.json is the speaker crop)
cd /workspace/dub/keysync; id=$1; SRC=$2; NAME=$3; K=/workspace/dub/keysync; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id
V2D=/workspace/dub/outbox/r13_ks/v2; CF=/workspace/dub/outbox/r13_ks/colorfix; mkdir -p $V2D $CF qc/$id
KS_OUTDIR=$V2D bash ks_final.sh $id $SRC $NAME > final/$id/final_v2.log 2>&1 || { echo "V2_FAIL $id"; tail -5 final/$id/final_v2.log; exit 1; }
V2=$V2D/${NAME}_KeySync.mp4; echo "V2_READY $V2 $(stat -c %s $V2)"
O=$K/final/${id}_cfhf; mkdir -p $O; ln -sfn $K/final/$id/ks $O/ks; cp final/$id/raw_sync.json final/$id/src.mediapipe.npz $O/
KS_CMS=1 KS_HF=3 KS_SHARP=0.3 KS_REDO=1 KS_FINALDIR=$O KS_OUTDIR=$CF KS_LAYERARGS="--end-fade 12" KS_FIXARGS="--skin-cm 12 --cm-robust 1 --guard-fade 10" \
  bash ks_final.sh $id $SRC ${NAME}_cfhf > $O/final.log 2>&1 || { echo "CFHF_FAIL $id"; tail -5 $O/final.log; exit 1; }
grep -a "==" $O/final.log | cut -c1-200; F=$CF/${NAME}_cfhf_KeySync.mp4; echo "CFHF_READY $F $(stat -c %s $F)"
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $SRC | python3 -c "import sys; n, d = sys.stdin.read().strip().split('/'); print(float(n) / float(d))")
meas() {  # <video> <tag>
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $1 --out $O/$2.mediapipe.npz > /dev/null 2>&1 )
  /venv/dabai/bin/python ks_colorcheck.py --a $1 --b $SRC --box data/box_$id.json --lm $O/$2.mediapipe.npz --out qc/$id/$2.json | sed "s/^/CC $2 /" | cut -c1-600
  SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  /venv/dabai/bin/python ks_lag.py --lm $O/$2.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps $FPS --shift $SH --out $CF/$2_lag | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('LAG $2', d['spike_rate'], 'src', d['source_rate'])"
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    python scripts/quality/r3_flash.py --video $1 --src $SRC --lm /workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz --out $O/$2_flash.json > /dev/null 2>&1
    python3 -c "import json; f=json.load(open('$O/$2_flash.json')); print('FLASH $2', sum(1 for e in f['flash_list'] if (e.get('mc_ratio') or 0) >= 2.0 and (e.get('mc_vs_src') or 0) >= 2.0))"
    ffmpeg -v error -y -i $1 -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$2.mp4
    flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$2.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$2.json > /tmp/s_$2.log 2>&1
    python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$2.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC $2', d['global'], round(st.mean(sp),3))" )
}
meas $V2 ${NAME}_v2; meas $F ${NAME}_cfhf
# 2-panel v2 | cfhf, full length <= 25 MB; + a 0.5x crop of the box around v2's worst upper-face jump (or the middle if none)
P=$CF/${NAME}_v2_vs_cfhf; LB="drawtext=font=DejaVu Sans:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:x=12:y=12:text"
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $F); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))")
ffmpeg -v error -y -i $V2 -i $F -filter_complex "[0:v]$LB='v2 (now)'[a];[1:v]$LB='colour + detail'[b];[a][b]hstack=2[v]" -map "[v]" -map 1:a \
  -c:v libx264 -preset slow -crf 20 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -c:a copy -movflags +faststart ${P}_2panel.mp4
read T0 BX <<< $(python3 -c "
import json; q=json.load(open('qc/$id/${NAME}_v2.json')); j=q['jumps']['up']; b=json.load(open('data/box_$id.json'))
t=max(j, key=lambda e: e['dE'])['t'] if j else $D/2; print(round(max(0, t-2), 2), '%d:%d:%d:%d' % (b['side'], b['side'], b['x'], b['y']))")
ffmpeg -v error -y -ss $T0 -t 4 -i $V2 -ss $T0 -t 4 -i $F -filter_complex \
  "[0:v]crop=$BX,scale=480:480:flags=lanczos,$LB='v2 0.5x'[a];[1:v]crop=$BX,scale=480:480:flags=lanczos,$LB='colour + detail 0.5x'[b];[a][b]hstack=2,setpts=2*PTS[v];[1:a]atempo=0.5[au]" \
  -map "[v]" -map "[au]" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart ${P}_crop_${T0}s_halfspeed.mp4
ls -la ${P}_*; echo CFHF_DONE $id
