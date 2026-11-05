#!/bin/bash
# r15 K: a 0.5x mouth / lower-face 2-panel LEFT | RIGHT of the pasted finals over one window (padded 0.5 s), the dub mix slowed to match
# (scripts/r12/clip_at.sh with this server's paths; the crop from the source landmarks).
#   bash k_clip.sh <id> LEFT.mp4 "<left label>" RIGHT.mp4 "<right label>" <t0-t1> OUT.mp4
set -u
id=$1; L=$2; LL=$3; R=$4; RL=$5; w=$6; O=$7; K=/workspace/dub/keysync; LM=$K/lm/${id}_lm.npz
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"
t0=${w%-*}; t1=${w#*-}; read s d x y c <<< $(/venv/dabai/bin/python - $LM $t0 $t1 $L <<'PY'
import sys, numpy as np, cv2
lm = np.load(sys.argv[1])["lm"]; t0, t1 = float(sys.argv[2]), float(sys.argv[3]); fps = cv2.VideoCapture(sys.argv[4]).get(cv2.CAP_PROP_FPS) or 30
s = max(0.0, t0 - 0.5); d = (t1 + 0.5) - s; q = np.nanmedian(lm[int(s * fps):int((s + d) * fps)], 0)
iod = np.linalg.norm(q[33, :2] - q[263, :2]); cx, cy = q[[61, 291, 13, 14], :2].mean(0); c = int(1.8 * iod) // 2 * 2
print(round(s, 2), round(d, 2), max(0, int(cx - c / 2)), max(0, int(cy - 0.45 * c)), c)
PY
)
DT="x=8:y=8:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
ffmpeg -loglevel error -y -ss $s -t $d -i $L -ss $s -t $d -i $R -filter_complex \
  "[0:v]crop=$c:$c:$x:$y,scale=480:480,setsar=1,setpts=2*PTS,drawtext=${FF}text='$LL  $w s  0.5x':$DT[l];[1:v]crop=$c:$c:$x:$y,scale=480:480,setsar=1,setpts=2*PTS,drawtext=${FF}text='$RL  $w s  0.5x':$DT[r];[l][r]hstack=inputs=2,fps=30[v];[0:a]atempo=0.5,aresample=44100[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart $O
echo "== k_clip $O $(ls -l $O | awk '{printf "%.1f", $5/1048576}') MB window $w"
