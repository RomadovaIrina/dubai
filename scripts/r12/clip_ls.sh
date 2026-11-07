#!/bin/bash
# r13 LS: user clips for an arm vs the k5 base of one video, demucs mix, H.264 + AAC, <= MAXMB (default 25):
#   1) full length, full frame, side by side  "k5 (now)" | "<label>"      -> <out>/<id>_k5_vs_<arm>.mp4
#   2) mouth crop of the window where the two differ most around the mouth, 2 panels, 0.5x (audio slowed too) -> <out>/<id>_k5_vs_<arm>_mouth05x.mp4
#   bash scripts/r12/clip_ls.sh <id> <arm> "<right label>" [out dir, default /workspace/dub/outbox/r13_ls] [window s, default 4]
set -u
id=$1; arm=$2; RL=$3; OD=${4:-/workspace/dub/outbox/r13_ls}; WS=${5:-4}; MAXMB=${MAXMB:-25}; mkdir -p $OD
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
B=/workspace/dub/artifacts/r4m; L=$B/r9f_$id/${id}_r9f_$id.mp4; R=$B/r9f_$id$arm/${id}_r9f_$id$arm.mp4; A=/workspace/dub/r10/runs/$id/${id}_DabAI_r9_demucs.mp4
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"
DT="x=12:y=12:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6"
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $L); H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $L)
ph=$H; [ $((2 * W)) -gt 1440 ] && ph=$(( H * 720 / W / 2 * 2 ))   # each panel <= 720 wide
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $A); kb=$(python3 -c "print(int($MAXMB*8*1024*0.92/$dur - 160))")
O=$OD/${id}_k5_vs_$arm.mp4
ffmpeg -loglevel error -y -i $L -i $R -i $A -filter_complex \
  "[0:v]scale=-2:$ph,setsar=1,drawtext=${FF}text='k5 (now)':$DT[l];[1:v]scale=-2:$ph,setsar=1,drawtext=${FF}text='$RL':$DT[r];[l][r]hstack=inputs=2[v]" \
  -map "[v]" -map 2:a:0 -c:v libx264 -preset slow -b:v ${kb}k -maxrate $((kb * 2))k -bufsize $((kb * 2))k -pix_fmt yuv420p -c:a aac -b:a 128k -shortest -movflags +faststart $O
echo "== clip $O $(ls -l $O | awk '{printf "%.1f", $5/1048576}') MB ${dur}s"
# the worst-diff mouth window: mean |L - R| in a box around the source mouth (source landmarks, same frame grid as the finals)
read t0 x y s <<< $(python3 - $L $R $B/r9g_$id/timing/${id}_source_lm_F.npz $WS <<'PY'
import sys, numpy as np, cv2
L, R, lmp, ws = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
lm = np.load(lmp)["lm"]; cl, cr = cv2.VideoCapture(L), cv2.VideoCapture(R); fps = cl.get(cv2.CAP_PROP_FPS) or 30; d = []; k = 0
while True:
    a, fa = cl.read(); b, fb = cr.read()
    if not (a and b): break
    p = lm[k] if k < len(lm) else None
    if p is not None and np.isfinite(p[[13, 14, 33, 263]]).all():
        iod = np.linalg.norm(p[33, :2] - p[263, :2]); cx, cy = p[[61, 291, 13, 14], :2].mean(0); r = int(0.6 * iod)
        d.append(float(np.abs(fa[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)].astype(np.int16) - fb[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)]).mean()))
    else: d.append(0.0)
    k += 1
d = np.array(d); n = max(1, int(ws * fps)); c = np.convolve(d, np.ones(n), "valid"); k0 = int(np.argmax(c))
q = np.nanmedian(lm[k0:k0 + n], 0); iod = np.linalg.norm(q[33, :2] - q[263, :2]); cx, cy = q[[61, 291, 13, 14], :2].mean(0); s = int(1.4 * iod) // 2 * 2
print(round(k0 / fps, 2), int(cx - s / 2), int(cy - s / 2), s)
PY
)
x=$(( x < 0 ? 0 : x )); y=$(( y < 0 ? 0 : y )); OM=$OD/${id}_k5_vs_${arm}_mouth05x.mp4
ffmpeg -loglevel error -y -ss $t0 -t $WS -i $L -ss $t0 -t $WS -i $R -ss $t0 -t $WS -i $A -filter_complex \
  "[0:v]crop=$s:$s:$x:$y,scale=540:540,setsar=1,setpts=2*PTS,drawtext=${FF}text='k5 (now)':$DT[l];[1:v]crop=$s:$s:$x:$y,scale=540:540,setsar=1,setpts=2*PTS,drawtext=${FF}text='$RL':$DT[r];[l][r]hstack=inputs=2[v];[2:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart $OM
echo "== mouth clip $OM window ${t0}s +${WS}s at 0.5x $(ls -l $OM | awk '{printf "%.1f", $5/1048576}') MB"
