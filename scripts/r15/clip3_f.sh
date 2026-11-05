#!/bin/bash
# r15 F2: 0.5x mouth 3-panel (or N-panel) clip over windows (+-0.5 s each) with the source audio, r14-d layout.
#   bash scripts/r15/clip3_f.sh <id> "<win,win>" OUT.mp4 "label=VIDEO" "label=VIDEO" ...
# A window "auto" = the 3 most open-mouth seconds of the source (teeth visible = where the eye judges lip line / teeth).
set -u
id=$1; WINS=$2; O=$3; shift 3; cd "$(dirname "${BASH_SOURCE[0]}")/../.."; . scripts/env.sh >/dev/null
SRC=$(ls /workspace/dub/r14d/eval/$id/e2e/work/in/$id.* | head -1); LM=/workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz
if [ "$WINS" = auto ]; then WINS=$(python3 - $LM $SRC <<'PY'
import sys, numpy as np, cv2
lm = np.load(sys.argv[1])["lm"]; fps = cv2.VideoCapture(sys.argv[2]).get(cv2.CAP_PROP_FPS) or 30
op = np.nan_to_num(np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / np.linalg.norm(lm[:, 61, :2] - lm[:, 291, :2], axis=1))
s = np.convolve(op, np.ones(int(fps)) / int(fps), "same"); out = []
for k in np.argsort(-s):
    t = k / fps
    if t < 1 or all(abs(t - u) > 3 for u in out): out.append(t)
    if len(out) == 3: break
print(",".join(f"{t - 1:.1f}-{t + 1:.1f}" for t in sorted(out)))
PY
); fi
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"
T=$(mktemp -d); i=0; : > $T/list.txt; N=$#
for w in ${WINS//,/ }; do
  t0=${w%-*}; t1=${w#*-}; read s d x y c <<< $(python3 - $LM $t0 $t1 $SRC <<'PY'
import sys, numpy as np, cv2
lm = np.load(sys.argv[1])["lm"]; t0, t1 = float(sys.argv[2]), float(sys.argv[3]); fps = cv2.VideoCapture(sys.argv[4]).get(cv2.CAP_PROP_FPS) or 30
s = max(0.0, t0 - 0.5); d = (t1 + 0.5) - s; q = np.nanmedian(lm[int(s * fps):int((s + d) * fps)], 0)
mw = np.linalg.norm(q[61, :2] - q[291, :2]); cx, cy = q[[61, 291, 13, 14], :2].mean(0); c = int(1.8 * mw) // 2 * 2
print(round(s, 2), round(d, 2), max(0, int(cx - c / 2)), max(0, int(cy - 0.5 * c)), c)
PY
)
  DT="x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6"; IN=""; FC=""; j=0; ST=""
  for v in "$@"; do lab=${v%%=*}; path=${v#*=}; IN="$IN -ss $s -t $d -i $path"
    FC="$FC[$j:v]crop=$c:$c:$x:$y,scale=420:420:flags=lanczos,setsar=1,setpts=2*PTS,drawtext=${FF}text='$lab  $w s':$DT[p$j];"; ST="$ST[p$j]"; j=$((j + 1)); done
  ffmpeg -loglevel error -y $IN -ss $s -t $d -i $SRC -filter_complex "$FC${ST}hstack=inputs=$N,fps=30[v];[$N:a]atempo=0.5,aresample=44100[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k $T/w$i.mp4 && echo "file '$T/w$i.mp4'" >> $T/list.txt
  i=$((i + 1))
done
ffmpeg -loglevel error -y -f concat -safe 0 -i $T/list.txt -c copy -movflags +faststart $O; rm -rf $T
echo "== clip3_f $O $(ls -l $O | awk '{printf "%.1f", $5/1048576}') MB windows $WINS"
