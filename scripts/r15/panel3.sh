#!/bin/bash
# r15 W0: 3-panel mouth clip at the user's eye spots, 0.5x: <A> | <B> | <C> (same crop as scripts/r13/distill/mouth_2panel.py, the left
# video's audio). For the TBDub jelly check: q2 | tb (TBDub through our q2 + F3 chain) | tbnat (TBDub native, its own paste-back, 25 fps).
#   bash scripts/r15/panel3.sh <id> <A> <B> <C> "label A" "label B" "label C"
set -u
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; OB=/workspace/dub/outbox/r15_w; id=$1; shift
declare -A SP=([old03]="2.75:4.25 4.75:6.25 20.55:21.25 23.5:24.5 26.75:28.25" [m03]="16.5:21.5" [m18]="1.5:6.5")
vid(){ [ "$1" = q2 ] && echo $X/q2_$id/q2_$id.mp4 || echo $X/${1}_$id/${1}_$id.mp4; }
A=$(vid $1); B=$(vid $2); C=$(vid $3); LA=$4; LB=$5; LC=$6; T=$(mktemp -d)
LM=$(find /workspace/dub/artifacts/r4m -path "*r9g_q2_${id}*/*" -name "q2_${id}_source_lm_F.npz" | head -1)
IFS=, read W H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $A)
read x0 y0 s < <(python3 - "$LM" "$W" "$H" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); L = z[next(k for k in ("lm", "lm_all", "landmarks") if k in z.files)]
L = L[:, 0] if L.ndim == 4 else L; W, H = int(sys.argv[2]), int(sys.argv[3])
P = L[:, [61, 291, 0, 17, 13, 14, 78, 308], :2].reshape(-1, 2); P = P[np.isfinite(P).all(1)]
P = P * np.array([W, H]) if np.nanmax(P) <= 1.5 else P
cx, cy = np.median(P, 0); mw = np.percentile(P[:, 0], 95) - np.percentile(P[:, 0], 5); s = int(max(64, 2.6 * mw)); s -= s % 2
print(int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), s)
PY
)
lab(){ echo "drawtext=text='$1':x=8:y=8:fontsize=18:fontcolor=yellow:box=1:boxcolor=black@0.6"; }
ffmpeg -v error -y -i $A -i $B -i $C -filter_complex "[0:v]crop=$s:$s:$x0:$y0,scale=-2:360,$(lab "$LA")[a];[1:v]crop=$s:$s:$x0:$y0,scale=-2:360,$(lab "$LB")[b];[2:v]crop=$s:$s:$x0:$y0,scale=-2:360,fps=30,$(lab "$LC")[c];[a][b][c]hstack=inputs=3[v]" \
  -map "[v]" -map 0:a -c:v libx264 -crf 18 -preset fast -c:a aac -shortest $T/m.mp4
parts=""; k=0
for span in ${SP[$id]}; do s0=${span%:*}; e0=${span#*:}; k=$((k+1))
  ffmpeg -v error -y -ss $s0 -to $e0 -i $T/m.mp4 -filter_complex "[0:v]setpts=2*PTS,drawtext=text='t=$s0-$e0 s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac -ar 48000 $T/ts_$k.mp4 && parts="$parts -i $T/ts_$k.mp4"; done
n=$(echo $parts | grep -o "\-i" | wc -l)
ffmpeg -v error -y $parts -filter_complex "$(for i in $(seq 0 $((n-1))); do echo -n "[$i:v][$i:a]"; done)concat=n=$n:v=1:a=1[v][a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac $OB/${id}_mouth3_$1_$2_$3_spots_half_speed.mp4
rm -rf $T; ls -la $OB/${id}_mouth3_$1_$2_$3_spots_half_speed.mp4 | awk '{print $5, $9}'
