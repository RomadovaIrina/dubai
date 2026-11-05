#!/bin/bash
# r15 T: the clips for the eye -> /workspace/dub/outbox/r15_t/ (each made only when its inputs exist; re-run = fills the rest)
#   <id>_mouth_current_vs_clean_half_speed.mp4   profile current (CodeFormer + htdemucs) | clean (no CodeFormer, Spleeter), 0.5x   (id = old03_30s, m03_20s)
#   <id>_full_current_vs_clean.mp4               the same, full frame (audio = current)
#   <id>_current.mp4 / <id>_clean.mp4            each arm alone, with its own background bed (to LISTEN to the bed difference)
#   m03_20s_mouth_fast_vs_quality_half_speed.mp4 mode fast (student) | mode quality (teacher), 0.5x
#   m18_4k_mouth_fix_vs_source_half_speed.mp4    the 4K job's mouth | the 4K source, both in 4K pixels, 0.5x
#   long5_excerpts.mp4 / long10_excerpts.mp4     10 s from the start, the middle and the end of the long outputs
set -u
D=/workspace/dub/dubai-t; cd $D; . scripts/env.sh >/dev/null; J=/workspace/dub/r15t/jobs; O=/workspace/dub/outbox/r15_t; mkdir -p $O
lm(){ find /workspace/dub/artifacts/r4m/r9g_$1 -name "*source_lm_F.npz" 2>/dev/null | head -1; }
half(){ ffmpeg -v error -y -i $1 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast $2; }
lab(){ echo "drawtext=text='$1':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6"; }
pair(){   # pair <left> <right> <labelL> <labelR> <lm> <name>
  [ -s "$1" ] && [ -s "$2" ] && [ -n "$5" ] || { echo "skip $6"; return; }
  python scripts/r13/distill/mouth_2panel.py --left $1 --right $2 --lm $5 --out $O/$6.mp4 --labels "$3" "$4" && half $O/$6.mp4 $O/${6}_half_speed.mp4 && rm -f $O/$6.mp4 && echo "made $6"
}
for id in old03_30s m03_20s; do
  s=${id%_*}; C=$J/t15_ab_${s}_cur/t15_ab_${s}_cur.mp4; K=$J/t15_ab_${s}_cln/t15_ab_${s}_cln.mp4; L=$(lm t15_ab_${s}_cur)
  pair $C $K "current (CodeFormer + htdemucs)" "clean (no CodeFormer, Spleeter)" "$L" ${id}_mouth_current_vs_clean
  if [ -s $C ] && [ -s $K ]; then
    ffmpeg -v error -y -i $C -i $K -filter_complex "[0:v]scale=-2:960,$(lab current)[l];[1:v]scale=-2:960,$(lab clean)[r];[l][r]hstack=inputs=2[v]" \
      -map "[v]" -map 0:a -c:v libx264 -crf 20 -preset fast -c:a aac $O/${id}_full_current_vs_clean.mp4 && echo "made ${id}_full"
    cp $C $O/${id}_current.mp4; cp $K $O/${id}_clean.mp4
  fi
done
pair $J/t15_m03_fast/t15_m03_fast.mp4 $J/t15_ab_m03_cur/t15_ab_m03_cur.mp4 "mode fast (student 4 steps + c8)" "mode quality (teacher q2+F3+c8)" "$(lm t15_ab_m03_cur)" m03_20s_mouth_fast_vs_quality
K=$J/t15_4k_fix/t15_4k_fix.mp4; S=/workspace/dub/r15t/edge/in/res4k_vertical.mp4; L=$(lm t15_4k_fix)
if [ -s $K ] && [ -n "$L" ]; then   # the job's landmarks are at the face work size (1080x1920): x2 for the 4K frames
  read x y s < <(python - $L <<'PY'
import sys, numpy as np
L = np.load(sys.argv[1])["lm"]; L = L[:, 0] if L.ndim == 4 else L; P = L[:, [61, 291, 0, 17, 13, 14, 78, 308], :2].reshape(-1, 2); P = P[np.isfinite(P).all(1)]
if np.nanmax(P) <= 1.5: P = P * np.array([1080, 1920])
P = P * 2; cx, cy = np.median(P, 0); mw = np.percentile(P[:, 0], 95) - np.percentile(P[:, 0], 5); s = int(max(128, 2.6 * mw)) // 2 * 2
print(int(np.clip(cx - s / 2, 0, 2160 - s)), int(np.clip(cy - s / 2, 0, 3840 - s)), s)
PY
)
  ffmpeg -v error -y -i $K -i $S -filter_complex "[0:v]crop=$s:$s:$x:$y,scale=640:640:flags=lanczos,$(lab '4K job output')[l];[1:v]crop=$s:$s:$x:$y,scale=640:640:flags=lanczos,$(lab '4K source')[r];[l][r]hstack=inputs=2[v]" \
    -map "[v]" -map 0:a -c:v libx264 -crf 16 -preset fast -c:a aac $O/m18_4k_mouth.mp4 && half $O/m18_4k_mouth.mp4 $O/m18_4k_mouth_fix_vs_source_half_speed.mp4 && rm -f $O/m18_4k_mouth.mp4 && echo "made m18_4k_mouth"
fi
for n in long5 long10; do
  V=$J/t15_$n/t15_$n.mp4; [ -s $V ] || { echo "skip $n"; continue; }
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $V); parts=""; k=0
  for t in 5 $(python3 -c "print(round($d / 2 - 5, 1))") $(python3 -c "print(round($d - 15, 1))"); do
    k=$((k+1)); ffmpeg -v error -y -ss $t -t 10 -i $V -vf "scale=-2:960,$(lab "t=$t s")" -c:v libx264 -crf 20 -preset fast -c:a aac -ar 48000 $O/.${n}_$k.mp4 && parts="$parts -i $O/.${n}_$k.mp4"
  done
  ffmpeg -v error -y $parts -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]" -map "[v]" -map "[a]" -c:v libx264 -crf 20 -preset fast -c:a aac $O/${n}_excerpts.mp4 && rm -f $O/.${n}_*.mp4 && echo "made ${n}_excerpts"
done
ls -la $O
