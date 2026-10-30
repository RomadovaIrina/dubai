#!/bin/bash
# r9 diag: per window source | output side by side (full frame + x3 mouth zoom), the dub mix on the audio; and one mouth contact sheet per window
# (top row source, bottom row output, every 4th frame).   bash strips.sh <id> <out_dir> <t0:t1> ...
set -u
id=$1; O=$2; shift 2; mkdir -p $O
SRC=/workspace/dub/r9/runs/$id/work/in/$(ls /workspace/dub/r9/runs/$id/work/in | head -1); OUT=/workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4
LM=/workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz
for w in "$@"; do
  t0=${w%:*}; t1=${w#*:}; d=$(python3 -c "print($t1-$t0)")
  read cx cy r < <(python3 -c "
import numpy as np, subprocess, json
lm = np.load('$LM')['lm']; fps = 29.975; a, b = int($t0 * fps), int($t1 * fps); L = lm[a:b]
mo = np.nanmedian(L[:, [61, 291, 0, 17]], axis=(0, 1)); iod = np.nanmedian(np.linalg.norm(L[:, 33] - L[:, 263], axis=1))
print(int(mo[0]), int(mo[1]), int(0.9 * iod))")
  z=$((2 * r)); x0=$((cx - r)); y0=$((cy - r))
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $SRC -ss $t0 -t $d -i $OUT -filter_complex \
    "[0:v]split[s1][s2];[1:v]split[o1][o2];[s1]scale=-2:640[sf];[o1]scale=-2:640[of];[s2]crop=$z:$z:$x0:$y0,scale=320:320[sz];[o2]crop=$z:$z:$x0:$y0,scale=320:320[oz];[sz][oz]vstack[zz];[sf][of][zz]hstack=3,drawtext=text='$id %{pts\:hms} +$t0 s | source | output | mouth x3 (top source, bottom output)':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v]" \
    -map "[v]" -map 1:a:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 128k $O/${id}_${t0}-${t1}s_src_vs_out.mp4
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $SRC -ss $t0 -t $d -i $OUT -filter_complex \
    "[0:v]crop=$z:$z:$x0:$y0,scale=160:160,select='not(mod(n\,4))',tile=12x1[a];[1:v]crop=$z:$z:$x0:$y0,scale=160:160,select='not(mod(n\,4))',tile=12x1[b];[a][b]vstack" \
    -frames:v 1 $O/${id}_${t0}-${t1}s_mouth_sheet.jpg
  echo "== $w mouth box centre ($cx,$cy) r $r"
done
ls -la $O
