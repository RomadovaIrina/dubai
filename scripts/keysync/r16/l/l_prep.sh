#!/bin/bash
# r16 L: inputs. Test crops m03 / m18 (paste_inputs box.json, = the K1 paste grid), source audio, dub +40 ms (K1 step 0);
# eval segments; training jobs = Q's q-clips (512 crops) with their OWN audio (checked by envelope correlation vs the clip).
set -e
R=/workspace/dub/r16l; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs; M=/workspace/dub/inputs/mamakris; K=/workspace/dub/keysync
declare -A SRC=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4)
cd $R/data
for v in m03 m18; do
  read x y s < <(python3 -c "import json;b=json.load(open('$P/$v/box.json'));print(b['x'],b['y'],b['side'])")
  [ -s videos/${v}_src.mp4 ] || ffmpeg -v error -y -i ${SRC[$v]} -an -vf "crop=$s:$s:$x:$y,fps=25,scale=512:512:flags=lanczos" -c:v libx264 -crf 12 -pix_fmt yuv420p videos/${v}_src.mp4
  ffmpeg -v error -y -i ${SRC[$v]} -vn -ac 1 -ar 16000 audios/${v}_src.wav
  ffmpeg -v error -y -i $P/$v/dubbed_16k.wav -af adelay=40 -ac 1 -ar 16000 audios/${v}_dub40.wav
done
# eval segments: tag start dur  (A: m03 9-13 / m18 2-6 with 1 s margins; B: m03 23-27)
cut() { ffmpeg -v error -y -ss $3 -t $4 -i videos/$2_src.mp4 -an -c:v libx264 -crf 12 -pix_fmt yuv420p videos/$1.mp4
        ffmpeg -v error -y -ss $3 -t $4 -i audios/$2_dub40.wav audios/$1.wav; }
cut m03a m03 8 6; cut m18a m18 1 6; cut m03b m03 22 6
for t in m03a m18a m03b; do echo "$t $R/data/videos/$t.mp4 $R/data/audios/$t.wav"; done > $R/eval/jobs_eval.txt
# training: q crops + own audio
: > $R/dump/jobs_train.txt
for f in $K/data/videos/q0*_src.mp4; do t=$(basename $f .mp4); c=/workspace/dub/r16q/clips/${t%_src}.mp4
  r=$(python3 - $K/data/audios/$t.wav $c <<'EOF'
import sys, subprocess, numpy as np
def env(p):
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-t", "40", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout, np.int16).astype(np.float32)
    return np.log(np.sqrt((a[: len(a) // 640 * 640].reshape(-1, 640) ** 2).mean(1)) + 1)
a, b = env(sys.argv[1]), env(sys.argv[2]); n = min(len(a), len(b)); print(round(float(np.corrcoef(a[:n], b[:n])[0, 1]), 3))
EOF
)
  echo "$t own-audio corr $r"
  python3 -c "import sys; sys.exit(0 if float('$r') > 0.8 else 1)" && echo "$t $f $K/data/audios/$t.wav" >> $R/dump/jobs_train.txt
done
wc -l $R/dump/jobs_train.txt; ls -la videos audios | head -20
