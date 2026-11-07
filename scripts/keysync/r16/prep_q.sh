# prep_q.sh <id> <source>: master 25 fps, own audio 16k, mediapipe landmarks, static (ks_prep rule) + tracked crops
id=$1; src=$2; cd /workspace/dub/r16k2/q; MP=/venv/mediapipe/bin/python
ffmpeg -v error -y -i $src -an -vf fps=25 -c:v libx264 -crf 10 -preset fast ${id}_master.mp4
if [ -n "$3" ]; then ffmpeg -v error -y -i $3 -ac 1 -ar 16000 ${id}_16k.wav; ffmpeg -v error -y -i $3 -c:a aac -b:a 192k ${id}_mix.m4a; else ffmpeg -v error -y -i $src -vn -ac 1 -ar 16000 ${id}_16k.wav; ffmpeg -v error -y -i $src -vn -c:a aac -b:a 192k ${id}_mix.m4a; fi
$MP /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video ${id}_master.mp4 --out ${id}_lm.npz > /dev/null 2>&1
$MP - <<PY
import numpy as np; z=np.load("${id}_lm.npz"); np.savez("${id}_lmF.npz", lm=z["lm"][..., :2].astype(np.float32), j=np.arange(len(z["lm"])))
PY
mkdir -p static/videos static/audios; $MP /workspace/dub/keysync/ks_prep.py ${id}_master.mp4 ${id}_lmF.npz ${id}_16k.wav static $id
$MP /workspace/dub/r16k2/k2_prep_track.py --master ${id}_master.mp4 --lm ${id}_lmF.npz --out ${id}_track
echo PREP_DONE $id
