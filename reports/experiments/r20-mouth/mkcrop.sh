#!/bin/bash
# r20: rebuild the KeySync input (static 512 crop + the 60 mirrored pad frames) of a saved job from its master + box (the commands of ks_prep.py / ks_face.sh)
#   mkcrop.sh <job>   (inside the image; /r18/data/jobs/<job>/e2e/work/ks) -> /r18/crops/<job>/{crop,pad}_f0.mp4 + crop_lm_f0.npz
set -euo pipefail
J=$1; KW=${2:-/r18/data/jobs/$J/e2e/work/ks}; O=/r18/crops/$J; mkdir -p $O; MPY=/venv/mediapipe/bin/python
read x y s < <(python3 -c "import json; b=json.load(open('$KW/data/box_f0.json')); print(b['x'], b['y'], b['side'])")
ffmpeg -v error -y -i $KW/master_25fps.mp4 -an -vf "crop=$s:$s:$x:$y,fps=25,scale=512:512:flags=lanczos" -c:v libx264 -crf 12 -pix_fmt yuv420p $O/crop_f0.mp4
$MPY - $O/crop_f0.mp4 $O/pad_f0.mp4 60 <<'PY'
import subprocess, sys, cv2
src, out, pad = sys.argv[1], sys.argv[2], int(sys.argv[3]); c = cv2.VideoCapture(src); F = []
while True:
    ok, f = c.read()
    if not ok: break
    F.append(f)
F = F + F[-2:-2 - pad:-1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
for f in F: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait(); print("pad", len(F), "frames")
PY
[ -s $KW/crop_lm_f0.npz ] && cp $KW/crop_lm_f0.npz $O/ || $MPY /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $O/crop_f0.mp4 --out $O/crop_lm_f0.npz > $O/lm.log 2>&1
cp $KW/data/dubp_f0.wav $O/; ls -la $O | tail -n +2 | awk '{print $5, $9}' | tr '\n' ' '; echo
