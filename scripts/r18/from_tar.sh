#!/bin/bash
# r18: bring saved layers back for CPU re-pastes on a fresh docker VM: no GPU, no models volume (only MediaPipe's face_landmarker.task, the
# project's pinned URL + sha256, for the multi-shot landmark slice). The tars are C:\repos\chkpts\r18_quality\layers_<id>.tar -> <root>/keep/.
#   from_tar.sh <id> [input in in/ for a job whose face work was downscaled (q05, q02): it becomes work/hires_src.<ext>]
# Then: r18.sh repaste <id> <variant> K=V ...  (KS_POST_REUSE=1: the saved post layer; only the paste runs).
set -uo pipefail
B=${R18_ROOT:-/root/r18}; CODE=${R18_CODE:-$B/code}; id=$1; HI=${2:-}; W=$B/data/jobs/$id/e2e/work; mkdir -p $B/data/jobs $B/out/$id $B/log
tar xf $B/keep/layers_$id.tar -C $B/data/jobs || { echo "FROM_TAR $id: no tar"; exit 1; }
n=0; for P in $(find $W -type d -name "post_f*"); do [ -s $P/seam.mp4 ] || continue
  echo "/data/jobs/${P#$B/data/jobs/}/seam.mp4" > $P/layer.txt; mkdir -p $(dirname $P)/kit $(dirname $P)/out; n=$((n + 1)); done
mkdir -p $W/ks/kit
for d in $(find $W -name "dubp_f*.wav"); do t=$(dirname $(dirname $d))/$(basename $d | sed s/dubp_/dub_/); [ -s $t ] || cp $d $t; done   # dub_<f>.wav (KS_TAIL_FADE reads it) was not saved: the dub KeySync heard (delayed <= 40 ms, padded with silence) stands in
[ -n "$HI" ] && cp $B/in/$HI $W/hires_src.${HI##*.}
if [ "$(python3 -c "import json; print(len(json.load(open('$W/ks/shots.json'))['segments']))")" -gt 1 ] && [ ! -s $W/ks/master_mp.npz ]; then   # ks_shots.sh slices it per shot (the saved per-shot landmarks are what the paste uses)
  docker volume create dabai-models > /dev/null
  bash $CODE/scripts/r18/r18.sh sh "set -e; M=models/mediapipe; mkdir -p \$M; [ -s \$M/face_landmarker.task ] || curl -fsSL --retry 3 -o \$M/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    echo '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  models/mediapipe/face_landmarker.task' | sha256sum -c - > /dev/null
    GLOG_minloglevel=2 OMP_NUM_THREADS=8 /venv/mediapipe/bin/python scripts/pilot/mp_landmarks.py /data/jobs/$id/e2e/work/ks/master_25fps.mp4 /data/jobs/$id/e2e/work/ks/master_mp.npz --num-faces 4" > $B/log/from_tar_$id.log 2>&1 || { echo "FROM_TAR $id: master landmarks failed (log/from_tar_$id.log)"; exit 1; }
fi
for sd in $W/shots/s*; do [ -d $sd/ks ] && [ ! -s $sd/ks/master_25fps.mp4 ] || continue   # a shot's own master (= its lossless cut of the parent master) was not saved either
  i=${sd##*/s}; read -r k0 k1 <<< "$(python3 -c "import json; s = json.load(open('$W/ks/shots.json'))['segments'][$i]; print(s[0], s[1])")"
  bash $CODE/scripts/r18/r18.sh sh "ffmpeg -v error -y -i /data/jobs/$id/e2e/work/ks/master_25fps.mp4 -vf 'select=between(n\,$k0\,$((k1 - 1))),setpts=N/25/TB' -r 25 -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p /data/jobs/$id/e2e/work/shots/s$i/ks/master_25fps.mp4" >> $B/log/from_tar_$id.log 2>&1 || echo "FROM_TAR $id: shot $i master cut failed"
done   # (a shot pasted through a clean dissolve, ks_alpha.py, has an UNMIXED master: not rebuilt here; q03)
echo "FROM_TAR $id: $n face layer(s) restored under $W"
