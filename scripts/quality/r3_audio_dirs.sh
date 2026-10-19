#!/bin/bash
# Round 3: per-video audio-r2 work dir + merged manifest for candidate_video_stage / ls_window_ab (track C audio, A's layout).
#   r3_audio_dirs.sh <id>...   needs r3_prep.sh <id> first; C's audio in /workspace/dub/handoff/in/C/audio-r2/<id>
#   -> artifacts/r3/audio-r2/work_<id> (canonical source audio + C's dub / mix + master links), artifacts/r3/audio-r2/<id>.manifest.json
set -e
cd /workspace/dub/dubai
for v in "$@"; do
  A=/workspace/dub/artifacts/r3/audio-r2; W=$A/work_$v; C=/workspace/dub/handoff/in/C/audio-r2/$v; M=/workspace/dub/artifacts/r3/masters/$v; mkdir -p $W
  for f in /workspace/dub/canonical/artifacts/baseline/$v/work/work_$v/*; do ln -sfn $f $W/; done; rm -f $W/tts_* $W/aligned_*
  for f in $C/*; do ln -sfn $f $W/; done
  for f in master_25fps.mp4 master_mediapipe.npz master_rigid.npz master_mediapipe_probe2.npz; do [ -e $M/$f ] && ln -sfn $(readlink -f $M/$f) $W/$f; done
  /venv/dabai/bin/python - $v $C $W $A <<'PY'
import json, subprocess, sys
v, C, W, A = sys.argv[1:]
m = json.load(open(f"{C}/{v}_r2.manifest.json")); m["work_dir_audio_c"] = m["work_dir"]; m["work_dir"] = W; m["audio"] = "audio-r2 (track C, Spleeter bed)"
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", f"{W}/master_25fps.mp4"],
                       capture_output=True, text=True).stdout.strip())
m["lipsync"] = {"master_frames": n, "note": "master frame count only"}
json.dump(m, open(f"{A}/{v}.manifest.json", "w")); print(v, n)
PY
done
