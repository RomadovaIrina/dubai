#!/bin/bash
# r3 B: audio-r2 work dir + C's manifest (its own speech intervals / gate) per video, as A's r3_audio_dirs.sh but without A's -r 25 masters
# (each render builds its own master in its work dir).  -> artifacts/r3-face/audio-r2/{work_<id>, <id>.manifest.json}
set -e; cd /workspace/dub/dubai
for v in "$@"; do
  A=/workspace/dub/artifacts/r3-face/audio-r2; W=$A/work_$v; C=/workspace/dub/handoff/in/C/audio-r2/$v; mkdir -p $W
  for f in /workspace/dub/canonical/artifacts/baseline/$v/work/work_$v/*; do ln -sfn $f $W/; done; rm -f $W/tts_* $W/aligned_*
  for f in $C/*; do ln -sfn $f $W/; done
  /venv/dabai/bin/python - $v $C $W $A <<'PY'
import json, sys
v, C, W, A = sys.argv[1:]
m = json.load(open(f"{C}/{v}_r2.manifest.json")); m["work_dir_audio_c"] = m["work_dir"]; m["work_dir"] = W; m["audio"] = "audio-r2 (track C, Spleeter bed)"
json.dump(m, open(f"{A}/{v}.manifest.json", "w")); print(v, "manifest ->", f"{A}/{v}.manifest.json", "input", m["input"])
PY
done
