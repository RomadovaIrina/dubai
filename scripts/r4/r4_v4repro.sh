#!/bin/bash
# Round 4 V: v4-repro = the r3final v4 pipeline of 904c5d2 end to end from the raw inputs (server D and its artifacts are gone):
#   audio-r4 (r3_audio_r4.sh) -> audio-r4b bed remix + static loudness (r3_remix.py) -> A R3 generator on B's fps master + r3fix composite
#   (r4_video.sh, the r3_fix30.sh r3fix defaults) -> mux audio-r4b. Default flags only; LatentSync seed 1247 (face_aware_latentsync).
#   bash scripts/r4/r4_v4repro.sh [ids]   -> /workspace/dub/artifacts/r4v/v4repro/<id>_v4repro.mp4 (+ manifest), audio in .../audio-r4{,b}/<id>
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=${R4_BASE:-/workspace/dub/artifacts/r4v}; O=$B/v4repro; mkdir -p $O
for id in ${*:-04 03 05 01 02}; do
  s=$(date +%s)
  R4_OUT=$B/audio-r4 bash scripts/quality/r3_audio_r4.sh $id | grep -v AUDIO_R4_DONE
  [ -e $B/audio-r4/$id ] || { echo "FAIL audio $id"; continue; }
  python scripts/quality/r3_remix.py --src $B/audio-r4 --out $B/audio-r4b $id > $O/${id}_remix.log 2>&1 || { echo "FAIL remix $id"; continue; }
  a=$(date +%s)
  bash scripts/r4/r4_video.sh $id $B/audio-r4b/$id $B/audio-r4b/$id/${id}_r4b.manifest.json $O v4repro
  python - $O/${id}_v4repro.manifest.json $s $a <<'PY'
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1]); s, a = float(sys.argv[2]), float(sys.argv[3])
if p.exists():
    m = json.loads(p.read_text()); vmin = m["source"]["duration_s"] / 60; e = time.time()
    m["r4v"]["e2e_wall_s"] = {"audio_incl_remix": round(a - s, 1), "video": round(e - a, 1), "total": round(e - s, 1)}
    m["r4v"]["e2e_s_per_video_min"] = round((e - s) / vmin, 1); p.write_text(json.dumps(m, indent=1))
    print(f"== {p.stem}: e2e {e - s:.0f} s = {(e - s) / vmin:.0f} s/video-min (audio {(a - s) / vmin:.0f})")
PY
done
echo V4REPRO_DONE
