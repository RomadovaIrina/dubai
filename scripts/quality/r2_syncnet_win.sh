#!/bin/bash
# Round 2 track B: SyncNet dub-only on a time window of any-fps output (cut + dub voice, then syncnet_fps.py: fps=25 rendition for 30 fps).
#   bash scripts/quality/r2_syncnet_win.sh <id> <t0_s> <dur_s> <out.json> <video.mp4>
set -eu
id=$1; t0=$2; dur=$3; out=$4; vid=$5
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
dub=/workspace/dub/artifacts/restoration-temporal/r2/work/work_$id/dubbed_aac.m4a; tmp=$(mktemp -d)
ffmpeg -loglevel error -y -ss $t0 -t $dur -i $vid -ss $t0 -t $dur -i $dub -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 12 -pix_fmt yuv420p -c:a aac -shortest $tmp/w.mp4
flock /workspace/dub/gpu.lock python scripts/quality/syncnet_fps.py $tmp/w.mp4 --json $out 2>/dev/null | tail -1
