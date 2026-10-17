#!/bin/bash
# Round 2 (track B): common slide metric on a composite_register / decompose window dir, with track A's npz + rigid (same as the other server).
#   bash scripts/quality/r2_slide_win.sh <id> <win dir> <name> [name ...]   -> <win>/<name>.slide.json
set -u
id=$1; win=$2; shift 2
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
FA=/workspace/dub/common/from_A/artifacts/lipsync-temporal/r2; MASTER=$(python -c "import json;print(json.load(open('$win/window_manifest.json'))['source_manifest'])" | xargs -I{} python -c "import json;print(json.load(open('{}'))['work_dir'])")/master_25fps.mp4
A=$(python -c "import json;print(json.load(open('$win/window_manifest.json'))['window'][0])"); B=$(python -c "import json;print(json.load(open('$win/window_manifest.json'))['window'][1])")
python - "$win" <<'PY'
import json, sys, pathlib
w = pathlib.Path(sys.argv[1]); wm = json.loads((w / "window_manifest.json").read_text())
segs = [{"action": "LATENT_SYNC", "start_frame": p["start"], "end_frame": p["end"]} for p in wm["pieces"]]
(w / "full_index.manifest.json").write_text(json.dumps({"lipsync": {"segments": segs}}))
PY
for n in "$@"; do
  bx=""; [ -f $win/$n.boxes.npz ] && bx="--boxes $win/$n.boxes.npz"; [ -z "$bx" ] && [ -f $win/V0.boxes.npz ] && bx="--boxes $win/V0.boxes.npz"
  python scripts/quality/slide_metric.py --video $win/$n.mp4 --master $MASTER --npz $FA/masters/$id/master_mediapipe.npz --offset $A --frames $A:$B \
    --manifest $win/full_index.manifest.json --rigid $FA/anchor/${id}_rigid.npz $bx --out $win/$n.slide.json --series 2>&1 | tail -1
done
