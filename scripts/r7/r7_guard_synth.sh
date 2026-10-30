#!/bin/bash
# r7 F: synthetic check of --paste-guard on a finished run: the SOURCE frames F0..F1 get the lower face covered by a background patch from the
# same frame (the camera "looked away" but the landmarks / layer still say face) -> the composite must paste NOTHING there (whole-frame gate).
#   bash scripts/r7/r7_guard_synth.sh [id=03] [tag=m6] [F0=600] [F1=720]   -> /workspace/dub/artifacts/r7f/guard_synth/<id>_cover_<arm>.mp4 + gate lines
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
id=${1:-03}; TAG=${2:-m6}; F0=${3:-600}; F1=${4:-720}; B=/workspace/dub/artifacts/r4m/$TAG; W=$B/work_${id}_$TAG; O=/workspace/dub/artifacts/r7f/guard_synth; mkdir -p $O
src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1); COV=$O/${id}_cover.mp4
python - "$src" "$COV" $F0 $F1 /workspace/dub/artifacts/r7f/lm/${id}_src_mp.npz <<'PY'
import sys, numpy as np, cv2, av
src, out, f0, f1, lmp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; lm = np.load(lmp)["lm"]
c = av.open(src); s = c.streams.video[0]; o = av.open(out, "w"); st = o.add_stream("libx264", rate=s.average_rate); st.pix_fmt = "yuv420p"; st.options = {"crf": "13"}
for k, fr in enumerate(c.decode(s)):
    f = fr.to_ndarray(format="bgr24"); H, W = f.shape[:2]
    if k == 0: st.width, st.height = W, H
    if f0 <= k < f1 and np.isfinite(lm[k, 0, 0]):
        p = lm[k]; iod = np.linalg.norm(p[33, :2] - p[263, :2]); x0, x1 = int(max(0, p[:, 0].min() - 0.3 * iod)), int(min(W, p[:, 0].max() + 0.3 * iod)); y0, y1 = int(max(0, p[2, 1] - 0.5 * iod)), int(min(H, p[152, 1] + 0.6 * iod))
        patch = cv2.resize(f[0:max(40, (y1 - y0) // 2), 0:max(40, (x1 - x0) // 2)], (x1 - x0, y1 - y0)); f[y0:y1, x0:x1] = patch   # the frame's own top-left background, scaled
    nf = av.VideoFrame.from_ndarray(f, format="bgr24")
    for pk in st.encode(nf): o.mux(pk)
for pk in st.encode(): o.mux(pk)
o.close(); c.close(); print("cover", out, f0, f1)
PY
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38"
M=$O/${id}_cover.manifest.json; python -c "import json; m=json.load(open('$W/${id}_${TAG}_25fps.manifest.json')); m['input']='$COV'; json.dump(m, open('$M','w'))"
for arm in guard0 guard05; do
  [ $arm = guard0 ] && PG="--paste-guard 0" || PG="--paste-guard 0.5"
  python scripts/quality/fps_composite.py --manifest $M --ls-raw $W/${id}_${TAG}_25fps_ls_raw.mp4 --out $O/${id}_cover_$arm.mp4 --master-timing $B/timing/${id}_master_timing_F.json --source-lm $B/timing/${id}_source_lm_F.npz --threads 12 $V1 $PG > $O/${id}_cover_$arm.log 2>&1
  grep "paste-guard" $O/${id}_cover_$arm.log
  python scripts/r7/r7_frame_gate.py --src $COV --out-video $O/${id}_cover_$arm.mp4 --lm /workspace/dub/artifacts/r7f/lm/${id}_src_mp.npz --out $O/${id}_cover_$arm.gate.json --label cover_$arm --id $id --sheet $O/${id}_cover_${arm}_sheet.jpg --top 6 | grep "^=="
done
