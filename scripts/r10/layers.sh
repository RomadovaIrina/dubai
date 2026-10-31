#!/bin/bash
# r10 FACE: lip flicker + chin slide, layer by layer, on one finished r10 baseline (base.sh), CPU only.
#   bash scripts/r10/layers.sh <id> [ablations]   -> /workspace/dub/r10/metrics/<id>/<name>.json
# Layers (25 fps master grid, vs master_25fps.mp4): ls_pre (raw LatentSync), ls_raw (+CodeFormer), cfd (detail 0.4 inside), srt1 (layer SR),
#   rep2 (layer repair) / stab (stabiliser + repair). Finals (source fps, vs the source): fx, stab.
# Ablations (fps_composite on the fx layer, same args as the fx final minus ONE stage): nosharp, nearest, rigidreg, notex, noregion.
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
id=$1; B=/workspace/dub/artifacts/r4m; G=r9g_$id; W=$B/$G/work_${id}_$G; P=$W/${id}_${G}_25fps; O=/workspace/dub/r10/metrics/$id; mkdir -p $O
SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SLM=$B/$G/timing/${id}_source_lm_F.npz; MM=$W/video/master_25fps.mp4; MLM=$W/video/master_mediapipe.npz
[ -f $MLM ] || MLM=$W/master_mediapipe.npz
fm() { [ -s $O/$1.json ] || python scripts/r10/face_metrics.py --video $2 --src $3 --lm-src $4 ${5:+--lm-out $5} --out $O/$1.json --per-frame 2>&1 | grep "^=="; }
fm L0_pre ${P}_ls_pre.mp4 $MM $MLM
fm L1_cf ${P}_ls_raw.mp4 $MM $MLM ${P}_ls_raw.mediapipe.npz
fm L2_cfd ${P}_ls_cfd_ia04.mp4 $MM $MLM ${P}_ls_cfd_ia04.mediapipe.npz
fm L3_sr ${P}_ls_cfd_ia04_srt1.mp4 $MM $MLM ${P}_ls_cfd_ia04_srt1.mediapipe.npz
fm L4_rep ${P}_ls_cfd_ia04_srt1_rep2.mp4 $MM $MLM
fm L4_stab ${P}_ls_cfd_ia04_srt1_stab2_stab.mp4 $MM $MLM
fm F_fx $B/r9f_$id/${id}_r9f_$id.mp4 $SRC $SLM
fm F_stab $B/r9f_${id}stab/${id}_r9f_${id}stab.mp4 $SRC $SLM
shift
if [ $# -gt 0 ]; then   # composite ablations on the fx layer (the fx final's exact args, minus one stage); outputs are scratch
  A=$(python -c "import json;a=json.load(open('$B/r9f_$id/${id}_r9f_$id.fps_composite.json'))['args'];print(json.dumps(a))")
  for ab in "$@"; do
    out=/workspace/dub/r10/abl/$id/${id}_$ab.mp4; mkdir -p $(dirname $out)
    [ -s $out ] || python scripts/r10/recomp.py --args-json "$A" --set "$ab" --out $out > $O/abl_$ab.log 2>&1 || { echo "FAIL abl $ab"; continue; }
    fm A_$ab $out $SRC $SLM
  done
fi
echo "R10_LAYERS_DONE $id"
