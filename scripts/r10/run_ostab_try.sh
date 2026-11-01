#!/bin/bash
# bash ostab_try.sh <id> <label> <out_stab args...>   (on the fx final) -> abl/<id>/<id>_<label>.mp4 + metrics
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
id=$1; lab=$2; shift 2; B=/workspace/dub/artifacts/r4m; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SLM=$B/r9g_$id/timing/${id}_source_lm_F.npz
IN=${OSTAB_IN:-$B/r9f_$id/${id}_r9f_$id.mp4}; O=/workspace/dub/r10/abl/$id; mkdir -p $O /workspace/dub/r10/metrics/$id
[ -s $O/${id}_$lab.mp4 ] || python scripts/r10/r10_out_stab.py --video $IN --src $SRC --lm $SLM --out $O/${id}_$lab.mp4 "$@" | grep "^=="
python scripts/r10/face_metrics.py --video $O/${id}_$lab.mp4 --src $SRC --lm-src $SLM --out /workspace/dub/r10/metrics/$id/O_$lab.json --per-frame 2>&1 | grep "^==" | cut -c1-600
