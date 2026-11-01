#!/bin/bash
# r10: the face agent's r10_out_stab.py (os_sh3fk args, unmodified copy in /workspace/dub/r10/ext) on face_only outputs -> artifacts/r4m/r9f_<id><v>os
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai
B=/workspace/dub/artifacts/r4m
for iv in "$@"; do id=${iv%%:*}; v=${iv#*:}; SRC=$(ls /workspace/dub/r9/runs/$id/work/in/$id.* | head -1); SLM=$B/r9g_$id$v/timing/${id}_source_lm_F.npz
  T=r9f_$id${v}os; mkdir -p $B/$T; ln -sfn $B/r9g_$id$v/timing $B/$T/timing 2>/dev/null
  python /workspace/dub/r10/ext/r10_out_stab.py --video $B/r9f_$id$v/${id}_r9f_$id$v.mp4 --src $SRC --lm $SLM --out $B/$T/${id}_$T.mp4 \
    --region paste --shade on --hf off --radius 3 --shade-sigma 0.07 --keep-cavity on | grep "^=="
  python - $B/r9f_$id$v/${id}_r9f_$id$v.manifest.json $B/$T/${id}_$T.manifest.json $B/$T/${id}_$T.mp4 <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m["output"] = sys.argv[3]; json.dump(m, open(sys.argv[2], "w"), indent=1)
PY
done
echo OSTAB_DONE
