#!/bin/bash
# r14 F: composite-only ablations of the q2 final (cached layers of r9g_<id>) + tone_flicker on each.
#   bash scripts/r12/abl.sh <id> "tag=extra fps_composite flags" ...   -> /workspace/dub/r14/abl/<id>.txt
cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; id=$1; shift; mkdir -p /workspace/dub/r14/abl
Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
for arm in "$@"; do t=${arm%%=*}; x=${arm#*=}; (
  FCX="$Q2 $x" bash scripts/r12/comp_k5.sh $id $t FC_CMS=1 LS_CMS_READ=1 > /dev/null
  . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1)
  python scripts/r12/tone_flicker.py --video $B/r9f_$id$t/${id}_r9f_$id$t.mp4 --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt ) &
done; wait; echo ABL_DONE
