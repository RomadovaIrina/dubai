#!/bin/bash
# gate rows for composite arms: r4_gate score (SyncNet, flashes, mismatch, ...) + pressed lips (r7_closed_mouth) + the artefact classes
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag
for spec in "$@"; do IFS=: read id T G <<< "$spec"; export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in
  python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $D/score_$T.log 2>&1
  python scripts/r7/r7_closed_mouth.py --ids $id --variants $T:$A/r9_$id --out $D/closed_$T.json > /dev/null 2>&1
  [ -f $D/art_${T#r9f_}.json ] || { L3=$(ls $B/$G/work_${id}_$G/${id}_${G}_25fps_ls_cfd_ia04_srt1_rep*.mp4 2>/dev/null | tail -1); CUDA_VISIBLE_DEVICES= python $D/artefacts.py $id --tag-suffix ${T#r9f_$id} --gen $G ${L3:+--l3 $L3} --out $D/art_${T#r9f_}.json > /dev/null 2>&1; }
  echo "SCORED $T $(date +%T)"
done; echo SCORE_ARMS_DONE
