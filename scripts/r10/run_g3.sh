#!/bin/bash
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; R=/workspace/dub/r10; B=/workspace/dub/artifacts/r4m
OS="--region paste --shade on --hf off --radius-s 0.1 --shade-sigma 0.07 --keep-cavity on"
for id in old05 m03 old03 m18; do mkdir -p $R/metrics/$id $R/abl/$id
  A=$(python -c "import json;print(json.dumps(json.load(open(\"$B/r9f_$id/${id}_r9f_$id.fps_composite.json\"))[\"args\"]))")
  RC=$R/abl/$id/${id}_reg_smooth=4.mp4
  [ -s $RC ] || python scripts/r10/recomp.py --args-json "$A" --set "reg_smooth=4" --out $RC > $R/metrics/$id/abl_rs4.log 2>&1
  [ -s $R/gate/${id}R/face_gate.json ] && grep -q sync $R/gate/${id}R/face_gate.json || bash scripts/r10/gate.sh $id R $RC
  cd $R; OSTAB_IN=$RC bash ostab_try.sh $id rs4_p0 $OS --shade-model poly0; OSTAB_IN=$RC bash ostab_try.sh $id rs4_p1 $OS --shade-model poly1
  OSTAB_IN=$RC bash ostab_try.sh $id rs4_p1c $OS --shade-model poly1 --chin-reg on --chin-reg-vert 1.0; cd /workspace/dub/dubai
  for v in p0:P0 p1:P1 p1c:P1C; do bash scripts/r10/gate.sh $id ${v#*:} $R/abl/$id/${id}_rs4_${v%%:*}.mp4; done
  for v in R P0 P1 P1C; do python scripts/r10/gate_cmp.py --base $R/gate/${id}base --cand $R/gate/$id$v --md $R/gate/gate3.md | head -n 1; done
done 2>&1 | grep "^== gate\|face_metrics\|Traceback"
echo G3_DONE
