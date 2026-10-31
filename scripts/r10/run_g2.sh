#!/bin/bash
# candidates per id: S = fx + out_stab(sh3fk); R = composite reg_smooth=4; RS = R + out_stab(sh3fk); RSn = R + out_stab(sh3f, no cavity guard)
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; R=/workspace/dub/r10; B=/workspace/dub/artifacts/r4m
OS="--region paste --shade on --hf off --radius-s 0.1 --shade-sigma 0.07"
for id in old03 m03 m18 old05; do
  A=$(python -c "import json;print(json.dumps(json.load(open(\"$B/r9f_$id/${id}_r9f_$id.fps_composite.json\"))[\"args\"]))")
  RC=$R/abl/$id/${id}_reg_smooth=4.mp4
  [ -s $RC ] || python scripts/r10/recomp.py --args-json "$A" --set "reg_smooth=4" --out $RC > $R/metrics/$id/abl_rs4.log 2>&1
  cd $R; OSTAB_IN=$RC bash ostab_try.sh $id rs4_sh3fk $OS --keep-cavity on; OSTAB_IN=$RC bash ostab_try.sh $id rs4_sh3f $OS; bash ostab_try.sh $id os_sh3fk $OS --keep-cavity on; cd /workspace/dub/dubai
  [ -s $R/gate/${id}base/face_gate.json ] || bash scripts/r10/gate.sh $id ""
  bash scripts/r10/gate.sh $id S $R/abl/$id/${id}_os_sh3fk.mp4
  bash scripts/r10/gate.sh $id R $RC
  bash scripts/r10/gate.sh $id RS $R/abl/$id/${id}_rs4_sh3fk.mp4
  bash scripts/r10/gate.sh $id RSn $R/abl/$id/${id}_rs4_sh3f.mp4
  for v in S R RS RSn; do python scripts/r10/gate_cmp.py --base $R/gate/${id}base --cand $R/gate/$id$v --md $R/gate/gate.md | head -n 3; done
done 2>&1 | grep "^== gate\|   base\|   cand\|face_metrics\|Traceback"
echo G2_DONE
