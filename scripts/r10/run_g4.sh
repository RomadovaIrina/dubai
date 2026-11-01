#!/bin/bash
# per id (run ids in parallel): F1 = composite reg_smooth=4 + out_stab field shading (keep-cavity); F1C = F1 + chin re-registration (vert 1.0)
id=$1; cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; R=/workspace/dub/r10; B=/workspace/dub/artifacts/r4m
OS="--region paste --shade on --hf off --radius-s 0.1 --shade-sigma 0.07 --keep-cavity on"; RC=$R/abl/$id/${id}_reg_smooth=4.mp4
cd $R; rm -f abl/$id/${id}_f1.mp4 abl/$id/${id}_f1c.mp4
OSTAB_IN=$RC bash ostab_try.sh $id f1 $OS; OSTAB_IN=$RC bash ostab_try.sh $id f1c $OS --chin-reg on --chin-reg-vert 1.0; cd /workspace/dub/dubai
if [ $id = old05 ]; then A=$(python -c "import json;print(json.dumps(json.load(open(\"$B/r9f_$id/${id}_r9f_$id.fps_composite.json\"))[\"args\"]))")
  [ -s $R/abl/$id/${id}_reg_smooth=2.mp4 ] || python scripts/r10/recomp.py --args-json "$A" --set "reg_smooth=2" --out $R/abl/$id/${id}_reg_smooth=2.mp4 > $R/metrics/$id/abl_rs2.log 2>&1
  bash scripts/r10/gate.sh $id R2 $R/abl/$id/${id}_reg_smooth=2.mp4; fi
bash scripts/r10/gate.sh $id F1 $R/abl/$id/${id}_f1.mp4
bash scripts/r10/gate.sh $id F1C $R/abl/$id/${id}_f1c.mp4
for v in R R2 F1 F1C; do [ -d $R/gate/$id$v ] && python scripts/r10/gate_cmp.py --base $R/gate/${id}base --cand $R/gate/$id$v --md $R/gate/gate4.md | head -n 3; done
echo G4_DONE $id
