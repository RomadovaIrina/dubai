#!/bin/bash
# r3 B, task 3: generator-side window-joint fix (A's bridges) vs / with the post-side fix. Joint visibility (r3_joints.py) on the 25 fps generator
# output and on the R3 composite, + SyncNet. 05, audio-r2.
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; id=05; PY=/venv/dabai/bin/python
R3FLAGS="--colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"
for g in a3F a3Fbr8 a3Fbr16; do
  until grep -q "^== $id $g\|^FAIL $id $g" $R/gen/${id}_$g.log 2>/dev/null; do sleep 30; done
  $PY scripts/quality/r3_joints.py --video $R/gen/${id}_$g.mp4 --manifest $R/gen/${id}_$g.manifest.json --lm $R/timing/${id}_source_lm.npz --ref-master $R/timing/${id}_master_timing_F.json --out $R/metrics/${id}_${g}.joints.json | cut -c1-10 > /dev/null
  t=${g/a3F/a3_R3}; [ -f $R/comp/${id}_$t.mp4 ] || GEN=$g scripts/quality/r3_compF.sh $id $t $R3FLAGS > /dev/null 2>&1
  $PY scripts/quality/r3_joints.py --video $R/comp/${id}_$t.mp4 --manifest $R/gen/${id}_$g.manifest.json --lm $R/timing/${id}_source_lm_F.npz --out $R/metrics/${id}_$t.joints.json > /dev/null
  scripts/quality/r3_sync.sh $id:$g $id:$t > /dev/null
  echo "$g: gen25 $(grep -o '"lowfreq": {[^}]*}' $R/metrics/${id}_${g}.joints.json) | R3 $(grep -o '"lowfreq": {[^}]*}' $R/metrics/${id}_$t.joints.json) | ls_s $(grep -o '"total": [0-9.]*' $R/gen/${id}_$g.manifest.json | tail -1)"
done
