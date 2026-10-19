#!/bin/bash
# Round 3 track B: on-time-master (genF) source-fps composite of <id> + evaluation.  r3_compF.sh <id> <tag> [extra fps_composite flags]
set -u
id=$1; tag=$2; shift 2; cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; T=$R/timing; PY=/venv/dabai/bin/python; SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.*)
[ -f $T/${id}_master_timing_F.json ] || $PY scripts/quality/r3_master_timing.py --src $SRC --master $R/gen/work_${id}_genF/master_25fps.mp4 --vf fps=25 --out $T/${id}_master_timing_F.json
[ -f $T/${id}_source_lm_F.npz ] || $PY -c "
import numpy as np, json
z=dict(np.load('$T/${id}_source_lm.npz')); tm=json.load(open('$T/${id}_master_timing_F.json')); tau=np.array(tm['tau_s'])
z['mu']=np.interp(z['t'],tau,np.arange(len(tau),dtype=float)); z['j']=np.array(tm['j']); np.savez_compressed('$T/${id}_source_lm_F.npz',**z)"
[ -e $T/${id}_source_lm_F_rigid.npz ] || { [ -f $T/${id}_source_lm_rigid.npz ] && ln -sf ${id}_source_lm_rigid.npz $T/${id}_source_lm_F_rigid.npz; }
$PY scripts/quality/fps_composite.py --manifest $R/gen/${id}_genF.manifest.json --ls-raw $R/gen/${id}_genF_ls_raw.mp4 --out $R/comp/${id}_$tag.mp4 \
  --master-timing $T/${id}_master_timing_F.json --register sim_ring --source-lm $T/${id}_source_lm_F.npz --threads 12 "$@" | tail -1
bash scripts/quality/r3_eval.sh $id $tag $R/comp/${id}_$tag.mp4 $R/gen/${id}_genF.manifest.json
