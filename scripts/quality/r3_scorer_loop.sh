#!/bin/bash
# score every finished output once; several copies may run (per-item claim dir); pause while a GPU audit / E2E timing run is active
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; V=/workspace/dub/artifacts/r3d/video; S=/workspace/dub/artifacts/r3d/score; mkdir -p $S/.claim
while true; do
  for m in $V/??_clean0.manifest.json $V/??_r2m.manifest.json $V/??_r2m_fix30.manifest.json $V/??_r2mC.manifest.json $V/??_r2mC_fix30.manifest.json $V/??_r3F.manifest.json $V/??_r3F_fix30.manifest.json $V/??_r3F_r3cand.manifest.json $V/??_r3cand.manifest.json $V/??_r3cand_fix30.manifest.json $V/??_r3Fa3.manifest.json $V/??_clean0a3.manifest.json $V/??_r3candb_final.manifest.json $V/??_r3final.manifest.json $V/??_r3final_fix30.manifest.json; do
    [ -f "$m" ] || continue; b=$(basename $m .manifest.json); id=${b%%_*}; arm=${b#*_}; lab=${arm//_/}; [ $arm = r2m ] && lab=r2m25; [ $arm = r2mC ] && lab=r2mC25; [ $arm = r3F ] && lab=r3F25; [ $arm = r3cand ] && lab=r3cand25; [ $arm = r3final ] && lab=r3final25
    [ -f $S/${id}_${lab}.score.json ] && continue
    mkdir $S/.claim/${id}_${lab} 2>/dev/null || continue
    Q=/workspace/dub/artifacts/r3d/q4.log   # yield to the GPU audit (q4 waits for an idle scorer) and never overlap an audit / E2E timing run
    while pgrep -f "r3_e2e[.]sh|r3_gpu_audit[.]sh|r3_queue_v9[.]sh" >/dev/null; do sleep 30; done
    python scripts/quality/r3_scorecard.py score --id $id --label $lab --video $V/$b.mp4 --manifest $m --out $S > $S/${id}_${lab}.log 2>&1; echo "scored $id $lab rc=$?"
  done
  [ -f /workspace/dub/artifacts/r3d/SCORER_STOP ] && break; sleep 60
done
echo SCORER_DONE
