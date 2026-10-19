#!/bin/bash
# score every finished video-stage output once (skips existing); pauses while an E2E timing run is active
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; V=/workspace/dub/artifacts/r3d/video; S=/workspace/dub/artifacts/r3d/score
while true; do
  for m in $V/??_clean0.manifest.json $V/??_r2m.manifest.json $V/??_r2m_fix30.manifest.json $V/??_r2mC.manifest.json $V/??_r2mC_fix30.manifest.json; do
    [ -f "$m" ] || continue; b=$(basename $m .manifest.json); id=${b%%_*}; arm=${b#*_}; lab=${arm//_/}; [ $arm = r2m ] && lab=r2m25; [ $arm = r2mC ] && lab=r2mC25
    [ -f $S/${id}_${lab}.score.json ] && continue
    while pgrep -f r3_e2e.sh >/dev/null; do sleep 30; done
    python scripts/quality/r3_scorecard.py score --id $id --label $lab --video $V/$b.mp4 --manifest $m --out $S > $S/${id}_${lab}.log 2>&1; echo "scored $id $lab"
  done
  grep -q E2E04_DONE /workspace/dub/artifacts/r3d/q3.log 2>/dev/null && break; sleep 60
done
echo SCORER_DONE
