#!/bin/bash
# r6 G: metrics of every finished window clip (CPU + SyncNet): r4_sweep_metrics.py then r6_win_extra.py; loops until R6_SWEEP_DONE (LOOP=1)
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=6
S=/workspace/dub/artifacts/r6g; O=$S/win; M3=/workspace/dub/artifacts/r4m/m3
while :; do
  for m in $O/*.manifest.json; do [ -e "$m" ] || continue; b=$(basename $m .manifest.json); id=${b%%_*}; tag=${b#*_}; [ -f $O/$b.metrics.json ] && continue
    t0=${tag##*_}; t1=$(python3 -c "import json;print([w['t1'] for w in json.load(open('$S/wins_$id.json'))['windows'] if int(w['t0'])==$t0][0])")
    MST=$(ls $M3/work_${id}_m3/video/master_25fps.mp4 $M3/work_${id}_m3/*/master_25fps.mp4 2>/dev/null | head -1)
    python scripts/quality/r4_sweep_metrics.py --win-dir $O --id $id --tag $tag --master $MST --t0 $t0 --t1 $t1 --json $O/$b.metrics.tmp.json > /dev/null 2>$O/$b.metrics.err \
      && mv $O/$b.metrics.tmp.json $O/$b.metrics.json && python scripts/r6/r6_win_extra.py --win-dir $O --id $id --tag $tag --t0 $t0 >/dev/null 2>>$O/$b.metrics.err || echo "FAIL metrics $b"
  done
  [ "${LOOP:-0}" = 1 ] || break; grep -qs R6_SWEEP_DONE $S/sweep.out && break; sleep 60
done
echo METRICS_DONE
