cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=6
S=/workspace/dub/artifacts/r4sweep; O=$S/win
for m in $O/*.manifest.json; do b=$(basename $m .manifest.json); id=${b%%_*}; tag=${b#*_}; [ -f $O/$b.metrics.json ] && continue
  t0=${tag##*_}; t1=$(python3 -c "import json;print([w['t1'] for w in json.load(open('$S/wins_$id.json'))['windows'] if int(w['t0'])==$t0][0])")
  python scripts/quality/r4_sweep_metrics.py --win-dir $O --id $id --tag $tag --master $S/work_$id/master_25fps.mp4 --t0 $t0 --t1 $t1 --json $O/$b.metrics.json > /dev/null 2>&1 || echo "FAIL metrics $b"
done
