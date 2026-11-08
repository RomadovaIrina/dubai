#!/bin/bash
# after run_all: re-run m03 (its paste hit the flaky parallel failure) and q02 unless it was SUCCESS, then compare all four
G=/workspace/dub/r16t/golden; while [ ! -f $G/DONE ]; do sleep 20; done
for v in m03 q02; do
  if [ $v = q02 ] && grep -q '"status": "SUCCESS"' $G/jobs/gold_q02/status.json 2>/dev/null; then continue; fi
  mv $G/jobs/gold_$v $G/jobs/gold_${v}_prev_$(date +%H%M) 2>/dev/null; bash /workspace/dub/r16t/runjob.sh $G/gold_$v.json
done
sed -i 's|while \[ ! -f $G/DONE \]; do sleep 20; done||' $G/cmp_all.sh
bash $G/cmp_all.sh
