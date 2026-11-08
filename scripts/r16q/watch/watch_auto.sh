#!/bin/bash
# r16 Q 08.11: wait for T's final_0811 jobs (/workspace/dub/r16t/final/f0811_*.json), score each when its status is final (CPU SyncNet: T has
# the GPU), regenerate /workspace/dub/outbox/r16_q/WATCH_0811.md after each. Stops when every f0811 job is scored, or after 6 h.
W=/workspace/dub/r16q/watch; FJ=/workspace/dub/r16t/final; GJ=/workspace/dub/r16t/golden/jobs; B=/workspace/dub/outbox/r16_t
BASE="full_m18:$B/full/full_m18.mp4:$GJ/full_m18 full_old03:$B/full/full_old03.mp4:$GJ/full_old03 long5:$B/full/long5.mp4:$GJ/long5 q03_alpha:$B/alpha/q03_e2e_alpha_full.mp4:$GJ/q03_alpha q05_e2e:$B/alpha/q05_e2e_full.mp4:$GJ/q05_e2e"
gen() { cd $W; python3 watch_sum.py $(cat $W/scored.txt 2>/dev/null) $BASE 2>/dev/null > $W/WATCH_0811.md.tmp && mv $W/WATCH_0811.md.tmp /workspace/dub/outbox/r16_q/WATCH_0811.md; }
touch $W/scored.txt; gen; T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt 21600 ]; do
  left=0
  for js in $FJ/f0811_*.json; do id=$(basename $js .json); grep -q "^$id:" $W/scored.txt && continue; left=1
    S=$FJ/jobs/$id/status.json; [ -s $S ] || continue
    st=$(python3 -c "import json;print(json.load(open('$S')).get('status',''))" 2>/dev/null)
    case "$st" in SUCCESS|PARTIAL_SUCCESS|FAILED|ERROR) ;; *) continue;; esac
    F=$(python3 -c "import json;s=json.load(open('$S'));print(s.get('output') or '')" 2>/dev/null)
    [ -s "$F" ] || F=$(ls $B/final_0811/*${id#f0811_}*.mp4 2>/dev/null | head -1)
    if [ ! -s "$F" ]; then echo "$id: status $st, no output" >> $W/auto.log; echo "$id:NOOUT" >> $W/noout.txt; echo "$id:" >> $W/scored.txt; continue; fi
    DEV=cpu bash $W/watch_one.sh $id "$F" $FJ/jobs/$id >> $W/auto.log 2>&1
    echo "$id:$F:$FJ/jobs/$id" >> $W/scored.txt; sed -i "/^$id:$/d" $W/scored.txt; gen
  done
  [ $left = 0 ] && break; sleep 120
done
gen; echo AUTO_DONE >> $W/auto.log
