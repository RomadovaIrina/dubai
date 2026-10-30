#!/bin/bash
# face twitch (r5_twitch.py, the round-5/6 whole-video detector vs the raw source) on r9 vs r9fx, per video, + top-20 sheets for the r9 / r9fx output
R=${R9ROOT:-/workspace/dub/dubai}; cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R OMP_NUM_THREADS=6
B=/workspace/dub/artifacts/r4m; L=/workspace/dub/artifacts/r6f/lm; O=/workspace/dub/artifacts/r6f/twitch; mkdir -p $L
for id in "$@"; do src=$(ls /workspace/dub/r9/runs/$id/work/in/$id.* | head -1)
  [ -f $L/${id}_src_lm.npz ] || python scripts/r5/r5_srclm.py --src $src --out $L/${id}_src_lm.npz > /dev/null 2>&1
  FX=r9f_${id}fx; [ $id = m03 ] && FX=r9f_m03fx3
  while [ ! -f $B/$FX/${id}_$FX.mp4 ]; do sleep 30; done
  for T in r9f_$id $FX; do mkdir -p $O/$T
    python scripts/r5/r5_twitch.py score --video $B/$T/${id}_$T.mp4 --src $src --lm $L/${id}_src_lm.npz --manifest $B/$T/${id}_$T.manifest.json --out $O/$T/$id.twitch.json > $O/$T/$id.log 2>&1 || echo "FAIL $T"
    python scripts/r5/r5_twitch.py sheet --twitch $O/$T/$id.twitch.json --lm $L/${id}_src_lm.npz --top 12 --outdir /workspace/dub/outbox/r9fx/twitch_sheets --stem ${id}_$T --whole-frame > /dev/null 2>&1
    echo "== $T $(tail -1 $O/$T/$id.log | python3 -c "import json,sys; d=json.load(sys.stdin); print({k: d.get(k) for k in (\"twitch_sum\",\"n_events\",\"n_events_ge2\",\"top20_mean\",\"p99\")})" 2>/dev/null)"
  done
done; echo TWITCH_DONE
