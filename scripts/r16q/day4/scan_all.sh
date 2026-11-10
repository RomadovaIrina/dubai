#!/bin/bash
# scan_all.sh SET  -> out/SET/scan/<id>.breaks.json for every finished id
SET=$1; Q=/workspace/dub/q; R=/workspace/dub/r16in/src; QS=$Q/src
declare -A SRC=([m03]=$QS/story-1774249919482-367803614.MOV [m18]=$QS/story-1774250817106-616905094.mp4 [old03]=$R/old03.mp4 [q01]=$R/q01_amir.mp4 [q01_subs]=$R/q01_amir.mp4 [q02]=$R/q02_vera.mp4 [q04]=$R/q04_farikh.mp4 [q06]=$R/q06_suvorov.mp4)
declare -A F0811=([m03]=f0811_m03 [m18]=full_m18 [old03]=full_old03 [q01]=f0811_q01 [q01_subs]=f0811_q01_subs [q02]=f0811_q02 [q04]=f0811_q04 [q06]=f0811_q06)
cd $Q
for id in $(grep 'rc=0' out/$SET/done.txt | cut -d' ' -f1); do
  if [ "$SET" = now ]; then FIN=/workspace/dub/r16in/day3/now_1011/now_$id.mp4; else FIN=/workspace/dub/r16in/day3/e2e_0811/${F0811[$id]}.mp4; fi
  /workspace/dub/qenv/bin/python scripts/r16q/scan_breaks.py --series out/$SET/$id.series.npz --final $FIN --src ${SRC[$id]} --id $id --out out/$SET/scan 2>&1 | tail -1
done
