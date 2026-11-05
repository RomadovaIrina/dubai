#!/bin/bash
# r15 W round 2: once the round-2 chains are done (m03 face units, selective best-of-N v1 / v2, teacher + keyframes + c8), the metrics of every
# arm per video (SyncNet cached, flicker, mouth opening, tremble; the first arm = the reference: m03 = q2u, the face-units production)
# -> /workspace/dub/r15w/eval/TABLE.md + BON_STATS.txt
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; R=/workspace/dub/r15w; X=$R/eval
until grep -q FU_CHAIN_DONE $R/chain_fu.log && grep -q BONSEL_DONE $R/chain_bonsel.log && grep -q BONSEL2_DONE $R/chain_bonsel2.log && grep -q TKF_DONE $R/chain_tkf.log; do sleep 30; done 2>/dev/null
declare -A ARMS=(
  [old03]="q2 q2r s4c8 s4c8n2 tkf2 tkf2c8 kf2 kf3 kfv3 bon3 bon3e bon2e bonsel bonsel2 tb tbw tbnat"
  [m03]="q2u q2 s4c8u s4c8 tkf2u tkf2 tkf2c8 kfv3u kfv3 kf2 kf3 bon3 bon3e bon2e bonsel2 s4c8n2 tb tbw tbnat"
  [m18]="q2 q2r s4c8 s4c8n2 tkf2 tkf2c8 kf2 kf3 kfv3 bon3 bon3e bon2e bonsel2 tb tbw tbnat")
for id in old03 m03 m18; do
  bash scripts/r15/metrics.sh $id ${ARMS[$id]} > /dev/null 2>&1
  SRC=$(ls $X/q2_$id/e2e/work/in/q2_$id.* | head -1); V=""
  for a in ${ARMS[$id]}; do [ $a = tbnat ] && continue; v=$X/${a}_$id/${a}_$id.mp4; [ $a = q2 ] && v=$X/q2_$id/q2_$id.mp4; [ -s $v ] && V="$V $a=$v"; done
  python scripts/r15/tremble.py --src $SRC --video $V --json $X/tremble_$id.json > $X/tremble_$id.txt 2>&1
  echo "metrics $id done $(date +%T)"
done
{ for id in old03 m03 m18; do python3 scripts/r15/table.py --ids $id --arms ${ARMS[$id]} tbnc tbmr0 | { [ $id = old03 ] && cat || tail -n +3; }; done; } > $X/TABLE.md; cat $X/TABLE.md
python3 scripts/r15/bon_stats.py > $X/BON_STATS.txt; cat $X/BON_STATS.txt
echo FINAL2_DONE
