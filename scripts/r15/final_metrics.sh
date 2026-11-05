#!/bin/bash
# r15 W: once every arm chain is done, the metrics of ALL arms per held-out video + the markdown table -> /workspace/dub/r15w/eval/TABLE.md
cd /workspace/dub/dubai; R=/workspace/dub/r15w; X=$R/eval
until grep -q W12_CHAIN_DONE $R/chain_w12.log && grep -q BON2E_DONE $R/bon2e.log && grep -q Q2R_DONE $R/q2r.log && grep -q "tbnat_m18 done" $R/native_m.log \
      && grep -q W0_CHAIN_DONE $R/chain_w0.log && grep -q SEED_DONE $R/s4c8n2.log; do sleep 30; done 2>/dev/null
ARMS="q2 q2r s4c8 s4c8n2 tb tbw tbnat bon3 bon3e bon2e kf2 kf3 kfv3 tkf2"
for id in old03 m03 m18; do bash scripts/r15/metrics.sh $id $ARMS > /dev/null 2>&1; echo "metrics $id done $(date +%T)"; done
python3 scripts/r15/table.py --arms $ARMS tbnc tbmr0 > $X/TABLE.md; cat $X/TABLE.md
python3 scripts/r15/bon_stats.py > $X/BON_STATS.txt; cat $X/BON_STATS.txt
echo FINAL_METRICS_DONE
