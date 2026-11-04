#!/bin/bash
# r13 LS: one-command E2E (audio + face + k5 composite), warm box, per video: k5 control (ID <id>e0) and k5 + E2E_LS_RATE=2/3 (ID <id>e23);
# the e2e NVML GPU-busy s/vmin line is the number vs the 300 norm.  setsid nohup bash scripts/r12/r13_e2e.sh > /workspace/dub/r12/logs/e2e.out 2>&1 &
cd /workspace/dub/dubai; export E2E_METRICS=off DUB_ALLOW_NC=1
until grep -q R13_CHAIN_DONE /workspace/dub/r12/logs/chain.out; do sleep 30; done
M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw; O=/workspace/dub/r13/e2e; mkdir -p $O
declare -A F=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4)
for id in m03 old03 m18; do
  for v in e0 e23; do X=""; [ $v = e23 ] && X="E2E_LS_RATE=2/3"
    env $X E2E_FX=1 E2E_SYNC=1 ID=$id$v bash scripts/r9/e2e.sh ${F[$id]} $O/$id$v > $O/$id$v.out 2>&1
    echo "== $id$v $(grep -h "NVML GPU-busy" $O/$id$v/logs/e2e.log | tail -1) | $(grep -h "SUCCESS flags" $O/$id$v/logs/e2e.log | tail -1 | cut -c1-220)"
  done
done
echo R13_E2E_DONE
