#!/bin/bash
# all 6 cfhf finals (prod + fast2 x m18 / m03 / old03) in parallel (CPU-bound; the raw SyncNet calls take gpu.lock)
K=/workspace/dub/keysync; cd $K
for id in q18 m03 qo3; do
  bash k_cfhf.sh $id prodKS $K/fast/prod/out/$id/${id}_$id.mp4 > final/cfhf_prodKS_$id.out 2>&1 &
  bash k_cfhf.sh $id fast2 $K/fast/fast2v/out/$id/${id}_$id.mp4 > final/cfhf_fast2_$id.out 2>&1 &
done; wait; cat final/cfhf_*.out | grep -E "CFHF_DONE|FAIL"; echo K_CFHF_ALL_DONE
