#!/bin/bash
# r13 KS: the r9 E2E (production stages only) on old03 / m18 / m03 -> the dub voice, the demucs mix, master_25fps and source landmarks that the KeySync eval / finals read
cd /workspace/dub/dubai; M=/workspace/dub/inputs/mamakris; mkdir -p /workspace/dub/r13k/runs
while pgrep -f "setup_daba[i]" > /dev/null; do sleep 30; done
bash scripts/r9/r9_models.sh > /workspace/dub/r13k/models.log 2>&1; echo "MODELS rc=$?"
for s in "qo3 /workspace/dub/inputs/test_videos_raw/03.mp4" "q18 $M/story-1774250817106-616905094.mp4" "m03 $M/story-1774249919482-367803614.MOV"; do set -- $s
  until [ -s $2 ] && [ $(ls $M | wc -l) -ge 9 -o "$1" = qo3 ]; do sleep 30; done
  t0=$(date +%s); ID=$1 E2E_METRICS=off bash scripts/r9/e2e.sh $2 /workspace/dub/r13k/runs/$1 > /workspace/dub/r13k/e2e_$1.log 2>&1; echo "E2E $1 rc=$? $(( $(date +%s) - t0 ))s"
done; echo R13_E2E_DONE
