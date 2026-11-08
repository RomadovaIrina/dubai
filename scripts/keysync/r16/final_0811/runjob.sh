#!/bin/bash
# runjob.sh <job.json>: dabai_job + the job tree RSS / free disk every 5 s -> <job>.mem.tsv
J=$1; cd /workspace/dub/dubai
env DUB_ALLOW_NC_BED=0 R10_HEAVY_LOCK=/workspace/dub/gpu.lock /venv/dabai/bin/python scripts/job/dabai_job.py $J > ${J%.json}.log 2>&1 & P=$!; M=${J%.json}.mem.tsv; : > $M
while kill -0 $P 2>/dev/null; do echo -e "$(date +%s)\t$(/workspace/dub/r16t/treerss.sh $P)\t$(df -BG --output=avail /workspace | tail -1 | tr -d " G")" >> $M; sleep 5; done
echo "JOB_DONE peak_rss_gb $(sort -k2 -g $M | tail -1 | cut -f2) min_free_gb $(sort -k3 -g $M | head -1 | cut -f3)" | tee -a ${J%.json}.log
