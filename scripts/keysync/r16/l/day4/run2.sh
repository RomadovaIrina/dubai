#!/bin/bash
# r16 L day 4 (2): the adapt LoRA at half strength (p1h = up x 0.5) on m03 / old03, after run.sh
D=/workspace/dub/r16l/d4; cd /workspace/dub/r16l/int
until grep -q D4_DONE $D/run.log; do sleep 20; done
[ -s $D/lora/old03_p1h/kf_lora.pt ] || /venv/dabai/bin/python $D/scale_lora.py /workspace/dub/r16in/L_adapt/old03_p1 $D/lora/old03_p1h 0.5
job() { local arm=$1 v=$2; shift 2; J=$(python3 $D/mkjob.py $arm $v "$@"); T0=$(date +%s)
  env DUB_ALLOW_NC_BED=0 R10_HEAVY_LOCK=/workspace/dub/gpu.lock /venv/dabai/bin/python scripts/job/dabai_job.py $J > ${J%.json}.log 2>&1
  echo "$arm $v rc=$? wall $(( $(date +%s) - T0 )) s $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(\"status\"), \"gpu_busy_s\", d.get(\"gpu_busy_s\"))" $D/${arm}_$v/status.json 2>/dev/null)"; }
job p1h m03 KS_KF_LORA=$D/lora/m03_p1h/kf_lora.pt
job p1h old03 KS_KF_LORA=$D/lora/old03_p1h/kf_lora.pt
echo D4B_DONE
