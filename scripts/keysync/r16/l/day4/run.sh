#!/bin/bash
# r16 L day 4: base CP vs adapt CP (per-speaker keyframe LoRA) on m03 / old03 through int 0336ef6 + the KS_KF_LORA hook; GPU stages under the lock
D=/workspace/dub/r16l/d4; A=/workspace/dub/r16in/L_adapt; cd /workspace/dub/r16l/int
job() { local arm=$1 v=$2; shift 2; J=$(python3 $D/mkjob.py $arm $v "$@"); T0=$(date +%s)
  env DUB_ALLOW_NC_BED=0 R10_HEAVY_LOCK=/workspace/dub/gpu.lock /venv/dabai/bin/python scripts/job/dabai_job.py $J > ${J%.json}.log 2>&1
  echo "$arm $v rc=$? wall $(( $(date +%s) - T0 )) s $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(\"status\"), \"gpu_busy_s\", d.get(\"gpu_busy_s\"), [w[\"code\"] for w in d.get(\"warnings\",[])])" $D/${arm}_$v/status.json 2>/dev/null)"; }
job base m03
job p1 m03 KS_KF_LORA=$A/m03_p1/kf_lora.pt
job base old03
until [ -s $A/old03_p1/kf_lora.pt ] && [ -s $A/old03_p1/lora_meta.json ]; do sleep 20; done; sleep 30
job p1 old03 KS_KF_LORA=$A/old03_p1/kf_lora.pt
echo D4_DONE
