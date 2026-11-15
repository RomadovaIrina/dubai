#!/bin/bash
# r17: the Vast Serverless TEST endpoint (Irina 12.11: max 1 worker, 1 cold, 15 min idle, <= $3). RUN BY HAND: it creates PAID Vast resources.
# Needs the vastai CLI (pip install vastai) and /workspace/.env with HF_TOKEN and VAST_API_KEY. It prints no secret.
# The HF token goes to the Vast ACCOUNT env vars (injected into the account's instances), not into the template text.
# Delete afterwards: vastai delete workergroup <id>; vastai delete endpoint <id>   (the ids are in /root/vast_sl/*.json)
set -uo pipefail
set -a; . /workspace/.env; set +a; K=(--api-key "$VAST_API_KEY"); SHA=${SHA:?the commit whose start.sh the workers run}; TAG=${TAG:?image tag, e.g. 5593589c6164-clean}; O=${O:-/root/vast_sl}; mkdir -p $O
vastai "${K[@]}" create env-var HF_TOKEN "$HF_TOKEN" > /dev/null 2>&1 && echo "account env-var HF_TOKEN set" || echo "env-var HF_TOKEN: exists or failed"
# r20: cuda_max_good < 13.05 = drivers 570-580 (the release was verified on 580.95.05). On 615.71.09 the KeySync kf TensorRT engine verified at
# 55.82 dB < the 56 dB floor (KS_TRT_MIN_DB), so the resident failed: KS_ENGINES_FAIL kf verify.
SP="gpu_name=RTX_5090 num_gpus=1 gpu_ram>=30 cpu_ram>=64 disk_space>=200 inet_down>=500 reliability>0.97 cuda_max_good>=12.8 cuda_max_good<13.05"
vastai "${K[@]}" create template --name dabai-serverless-$TAG --image ghcr.io/romadovairina/dubai --image_tag $TAG \
  --env "-e DUB_PROFILE=clean -e DABAI_RESIDENT=1 -e KS_KIT_REPO=RomadovaIrina/dabai-keysync-kit -e WORKER_PORT=3000 -p 8000:8000 -p 3000:3000" \
  --onstart-cmd "curl -fsSL https://raw.githubusercontent.com/RomadovaIrina/dubai/$SHA/infra/vast_serverless/start.sh -o /tmp/start.sh && bash /tmp/start.sh $SHA" \
  --disk_space 200 --ssh --direct --search_params "$SP" --desc "DabAI dubbing worker (r17 serverless test)" --raw > $O/template.json 2>&1
HASH=$(grep -oE "'hash_id': '[0-9a-f]{32}'|\"hash_id\": \"[0-9a-f]{32}\"" $O/template.json | grep -oE "[0-9a-f]{32}" | head -1)   # r20: the CLI prints "New Template: {python dict}", not JSON
echo "template hash: ${HASH:-NONE}"; [ -n "$HASH" ] || { echo "template creation failed: see $O/template.json"; exit 1; }
vastai "${K[@]}" create endpoint --endpoint_name dabai-dub --cold_workers 1 --max_workers 1 --min_load 0 --target_util 0.9 --inactivity_timeout 900 --max_queue_time 3600 --raw > $O/endpoint.json 2>&1
echo "endpoint: $(head -c 300 $O/endpoint.json | tr -d '\n')"
vastai "${K[@]}" create workergroup --endpoint_name dabai-dub --template_hash $HASH --gpu_ram 30 --search_params "$SP" --raw > $O/workergroup.json 2>&1
echo "workergroup: $(head -c 300 $O/workergroup.json | tr -d '\n')"
echo "Next: watch the worker in the Vast console (first start ~40 min: image pull + fetch-models 38 GB + TensorRT build); /var/log/portal/dabai_start.log on it."
