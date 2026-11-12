#!/bin/bash
# DabAI on Vast Serverless (TZ 10.1): the worker's launch command. The models volume = this machine's disk (kept while the worker is STOPPED):
# the first start fills it (fetch-models: HF_TOKEN + KS_KIT_REPO from the template secrets), every start runs the API + the resident worker,
# then the Vast PyWorker (worker.py) in front of it. Nothing private is in the image.
set -uo pipefail
R=/workspace/dub/dubai; export DUB_PROFILE=${DUB_PROFILE:-clean} DABAI_RESIDENT=${DABAI_RESIDENT:-1}
mkdir -p /data/resident /var/log/portal
bash $R/infra/docker/check.sh --weights-only > /data/weights_check.log 2>&1 || bash $R/infra/docker/fetch_models.sh > /data/fetch_models.log 2>&1 || { echo "RESIDENT: fetch-models failed" >> /data/resident/start.log; }
( bash $R/infra/docker/entrypoint.sh api > /var/log/portal/api.log 2>&1 & )
for i in $(seq 120); do curl -sf http://127.0.0.1:8000/health > /dev/null && break; sleep 1; done
pip install -q -r $R/infra/vast_serverless/requirements.txt 2>/dev/null || /venv/api/bin/pip install -q vastai
cd $R/infra/vast_serverless && exec python3 worker.py
