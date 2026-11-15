#!/bin/bash
# DabAI on Vast Serverless (TZ 10.1): the worker's onstart. The models live on THIS machine's disk (kept while the worker is STOPPED):
# the first start fills them (fetch-models: HF_TOKEN + KS_KIT_REPO from the template env), every start runs the API + the resident worker,
# then the Vast PyWorker (worker.py, in its own venv) in front of it. Nothing private is in the image.
# Onstart (template): curl -fsSL https://raw.githubusercontent.com/RomadovaIrina/dubai/<sha>/infra/vast_serverless/start.sh -o /tmp/start.sh && bash /tmp/start.sh <sha>
set -uo pipefail
SHA=${1:-experiment/r16-docker}; RAW=https://raw.githubusercontent.com/RomadovaIrina/dubai/$SHA/infra/vast_serverless
R=/workspace/dub/dubai; export DUB_PROFILE=${DUB_PROFILE:-clean} DABAI_RESIDENT=${DABAI_RESIDENT:-1} DUB_ROOT=$R
# r20: the Vast PyWorker listens on WORKER_PORT and reports VAST_TCP_PORT_$WORKER_PORT; Vast Serverless maps 3000 (VAST_TCP_PORT_3000) but sets no WORKER_PORT
export WORKER_PORT=${WORKER_PORT:-3000}
mkdir -p /data/resident /data/jobs /var/log/portal; L=/var/log/portal/dabai_start.log; exec >> $L 2>&1
echo "== $(date -u +%FT%TZ) start ($SHA)"
curl -fsSL $RAW/worker.py -o /opt/dabai_worker.py && curl -fsSL $RAW/requirements.txt -o /opt/dabai_req.txt || echo "RESIDENT: cannot fetch worker.py"
# the benchmark input: a 5 s synthetic clip (no face -> the job ends fast with a clear status; the PyWorker only needs a 200 round trip)
ffmpeg -v error -y -f lavfi -i testsrc=duration=5:size=720x1280:rate=25 -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest /data/bench_5s.mp4
if ! bash $R/infra/docker/check.sh --weights-only > /data/weights_check.log 2>&1; then
  # r20: a new host can hit HF CDN read timeouts (10 s default): longer timeouts + 3 tries (fetch_models.sh resumes, as in scripts/r18/vm_setup.sh)
  export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}; ok=0
  for i in 1 2 3; do echo "weights missing -> fetch-models (try $i)"; bash $R/infra/docker/fetch_models.sh > /data/fetch_models_$i.log 2>&1 && { ok=1; break; }; sleep 20; done
  [ $ok = 1 ] && echo "fetch-models ok (try $i)" || echo "RESIDENT: fetch-models failed 3 times (/data/fetch_models_*.log)"; fi
( cd $R && bash infra/docker/entrypoint.sh api > /var/log/portal/api.log 2>&1 & )
for i in $(seq 180); do curl -sf http://127.0.0.1:8000/health > /dev/null && break; sleep 1; done; echo "api up"
[ -x /opt/pyworker/bin/python ] || { uv venv -q -p 3.11 /opt/pyworker && uv pip install -q --python /opt/pyworker/bin/python -r /opt/dabai_req.txt; }
cd /opt && BENCH_CLIP=/data/bench_5s.mp4 exec /opt/pyworker/bin/python /opt/dabai_worker.py
