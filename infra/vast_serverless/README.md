# DabAI on Vast Serverless (TZ 10.1) — plan and config

Irina 11.11: serverless runs on **Vast.ai**, not RunPod. **Nothing here has been deployed.** It needs the image in GHCR (the token is pending) and a few
hours of tests on a real endpoint. Vast's docs disagree on some autoscaler defaults, so set every parameter explicitly and check it in the console.

## How it fits together
```
client (backend) ──POST /jobs/sync──▶ Vast Serverless endpoint "dabai-dub" ──▶ a worker in the worker group
                                                                   │  Vast PyWorker (infra/vast_serverless/worker.py, port from the template)
                                                                   ▼
                                   ghcr.io/romadovairina/dubai:<tag>  — job API :8000 + resident worker (DABAI_RESIDENT=1)
```
- **One request = one whole job.** `POST /jobs/sync` (scripts/job/job_api.py) blocks until the job ends and returns the final status.json. The
  router may send each request to any worker, so the async `POST /jobs` + `GET /jobs/{id}` pair cannot be used through it.
- **Delivery:** the request carries a presigned `output_put_url` (and optionally `output_put_subs_url`) in the backend's own object storage. The worker
  PUTs the result there. The input is an `https://` URL (the job interface downloads it). No storage credentials are on the worker.
- **One job per GPU** (`allow_parallel_requests=False`): the 32 GB card fits one pipeline (audio ~30 GB peak, KeySync ~28 GB, never at once).
- **Load unit:** input video minutes. The client sends `video_minutes`; the PyWorker's `workload_calculator` uses it.

## Weights and cold start
Vast workers are separate machines, and there is no shared network volume, so each worker needs the 37.6 GB of weights and its own TensorRT plans.
- **Recommended: let stopped ("cold") workers keep their disk.** The first start on a new machine runs `fetch-models` (~38 GB, a few minutes at
  datacentre speed) and the first job builds the TensorRT plans + the VAE autotune (~14 min on a 5090 / driver). A worker that is then STOPPED
  (scale to zero) keeps its disk, so its next wake-up is container start + resident load. Measured 11.11: API up 5.7 s; resident load and
  the first-job numbers come from `infra/docker/resident_bench.sh` (to run).
- **Alternative:** a derived image with the weights baked in (`FROM ghcr.io/romadovairina/dubai:<tag>` + `COPY models`): ~55 GB per pull. Slower to
  pull on every new machine; the TensorRT plans still have to be built per GPU / driver.
- The entrypoint for serverless: `fetch-models` if `check --weights-only` fails, then `api` with `DABAI_RESIDENT=1` (a small wrapper script; to write
  with the template).

## Endpoint / worker group settings (to set explicitly)
| Setting | Value | Why |
|---|---|---|
| image | `ghcr.io/romadovairina/dubai:<tag>` (public) | |
| GPU filter | RTX 5090 (or ≥ 32 GB, compute ≥ 12.0), `cuda_max_good ≥ 12.8`, driver ≥ 570 | sm_120 build |
| disk | ≥ 120 GB | image ~30 GB unpacked + weights 37.6 GB + TRT / inductor 18 GB + jobs (≥ 10 GB per video minute while a job runs) |
| RAM | ≥ 64 GB | long videos |
| min_load | 0 | scale to zero |
| cold_workers | 1 (or more) | stopped workers that keep their disk (weights + TRT plans) → wake without the 38 GB download / 14 min build |
| max_workers | per budget | |
| inactivity → stop | 15 min (Irina) | the docs call this the inactivity timeout: check its exact name in the console |
| env | `DABAI_RESIDENT=1`, `HF_TOKEN` (only for fetch-models; Vast template secret), `PYWORKER_REPO` | never bake tokens into the image |

## What is still to do
1. Push the image to GHCR (the token). The image is already checked: push_ghcr.sh PASS on 6b50043.
2. Add a 10 s CC0 benchmark clip to the image (the PyWorker benchmark runs one real job; this also warms a new worker's caches).
3. The serverless start wrapper (fetch-models if needed → api + resident).
4. A real endpoint test: scale 0 → 1 on the first request (measure the wake time on a fresh machine and on a stopped one), the 15-min idle stop,
   two requests queued on one worker, a worker killed mid-job (the client must retry: the job id is idempotent → 409 on a duplicate).
5. Cost: Vast serverless bills the worker's GPU time while active and only storage while stopped (check the current price sheet).
