# Pilot 0.10 — RunPod Serverless control scripts

Provider-specific commands for `scripts/pilot/benchmark_10_cold_start.py`. They only talk to an **existing** endpoint
(they never create/delete one) through two RunPod APIs:

* job API `https://api.runpod.ai/v2/<ENDPOINT_ID>/{run,status/<id>,cancel/<id>,purge-queue,health}`
* REST API `https://rest.runpod.io/v1/endpoints/<ENDPOINT_ID>` (`GET`, `PATCH {workersMin, workersMax, idleTimeout, flashboot, ...}`)

Credentials come from the environment only: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`.

| script | benchmark flag | does | exit 0 means |
|---|---|---|---|
| `ensure_stopped.sh` | `--ensure-stopped-command` | asserts scale-to-zero (`workersMin=0`, `workersMax≥1`), **refuses FlashBoot**, cancels our previous job, purges the queue, waits until `/health` shows zero workers in any state and an empty queue | the next start is a true cold start |
| `start_node.sh` | `--start-command` | `POST /run {"input":{"action":"ready"}}` — with `workersMin=0` this is the provider start request; saves the job id | job accepted (worker boot begins) |
| `is_ready.sh` | `--ready-command` | `GET /status/<job>`; prints `{status, ready, gpu, sm, missing, delayTime_ms, executionTime_ms, workerId}` | job `COMPLETED` **and** `output.ready == true` (CUDA + all weights + offline env verified inside the worker) |
| `stop_node.sh` | `--stop-command` | cancels the job if still running, purges the queue, waits for zero workers (idle timeout elapsed) | endpoint is cold again |

Exit codes of `is_ready.sh`: 1 still `IN_QUEUE`/`IN_PROGRESS`, 3 completed but `ready=false` (topology/weights problem —
inspect the printed `missing`), 4 job `FAILED`/`CANCELLED`/`TIMED_OUT`. The benchmark keeps polling on any non-zero code
until `--timeout`.

Timing definition preserved: t0 is taken by the benchmark immediately before `start_node.sh`; READY is the first
`is_ready.sh` exit 0, i.e. container pulled/started + RunPod automatic system checks (memory/disk/network/CUDA/GPU benchmark) +
our fitness checks + the `ready` job executed (torch import, CUDA context, weight/path validation on the volume).

```bash
export RUNPOD_API_KEY=...          # never commit
export RUNPOD_ENDPOINT_ID=...      # endpoint created manually with the settings in infra/serverless/README.md
I=scripts/pilot/infra_10
python scripts/pilot/benchmark_10_cold_start.py \
  --start-command "$I/start_node.sh" --ready-command "$I/is_ready.sh" \
  --stop-command "$I/stop_node.sh" --ensure-stopped-command "$I/ensure_stopped.sh" \
  --repeat 3 --timeout 600 --poll-interval 2 --stop-timeout 900 \
  --confirm-cold-node --confirm-multistage-image --confirm-network-nvme \
  --topology-note 'RunPod Serverless RTX 5090 (sm_120); ghcr.io/romadovairina/dabai-worker:<tag>; weights on network volume dabai-models (/runpod-volume/models); FlashBoot off; workersMin=0'
```
(`README_PILOT.md` still shows a single `--confirm-target-topology` flag; the script takes the three flags above.)

Prerequisites before the first formal run: image pushed; endpoint created with FlashBoot **off**, `workersMin=0`,
`workersMax=1`, idle timeout 5 s, RTX 5090 only, volume attached; one manual `{"action":"ready"}` returned `ready:true`
with `sm_120`; `stop_node.sh` reached zero workers. Cold-start numbers from the RTX 2000 Ada preparation Pod are not valid.
