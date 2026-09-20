# Pilot 0.10 — RunPod Serverless control scripts

Provider-specific commands for `scripts/pilot/benchmark_10_cold_start.py`. They only talk to an **existing** endpoint
(they never create/delete one) through two RunPod APIs:

* job API `https://api.runpod.ai/v2/<ENDPOINT_ID>/{run,status/<id>,cancel/<id>,purge-queue,health}`
* REST API `https://rest.runpod.io/v1/endpoints/<ENDPOINT_ID>` (`GET`, `PATCH {workersMin, workersMax, idleTimeout, flashboot, ...}`)

Credentials come from the environment only: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`.

| script | benchmark flag | does | exit 0 means |
|---|---|---|---|
| `ensure_stopped.sh` | `--ensure-stopped-command` | asserts scale-to-zero (`workersMin=0`, `workersMax≥1`), **refuses FlashBoot**, cancels our previous job, purges the queue, waits until no worker container is alive (`/health` `initializing+running == 0` **and** every endpoint pod is `EXITED` with uptime 0 via GraphQL) and the queue is empty | the next start is a scale-to-zero cold container start (see *Cold precondition*) |
| `start_node.sh` | `--start-command` | `POST /run {"input":{"action":"ready"}}` — with `workersMin=0` this is the provider start request; saves the job id | job accepted (worker boot begins) |
| `is_ready.sh` | `--ready-command` | `GET /status/<job>`; prints `{status, ready, gpu, sm, missing, delayTime_ms, executionTime_ms, workerId}` | job `COMPLETED` **and** `output.ready == true` (CUDA + all weights + offline env verified inside the worker) |
| `stop_node.sh` | `--stop-command` | cancels the job if still running, purges the queue, waits until no container is alive (idle timeout elapsed, pod `EXITED`) | endpoint is cold again |

Exit codes of `is_ready.sh`: 1 still `IN_QUEUE`/`IN_PROGRESS`, 3 completed but `ready=false` (topology/weights problem —
inspect the printed `missing`), 4 job `FAILED`/`CANCELLED`/`TIMED_OUT`. The benchmark keeps polling on any non-zero code
until `--timeout`.

**Cold precondition (accepted 2026-09-20; Pilot 0.10 was closed with it):** RunPod keeps a standby worker *slot* assigned to the endpoint even with zero traffic (`/health` shows it as `idle` = "scaled down, waiting for requests", or `throttled` = host GPU busy; the pod is `EXITED`, uptime 0, not billed, image cached on the host). A standby idle/throttled slot is a logical RunPod slot and is **not** counted as a live worker/container. The original "sum of all `/health` worker states == 0" precondition could only be met transiently (while RunPod is between hosts), never on demand.

Cold state = FlashBoot **off** · `pending_jobs = 0` · `initializing = 0` · `running = 0` · all endpoint pods `EXITED` / uptime 0 (GraphQL). What is measured is therefore a **scale-to-zero Serverless cold container start** (RunPod scheduling + container start + network-volume mount + handler/CUDA init + readiness checks). It is **not** a guarantee of a fresh physical host or of an empty Docker/image cache: RunPod may start the container on the same host with the image already cached (formal runs 2–3), or may have to re-initialize the worker first (formal run 1: throttled → initializing → idle → running).

Formal result (2026-09-20): `CLOSED — FAIL_TARGET` — start→READY 271.170 / 15.701 / 11.280 s (mean 99.384, median 15.701, target <60 s for every run); RunPod provider-reported pre-execution delay 269.387 / 14.616 / 8.710 s (kept separately, never subtracted); readiness execution 0.28–0.35 s. See `reports/pilot/0.10_cold_start.md` and `reports/pilot/0.10_cold_start_breakdown.md`.

Timing definition preserved: t0 is taken by the benchmark immediately before `start_node.sh`; READY is the first
`is_ready.sh` exit 0, i.e. container started (image pull only when RunPod (re)initializes the worker) + RunPod automatic system checks (memory/disk/network/CUDA/GPU benchmark) +
our fitness checks + the `ready` job executed (torch import, CUDA context, weight/path validation on the volume).

```bash
export RUNPOD_API_KEY=...          # never commit
export RUNPOD_ENDPOINT_ID=...      # endpoint created manually with the settings in infra/serverless/README.md
I=scripts/pilot/infra_10
python scripts/pilot/benchmark_10_cold_start.py \
  --start-command "$I/start_node.sh" --ready-command "$I/is_ready.sh" \
  --stop-command "$I/stop_node.sh" --ensure-stopped-command "$I/ensure_stopped.sh" \
  --repeat 3 --timeout 3600 --poll-interval 2 --stop-timeout 900 \
  --confirm-cold-node --confirm-multistage-image --confirm-network-nvme \
  --topology-note 'RunPod Serverless RTX 5090 (sm_120); ghcr.io/romadovairina/dabai-worker:<tag>; weights on network volume dabai-models (/runpod-volume/models); FlashBoot off; workersMin=0'
```
`--timeout 3600` (not 600): RTX 5090 capacity waits of 20–30 min have been observed in EUR-IS-1 and are part of the measurement.

Prerequisites before the first formal run: image pushed; endpoint created with FlashBoot **off**, `workersMin=0`,
`workersMax=1`, idle timeout 5 s, RTX 5090 only, volume attached; one manual `{"action":"ready"}` returned `ready:true`
with `sm_120`; `stop_node.sh` reached zero workers. Cold-start numbers from the RTX 2000 Ada preparation Pod are not valid.
