# Worker monitoring (TZ 10.4, r18)

What it watches: worker liveness, queue depth, stuck jobs, GPU hours, VRAM peak, wall time per video minute.
Code: `monitor.py` (the numbers), `job_api.py` (`/health`, `/metrics`), `alert.py` (alerts). Tests: `test_monitor.py` (6 cases) + the r18 part of `test_interface.sh` (8 checks). CPU only.

## Endpoints

| Path | Auth | What |
|---|---|---|
| `GET /health` | none | As before (200 / 503) + a `monitor` block: `worker` (thread alive, heartbeat age, uptime), `queue_depth`, `queue_oldest_wait_s`, `running` (id, step, %, elapsed), `stuck[]` (job id + reason), `totals` (jobs by status, video minutes, wall s, GPU busy s / hours), `last_job`, `recent` (last 20 jobs: error / partial share, wall and GPU s per video minute, VRAM peak max). `ok` is now also false when the worker thread is dead. |
| `GET /metrics` | the API token (`DABAI_METRICS_PUBLIC=1` = none) | The same numbers in the Prometheus text format. Names start with `dabai_`. No job ids in labels. |

Main series:

| Series | Meaning |
|---|---|
| `dabai_worker_alive` | 1 = the worker thread lives and its idle loop ticked < 60 s ago |
| `dabai_health_ok` | the `/health` verdict |
| `dabai_queue_depth`, `dabai_queue_oldest_wait_seconds` | the FIFO queue of this node (TZ 3.7 scale-out: > 3) |
| `dabai_jobs_stuck` | stuck jobs (ids: `/health` → `monitor.stuck`) |
| `dabai_jobs_finished_total{status}` | finished jobs by status (TZ 11.3: PARTIAL_SUCCESS lands here) |
| `dabai_gpu_busy_seconds_total`, `dabai_gpu_busy_hours_total` | GPU time of the finished jobs (NVML device busy) |
| `dabai_gpu_weighted_seconds_total` | the jobs' own processes only (NVML weighted) |
| `dabai_video_minutes_total`, `dabai_job_wall_seconds_total` | for the rates: wall / video minute = ratio of the two |
| `dabai_last_job_vram_peak_mib`, `dabai_recent_vram_peak_mib_max` | the VRAM peak (card 32607 MiB; the open risk is 31.6 GB) |
| `dabai_last_job_wall_seconds_per_video_minute`, `dabai_last_job_gpu_seconds_per_video_minute` | the last job against the 300 s norm |
| `dabai_uptime_seconds` | a serverless node is paid for its uptime, not its busy time |
| `dabai_gpu_memory_used_mib`, `dabai_disk_free_gb`, `dabai_resident_state{state}` | live node state |

The totals come from a ledger, `$DABAI_MONITOR_DIR/ledger.jsonl` (default `$DABAI_JOBS_DIR/.monitor/`): one line per finished job. They survive deleted job directories. At start the ledger is back-filled from the finished jobs on disk. A container without a persistent `/data/jobs` volume starts from zero; Prometheus keeps the history.

## Stuck job: the rules

| Reason | When | Default |
|---|---|---|
| `stale` | RUNNING, `status.json` not rewritten (a live job rewrites it every ~3 s) | 120 s (`DABAI_STUCK_STALE_S`) |
| `orphan` | RUNNING on disk, but not the worker's job | > 30 s |
| `timeout` | running past its `timeout_s` + grace: the kill did not work | 300 s (`DABAI_STUCK_TIMEOUT_GRACE_S`) |
| `step` | one step runs longer than max(1800 s, 4 × its expected time) (`DABAI_STUCK_STEP_MIN_S`, `DABAI_STUCK_STEP_FACTOR`) | e.g. KeySync of a 10-min video: 4 × 2900 s |
| `lost_queued` | QUEUED on disk, but not in the queue | > 30 s |

`step` is the only rule that sees a hung stage early: its status stays fresh, and the timeout is 1200 s per video minute (3.3 h for 10 min).

## Alerts

```bash
python3 scripts/job/alert.py --url http://worker:8000                        # one check: exit 0 OK, 1 alerts (cron)
python3 scripts/job/alert.py --url http://w1:8000 --url http://w2:8000 --every 60   # a loop: notify on change only
```
Standard library only. Optional channels: `DABAI_ALERT_WEBHOOK` (POST `{"text": …}`: Slack / Mattermost / any JSON hook), `DABAI_ALERT_TG_TOKEN` + `DABAI_ALERT_TG_CHAT` (Telegram). The tokens are never printed.

| Level | Rule | Threshold (env) |
|---|---|---|
| CRIT | down: no answer | – |
| CRIT | unhealthy: `/health` 503 (GPU / weights / disk / worker thread) | `DABAI_MIN_FREE_GB` 30 |
| CRIT | worker_stalled: the thread lives, its loop does not tick | 60 s |
| CRIT | stuck: one alert per stuck job | the table above |
| WARN | queue depth | > 3 (`DABAI_ALERT_QUEUE`) |
| WARN | oldest queued job waits | > 1800 s (`DABAI_ALERT_QUEUE_WAIT_S`) |
| WARN | last job VRAM peak | > 31500 MiB (`DABAI_ALERT_VRAM_MIB`) |
| WARN | recent GPU s per video minute | > 300 (`DABAI_ALERT_GPU_PER_VMIN`) |
| WARN | recent wall s per video minute | > 900 (`DABAI_ALERT_WALL_PER_VMIN`) |
| WARN | recent ERROR + TIMEOUT share (≥ 5 jobs) | > 0.2 (`DABAI_ALERT_ERROR_RATIO`) |

With Prometheus, the same as rules:
```yaml
groups:
- name: dabai
  rules:
  - {alert: DabaiDown,      expr: up{job="dabai"} == 0 or dabai_health_ok == 0 or dabai_worker_alive == 0, for: 2m}
  - {alert: DabaiStuckJob,  expr: dabai_jobs_stuck > 0, for: 1m}
  - {alert: DabaiQueue,     expr: sum(dabai_queue_depth) > 3, for: 5m}
  - {alert: DabaiVram,      expr: dabai_last_job_vram_peak_mib > 31500}
  - {alert: DabaiGpuNorm,   expr: increase(dabai_gpu_busy_seconds_total[1d]) / increase(dabai_video_minutes_total[1d]) > 300}
  - {alert: DabaiErrors,    expr: dabai_recent_error_ratio > 0.2 and dabai_recent_jobs >= 5}
```
GPU hours per day: `increase(dabai_gpu_busy_hours_total[1d])`. Node hours (what a serverless node costs): `dabai_uptime_seconds / 3600`.

## Caveats

- `gpu_busy` is NVML device busy: exact on a dedicated card; with the resident worker it includes the parked daemons (INTERFACE.md §8). `gpu_weighted` misses the resident processes. On the 10-min job of 12.11 (`C:\repos\chkpts\r17_docker\long10_1211`): busy 301.1, weighted 170.6 GPU-s per video minute.
- Across nodes (the backend's Redis / Celery queue, TZ 3.4 / 3.7), the queue depth is the backend's number; this one is the node's own queue.
