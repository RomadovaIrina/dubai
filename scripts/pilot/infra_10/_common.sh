#!/usr/bin/env bash
# Shared helpers for the Pilot 0.10 RunPod Serverless control scripts. Sourced, not executed.
# Contract consumer: scripts/pilot/benchmark_10_cold_start.py (start/ready/stop/ensure-stopped commands, exit codes).
#
# Required environment (never hard-code them; export them in the benchmark shell):
#   RUNPOD_API_KEY       RunPod API key (Settings -> API Keys)
#   RUNPOD_ENDPOINT_ID   the Serverless endpoint that runs ghcr.io/romadovairina/dabai-worker with the dabai-models volume
# Optional:
#   DABAI_INFRA_STATE    state dir for the current job id             (default /tmp/dabai_infra_10)
#   DABAI_HTTP_TIMEOUT   per-request curl timeout in seconds          (default 20)
#   DABAI_STOP_TIMEOUT   max seconds to wait for zero workers         (default 900)
#   DABAI_ALLOW_FLASHBOOT=1  accept an endpoint with FlashBoot enabled (NOT a true cold start; default: refuse)
# shellcheck shell=bash
set -euo pipefail
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be exported (not stored in the repo)}"
: "${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID must be exported}"

JOB_API="https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}"
REST_API="https://rest.runpod.io/v1"
STATE_DIR="${DABAI_INFRA_STATE:-/tmp/dabai_infra_10}"
JOB_FILE="${STATE_DIR}/job_id"
HTTP_TIMEOUT="${DABAI_HTTP_TIMEOUT:-20}"
mkdir -p "${STATE_DIR}"

log() { printf '[%s %s] %s\n' "$(basename "$0")" "$(date -u +%H:%M:%S)" "$*" >&2; }

# api <curl args...>  — authenticated JSON request; prints the body; non-2xx -> non-zero exit
api() {
  curl -sS --fail-with-body --max-time "${HTTP_TIMEOUT}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json" "$@"
}

# jq-free JSON field access: json_get '<python expression over d>'  (stdin = JSON document)
json_get() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }

health() { api "${JOB_API}/health"; }

# Workers with a LIVE container. RunPod /health worker states (docs.runpod.io/serverless/workers/overview):
#   initializing = downloading image / starting;  running = executing a job;
#   idle         = "scaled down, waiting for requests" (container EXITED, not billed, image cached on the host);
#   throttled    = slot assigned but the host has no free GPU (no container);  ready = aggregate of idle+running.
# With workersMax>=1 RunPod keeps a standby slot assigned (idle/throttled) even with zero traffic, so the old
# "sum of all states == 0" precondition could never be met on demand. A standby idle/throttled slot is a logical
# RunPod slot, NOT a live worker/container. Alive == initializing + running.
# Accepted cold state (Pilot 0.10, 2026-09-20): FlashBoot off, pending_jobs=0, initializing=0, running=0, all endpoint
# pods EXITED / uptime 0. This guarantees a scale-to-zero Serverless cold CONTAINER start; it does NOT guarantee a
# fresh physical host or an empty Docker/image cache (RunPod may reuse the same host with the image already cached).
workers_alive() {
  health | json_get 'int(d.get("workers", {}).get("initializing", 0)) + int(d.get("workers", {}).get("running", 0))'
}
workers_slots() {  # informational: idle/throttled standby slots (no container)
  health | json_get 'str(d.get("workers", {}).get("idle", 0)) + "/" + str(d.get("workers", {}).get("throttled", 0))'
}
# Second, independent check through GraphQL: every pod (worker) of this endpoint must be EXITED with uptime 0,
# i.e. no container process survives from a previous run (FlashBoot is off, so nothing is retained).
pods_alive() {
  api -X POST "https://api.runpod.io/graphql" \
      -d '{"query":"{ myself { endpoints { id pods { id desiredStatus uptimeSeconds } } } }"}' \
    | json_get 'sum(1 for e in d["data"]["myself"]["endpoints"] if e["id"] == "'"${RUNPOD_ENDPOINT_ID}"'" for p in (e.get("pods") or []) if p.get("desiredStatus") != "EXITED" or (p.get("uptimeSeconds") or 0) > 0)'
}
jobs_pending() {
  health | json_get 'int(d.get("jobs", {}).get("inQueue", 0)) + int(d.get("jobs", {}).get("inProgress", 0))'
}

endpoint_get() { api "${REST_API}/endpoints/${RUNPOD_ENDPOINT_ID}"; }
endpoint_patch() { api -X PATCH "${REST_API}/endpoints/${RUNPOD_ENDPOINT_ID}" -d "$1"; }

cancel_saved_job() {
  [ -f "${JOB_FILE}" ] || return 0
  local id; id="$(cat "${JOB_FILE}")"
  [ -n "${id}" ] || return 0
  local st; st="$(api "${JOB_API}/status/${id}" | json_get 'd.get("status")' 2>/dev/null || echo UNKNOWN)"
  case "${st}" in
    COMPLETED|FAILED|CANCELLED|TIMED_OUT) log "previous job ${id} is ${st}" ;;
    *) log "cancelling previous job ${id} (${st})"; api -X POST "${JOB_API}/cancel/${id}" >/dev/null || true ;;
  esac
}

# wait_zero_workers <timeout_s> — exit 0 when no live worker container exists (health: initializing+running == 0,
# GraphQL: every endpoint pod EXITED with uptime 0) and the queue is empty. Standby idle/throttled slots are allowed
# (they hold no container) and are logged for evidence. Transient API errors count as "not cold yet".
wait_zero_workers() {
  local timeout="$1" t0 now w p j slots
  t0=$(date +%s)
  while :; do
    w="$(workers_alive 2>/dev/null || echo NA)"; p="$(pods_alive 2>/dev/null || echo NA)"
    j="$(jobs_pending 2>/dev/null || echo NA)"; slots="$(workers_slots 2>/dev/null || echo NA)"
    if [ "${w}" = "0" ] && [ "${p}" = "0" ] && [ "${j}" = "0" ]; then
      log "endpoint is cold: live_workers=0 pods_alive=0 pending_jobs=0 (standby idle/throttled slots=${slots})"; return 0
    fi
    now=$(date +%s)
    if [ $((now - t0)) -ge "${timeout}" ]; then log "TIMEOUT after ${timeout}s: live_workers=${w} pods_alive=${p} pending_jobs=${j} slots=${slots}"; return 1; fi
    log "waiting for scale-down: live_workers=${w} pods_alive=${p} pending_jobs=${j} slots(idle/throttled)=${slots}"; sleep 5
  done
}
