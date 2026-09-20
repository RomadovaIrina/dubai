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

# total workers in ANY state (idle/initializing/ready/running/throttled/unhealthy...): 0 == nothing is alive
workers_total() {
  health | json_get 'sum(v for v in d.get("workers", {}).values() if isinstance(v, (int, float)))'
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

# wait_zero_workers <timeout_s> — exit 0 when no worker exists and the queue is empty
wait_zero_workers() {
  local timeout="$1" t0 now w j
  t0=$(date +%s)
  while :; do
    w="$(workers_total)"; j="$(jobs_pending)"
    if [ "${w}" = "0" ] && [ "${j}" = "0" ]; then log "endpoint is cold: workers=0 pending_jobs=0"; return 0; fi
    now=$(date +%s)
    if [ $((now - t0)) -ge "${timeout}" ]; then log "TIMEOUT after ${timeout}s: workers=${w} pending_jobs=${j}"; return 1; fi
    log "waiting for scale-down: workers=${w} pending_jobs=${j}"; sleep 5
  done
}
