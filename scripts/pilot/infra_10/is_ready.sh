#!/usr/bin/env bash
# Pilot 0.10 --ready-command: exit 0 ONLY when the worker completed the readiness job with ready=true, i.e. the container
# is up, CUDA works, every production weight is reachable on the volume and no download is needed. Anything else -> non-zero
# (1 = still IN_QUEUE/IN_PROGRESS, 3 = job finished but ready=false, 4 = job FAILED/CANCELLED/TIMED_OUT).
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[ -f "${JOB_FILE}" ] || { log "no job id (start_node.sh not run)"; exit 1; }
id="$(cat "${JOB_FILE}")"
resp="$(api "${JOB_API}/status/${id}")" || { log "status request failed"; exit 1; }
st="$(printf '%s' "${resp}" | json_get 'd.get("status")')"
case "${st}" in
  COMPLETED)
    ready="$(printf '%s' "${resp}" | json_get '(d.get("output") or {}).get("ready")')"
    printf '%s\n' "${resp}" | python3 -c 'import json,sys; d=json.load(sys.stdin); o=d.get("output") or {}; print(json.dumps({"status": d.get("status"), "ready": o.get("ready"), "gpu": o.get("gpu"), "sm": o.get("sm"), "missing": o.get("missing"), "delayTime_ms": d.get("delayTime"), "executionTime_ms": d.get("executionTime"), "workerId": d.get("workerId")}))'
    [ "${ready}" = "True" ] && exit 0
    log "job completed but ready=false"; exit 3 ;;
  FAILED|CANCELLED|TIMED_OUT)
    log "job ${id} ended with ${st}: $(printf '%s' "${resp}" | json_get 'str(d.get("error") or d.get("output"))[:400]')"; exit 4 ;;
  *)
    log "job ${id}: ${st}"; exit 1 ;;
esac
