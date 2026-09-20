#!/usr/bin/env bash
# Pilot 0.10 --start-command: the provider start request. With workersMin=0 the ONLY way a worker starts is a job in
# the queue, so we submit the readiness job itself: POST /run {"input":{"action":"ready"}}. Returns immediately with the
# job id (saved for is_ready.sh / stop_node.sh). The benchmark's t0 is taken right before this script runs.
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

resp="$(api -X POST "${JOB_API}/run" -d '{"input": {"action": "ready"}}')"
id="$(printf '%s' "${resp}" | json_get 'd["id"]')"
[ -n "${id}" ] || { log "no job id in response: ${resp}"; exit 1; }
printf '%s' "${id}" > "${JOB_FILE}"
log "submitted readiness job ${id} (status $(printf '%s' "${resp}" | json_get 'd.get("status")'))"
echo "${id}"
