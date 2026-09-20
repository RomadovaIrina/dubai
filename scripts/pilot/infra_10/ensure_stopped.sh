#!/usr/bin/env bash
# Pilot 0.10 preflight (--ensure-stopped-command): guarantee the next start is a TRUE cold start.
#   * endpoint must be scale-to-zero (workersMin=0) with at least one allowed worker;
#   * FlashBoot must be OFF (it revives retained worker state -> not a cold start) unless DABAI_ALLOW_FLASHBOOT=1;
#   * no job of ours is pending; queue purged; zero workers alive (idle timeout elapsed).
# Exit 0 only when all of that holds. Never creates or deletes endpoints.
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cfg="$(endpoint_get)"
wmin="$(printf '%s' "${cfg}" | json_get 'd.get("workersMin")')"
wmax="$(printf '%s' "${cfg}" | json_get 'd.get("workersMax")')"
flash="$(printf '%s' "${cfg}" | json_get 'd.get("flashboot")')"
idle="$(printf '%s' "${cfg}" | json_get 'd.get("idleTimeout")')"
name="$(printf '%s' "${cfg}" | json_get 'd.get("name")')"
log "endpoint ${RUNPOD_ENDPOINT_ID} (${name}): workersMin=${wmin} workersMax=${wmax} flashboot=${flash} idleTimeout=${idle}s"

if [ "${flash}" = "True" ] && [ "${DABAI_ALLOW_FLASHBOOT:-0}" != "1" ]; then
  log "FAIL: FlashBoot is enabled -> starts would reuse retained state, not a cold start. Disable it (PATCH flashboot=false) or set DABAI_ALLOW_FLASHBOOT=1 to measure warm-ish starts explicitly."
  exit 2
fi
if [ "${wmin}" != "0" ]; then
  log "workersMin=${wmin}: setting workersMin=0 (scale to zero)"
  endpoint_patch '{"workersMin": 0}' >/dev/null
fi
if [ "${wmax}" = "0" ] || [ "${wmax}" = "None" ]; then
  log "FAIL: workersMax=${wmax}; the endpoint cannot start a worker. Set workersMax>=1."
  exit 2
fi

cancel_saved_job
api -X POST "${JOB_API}/purge-queue" >/dev/null || log "purge-queue returned non-2xx (ignored)"
rm -f "${JOB_FILE}"
wait_zero_workers "${DABAI_STOP_TIMEOUT:-900}"
