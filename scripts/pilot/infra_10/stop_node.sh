#!/usr/bin/env bash
# Pilot 0.10 --stop-command: return the endpoint to cold. Cancels our job if still running, purges the queue and waits until
# RunPod has torn the worker down (idle timeout elapsed, zero workers). Set DABAI_STOP_NOWAIT=1 to skip the wait
# (ensure_stopped.sh will then do it before the next run).
# shellcheck source=_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cancel_saved_job
api -X POST "${JOB_API}/purge-queue" >/dev/null || log "purge-queue returned non-2xx (ignored)"
if [ "${DABAI_STOP_NOWAIT:-0}" = "1" ]; then log "not waiting for scale-down (DABAI_STOP_NOWAIT=1)"; exit 0; fi
wait_zero_workers "${DABAI_STOP_TIMEOUT:-900}"
