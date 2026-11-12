#!/bin/bash
# r17 speed (KS_SYNC_SERVE=1, off by default until the VM guard): sync_perframe.py through ONE resident process per video (scripts/quality/sync_serve.py).
# Sourced by ks_face.sh / ks_shots.sh. Needs R and PY.
#   sync_serve_start <dir>   (no-op unless KS_SYNC_SERVE=1 and no server is set yet; exports KS_SYNC_SERVE_DIR for the child ks_face.sh runs)
#   sync_serve_stop
#   sync_run <log> <sync_perframe args...>   the server when alive, else the cold `python sync_perframe.py` (same argv, same log)
sync_serve_start() {
  [ "${KS_SYNC_SERVE:-0}" = 1 ] && [ -z "${KS_SYNC_SERVE_DIR:-}" ] || return 0
  export KS_SYNC_SERVE_DIR=$1; mkdir -p $1; rm -f $1/stop $1/alive $1/pid
  ( cd $R && exec $PY scripts/quality/sync_serve.py --spool $1 ) > $1/serve.log 2>&1 & SYNC_SRV=$!   # (exec: its parent = this script; it exits when the script is gone)
  for i in $(seq 100); do [ -s $1/alive ] && return 0; sleep 0.1; done; return 0
}
sync_serve_stop() {
  [ -n "${SYNC_SRV:-}" ] || return 0
  touch $KS_SYNC_SERVE_DIR/stop; for i in $(seq 50); do kill -0 $SYNC_SRV 2>/dev/null || break; sleep 0.1; done; kill $SYNC_SRV 2>/dev/null; SYNC_SRV=""
}
sync_run() {
  local LG=$1; shift; local D=${KS_SYNC_SERVE_DIR:-}
  if [ -n "$D" ] && [ -s $D/alive ] && [ -s $D/pid ] && kill -0 $(cat $D/pid) 2>/dev/null; then
    local id=$(date +%s%N)_$$_$RANDOM; local P=$(cat $D/pid)
    python3 - "$D" "$id" "$R" "$LG" "$@" <<'PY'
import json, os, sys
d, i, cwd, log, *argv = sys.argv[1:]
tmp = os.path.join(d, "jobs", f".{i}.tmp"); json.dump({"argv": argv, "cwd": cwd, "log": os.path.abspath(log)}, open(tmp, "w")); os.replace(tmp, os.path.join(d, "jobs", f"{i}.json"))
PY
    until [ -f $D/out/$id.rc ]; do kill -0 $P 2>/dev/null || break; sleep 0.1; done
    [ -f $D/out/$id.rc ] && return $(cat $D/out/$id.rc)
    rm -f $D/jobs/$id.json; echo "KS WARN sync server died: the cold sync_perframe call" >&2
  fi
  ( cd $R && $PY scripts/quality/sync_perframe.py "$@" ) > $LG 2>&1
}
