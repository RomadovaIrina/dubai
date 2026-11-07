# r16 T: the shared GPU lock WITHOUT fd inheritance (source this file). Audit 08.11: `exec 8>lock; flock 8` in e2e.sh let every later child
# inherit the locked fd (a killed parent left an orphan holding gpu.lock for hours) and a nested `flock gpu.lock` deadlocked against fd 8.
# Here a small HOLDER process owns the lock on its own fd; the caller's children never see it. The holder exits (= unlocks) when
#   - gpu_release is called (by the owner, or by a child that was handed GPU_RDY: ks_face.sh releases before its CPU post), or
#   - its owner process dies (checked every second), so a killed job never leaves the lock behind.
#   gpu_acquire [owner_pid]   wait for the lock (owner default: the calling shell); exports GPU_RDY (the hand-off token)
#   gpu_release               unlock (idempotent; works from any process that has GPU_RDY)
#   gpu_held                  true while the lock is held for this job
#   gpu_run CMD...            CMD under the lock: directly when it is already held (no nested flock), else `flock GPU_LOCK CMD`
GPU_LOCK=${GPU_LOCK:-${KS_GPU_LOCK:-/workspace/dub/gpu.lock}}
gpu_held() { [ -n "${GPU_RDY:-}" ] && [ -e "$GPU_RDY" ] && [ ! -e "$GPU_RDY.rel" ]; }
gpu_acquire() {
  gpu_held && return 0
  local owner=${1:-$$}; GPU_RDY=$(mktemp -u /tmp/gpulock.XXXXXXXX)
  ( exec 0</dev/null >/dev/null 2>&1
    exec flock "$GPU_LOCK" bash -c 'touch "$0"; while kill -0 "$1" 2>/dev/null && [ ! -e "$0.rel" ]; do sleep 1; done; rm -f "$0" "$0.rel"' "$GPU_RDY" "$owner" ) &
  local h=$!
  while [ ! -e "$GPU_RDY" ]; do kill -0 $h 2>/dev/null || { GPU_RDY=""; return 1; }; sleep 0.5; done
  export GPU_RDY
}
gpu_release() {
  [ -n "${GPU_RDY:-}" ] || return 0
  if [ -e "$GPU_RDY" ]; then touch "$GPU_RDY.rel"; for _ in $(seq 40); do [ -e "$GPU_RDY" ] || break; sleep 0.1; done; fi
  rm -f "$GPU_RDY.rel"; GPU_RDY=""
}
gpu_run() { if gpu_held; then "$@"; else flock "$GPU_LOCK" "$@"; fi; }
