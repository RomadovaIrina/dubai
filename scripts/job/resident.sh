#!/bin/bash
# r17 resident worker (Irina 11.11: models stay loaded between jobs; TZ 0.4 / 0.10): the two resident halves of ONE GPU worker.
#   bash scripts/job/resident.sh start [dir]    KeySync daemon (scripts/keysync/r16/ks_serve.sh: loads, builds the TRT plans if missing, parks) and
#                                               then the audio worker (scripts/r9/audio_worker.py, R9_WORKER_PARK=1); writes <dir>/ready (timings)
#   bash scripts/job/resident.sh stop  [dir]
#   bash scripts/job/resident.sh status [dir]   exit 0 when both are up
# VRAM: never both on the GPU. The audio worker parks at the end of its stage (Qwen freed, the rest on the CPU); the KeySync daemon unparks
# for its request and parks at END. Jobs are serial on the worker (the API runs one at a time), so the stages never overlap.
# dabai_job.py passes R9_WORKER_SPOOL / KS_SERVE_DIR to the pipeline when <dir> is up; a dead or busy half -> the job runs that half cold.
set -u
cd "$(dirname "$0")/../.."; R=$(pwd); D=${2:-${DABAI_RESIDENT_DIR:-/data/resident}}; mkdir -p $D
now() { date +%s.%N; }
case "${1:?start|stop|status}" in
  start)
    rm -f $D/ready; t0=$(now)
    bash scripts/keysync/r16/ks_serve.sh start $D/ks > $D/ks_start.log 2>&1 || { echo "RESIDENT: KeySync daemon failed (see $D/ks_start.log, $D/ks/serve.log)"; tail -5 $D/ks_start.log; exit 1; }
    t1=$(now)
    R9_WORKER_PARK=1 bash scripts/r9/audio_worker.sh start $D/audio > $D/audio_start.log 2>&1 || { echo "RESIDENT: audio worker failed (see $D/audio/worker.log)"; exit 1; }
    t2=$(now)
    python3 - $D $t0 $t1 $t2 <<'PY'
import json, sys
d, t0, t1, t2 = sys.argv[1], *map(float, sys.argv[2:])
ks = json.load(open(f"{d}/ks/ready"))
r = {"ks_ready_s": round(t1 - t0, 1), "audio_up_s": round(t2 - t1, 1), "total_s": round(t2 - t0, 1), "ks": ks,
     "note": "the audio models load in the worker's first job (Qwen / Whisper / pyannote / Chatterbox), then stay (parked between jobs)"}
json.dump(r, open(f"{d}/ready", "w")); print("RESIDENT_READY " + json.dumps(r))
PY
    ;;
  stop) bash scripts/r9/audio_worker.sh stop $D/audio; bash scripts/keysync/r16/ks_serve.sh stop $D/ks; rm -f $D/ready;;
  status)
    . scripts/keysync/r16/ks_serve.sh; ok=0
    ks_serve_live $D/ks && echo "ks: up $(cat $D/ks/ready)" || { echo "ks: down"; ok=1; }
    python3 -c "import sys, time; sys.exit(0 if time.time() - float(open('$D/audio/alive').read()) < 5 else 1)" 2>/dev/null && echo "audio: up" || { echo "audio: down"; ok=1; }
    exit $ok;;
esac
