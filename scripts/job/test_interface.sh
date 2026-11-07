#!/bin/bash
# r15 T: CPU-only tests of the job interface v2 (no GPU, ~1 min). A stand-in pipeline (DABAI_ROOT=<tmp> with a sleeping scripts/r9/e2e.sh)
# covers cancel / timeout / worker restart; the real pipeline covers a no-audio input (precheck only). Then the HTTP API end to end.
#     API_PY=<venv with fastapi + uvicorn>/bin/python bash scripts/job/test_interface.sh [no_audio.mp4]
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); J=$R/scripts/job; T=${TMPDIR:-/tmp}/dabai_iface_test.$$; mkdir -p $T; PY=${API_PY:-python3}; PASS=0; FAIL=0
NOA=${1:-$T/no_audio.mp4}
[ -f "$NOA" ] || ffmpeg -loglevel error -f lavfi -i testsrc2=s=320x240:r=25 -t 3 -pix_fmt yuv420p -c:v libx264 $NOA
ok(){ if eval "$2"; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi; }
st(){ python3 -c "import json,sys; s=json.load(open('$1/status.json')); print(s['status'], (s.get('error') or {}).get('code'))"; }
# the stand-in pipeline: writes stages like e2e.sh, then sleeps
FR=$T/fake; mkdir -p $FR/scripts/r9; cat > $FR/scripts/r9/e2e.sh <<'EOF'
#!/bin/bash
mkdir -p "$2/work"; printf "precheck\t%s\n" "$(date +%s.%N)" >> "$2/work/stages.tsv"; sleep 1; printf "audio\t%s\n" "$(date +%s.%N)" >> "$2/work/stages.tsv"
sleep 317 & wait
EOF
cp $NOA $T/in.mp4
# 1. validation
python3 $J/dabai_job.py --validate --input $T/in.mp4 --out $T/v > /dev/null; ok "validate a minimal CLI job" "[ $? = 0 ]"
echo '{"input": "x.mp4", "out_dir": "'$T'/bad", "mode": "turbo", "foo": 1}' > $T/bad.json; python3 $J/dabai_job.py $T/bad.json > /dev/null; e=$?
ok "bad request -> exit 2, E_BAD_REQUEST" "[ $e = 2 ] && [ \"\$(st $T/bad)\" = 'ERROR E_BAD_REQUEST' ]"
python3 $J/dabai_job.py --input $T/missing.mp4 --out $T/miss > /dev/null; ok "missing input -> E_INPUT_NOT_FOUND" "[ \"\$(st $T/miss)\" = 'ERROR E_INPUT_NOT_FOUND' ]"
# 2. timeout + cancel (stand-in pipeline): the process group dies, the status says so
echo '{"input": "'$T'/in.mp4", "out_dir": "'$T'/to", "job_id": "t_to", "timeout_s": 6}' > $T/to.json
DABAI_ROOT=$FR python3 $J/dabai_job.py $T/to.json > /dev/null; ok "timeout -> TIMEOUT E_TIMEOUT" "[ \"\$(st $T/to)\" = 'TIMEOUT E_TIMEOUT' ]"
ok "timeout killed the pipeline (no sleep left)" "! pgrep -f 'sleep 317' > /dev/null"
DABAI_ROOT=$FR python3 - <<EOF
import sys, threading, time, json; sys.path.insert(0, "$J"); import dabai_job as dj
ev = threading.Event(); threading.Timer(5, ev.set).start(); seen = []
st = dj.run_job({"input": "$T/in.mp4", "out_dir": "$T/cx", "job_id": "t_cx"}, on_event=seen.append, cancel=ev)
json.dump({"status": st["status"], "code": (st["error"] or {}).get("code"), "steps": [e.get("step") for e in seen if e["type"] in ("step", "progress")]}, open("$T/cx.r", "w"))
EOF
ok "cancel -> ERROR E_CANCELLED" "grep -q '\"code\": \"E_CANCELLED\"' $T/cx.r"
ok "progress events saw precheck + audio" "grep -q precheck $T/cx.r && grep -q audio $T/cx.r"
ok "events.jsonl ends with done" "tail -1 $T/cx/events.jsonl | grep -q '\"type\": \"done\"'"
# 3. the real pipeline on a no-audio input (precheck only, CPU)
python3 $J/dabai_job.py --input $NOA --out $T/noa --job-id t_noa > /dev/null; ok "no audio -> ERROR E_INPUT_UNREADABLE (real e2e.sh)" "[ \"\$(st $T/noa)\" = 'ERROR E_INPUT_UNREADABLE' ]"
ok "metrics.json written" "[ -s $T/noa/metrics.json ]"
# 4. HTTP API
if $PY -c "import fastapi, uvicorn" 2>/dev/null; then
  JD=$T/jobs; mkdir -p $JD/stale; echo '{"schema": "dabai-job/2", "job_id": "stale", "status": "RUNNING", "step": "audio", "progress_pct": 40}' > $JD/stale/status.json
  P=18766; H="Authorization: Bearer test-token-123"
  ( cd $J; exec env DABAI_JOBS_DIR=$JD DABAI_API_TOKEN=test-token-123 DABAI_MIN_FREE_GB=1 $PY -m uvicorn job_api:app --host 127.0.0.1 --port $P > $T/api.log 2>&1 ) & AP=$!
  for i in $(seq 40); do curl -s -o /dev/null http://127.0.0.1:$P/health && break; sleep 0.5; done
  c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$P/jobs); ok "no token -> 401" "[ $c = 401 ]"
  c=$(curl -s -o $T/h.json -w '%{http_code}' http://127.0.0.1:$P/health); ok "health answers without a token (200 or 503)" "[ $c = 200 ] || [ $c = 503 ]"
  ok "restart: a RUNNING job left by a dead worker -> E_WORKER_RESTART" "curl -s -H '$H' http://127.0.0.1:$P/jobs/stale | grep -q E_WORKER_RESTART"
  c=$(curl -s -o $T/p.json -w '%{http_code}' -H "$H" -H 'content-type: application/json' -d '{"input": "'$NOA'", "job_id": "a1"}' http://127.0.0.1:$P/jobs)
  ok "POST /jobs -> 202 QUEUED" "[ $c = 202 ] && grep -q QUEUED $T/p.json"
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" -H 'content-type: application/json' -d '{"input": "'$NOA'", "job_id": "a1"}' http://127.0.0.1:$P/jobs); ok "same job_id again -> 409" "[ $c = 409 ]"
  c=$(curl -s -o $T/p2.json -w '%{http_code}' -H "$H" -H 'content-type: application/json' -d '{"input": "'$NOA'", "mode": "turbo"}' http://127.0.0.1:$P/jobs); ok "invalid job -> 422 E_BAD_REQUEST" "[ $c = 422 ] && grep -q E_BAD_REQUEST $T/p2.json"
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" -H 'content-type: application/json' -d '{"input": "'$NOA'", "out_dir": "/etc"}' http://127.0.0.1:$P/jobs); ok "client out_dir refused -> 422" "[ $c = 422 ]"
  curl -s -o /dev/null -H "$H" -H 'content-type: application/json' -d '{"input": "'$NOA'", "job_id": "a2"}' http://127.0.0.1:$P/jobs
  curl -s -o /dev/null -X DELETE -H "$H" http://127.0.0.1:$P/jobs/a2
  for i in $(seq 60); do curl -s -H "$H" http://127.0.0.1:$P/jobs/a1 | grep -q '"status": "ERROR"\|"status":"ERROR"' && break; sleep 1; done
  ok "a1 ran -> ERROR E_INPUT_UNREADABLE" "curl -s -H '$H' http://127.0.0.1:$P/jobs/a1 | grep -q E_INPUT_UNREADABLE"
  ok "a2 cancelled while queued -> E_CANCELLED" "curl -s -H '$H' http://127.0.0.1:$P/jobs/a2 | grep -q E_CANCELLED"
  ok "GET /jobs/a1/events?after=1 -> later events" "curl -s -H '$H' 'http://127.0.0.1:$P/jobs/a1/events?after=1' | grep -q '\"done\"'"
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" http://127.0.0.1:$P/jobs/a1/files/metrics); ok "GET files/metrics -> 200" "[ $c = 200 ]"
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" http://127.0.0.1:$P/jobs/a1/files/video); ok "GET files/video of an ERROR job -> 404" "[ $c = 404 ]"
  c=$(curl -s -o /dev/null -w '%{http_code}' -H "$H" http://127.0.0.1:$P/jobs/..%2Fetc); ok "path traversal job id -> 404" "[ $c = 404 ]"
  ok "GET /version + /schema/job" "curl -s -H '$H' http://127.0.0.1:$P/version | grep -q dabai-job/2 && curl -s -H '$H' http://127.0.0.1:$P/schema/job | grep -q additionalProperties"
  ok "the token never appears in the log / job files" "! grep -rq test-token-123 $T/api.log $JD"
  kill $AP 2>/dev/null; wait $AP 2>/dev/null
else echo "SKIP the HTTP tests (no fastapi / uvicorn in $PY)"; fi
pkill -f "[s]leep 317" 2>/dev/null
echo "INTERFACE_TESTS pass=$PASS fail=$FAIL ($T)"
[ $FAIL = 0 ]
