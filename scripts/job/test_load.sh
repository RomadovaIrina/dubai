#!/bin/bash
# r15 T, TZ 12.4 (the worker side): the HTTP service under a burst of jobs and a hard restart, with a stand-in pipeline (no GPU, ~1.5 min).
#   1. 10 jobs posted at once -> all SUCCESS, FIFO, never two pipelines at the same time (one GPU = one job), none lost
#   2. kill -9 of the service while a job runs and 4 wait -> after the restart: the running one = E_WORKER_RESTART and its orphan pipeline
#      is killed (it would hold the GPU under the next job), the 4 queued ones run to SUCCESS in submit order
#     API_PY=<venv with fastapi + uvicorn>/bin/python bash scripts/job/test_load.sh
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); J=$R/scripts/job; T=${TMPDIR:-/tmp}/dabai_load_test.$$; mkdir -p $T/jobs; PY=${API_PY:-python3}; PASS=0; FAIL=0
ok(){ if eval "$2"; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi; }
FR=$T/fake; mkdir -p $FR/scripts/r9; cat > $FR/scripts/r9/e2e.sh <<'EOF'
#!/bin/bash
mkdir -p "$2/work"; echo "start $ID $(date +%s.%N)" >> $FAKE_LOG
printf "precheck\t%s\n" "$(date +%s.%N)" >> "$2/work/stages.tsv"; sleep 0.5; printf "audio\t%s\n" "$(date +%s.%N)" >> "$2/work/stages.tsv"; sleep $FAKE_S
cp "$1" "$2/${ID}_DabAI_r9_spleeter.mp4"; echo '{"status": "SUCCESS", "flags": [], "errors": [], "timing": {}, "outputs": {}}' > "$2/manifest.json"
printf "end\t%s\n" "$(date +%s.%N)" >> "$2/work/stages.tsv"; echo "end $ID $(date +%s.%N)" >> $FAKE_LOG
EOF
ffmpeg -loglevel error -f lavfi -i testsrc2=s=160x120:r=25 -f lavfi -i sine -t 2 -pix_fmt yuv420p -c:v libx264 -c:a aac $T/in.mp4
P=18767; H=http://127.0.0.1:$P
start_api(){ ( cd $J; exec env DABAI_ROOT=$FR DABAI_JOBS_DIR=$T/jobs DABAI_MIN_FREE_GB=1 FAKE_LOG=$T/runs.log FAKE_S=$1 $PY -m uvicorn job_api:app --host 127.0.0.1 --port $P > $T/api_$2.log 2>&1 ) & AP=$!
  for i in $(seq 40); do curl -s -o /dev/null $H/version && break; sleep 0.5; done; }
post(){ curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' -d '{"input": "'$T/in.mp4'", "job_id": "'$1'"}' $H/jobs; }
stat(){ curl -s $H/jobs/$1 | python3 -c "import json,sys; s=json.load(sys.stdin); print(s['status'], (s.get('error') or {}).get('code'))"; }
# 1. burst
start_api 1 a; codes=""; for i in $(seq -w 1 10); do codes="$codes$(post b$i) "; done
ok "10 posts accepted (202)" "[ \"\$(echo $codes | tr ' ' '\n' | sort -u)\" = 202 ]"
for i in $(seq 120); do n=$(curl -s $H/jobs | python3 -c "import json,sys; print(sum(1 for j in json.load(sys.stdin)['jobs'] if j['status'] == 'SUCCESS'))"); [ "$n" = 10 ] && break; sleep 1; done
ok "all 10 SUCCESS" "[ \"$n\" = 10 ]"
python3 - $T/runs.log > $T/burst.txt <<'EOF'
import sys
ev = [l.split() for l in open(sys.argv[1])]; st = {i: float(t) for k, i, t in ev if k == "start"}; en = {i: float(t) for k, i, t in ev if k == "end"}
order = sorted(st, key=st.get); iv = sorted((st[i], en[i]) for i in st)
over = max(sum(1 for a, b in iv if a <= s < b) for s, _ in iv)
print("order", " ".join(order)); print("max_concurrent", over); print("fifo", order == sorted(order))
EOF
ok "never two pipelines at once" "grep -q 'max_concurrent 1' $T/burst.txt"
ok "FIFO order" "grep -q 'fifo True' $T/burst.txt"
# 2. hard restart while r1 runs and r2..r5 wait
kill $AP; wait $AP 2>/dev/null; : > $T/runs.log; start_api 6.5 b
for i in 1 2 3 4 5; do post r$i > /dev/null; done
for i in $(seq 40); do grep -q "start r1" $T/runs.log && break; sleep 0.25; done; sleep 1
pkill -9 -f "uvicorn job_api:app --host 127.0.0.1 --port $P"; sleep 1
ok "the r1 pipeline outlived the dead service (an orphan)" "pgrep -f '^sleep 6[.]5$' > /dev/null"
start_api 1 c; sleep 1
ok "after restart: r1 = E_WORKER_RESTART" "[ \"\$(stat r1)\" = 'ERROR E_WORKER_RESTART' ]"
ok "after restart: the orphan pipeline was killed" "! pgrep -f '^sleep 6[.]5$' > /dev/null"
for i in $(seq 60); do [ "$(stat r5)" = "SUCCESS None" ] && break; sleep 1; done
ok "after restart: r2..r5 resumed -> SUCCESS" "for j in r2 r3 r4 r5; do [ \"\$(stat \$j)\" = 'SUCCESS None' ] || exit 1; done"
ok "after restart: resumed in submit order (r1 never finished)" "[ \"\$(grep start $T/runs.log | awk '{print \$2}' | tr '\n' ' ')\" = 'r1 r2 r3 r4 r5 ' ] && ! grep -q 'end r1' $T/runs.log"
kill $AP 2>/dev/null; wait $AP 2>/dev/null; pkill -f '^sleep 6[.]5$' 2>/dev/null
echo "LOAD_TESTS pass=$PASS fail=$FAIL ($T)"; [ $FAIL = 0 ]
