#!/bin/bash
# r17 resident worker, the KeySync half (scripts/keysync/ks_serve.py). Sourced by ks_face.sh / ks_shots.sh; run directly to start the daemon.
#   bash ks_serve.sh start <spool>     load the KeySync-fast models ONCE (TRT plans built on the first start if missing), park them off the GPU, serve
#   bash ks_serve.sh stop  <spool>
#   (sourced) ks_fast_launch <jobs> <out> <log>   -> KSP = a pid alive while the run is active (wait $KSP = its result, kill $KSP = abort),
#             KS_SERVED=1 (the resident daemon took it) | 0 (a cold `ks_fast.py --follow` as before). KS_SERVE_DIR unset / dead / other settings -> cold.
# A request is served only when its settings signature (ks_sig: the ks_fast args, the KS_* env that patches the models at start, the kit code)
# equals the daemon's, so a served job runs exactly the code + settings a cold one would.
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; KS=${KS_ROOT:-/workspace/dub/keysync}

ks_kit_md5() {   # the KeySync python the run imports (a daemon started from other code never serves this checkout)
  cat $R/scripts/keysync/ks_fast.py $R/scripts/keysync/ks_serve.py $R/scripts/keysync/r15/*.py $R/scripts/keysync/r16/ks_gate.py $R/scripts/keysync/r16/ks_noise.py 2>/dev/null | md5sum | cut -c1-12; }
ks_sig() {   # ks_sig <engine dir> <lora> <lmk cache> <gate>
  echo "cfg=_s8 vae-half compile=unet trt=$1 lora=$2 steps=${KS_INTERP_STEPS:-2} vae-batch=8 vae-compile dc=2:0 lmk=$3 gate=$4 ind=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} \
noise=${KS_NOISE:-} kfar=${KS_KF_AR:-} kfrr=${KS_KF_RR:-} iguide=${KS_IGUIDE:-} ichunk=${KS_ICHUNK:-} kit=$(ks_kit_md5)" | md5sum | cut -c1-16; }

ks_fast_cmd() {   # ks_fast_cmd <kit dir> <extra args...>: the ONE ks_fast command line (cold run and daemon alike)
  local K=$1; shift
  cd $KS/code && . ../venv/bin/activate && env KS_DC=2:0 KS_LMK_CACHE=$LMK TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} ${GATE:+KS_GATE=$GATE} \
    python $K/ks_fast.py "$@" --cfg _s8 --vae-half --compile unet --trt $ENG --interp-lora $LORA --interp-steps ${KS_INTERP_STEPS:-2} --vae-batch 8 --vae-compile
}

ks_serve_live() {   # ks_serve_live <spool>: 0 when a daemon is up, loaded (ready) and beating
  local D=$1; [ -n "$D" ] && [ -s $D/ready ] && [ -s $D/alive ] && [ -s $D/pid ] && kill -0 $(cat $D/pid) 2>/dev/null || return 1
  python3 -c "import sys, time; sys.exit(0 if time.time() - float(open('$D/alive').read()) < 5 else 1)" 2>/dev/null
}

ks_fast_launch() {   # needs ENG LORA LMK GATE KT (as set by ks_face.sh / ks_shots.sh)
  local FJ=$1 OUT=$2 LOG=$3 D=${KS_SERVE_DIR:-}; KS_SERVED=0
  if ks_serve_live "$D"; then
    local SIG; SIG=$(ks_sig $ENG $LORA $LMK "$GATE")
    if python3 -c "import json, sys; sys.exit(0 if json.load(open('$D/ready')).get('sig') == '$SIG' else 1)" 2>/dev/null; then
      local id=$(date +%s%N)_$$; : > $LOG
      python3 - "$D/req" "$id" "$FJ" "$OUT" "$LOG" "$SIG" "$$" <<'PY'
import json, os, sys
d, i, fj, out, log, sig, owner = sys.argv[1:]
env = {k: os.environ[k] for k in ("KS_FAULT",) if k in os.environ}   # per-request env (the model-patching KS_* are in the signature)
tmp = os.path.join(d, f".{i}.tmp"); json.dump({"jobs": fj, "out": out, "log": log, "sig": sig, "env": env, "owner": int(owner)}, open(tmp, "w")); os.replace(tmp, os.path.join(d, f"{i}.json"))
PY
      local n=0; until grep -q -E "^KS_SERVE_(ACCEPT|REJECT)" $LOG 2>/dev/null; do sleep 0.2; n=$((n + 1)); [ $n -lt 150 ] && kill -0 $(cat $D/pid) 2>/dev/null || break; done
      if grep -q "^KS_SERVE_ACCEPT" $LOG; then
        local DP=$(cat $D/pid)
        ( trap 'echo END >> '"$FJ"'; exit 1' TERM   # abort: the daemon ends the request at its next read
          while kill -0 $DP 2>/dev/null; do grep -q "^KS_FAST_DONE" $LOG && exit 0; grep -q "^KS_SERVE_ERROR" $LOG && exit 1; sleep 0.3; done; exit 1 ) & KSP=$!
        KS_SERVED=1; echo "KS_SERVE resident ks_fast $(grep -m1 '^KS_SERVE_ACCEPT' $LOG | cut -c18-200)"; return 0
      fi
      rm -f $D/req/$id.json; echo "KS_SERVE not taken ($(grep -m1 '^KS_SERVE_REJECT' $LOG | cut -c1-160)) -> cold ks_fast"
    else echo "KS_SERVE resident ks_fast has other settings -> cold ks_fast"; fi
  fi
  ( ks_fast_cmd $KT --follow --jobs $FJ --out $OUT ) > $LOG 2>&1 & KSP=$!
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then   # run directly: start / stop the daemon
  cmd=${1:?start|stop <spool>}; D=${2:?spool dir}; mkdir -p $D
  case $cmd in
    stop) touch $D/stop; [ -s $D/pid ] && { for i in $(seq 60); do kill -0 $(cat $D/pid) 2>/dev/null || break; sleep 0.5; done; kill $(cat $D/pid) 2>/dev/null; }; exit 0;;
    start)
      rm -f $D/stop $D/ready $D/alive $D/pid; rm -rf $D/req; mkdir -p $D/req $D/kit
      for f in $R/scripts/keysync/*.py $R/scripts/keysync/r15/*.py $R/scripts/keysync/r16/*.py; do cp $f $D/kit/; done
      LORA=${KS_LORA:-$KS/student/lora_pick.pt}; LMK=${KS_LMK_CACHE:-$KS/lmk}; mkdir -p $LMK; [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME
      GATE="${KS_GATE-thr=0.12,cap=0.25,hi=8,lo=2}"
      ENG=$(KS_GPU_LOCK=/dev/null bash $R/scripts/keysync/r16/ks_engines.sh 2> $D/engines.log | tail -1); [ -d "$ENG" ] || { tail -5 $D/engines.log; exit 1; }
      export KS_SERVE_SIG=$(ks_sig $ENG $LORA $LMK "$GATE")
      echo "KS_SERVE start sig $KS_SERVE_SIG engines $ENG"
      ( ks_fast_cmd $D/kit --serve $D ) >> $D/serve.log 2>&1 &
      for i in $(seq 1800); do [ -s $D/ready ] && { cat $D/ready; echo; exit 0; }; kill -0 $! 2>/dev/null || { tail -20 $D/serve.log; exit 1; }; sleep 1; done
      echo "KS_SERVE: not ready after 30 min"; exit 1;;
  esac
fi
