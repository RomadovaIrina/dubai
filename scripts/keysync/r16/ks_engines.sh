#!/bin/bash
# r16 T: the KeySync-fast TensorRT engines, built on first use into a cache keyed by GPU + driver + TensorRT (plans are not portable; ~2 min each on a 5090).
#   bash ks_engines.sh   -> prints the engine dir (for ks_fast.py --trt) on the LAST line; exit 1 if a build / verify failed.
# KS_ROOT (/workspace/dub/keysync)  KS_TRT_CACHE ($KS_ROOT/trt/cache)  KS_TRT_CAP ($KS_ROOT/trt/cap: kf.pt + interp.pt, the captured prod UNet inputs)
# KS_LORA (the LCM student, lora_meta.json next to it)  KS_TRT_MIN_DB (56: TRT_VERIFY PSNR floor)  KS_GPU_LOCK (/workspace/dub/gpu.lock)
set -o pipefail
KS=${KS_ROOT:-/workspace/dub/keysync}; CAP=${KS_TRT_CAP:-$KS/trt/cap}; LORA=${KS_LORA:-$KS/student/lora_pick.pt}; LOCK=${KS_GPU_LOCK:-/workspace/dub/gpu.lock}
. $KS/venv/bin/activate
KEY=$(python - <<'PY'
import re, subprocess, tensorrt
g = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip()
print(re.sub(r"[^A-Za-z0-9.]+", "_", f"{g}_trt{tensorrt.__version__}"))
PY
)
D=${KS_TRT_CACHE:-$KS/trt/cache}/$KEY; mkdir -p $D
ok() { python3 - "$1" "${KS_TRT_MIN_DB:-56}" <<'PY'
import json, sys
try: v = json.load(open(sys.argv[1] + "/verify.json"))
except Exception: sys.exit(1)
db = [x[0] for k, x in v.items() if k.startswith(("full_s", "shallow_s")) and isinstance(x, list)]
sys.exit(0 if db and min(db) >= float(sys.argv[2]) else 1)
PY
}
[ -s $LORA ] && [ -s $(dirname $LORA)/lora_meta.json ] || { echo "KS_ENGINES_FAIL no student LoRA / lora_meta.json at $LORA" >&2; exit 1; }
for w in kf interp; do
  [ -s $CAP/$w.pt ] || { echo "KS_ENGINES_FAIL no capture $CAP/$w.pt" >&2; exit 1; }
done
cd $KS/code
exec 9>$D/.build.lock; flock 9   # two workers on one cache: one builds, the other waits and reuses
if ! { [ -s $D/eng/kf/full.plan ] && ok $D/eng/kf; }; then
  echo "== KS engines: building keyframes into $D/eng (one-time per GPU / driver / TensorRT)" >&2; t0=$(date +%s)
  flock $LOCK python ../ks_trt.py --cap $CAP --which kf --out $D/eng >&2 || { echo "KS_ENGINES_FAIL kf build" >&2; exit 1; }
  ok $D/eng/kf || { echo "KS_ENGINES_FAIL kf verify" >&2; exit 1; }; echo "== KS engines: kf $(( $(date +%s) - t0 )) s" >&2
fi
if ! { [ -s $D/eng_s/interp/full.plan ] && ok $D/eng_s/interp; }; then
  echo "== KS engines: building the student interpolation into $D/eng_s" >&2; t0=$(date +%s)
  flock $LOCK python ../ks_trt.py --cap $CAP --which interp --out $D/eng_s --lora $LORA --no-dc >&2 || { echo "KS_ENGINES_FAIL interp build" >&2; exit 1; }
  ok $D/eng_s/interp || { echo "KS_ENGINES_FAIL interp verify" >&2; exit 1; }; echo "== KS engines: interp $(( $(date +%s) - t0 )) s" >&2
fi
ln -sfn ../eng/kf $D/eng_s/kf
echo $D/eng_s
