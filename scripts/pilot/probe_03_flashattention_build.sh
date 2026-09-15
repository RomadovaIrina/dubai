#!/usr/bin/env bash
# Pilot 0.3 — isolated source-build + runtime probe for flash-attn on the current GPU.
# Does NOT install anything into /venv/dabai. The wheel is built and installed into /tmp.
set -uo pipefail

DUB_ROOT="${DUB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${DUB_ENV:-/venv/dabai}/bin/python"
PILOT="${DUB_ROOT}/reports/pilot"
mkdir -p "$PILOT"

SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.8.3.post1}"
MAX_JOBS="${MAX_JOBS:-4}"
TS="$(date +%Y%m%d_%H%M%S)"
WORK="/tmp/dabai-flashattn-${TS}"
WHEELS="$WORK/wheels"
TARGET="$WORK/site"
LOG="$PILOT/0.3_flashattention_build_${TS}.log"
JSON="$PILOT/0.3_flashattention_build.json"
mkdir -p "$WHEELS" "$TARGET"

exec > >(tee "$LOG") 2>&1

echo "== Pilot 0.3 FlashAttention source build =="
echo "spec: $SPEC"
echo "python: $PY"
"$PY" - <<'PY'
import json, torch
p=torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "torch CUDA", torch.version.cuda)
print("GPU", p.name, "capability", torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())
PY
nvcc --version | tail -n 4 || true

CAP="$($PY - <<'PY'
import torch
m,n=torch.cuda.get_device_capability(0)
print(f"{m}.{n}")
PY
)"
export TORCH_CUDA_ARCH_LIST="$CAP"
export MAX_JOBS
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PIP_NO_CACHE_DIR=1

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS CUDA_HOME=$CUDA_HOME"

BUILD_RC=0
"$PY" -m pip wheel --no-build-isolation --no-deps "$SPEC" -w "$WHEELS" || BUILD_RC=$?
WHEEL="$(find "$WHEELS" -maxdepth 1 -name 'flash_attn-*.whl' -o -name 'flash-attn-*.whl' | head -1)"

RUNTIME_RC=99
RUNTIME_MSG="NOT_REACHED"
if [[ $BUILD_RC -eq 0 && -n "${WHEEL:-}" && -s "$WHEEL" ]]; then
  echo "built wheel: $WHEEL"
  "$PY" -m pip install --no-deps --target "$TARGET" "$WHEEL"
  set +e
  PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import json, time, traceback
import torch
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
    B,L,H,D=1,256,8,64
    q=torch.randn(B,L,H,D,device="cuda",dtype=torch.float16)
    k=torch.randn_like(q); v=torch.randn_like(q)
    torch.cuda.synchronize()
    for _ in range(3):
        y=flash_attn_func(q,k,v,causal=False)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(20):
        y=flash_attn_func(q,k,v,causal=False)
    torch.cuda.synchronize()
    print("runtime PASS", "shape", tuple(y.shape), "ms", (time.perf_counter()-t0)/20*1000)
    if not torch.isfinite(y).all():
        raise RuntimeError("non-finite output")
except Exception:
    traceback.print_exc()
    raise
PY
  RUNTIME_RC=$?
  set -e
  if [[ $RUNTIME_RC -eq 0 ]]; then RUNTIME_MSG="PASS"; else RUNTIME_MSG="FAIL"; fi
fi

"$PY" - "$JSON" "$SPEC" "$CAP" "$BUILD_RC" "$RUNTIME_RC" "$RUNTIME_MSG" "$LOG" "${WHEEL:-}" <<'PY'
import json, pathlib, subprocess, sys, torch, time
out,spec,cap,build_rc,runtime_rc,runtime_msg,log,wheel=sys.argv[1:]
rep={
  "task":"0.3",
  "at":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "spec":spec,
  "gpu":torch.cuda.get_device_name(0),
  "capability":cap,
  "torch":torch.__version__,
  "torch_cuda":torch.version.cuda,
  "nvcc":subprocess.run(["nvcc","--version"],capture_output=True,text=True).stdout.strip(),
  "build":{"status":"PASS" if int(build_rc)==0 else "FAIL", "returncode":int(build_rc), "wheel":wheel or None},
  "runtime":{"status":runtime_msg, "returncode":int(runtime_rc)},
  "closed": cap=="12.0",
  "verdict": ("USABLE" if int(build_rc)==0 and int(runtime_rc)==0 else ("NOT_USABLE_BUILD_FAIL" if int(build_rc)!=0 else "NOT_USABLE_RUNTIME_FAIL")),
  "log":log,
}
pathlib.Path(out).write_text(json.dumps(rep,indent=2,ensure_ascii=False))
print(json.dumps(rep,indent=2,ensure_ascii=False))
PY

echo "-> $JSON"
echo "-> $LOG"
# A build/runtime FAIL is still a valid experimental outcome for 0.3, so return 0 if
# the probe itself completed and produced JSON. The JSON verdict carries PASS/FAIL.
exit 0
