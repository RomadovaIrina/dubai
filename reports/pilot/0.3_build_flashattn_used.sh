#!/usr/bin/env bash
# Pilot 0.3: isolated source build of flash-attn for sm_120. Uses a throwaway venv that only *reads*
# /venv/dabai site-packages via a .pth file; nothing is installed into /venv/dabai.
set -uo pipefail
S=/tmp/claude-0/-workspace-dub/f2e597f3-2dc8-4b47-acf0-2a33a8049f3e/scratchpad
FA=$S/fa-venv; PY=$FA/bin/python
DUB=/workspace/dub; PILOT=$DUB/reports/pilot
SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.8.3.post1}"
WHEELS=$S/fa-wheels; mkdir -p "$WHEELS" "$PILOT"
export CUDA_HOME=/usr/local/cuda PATH=$FA/bin:/usr/local/cuda/bin:$PATH
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS="${MAX_JOBS:-8}" PIP_NO_CACHE_DIR=1 TMPDIR=$S/tmp; mkdir -p "$TMPDIR"
echo "== Pilot 0.3 flash-attn source build  $(date -Iseconds)"
echo "spec=$SPEC python=$PY MAX_JOBS=$MAX_JOBS FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2; gcc --version | head -1
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'cap',torch.cuda.get_device_capability(0),'cxx11abi',torch._C._GLIBCXX_USE_CXX11_ABI)"
BEFORE=$("$PY" -c "import torch;print(torch.__version__)")
t0=$(date +%s)
"$PY" -m pip wheel -v --no-build-isolation --no-deps "$SPEC" -w "$WHEELS"; BUILD_RC=$?
echo "== build rc=$BUILD_RC  elapsed=$(( $(date +%s)-t0 ))s"
AFTER=$("$PY" -c "import torch;print(torch.__version__)")
echo "torch before=$BEFORE after=$AFTER"
WHEEL=$(ls -t "$WHEELS"/flash_attn-*.whl 2>/dev/null | head -1)
RUNTIME=NOT_REACHED
if [ $BUILD_RC -eq 0 ] && [ -s "$WHEEL" ]; then
  echo "wheel: $WHEEL ($(du -h "$WHEEL" | cut -f1))"
  "$PY" -m pip install --no-deps "$WHEEL" && "$PY" "$S/fa_runtime_test.py"; RC=$?
  RUNTIME=$([ $RC -eq 0 ] && echo PASS || echo FAIL)
fi
echo "== RESULT build=$([ $BUILD_RC -eq 0 ] && echo PASS || echo FAIL) runtime=$RUNTIME torch_after=$AFTER"
