#!/bin/bash
# Reproducible setup of the `dabai` env for the dub pilot.
# Usage:  scripts/setup_dabai.sh            # all stages
#         scripts/setup_dabai.sh 2 3        # only the listed stages
# Stages: 0 env+torch  1 core  2 audio  3 faces  4 llama  5 chatterbox  6 latentsync  7 codeformer  8 kernel  9 freeze
# Idempotent: every stage can be re-run. Logs: logs/setup_<stage>_<ts>.log
set -euo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR=/venv/dabai
PY="${ENV_DIR}/bin/python"
REQ="${DUB_ROOT}/scripts/requirements"
LOGS="${DUB_ROOT}/logs"; mkdir -p "${LOGS}" "${DUB_ROOT}/third_party" "${DUB_ROOT}/models" "${DUB_ROOT}/reports"
TS="$(date +%Y%m%d_%H%M%S)"
PIP="uv pip install --python ${PY} -c ${REQ}/constraints.txt"
export UV_NO_CACHE=1 UV_LINK_MODE=copy

log(){ echo "[$(date +%H:%M:%S)] $*"; }
run_stage(){ local n="$1"; shift; log "=== stage $n: $* ==="; "stage_$n" 2>&1 | tee "${LOGS}/setup_${n}_${TS}.log"; log "=== stage $n done ==="; }

stage_0(){ # env + immutable torch baseline
  if [ ! -x "${PY}" ]; then  # conda `-n` would land wherever this conda keeps envs; we need exactly ${ENV_DIR}
    if command -v conda >/dev/null 2>&1; then conda create -y -p "${ENV_DIR}" python=3.11
    else uv venv --python 3.11 --seed "${ENV_DIR}"; fi   # image without conda (uv provides pip via --seed)
  fi
  "${PY}" --version
  uv pip install --python "${PY}" -r "${REQ}/00-torch.txt"
  "${PY}" "${DUB_ROOT}/scripts/check_torch.py"
}
stage_1(){ $PIP -r "${REQ}/10-core.txt"; }
stage_2(){ $PIP -r "${REQ}/20-audio.txt"; }
stage_3(){ $PIP -r "${REQ}/30-faces.txt"; }
stage_4(){ # prebuilt CPU wheel (AVX2/FMA = EPYC Zen2); fallback: native source build
  $PIP -r "${REQ}/40-llama.txt" || CMAKE_ARGS="-DGGML_NATIVE=ON" CMAKE_BUILD_PARALLEL_LEVEL=16 $PIP --no-binary llama-cpp-python llama-cpp-python==0.3.35
}
stage_5(){ # Chatterbox Multilingual v3 lives only in git master; it hard-pins torch 2.6 -> --no-deps
  local ref="${CHATTERBOX_REF:-master}"
  uv pip install --python "${PY}" --no-deps "git+https://github.com/resemble-ai/chatterbox.git@${ref}"
  uv pip install --python "${PY}" --no-deps s3tokenizer==0.3.0   # its metadata drags pre-commit etc.
  $PIP onnx ml_dtypes protobuf
  $PIP -r "${REQ}/50-chatterbox-deps.txt"
  "${PY}" -c "import chatterbox, importlib.metadata as m; print('chatterbox', m.version('chatterbox-tts'))"
}
stage_6(){ # LatentSync 1.6
  local d="${DUB_ROOT}/third_party/latentsync"
  [ -d "$d/.git" ] || git clone --depth 1 https://github.com/bytedance/LatentSync "$d"
  $PIP -r "${REQ}/60-latentsync.txt"
  $PIP cython setuptools wheel   # must exist BEFORE the no-isolation sdist build
  $PIP --no-build-isolation insightface==0.7.3 || { log "insightface 0.7.3 sdist build failed, falling back to 1.0.1"; $PIP insightface==1.0.1; }
  # onnxruntime (CPU) and onnxruntime-gpu share one import namespace: keep exactly one. GPU build is CUDA 12.8 (<=1.26).
  uv pip uninstall --python "${PY}" onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  uv pip install --python "${PY}" --no-deps onnxruntime-gpu==1.26.0
  if ! "${PY}" "${DUB_ROOT}/scripts/check_onnxruntime.py"; then
    log "onnxruntime-gpu CUDA provider failed on sm_120 -> falling back to CPU onnxruntime"
    uv pip uninstall --python "${PY}" onnxruntime-gpu >/dev/null 2>&1 || true
    $PIP "onnxruntime>=1.16,<2"
  fi
  # insightface 0.7.3 -> albumentations -> opencv-python-headless, which shadows the pinned opencv-python 4.9.0.80 (two cv2 builds)
  uv pip uninstall --python "${PY}" opencv-python-headless >/dev/null 2>&1 || true
  uv pip install --python "${PY}" --reinstall --no-deps opencv-python==4.9.0.80   # both dists share cv2/, uninstall leaves it broken
  "${PY}" -c "import cv2; print('cv2', cv2.__version__)"
  mkdir -p "$d/checkpoints"
}
stage_7(){ # CodeFormer: vendored basicsr+facelib, run from clone root via PYTHONPATH
  local d="${DUB_ROOT}/third_party/CodeFormer"
  [ -d "$d/.git" ] || git clone --depth 1 https://github.com/sczhou/CodeFormer "$d"
  $PIP -r "${REQ}/70-codeformer.txt"
  # `python basicsr/setup.py develop` is deprecated; it only generated basicsr/version.py -> write it directly
  ( cd "$d" && printf "# GENERATED VERSION FILE (setup_dabai.sh stage 7)\n__version__ = '%s'\n__gitsha__ = '%s'\nversion_info = (%s)\n" \
      "$(cat basicsr/VERSION)" "$(git rev-parse HEAD)" "$(tr . , < basicsr/VERSION | sed 's/,/, /g')" > basicsr/version.py )
  ( cd "$d" && PYTHONPATH=. "${PY}" -c "import basicsr, facelib; print('basicsr', basicsr.__version__)" )
}
stage_8(){ # Jupyter kernel for the notebook
  "${PY}" -m ipykernel install --name dabai --display-name "Python (dabai)"
  grep -q '^ACTIVE_VENV=' /workspace/.env 2>/dev/null || echo 'ACTIVE_VENV=dabai' >> /workspace/.env
  grep -q '^HF_HOME=' /workspace/.env 2>/dev/null || echo "HF_HOME=${DUB_ROOT}/models/hf" >> /workspace/.env
}
stage_9(){ # freeze
  uv pip freeze --python "${PY}" > "${DUB_ROOT}/reports/pip-freeze.lock.txt"
  { for r in latentsync CodeFormer; do echo "$r $(git -C "${DUB_ROOT}/third_party/$r" rev-parse HEAD)"; done; } > "${DUB_ROOT}/reports/third_party-commits.txt"
  "${PY}" -m pip check > "${DUB_ROOT}/reports/pip-check.txt" 2>&1 || true
  cat "${DUB_ROOT}/reports/pip-check.txt"
}

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(0 1 2 3 4 5 6 7 8 9)
for s in "${STAGES[@]}"; do run_stage "$s"; done
log "all requested stages finished"
