#!/bin/bash
# Clean background-bed separator (2026-10-17, Agent C r2): Deezer Spleeter 2stems in its own Python 3.10 / TensorFlow 2.9 CPU venv (the
# dabai env stays untouched) + the 2stems weights (GitHub release asset of the MIT repo, sha256 = the release's checksum.json).
# Used by scripts/pilot/background_bed.py (separator "spleeter", the default of run_clean_pipeline_05.py --bed-separator).
set -euo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DUB_SPLEETER_VENV:-/workspace/dub/venv-spleeter}"; M="${DUB_ROOT}/models/spleeter/pretrained_models"
SHA=f3a90b39dd2874269e8b05a48a86745df897b848c61f3958efc80a39152bd692
[ -x "$VENV/bin/python" ] || uv venv --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" "spleeter==2.3.2" "tensorflow==2.9.3" "numpy==1.22.4" "numba==0.55.2" "llvmlite==0.38.1" \
    "wrapt<1.15" "setuptools<70" soundfile 2>&1 | tail -2
if [ ! -f "$M/2stems/model.index" ]; then
  mkdir -p "$M/2stems"; tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/2stems.tar.gz" https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz
  echo "$SHA  $tmp/2stems.tar.gz" | sha256sum -c -
  tar -xzf "$tmp/2stems.tar.gz" -C "$M/2stems"; rm -rf "$tmp"
fi
"$VENV/bin/python" -I -c "import spleeter, tensorflow as tf; print('spleeter ok, tf', tf.__version__)" 2>&1 | tail -1
