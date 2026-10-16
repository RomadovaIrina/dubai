#!/bin/bash
# Clean face stack env (2026-10-16): MediaPipe Face Landmarker (Apache-2.0) in its own venv, called as a subprocess by
# scripts/pilot/clean_face_stack.py. Kept out of /venv/dabai: mediapipe 0.10.21 needs protobuf 4.25 / opencv-contrib / jax.
set -euo pipefail
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
ENV_DIR="${DUB_MEDIAPIPE_ENV:-/venv/mediapipe}"
[ -x "${ENV_DIR}/bin/python" ] || uv venv --python 3.11 --seed "${ENV_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" mediapipe==0.10.21
"${ENV_DIR}/bin/python" -c "import mediapipe as mp, cv2, numpy; print('mediapipe', mp.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__)"
