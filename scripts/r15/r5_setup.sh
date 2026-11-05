#!/bin/bash
# r15 r5: the server stack, as on r14-d (04.11: ~25 min, ~23 GB models + ~15 GB venvs); log SETUP_FAIL <step> / SETUP_CHAIN_DONE
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
step(){ echo "[$(date +%T)] == $*"; }
step "1 setup_dabai"; CHATTERBOX_REF=5de7a54aa4e5e2baadb0182dde554908b48b85c2 bash scripts/setup_dabai.sh || echo "SETUP_FAIL dabai"
step "2 download_models"; DUB_SKIP_QWEN7B=1 bash scripts/download_models.sh || echo "SETUP_FAIL models"
step "3 mediapipe"; bash scripts/setup_mediapipe.sh || echo "SETUP_FAIL mediapipe"
step "4 llama_cuda"; JOBS=32 bash scripts/setup_llama_cuda.sh || echo "SETUP_FAIL llama"
step "5 r9_models"; bash scripts/r9/r9_models.sh || echo "SETUP_FAIL r9_models"
step "6 nvml"; uv pip install --python /venv/main/bin/python nvidia-ml-py 2>/dev/null || /venv/main/bin/python -m pip install -q nvidia-ml-py
echo "SETUP_CHAIN_DONE $(date +%T)"
