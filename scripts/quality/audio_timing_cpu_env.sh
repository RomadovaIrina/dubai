# Agent C (audio timing): CPU-only, <= 6 threads, own clone; source this, then run with `nice -n 10`.
export DUB_ROOT=/workspace/dub/dubai-audio
. /workspace/dub/dubai-audio/scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
export DUB_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMBA_NUM_THREADS=6 TORCH_NUM_THREADS=6
cd /workspace/dub/dubai-audio
