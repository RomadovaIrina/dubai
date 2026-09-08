# dub — pilot ML environment (`dabai`)

Immutable baseline: **PyTorch 2.7.1 + cu128 + sm_120** (RTX 5090). One conda env `/venv/dabai` (Python 3.11) hosts the whole stack:
Silero VAD, faster-whisper large-v3, pyannote 3.x diarization, facexlib RetinaFace, Qwen2.5-7B GGUF via llama.cpp (CPU),
Chatterbox Multilingual v3, librosa/atempo, LatentSync 1.6, CodeFormer, ffmpeg (CPU codecs only on this host).

The container disk is **not persistent** (no volume). Everything needed to rebuild lives in this directory — copy it off-box.

## Rebuild from scratch
```bash
scripts/setup_dabai.sh              # stages 0-9: env, torch, pins, git deps, kernel, freeze  (~6 min)
scripts/download_models.sh          # ~18 GB of weights into models/                          (~5 min)
echo 'HF_TOKEN=hf_...' >> /workspace/.env   # only for gated pyannote models
scripts/run_smoke.sh                # every component end-to-end -> reports/smoke.md          (~7 min)
```
Re-run single stages with `scripts/setup_dabai.sh 5 6`, single tests with `scripts/run_smoke.sh llama.cpp latentsync`.
Exact resolved versions: `reports/pip-freeze.lock.txt`; third-party commits: `reports/third_party-commits.txt`.

## Use the env
```bash
source scripts/env.sh     # activates /venv/dabai, sets HF_HOME, PYTHONPATH (CodeFormer), thread counts
```
Jupyter kernel: **Python (dabai)**. `test-pipeline.ipynb` prints all versions (no model loading).

## Layout
```
scripts/requirements/   per-stage pins + constraints.txt (torch/numpy/transformers/hub/diffusers are locked; xformers is forbidden)
scripts/check_*.py|sh   one smoke test per component; smoke_common.py has shared paths
third_party/latentsync  bytedance/LatentSync (checkpoints/ -> symlinks into models/latentsync)
third_party/CodeFormer  sczhou/CodeFormer (vendored basicsr + facelib; weights/ downloaded by download_models.sh)
models/                 whisper-large-v3, qwen2.5-7b-instruct-gguf (q4_k_m), chatterbox (v3), latentsync, sd-vae-ft-mse, facexlib, hf/ cache
logs/  reports/         setup + test logs; versions.txt, smoke.md, env-report.md, demo outputs
```

## Non-obvious decisions (see reports/env-report.md for details)
- `pyannote.audio==3.4.0` + `huggingface_hub==0.30.2`: pyannote 4.x needs torch ≥ 2.8; 3.x breaks on hub ≥ 1.0.
- Chatterbox from git master with `--no-deps` (v3 checkpoint is git-only; it hard-pins torch 2.6 / transformers 5).
- No xformers (no sm_120 kernels; replaces torch). LatentSync/Chatterbox use PyTorch SDPA.
- `onnxruntime-gpu==1.26.0` is the only onnxruntime dist (CUDA 12.8 build; ≥1.27 is CUDA 13). CUDA provider verified on sm_120.
- faster-whisper `int8_float16` executes as float16 on Blackwell (CTranslate2 disables int8 for sm_120).
- llama.cpp: always pass `n_threads=n_threads_batch=16`; `os.cpu_count()` reports the host's 128 cores and the cgroup quota is ~16.
