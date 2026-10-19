# Server B (dub-B) — Phase 0 environment record (2026-10-16)

```text
hostname        bcf772b4cb09 (vast.ai container 56277943, France), unprivileged Docker, /workspace NOT a volume (workspace_is_volume=false)
GPU             NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 (sm_120), exclusive (2 MiB used / 0 % before every run)
driver          580.178.04 (driver CUDA 13.0); system toolkit nvcc 13.0.88 (/usr/local/cuda, image-provided, NOT used by the project)
project env     /venv/dabai: Python 3.11.17, torch 2.7.1+cu128 (wheel CUDA 12.8, minor-version compatible with the 13.0 driver), cuda available
system python   3.12.15 (/venv/main = vast default, not used)
CPU             AMD Ryzen 9 9950X3D 16-Core; 32 logical CPUs visible, cgroup cpu.max 1536000/100000 = 15.36 vCPU quota; DUB_THREADS stays 16 (production setting)
RAM             cgroup memory.max 65346207744 B = 60.9 GiB (host MemTotal 89.7 GiB)
disk            120 GB container quota (overlay; df shows the host's 3.6 TB pool, so free space is tracked by du): /workspace/dub 24 GB after setup
                (models 20 GB, third_party 1.2 GB), /venv/dabai 8.4 GB
ffmpeg          6.1.1-3ubuntu5 (CPU codecs)
repo            RomadovaIrina/dubai @ tts-intelligibility-fix = 559d700e0bf14ffa5692d855e5eae10a42eadd11 (= coordinator's SHA; remote not moved)
branch          experiment/restoration-temporal created from 559d700
third_party     latentsync a229c3948406bc2cf6eaf4873e662e70c6a04746, CodeFormer b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b (= reports/third_party-commits.txt)
inputs          /workspace/dub/inputs/test_videos_raw/{01.MP4,02..05.mp4}, sha256 -c OK (5/5), symlinked as test_videos (excluded via .git/info/exclude)
setup           scripts/setup_dabai.sh stages 0-9 exit 0 (~70 s); scripts/download_models.sh exit 0 (~4 min, 20 GB);
                pip-freeze drift vs the committed lock = unpinned patch/minor bumps only (aiohttp, pydantic, lmdb 2->3, multidict 6->7, soupsieve 2->3, ...);
                torch / numpy / transformers / diffusers / huggingface_hub pins unchanged
smoke           scripts/run_smoke.sh 11/11 PASS (chatterbox, tts-intelligibility, faster-whisper, silero-vad, pyannote, librosa/atempo, llama.cpp,
                retinaface, latentsync, codeformer, ffmpeg-mux)
side effect     LatentSync's FaceDetector (insightface.app.FaceAnalysis, root checkpoints/auxiliary) auto-downloaded the InsightFace buffalo_l pack
                (det_10g + 2d106det) on first use -> license note in LICENSES.md
```
