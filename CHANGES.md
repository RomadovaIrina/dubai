# Agent C round 3: voice / audio (`experiment/r3-audio`)

Base: `experiment/r2-audio` @ a1fd764. Round-2 log: `reports/quality/audio_timing/r2/CHANGES.md`. Server A, shared GPU (≤ 10 GB VRAM for C).

## 1. Device-agnostic audio stage
- `pilot_common.dev()` (`DUB_DEVICE`, otherwise cuda if visible, otherwise cpu). It replaces the hard-coded `"cuda"` in the runner, `tts_burst` and `tts_qa`.
- On the CPU: Chatterbox runs fp32 only and Whisper uses int8.
- Why: the brief first said CPU only. Since the 19.10 coordinator update, the GPU is used and the CPU path is the fallback.
- Qwen stays on the CPU: the shared llama-cpp-python is a CPU build, and this keeps the 8 GB of weights off A's GPU.
- `scripts/quality/r3_audio_run.sh <id>`: audio-only run with the r2 defaults. The Spleeter venv is in `/workspace/dub/c/venv-spleeter`.
