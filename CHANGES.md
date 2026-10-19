# Agent C round 3: voice / audio (`experiment/r3-audio`)

Base: `experiment/r2-audio` @ a1fd764. Round-2 log: `reports/quality/audio_timing/r2/CHANGES.md`. Server A, shared GPU (≤ 10 GB VRAM for C).

## 1. Device-agnostic audio stage
- `pilot_common.dev()` (`DUB_DEVICE`, otherwise cuda if visible, otherwise cpu). It replaces the hard-coded `"cuda"` in the runner, `tts_burst` and `tts_qa`.
- On the CPU: Chatterbox runs fp32 only and Whisper uses int8.
- Why: the brief first said CPU only. Since the 19.10 coordinator update, the GPU is used and the CPU path is the fallback.
- Qwen stays on the CPU: the shared llama-cpp-python is a CPU build, and this keeps the 8 GB of weights off A's GPU.
- `scripts/quality/r3_audio_run.sh <id>`: audio-only run with the r2 defaults. The Spleeter venv is in `/workspace/dub/c/venv-spleeter`.

## 2. audio-r2 regenerated for 01, 03, 04, 05 (19.10), handed off to A / B / D
- `r2_deliver.py` reuses the runner's own Spleeter bed and mix when the flags match, so there is no second Spleeter pass.
- The face reference (`face_ref.py`) now comes from the r2 dub itself. In r2 it came from a cand5 run.
- Residual gaps (n / s / LS frames) vs the r2 delivery before refit:
  - 03: 7 / 4.6 / 118 (r2 8 / 5.6 / 143)
  - 05: 4 / 2.2 / 43 (r2 5 / 3.5 / 58)
  - 04: 5 / 2.6 / 67 (unchanged)
  - 01: 7 / 3.8 / 13 (r2 5 / 3.2 / 13)
- Not yet re-applied: the layer-3 refit (01 u23, 04 u05).
- Time: 478–632 s per video-minute, with 4 videos run in parallel on a cgroup quota of about 30 CPUs.
  - Spleeter: 160–404 s per video (r2: 5 s). The first `separate()` call takes about 80 s, and the CPU is contended.
  - Qwen on CPU: segmentation about 120 s.
  - Without Spleeter: 283–389 s/min. Speed work is next.
