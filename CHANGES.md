# r3-integration (Agent D, round 3) — changelog

Branch `experiment/r3-integration` from `clean-base` 326e807. Server: RTX 5090, 24 vCPU.

## Merges
| # | merge | conflicts / resolution |
|---|---|---|
| 1 | `r2-lipsync-position` 92de423 (A: fp32 AlignRestore, flow anchors, head-locked mask, 05 coverage) | none |
| 2 | `r2-composite-position` 6341e2e (B: FIX30 source-fps composite) | `anchor_filters.py` = A's superset (blend filter, rigid thread cap); `slide_metric.py` identical (mode); B's `r2_sbs.sh` kept, A's as `r2_sbs_A.sh`; B's dub-B3 pip freeze dropped. Tree = B's r3-face merge 98c8333 apart from env records |
| 3 | `r2-audio` a1fd764 (C: L1 + floor 0.875, Spleeter bed, declick, duck −4 dB / 250 ms) | runner argparse: C's `--background on` (Spleeter 2stems, MIT) + `--audio-policy/--declick/--bed-separator` taken; HDEMUCS stays behind `DUB_ALLOW_NC_BED` |

## Tooling
- `candidate_video_stage.py --dump-ls-raw on`: the same run also writes `<out>_ls_raw.mp4` (raw LatentSync frames on the master,
  lossless) = the `--ls-raw` input of `fps_composite.py` (FIX30), so no second LatentSync pass (B used `decompose_video_stage.py`).
- `master_timing.py`: regenerates B's `<id>_master_timing.json` (master frame → source frame; generator was not committed).
  03 / 04: master lags its clock by 66 ms mean (max 80 ms), matches B's finding.
- `r3_video.sh <arm> <ids>`: video stage on canonical cand5 audio (arms `clean0`, `r2m`).
- `r3_scorecard.py score|table`: per-video JSON + Markdown table (9.4, coverage, SyncNet, v4, slide, joints, feather, WER, LUFS, speed).

## Licences
Nothing new. Spleeter 2stems (MIT code, MIT release weights) set up in `/workspace/dub/venv-spleeter` via `scripts/setup_spleeter.sh`.
- `r3_fix30.sh <manifest>`: FIX30 (`fps_composite.py --master-timing --register sim_ring --ring-region upper --edge-px 12`) on a run with a raw layer.
- `r3_sbs.sh`: labelled side by side for mixed frame rates (+ half speed). `r3_scorer_loop.sh` / `r3_queue_q3.sh`: the queue used on this server.

## First numbers (canonical cand5 audio; speed contaminated by concurrent CPU jobs)
| | 03 clean0 | 04 clean0 | 05 clean0 | 01 clean0 | 02 clean0 | 04 r2m |
|---|---:|---:|---:|---:|---:|---:|
| LS frames | 2025 | 1074 | 1025 (cand5 949) | 260 | 0 (no face) | 1074 |
| LatentSync s/video-min | 315 | 298 | 288 | – | – | 281 |
| video stage s/video-min | 410 | 417 | 409 | – | – | 414 |
04 FIX30 (r2m): 0 fallback frames, edge 11.4 px, +105 s/video-min (CPU, 12 threads).

## CUDA llama-cpp-python (Qwen on the GPU) — recipe (for C too)
Same version as the env (0.3.35), CUDA 13.2 nvcc, sm_120 (RTX 5090), installed into `third_party/llama_cuda` so `/venv/dabai` stays CPU:
```bash
JOBS=8 bash scripts/setup_llama_cuda.sh        # CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120", uv pip --target third_party/llama_cuda; ~15 min with 8 jobs
PYTHONPATH=third_party/llama_cuda:$PYTHONPATH python scripts/pilot/run_clean_pipeline_05.py ... --qwen-gpu-layers -1 [--qwen-resident on]
```
Check: `PYTHONPATH=third_party/llama_cuda python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"` → True.
For a separate venv instead: `CMAKE_ARGS=... FORCE_CMAKE=1 uv pip install --python <venv>/bin/python --no-cache llama-cpp-python==0.3.35`.
