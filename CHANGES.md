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

## Licence guard (coordinator 19.10)
- `download_models.sh` never fetches HDEMUCS, CodeFormer, ParseNet, RetinaFace R50 or buffalo_l any more (the `DUB_ALLOW_NC=1` branches are
  gone); it ends with the tripwire check and exits 3 if a banned file is present. HDEMUCS had been re-fetched on this server: the coordinator
  quarantined it (`/workspace/dub/quarantine_nc/demucs/`).
- `clean_face_stack.assert_clean_tree()` (called first in `enforce_clean`, i.e. by `run_clean_pipeline_05.py`, `candidate_video_stage.py`,
  `ls_window_ab.py`): any banned weight file under `models/`, `third_party/`, the torch hub cache or `~/.insightface` stops the run with the list.
- cand5 is a reference only (numbers from the round-1/2 reports, never re-rendered). Candidates use C's clean audio (Spleeter bed); runs on the
  canonical cand5 audio are video-only controls. Scorecard / spec status label the audio source per row.
| 4 | `r3-face` 28491dd (B r3) | see 1a30201: `--dump-ls-raw` kept D's (accepts B's PATH form), B's `--master-timing-mode` kept |
| 5 | `r3-audio` 49e8002 (C r3: device-agnostic audio stage, Spleeter torch port on the GPU, `--qwen-quant`, `--qa-whisper-compute`, OOM backoff, `--tts-text-norm`) | runner `main()` globals: D's `VAD_DEVICE` + C's `QWEN_QUANT` / env flags combined; C's changelog kept as `reports/experiments/r3-audio/CHANGES_C.md` |

## Speed finding: the CPU quota, not the GPU, is the bottleneck (19.10 08:10)
- Container cgroup `cpu.max` = 2304000/100000 = **23 CPUs**. Under a quota, `nice` does not help: every thread is throttled once the quota is spent.
- With 3 video-stage preps (MediaPipe + rigid motion, ~5 cores each), FIX30 (`--threads 16`, 18 cores) and 3 v4 metric jobs at once, load hit 97,
  the GPU sat at 0 % and the prep of the run holding the GPU lock took > 10 min (normally ~2 min).
- Fixes used: `--prep-only` (prep outside the GPU lock), SIGSTOP / SIGCONT of side jobs while the critical run preps. Proposal for the worker:
  a CPU budget per stage (FIX30 threads ≤ 8, one prep at a time), the face-stack prep in parallel with the audio stage (it only needs the master).
| 6 | `r3-face` e9c3f58 (B r3 final: `--lip-guard on`, `--sharpen-auto 0.95 --skin-texture source`, dub-silent chains; carries A r3 eaaaaea: `--ref-mode`, bridges, `--dump-raw`, spec-11 2 % status) | `candidate_video_stage.py` raw dumps: D's `--dump-ls-raw` (on / PATH) + A's `--dump-raw` kept, B's duplicate block dropped; `r3_sbs.sh` = A's, D's side-by-side tool renamed `r3d_sbs.sh` |
| 7 | `r3-lipsync` ad7ada1 (A r3: `--guidance`, `--bridge-len/--bridge-steps`, `--max-ls-frames` call cap, README) | clean |

## r3cand (coordinator 19.10): A's R3 generator + B's final composite, rendered on C audio-r3, muxed with audio-r3b
Generator: r2best (`--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head`) + `--guidance 2.0 --bridge-len 8
--bridge-steps 4 --max-ls-frames 400` + B's fps master (`--master-timing-mode fps`). Composite: `r3_fix30.sh` mode r3cand
(`--register sim_ring --source-lm --colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on --edge-px 12`).
