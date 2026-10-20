# r4-audio (Agent A audio, round 4) — changelog

Branch `experiment/r4-audio` from v4 `904c5d2`. Server: RTX 5090 (alone). One command: `bash scripts/r4_audio.sh <tag>` (all five videos from the raw input,
v4 audio-r4 flags + `r3_remix.py` + new audio over the original video + metrics); gate + publish: `bash scripts/r4_audio_publish.sh <ver> <tag>`.

## Findings
- v4 reproduces: `v4repro` = published v4 within realization noise (LUFS, duration, residue +-2.3 dB, pause W1 +-0.33 s on 05: different Qwen text);
  two runs on this server are bit-identical (seeded TTS, Qwen temperature 0) -> every A/B here is free of TTS noise.
- The v5 bed regression is PLACEMENT, not the bed: 53cfb22's remix is in v4 too. v4 vs v5 finals (03 / 04): residue and bed levels equal
  (+-1 dB); "source speaks, dub silent" 14.5 -> 16.4 s / 5.7 -> 8.0 s, pause W1 0.28 -> 0.46 / 0.28 -> 0.57 s. Bisect: v4 code with
  `--sentence-flow-gap 1.0` alone gives 04 8.0 s / 0.58 s (= v5), 03 16.0 s / 0.56 s + a 20 dB level jump at pause onsets -> flow gap stays 0.2.
- Bed A/B on the same dub (remix only): residue duck -15 dB = no-op (Silero hears no speech in the bed); duck -4 dB doubles the bed gain jitter
  (r3 pumping); `--src-duck-db -6` (bed -6 dB where the source speaks and the dub is silent) lowers the exposed original-voice residue 1-5 dB, nothing else
  changes -> kept (one flag, no new layer).
- "Solki" (04): Qwen wrote "Sochi city", Chatterbox said "Soky" (CTC "soky", Whisper "It's OKCITY"); sentence CER 0.05 passed the QA.
  `--ne-guard on` (ne_guard.py): Qwen lists proper nouns, only rare capitalised words are guarded (capitalised form >= 2 Whisper BPE tokens:
  Sochi yes; Moscow / Finnish / sauna / banya no), the TTS QA's Whisper must hear them, retries synthesise phonetic respellings of the closest
  source word ("Sochee", "So-chee", Qwen's); 04: attempt 2 "Sochee" -> Whisper "Sochi". The llama state is restored after the extraction so no other
  unit changes text.
- Voice continuity: v4 is already whole-sentence synthesis (F0 at part joins = the uncut take); sentence-to-sentence level / F0 jumps are at or
  below the source speaker's own variation -> no new layer, gate axis only.

## Gate (`r4_audio_metrics.py`, `r4_audio_gate.py`)
Mix-only metrics (bed = c(t) x the source's Spleeter accompaniment, c by 50 ms least squares; validated vs stems within 0.3 dB), so a published mp4
scores on the same axes: residue under dub / exposed, bed VAD, bed level and level jumps in pauses, pause W1 / > 1 s, source-only / dub-only seconds,
LUFS / TP, duration; stem axes from the reproducing run: WER, hard cuts, joins, sentence continuity, names misheard. Tolerance = fixed + |published
previous - its reproduction|. Review sheet `r4_review.py` (10 s rows, LOOK flags).

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
| 8 | `r3-audio` 5ef9cbf (C r3: loudness normalisation -16 LUFS / TP <= -1, TTS lexicon, Qwen reload fix, peak-safe declick) | CHANGES.md: C's kept as reports/experiments/r3-audio/CHANGES_C.md |

## r3final (19.10 evening): the three user defects fixed; finals `outbox/<id>_r3final_fixed.mp4` (v4)
Pipeline: audio-r4 (C r3 settings + phrase-cut fix) -> A's R3 generator (guidance 2.0, bridge 8x4, 400-frame cap) on B's fps master ->
`fps_composite.py` (B's final composite + D fixes: `--retime flow_safe`, `--sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8`,
`--sharpen-lips-max 0.8`, `--face-clip on`) -> remux audio-r4b (bed remix + static loudness). Scripts: `r3_audio_r4.sh`, `r3_remix.py`,
`r3_fix30.sh` (mode r3fix), `r3_publish_v3.sh` (VER=v4).

| defect | cause | fix | numbers |
|---|---|---|---|
| torn / ghosted lips on single 30 fps frames | 25->30 flow interpolation averaged two disagreeing warps | `flow_safe`: low confidence -> the nearer frame alone, warped to t | 04 frames 670 / 675 / 680 clean |
| pale "pink blob" lips (1-3 frames) | in the RAW LatentSync layer (A R3 11.4 % of LS frames, r2best 8.4 %, no-bridge 11.8 %) | sanitize: flagged frames keep their own low frequencies (opening) + the neighbours' lip-line detail | 04 SyncNet dub 4.27 -> 4.17 (full replace: 3.77); 03 6.15 (replace 5.38) |
| background beside the cheek / jaw moves | paste ellipse / ring reached outside the face | paste weight capped inside the source face contour | ring excess 04 1.99 -> -0.10, 03 2.13 -> 0.06, 05 0.43 -> -0.14 (clean0 0.70 / -0.42 / -0.46) |
| abrupt phrase cuts | CTC word ends early -> cuts mid-voice, 5 ms linear fades | cut snap + merge voiced cuts + equal-power fades | hard cuts 36 -> 7 (01-05), voiced inner cuts 03 6 -> 0 |
| bed "breathing" | ffmpeg loudnorm dynamic mode (AGC) on 02 / 03 / 05 | static gain + oversampled limiter, duck -2 dB / 0.4 / 0.8 s | bed gain jitter p95 03 7.83 -> 3.11 dB (cand5 3.35), swing 0.9-1.9 dB, -16 LUFS, TP <= -1.3 |

Generator A/B on 04 (audio-r4, 25 fps, SyncNet dub / mix): A R3 4.51 / 4.20 (123 blob frames), without bridge 4.48 / 4.17 (128), r2best
3.81 / 3.58 (91) -> A R3 kept (the sanitizer handles its blobs). Rejected: full-replace sanitize (-0.5 SyncNet), per-channel colour match
(over-saturated), luminance match (-0.24 SyncNet on 03 for little visible change).
