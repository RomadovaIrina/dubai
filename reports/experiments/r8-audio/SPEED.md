# r8 audio speed (phase 2): lever → s/video-min → gate

One command from the raw input: `bash scripts/r8/r8_audio.sh <tag> [ids]` (defaults = the audio-r8-v4 configuration). Timing = production
seconds per stage (`scripts/r8/speed_table.py`): the r6 pipeline (ASR, diarization, segmentation, Qwen 14B, Chatterbox + QA, separation),
retake, clause takes + burst_align, close_inner + soften, lip_place, hush + final soften, remix. The faith judge, metrics and coverage rows are
not counted. One RTX 5090 (shared with the LTX probe under `flock /workspace/dub/gpu.lock`, never concurrently). 5 videos = 409 s = 6.82 video-min.

## Baseline (r8e2e0 + r8e2e0b, all 5): 303 s/video-min

| stage | s/video-min |
|---|---:|
| pipeline | 145.7 |
| · Chatterbox TTS (incl. the duration-aware candidates: 35–48 takes per video for 5–7 final parts) | 117.1 |
| · ASR (faster-whisper large-v3) | 4.0 |
| · diarization (pyannote) | 2.7 |
| · translation (Qwen 14B) + load | 1.3 + 3.3 |
| retake | 47.0 |
| clause takes + burst_align | 92.1 |
| close_inner + soften | 1.5 |
| lip_place | 5.1 |
| hush + final soften | 3.8 |
| remix | 8.0 |

The brief's Whisper / pyannote levers can save < 7 s/video-min together: the time is Chatterbox calls (one by one, ~100 tokens/s,
launch-bound) in the pipeline, retake and clause takes.

The quality reference for speed is the FROM-SCRATCH baseline r8e2e0b (its own takes; its quality ≈ audio-r8-v4: faith 4.81 / 4.64 /
4.55 / 5.0 / 4.71, coverage spans > 0.3 s 1.3 / 2.9 / 2.2 / 0.8 / 3.6 s, 0 cut words). audio-r8-v4 itself is built on the user-approved v3a1
takes, which a run from the raw input does not reproduce.

## Each lever alone (04 + 05, baseline there 377 s/video-min)

| lever | flag | s/video-min | pipeline / retake / clause | quality vs baseline (04 / 05) |
|---|---|---:|---|---|
| L1 | pipeline `--dur-top 1 --dur-rounds 1` | 338 | 114 / 55 / 145 | 04 coverage 0.8 → 1.5, gap excess 0.9 → 2.3 |
| L2 | fast T3 (`--tts-fast-t3 on`, `DUB_TTS_FAST_T3=1`) | 270 | 106 / 55 / 86 | other takes (not bit-identical); 05 faith 4.71 → 4.43, coverage 3.6 → 4.5 (one run) |
| L3 | retake `--max-tts 2`, clause `--lazy-cfg on --max-cuts 2` | 323 | 148 / 52 / 101 | 04 coverage 0.8 → 1.2 |
| L4 | pipeline `--dur-translate off` | 277 | 53 / 70 / 131 | 04 faith 5.0 → 4.8 |
| L5 | `R8_ONEPROC=on` (retake / clause / burst_align of all videos in one process) | 338 | 147 / 43 / 124 | identical (the pipeline is deterministic per flags) |
| L6 | clause `--plan-top 2` | 306 | 147 / 55 / 81 | 04 coverage 0.8 → 2.4 |

## Stacks (all 5)

| stack | s/video-min | gate vs r8e2e0b |
|---|---:|---|
| r8fast1 = L2 + L3 + L4 + L5 + L6 | 109 | FAIL: 03 coverage 2.2 → 11.4 s (retake rejects 6 / 7 units: plain base translation + 2 takes), 02 gap excess 1.35 → 4.7, 04 faith 5.0 → 4.6 |
| r8fast2 = L1 + L5 + clause lazy + plan-top 4 | 199 | FAIL: 03 coverage 2.2 → 4.9, 02 gap excess 1.35 → 4.8, 04 coverage 0.8 → 2.4 |
| r8fast3 = r8fast2 + L2 | 153 | FAIL: 03 coverage 2.2 → 8.4, 02 gap excess 1.35 → 4.8, 05 coverage 3.6 → 4.4 |
| r8fast4 = L2 + L5 + clause lazy | 188 | FAIL: 03 coverage 2.2 → 5.2, 04 coverage 0.8 → 2.0 / faith 5.0 → 4.8, 05 faith 4.71 → 4.43 / coverage 3.6 → 4.5 |

All stacks keep 0 cut-word ends and the dead-air counts (0–2 per video); what fails is coverage, gap excess and faith.

## Reading

- Proven safe: L5 only (identical output, the pipeline is deterministic per flags): ~ −15 s/video-min → ~288.
- Every lever that synthesises FEWER candidates (L1, L3's retake cut, L4, L6) costs coverage or timing somewhere, worst on 03, because the
  candidates are how the dub gets to the source length.
- L2 (fast T3) is deterministic but gives OTHER takes (not bit-identical by design); on this one sample they score worse on 04 / 05. One
  draw cannot separate "worse model" from "worse draw" (audio-r8-v4 vs the from-scratch baseline already differ by that much on 05);
  it needs several seeds before it can be gated.
- ≤ 80 s/video-min with today's quality is not reachable by these flags. The remaining lever is architectural: BATCHED Chatterbox
  inference (several units / candidates in one T3 forward, batch > 2 next to CFG, with the CUDA-graph decode) and synthesising each unit
  once for the pipeline + retake + clause stages instead of three times.
