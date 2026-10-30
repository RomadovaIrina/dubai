# r9 AUDIO speed (r9-f): lever → s/video-min → gate, and the stack r9fast-audio

**Decision (30.10): the user listened to the ear AB (`/workspace/dub/outbox/r9fast_audio_AB/`: m04, 03, 05, m01, baseline → r9fast-audio, demucs mix): "sounds the same, all great". r9fast-audio (112 s/video-min) is APPROVED and is now the default of `scripts/r9/e2e.sh` (`R9_FAST_AUDIO=2`; `=1` = the safe stack, `=0` = the r9 E2E audio path).**

Branch `experiment/r9-e2e` on r9-f (local commits 701c104, e2b9a77, + the report commit). Every lever is an env flag, default OFF: with no
flag set the audio stage is the r9 E2E one.

## Where the audio time went (r9 E2E baseline, m01 / m04)

| part | m01 (86 s) | m04 (21 s) |
|---|---:|---:|
| pipeline (ASR, diarization, Qwen 14B translation + duration-aware candidates + judge, batched Chatterbox + QA) | 150 s | 77 s |
| · of it Qwen 14B (146 / 36 greedy calls, ~1000 prompt + ~40 answer tokens each: decode-bound, one sequence per step) | 64 s | 16 s |
| · Chatterbox T3 decode / S3Gen / QA (Whisper + CTC), one take at a time after the batched T3 | ~46 s | ~17 s |
| source MediaPipe face track (CPU, ~1 s per video-second, on the critical path) | ~86 s | ~15 s |
| second process: retake / clause takes / burst_align (+ its start and model loads, ~14 s) | 61 s | 31 s |
| lip_place / hush / soften / remix (CPU) | 27 s | 13 s |

## The levers

| flag | what | equivalence |
|---|---|---|
| `DUB_UNIT_CONC=1` (U) | `scripts/r9/coop.py`: the units of the pipeline's TTS stage (and of retake) run as COOPERATIVE worker threads: one runs Python at a time, a worker that needs the GPU parks its request; when none can run, all parked requests of one kind run as one batch. The per-unit selection code is unchanged; its independent calls (budgets, piece candidates, the plain judge with round 1, the pool's judges) run as sub-tasks (`dur_translate.pmap`). Chatterbox takes of all units batch together. | same calls; batch composition changes the GEMM rounding of the batched T3 (as the accepted r8b5) |
| `DUB_QWEN_BATCH=1` (Q, needs U) | `scripts/r9/qwen_batch.py`: the parked greedy Qwen calls decoded together on the RESIDENT 14B (a 2nd llama.cpp context on the same weights: KV only, no 2nd load), prompt tokens exactly as `create_chat_completion`, the few-shot prefixes of each prompt kind resident in the unified KV and shared (seq_cp), flash attention. 150 recorded m01 calls: 65.6 → 11.2 s. | identity vs the recorded answers 99–105 / 150; a plain one-by-one REPLAY gives 111 / 150 (llama-cpp-python reuses the previous call's KV: one-by-one is not reproducible either) |
| `R9_MP_BG=on` (M) | the source MediaPipe track runs in the background while the GPU pipeline runs (same file) | identical |
| `DUB_S3_BATCH=1` (S) | `scripts/r9/s3_batch.py`: S3Gen of a ladder round batched: per take the single-take path up to the CFM (its noise drawn with the take's RNG state), the Euler solve of all takes in one batch (end-padded; the estimator is causal + masked), HiFiGAN per take with its RNG state. 0.30 → 0.08 s / take. | batch of 1 bit-identical; padded == unpadded batch; batch vs alone: mel mean abs diff 0.0014 (values ~4) = GEMM rounding |
| `DUB_QA_BATCH=1` (W) | `scripts/r9/qa_batch.py`: the round's Whisper QA in one batch: each take's own `transcribe()` as a cooperative worker, its `encode` / `generate` parked and made for all takes together. 0.179 → 0.140 s / take (beam-5 decode dominates). | texts 19 / 20 identical ("." vs "!") |
| `R9_ONEPROC=on` (L) | `scripts/r9/audio_proc.py`: retake → clause takes → burst_align inside the pipeline process; Qwen 14B (+ the warm batched engine) and the TTS QA stay resident. Chatterbox is still loaded fresh for the stage (the pipeline casts its copy to fp16 in place). | identical code paths |

Tried, not taken: the QA of take k in a thread while take k + 1 vocodes (GPU contention: vocode 18 → 28 s, QA 13 → 19 s, net −1 s).

## How it was measured

- `scripts/r9/speed_gate_run.sh <tag> <id> [seed offset]`: the e2e audio stage alone (`r8_audio.sh` from the raw input, the per-id caches
  cleared as `e2e.sh` does), timed under `/workspace/dub/gpu.lock` (one job on the GPU), then the htdemucs mix of the same dub and the e2e
  metrics block on BOTH mixes. Seeds +1 / +2 / +3 follow the r8 envelope convention (pipeline 1247, retake 7, clause 11).
- Videos: the r9 m01 (86 s, 1080p60) and m04 (21 s), and the old 01–05 (8.62 video-min in all). Every timing below is on r9-f with nothing
  else on the GPU. The runs between 17:05 and 18:00 overlapped another agent's composites and were all re-run under the lock.
- The baseline on this harness (flags off, the r9 E2E code):
  - m04 115.8 s (3 seeds 118.5 / 122.1 / 106.9) and m01 326.8 s (332.9 / 339.1 / 308.4);
  - 01 284.9, 02 320.7, 03 357.9, 04 246.6, 05 205.8 s;
  - = **215.7 s/video-min** over the 7. The r9 E2E report's 239 is over the 9 new videos, which include more short ones, where fixed loads weigh more.
- `scripts/r9/r9_gate.py` applies the r8 envelope rule: a row fails only when it is worse than the WORST baseline draw by more than the tolerance.
  - Draws: m01 / m04 = the r9 E2E baseline + seeds +1 / +2 / +3 on this harness; 01–05 = the r8 envelope (3 sequential draws + r8b5) + a
    baseline draw made here.
  - Axes on the Spleeter mix: faith, coverage, gap excess, pause W1, dub over silence, WER, cut-word ends, inserted gaps, dead air, gap words.
  - Axes on the htdemucs mix: dead air and gap words.
- Calibration: a 4th baseline draw passes the 3-draw envelope on m04 and m01, and fresh baseline draws of 01–05 made here pass the r8
  envelope. So the envelope is not too tight for the baseline itself.

## Lever → s/video-min → gate

Each lever ALONE on m04 + m01 (1.80 video-min; baseline 442.6 s = 246.5 s/vmin):

| lever | flag | m04 + m01 s | s/video-min | gate (both mixes) |
|---|---|---:|---:|---|
| baseline | – | 442.6 | 246.5 | 4 draws = the envelope |
| U alone | `DUB_UNIT_CONC=1` | 509.8 | 283.9 | FAIL on m01: WER 0.026 (≤ 0.014), 1 inserted gap (0). Also slower: without Q the parked Qwen calls still run one by one, plus a few speculative judges |
| U + Q | `+ DUB_QWEN_BATCH=1` | 344.5 | 191.9 | FAIL on m04: coverage 1.48 s (≤ 0.98), dead air 3 (≤ 2) on both mixes; m01 PASS |
| M | `R9_MP_BG=on` | 350.7 | 195.3 | PASS (identical file) |
| S | `DUB_S3_BATCH=1` | 437.8 | 243.8 | PASS (02 / 05 PASS too) |
| W | `DUB_QA_BATCH=1` | 446.1 | 248.5 | PASS; **not taken**: about 0 gain at the stage level, and batched beam-5 Whisper holds ~245 MB of cross-attention cache per beam (out of memory inside the full stack on m01) |
| L (+ M, which it requires) | `R9_ONEPROC=on` | 329.6 | 183.6 | PASS on m04 / m01; on 02, pause W1 0.394 (≤ 0.313), explained below |

Stacks over all 7 videos (8.62 video-min; baseline 1858.5 s = 215.7 s/vmin):

| stack | flags (`e2e.sh`) | s | **s/video-min** | per video, s/vmin (m04 / m01 / 01 / 02 / 03 / 04 / 05) | gate |
|---|---|---:|---:|---|---|
| baseline | – | 1858.5 | **215.7** | 323 / 227 / 168 / 174 / 238 / 275 / 236 | – |
| **r9fast-audio-safe** = M + S + L | `R9_FAST_AUDIO=1` | 1494.1 | **173.4** | 262 / 157 / 145 / 163 / 184 / 210 / 185 | 1 row fails: 02 pause W1 0.388 (≤ 0.313) |
| **r9fast-audio** = U + Q + M + S + L | `R9_FAST_AUDIO=2` | 967.8 | **112.3** | 198 / 102 / 86 / 93 / 124 / 147 / 132 | 7 rows fail: 02 W1 0.384; 03 coverage 6.0 (≤ 4.5) and htdemucs dead air 1; 05 gap excess 2.75 (≤ 2.35); m04 coverage 1.48 and dead air 3 on both mixes. 01 / 04 / m01 PASS |

Where the time goes on m01 now (s):

| | baseline | safe | full |
|---|---:|---:|---:|
| pipeline: ASR + diarization | 10.0 | 10.5 | 10.3 |
| pipeline: Chatterbox / QA / Qwen loads | 8.7 | 9.0 | 9.0 |
| pipeline: TTS stage (Qwen candidates + judge + takes) | 106.8 | 108.5 | 47.6 |
| · T3 decode / S3Gen / QA | 15.4 / 14.8 / 10.1 | 19.0 / 11.1 / 12.3 | 11.4 / 9.7 / 12.4 |
| MediaPipe source track | in BASE (~86, CPU) | background | background |
| retake / clause / burst_align | 66.8 (own process) | in-process (in BASE) | in-process (in BASE) |
| BASE (pipeline + MediaPipe [+ stage]) | 237 | 196 | 118 |
| lip_place / hush / soften / remix (CPU) | 26.8 | 26.6 | 26.7 |
| **audio stage** | **332.9** | **225.0** | **146.1** |

## What the failing gate rows mean

- **02 pause W1 (both stacks, and L alone) is one draw, not three.** Outputs are deterministic per flag set. L, the safe stack and the full
  stack give 0.394 / 0.388 / 0.384 because they share L's Qwen path: the retake's Qwen calls run on the model the pipeline already used,
  so llama-cpp-python's prefix-KV reuse differs (a sequential replay of 150 calls reproduces only 111 of them). That W1 is 0.08 above the
  worst of 5 draws; every other row on 02 is inside the envelope.
- **The full stack's coverage / dead-air rows (m04, 03, 05).**
  - m04 over 4 seeds: coverage 1.48 / 1.40 / 0 / 1.40 s vs the baseline's 0.98 / 0.90 / 0.40 / 0. In the same runs, faith is
    5.0 / 5.0 / 4.0 / 5.0 vs the baseline's 4.33 every time. The stack keeps picking a more faithful, slightly shorter English for one unit.
  - 03: coverage 6.0 / 4.5 (two seeds) vs 2.2–4.5.
  - 05: inserted gaps 3 / 3 (seeds +1 / +2) vs ≤ 2.
  - The batched Qwen itself is not biased. On the recorded m01 calls, its judge agrees with a sequential replay on 48 / 56 calls (the replay
    agrees with the recording on 49 / 56; net score shift −1 vs −4). Its candidates are not shorter: +0.15 words per call vs recorded (the
    replay −0.27); in budget 29 vs 31 of 61.
  - What differs is which candidate survives a 4-vs-5 judge score, plus the cross-unit batch composition of the T3 rounding. The more
    draw-sensitive videos (03, m04) land outside the 4-draw envelope more often. **This is not a clean pass.** As with r8b5, the user's ear
    should decide (AB). Until then the safe stack is the one that passes.

## Handoff

- **For the face agent / the E2E:** `R9_FAST_AUDIO=1 ID=<id> bash scripts/r9/e2e.sh <input> <out>` (the safe stack; gate-clean apart from
  the 02 W1 draw), or `R9_FAST_AUDIO=2` (the full stack). Same outputs and manifest layout as the r9 E2E. Branch `experiment/r9-e2e` on r9-f.
- Audio of the stacks on r9-f (tags in `/workspace/dub/artifacts/r4a/`): safe `r9safe_<id>`, full `r9fast3_<id>` (+ `_htd` = the htdemucs
  mix), ids m01 m04 01–05; copies of the final mixes in `/workspace/dub/handoff/out/r9fast-audio/`.
- VRAM: the full stack peaks at ~27 GB in the audio stage (resident Qwen 14B + its batched KV context of 20k cells + Chatterbox + QA). Like the
  r9 audio stage, it needs the GPU alone.

## Not reached, and what is left

- Target 60–80 s/vmin: the full stack is 112 over these 7 (86–102 on the long clips, 198 on the 21-s m04, where fixed costs dominate); the
  safe stack is 173.
- Left in the full stack on m01 (146 s):
  - ~27 s CPU chain: lip_place 9, hush 5, remix 12.5 (single-process numpy / noisereduce);
  - T3 decode 11 + S3Gen 10 + QA 12 (the QA = Whisper large-v3, beam 5, per take);
  - ASR + diarization + loads ~20; Qwen ~13.
- Next levers:
  - the CPU chain run in parallel per unit, or a vectorised noisereduce;
  - QA with a smaller beam or Whisper turbo (a quality lever, it needs its own gate);
  - one resident audio worker per GPU across videos (the loads, ~20 s per video).

## 02 "the pauses are long" (management, side item)

- On the exact r8v5 files:
  - The speaker is silent 45.7 s of 111 s: 23 pauses, walking from room to room (vocal stem < −80 dB).
  - The dub's silence is 50.5 s, and 40.2 s of it falls on those true silences (9.4 / 5.8 / 5.6 / 4.6 s).
  - The excess (the source still talks while the dub pauses) is spread over ~20 gap edges of 0.2–1.7 s: 0:50.8 1.7 s, 0:26.0 1.1 s,
    1:08.8 1.0 s (`scripts/r9/gaps_vs_source.py`).
- Two generic off-screen levers were tried (flags, default off, committed):
  - burst_align `--tmin-off 0.85 --tmax-off 1.15 --tempo-w-off 0.5`: excess 9.3 → 9.9 s, coverage 2.82 → 3.52 s;
  - lip_place `--off-early 1.0` (timing only): total dub silence 49.0 → 48.9 s, excess 8.8 → 11.4 s.
- **Neither is taken.** Total dub silence = video length − dub speech length, so timing only moves silence around. The long pauses are the
  speaker's own and stay mirrored, as designed. Only more dub speech off-screen (slower delivery or fuller wording) would shorten them.

## Round 2 (31.10): the e2e audio + sep stages toward 60–80

Reference = the APPROVED r9fast-audio. A round-2 lever must not move any gate row of it beyond the envelope tolerance (`scripts/r9/vsref.py`).
Timing now covers the e2e audio AND sep stages (`speed_gate_run.sh`: the htdemucs stems + the demucs remix, sequential or overlapped).

| lever | flag | what | effect | vs the approved stack | taken |
|---|---|---|---|---|---|
| O | `R9_SEP_OVERLAP=on` (`scripts/r9/sep_overlap.sh`) | the htdemucs stems while the audio stage runs (they need only the source); the demucs remix starts the moment the final dub is linked, in parallel with the Spleeter remix (`r4_audio.sh` hook) | sep stage m04 8.8 → 0 s, m01 18.3 → 2.0 s | same commands; the demucs mix differs only by htdemucs's own run-to-run noise (its bed stem differs 26 % rms between two runs on the same input; m04 demucs dead air 3 / 2 / 3 over 3 sequential re-mixes vs 3 overlapped) | **yes** |
| C | `R9_CHAIN_INPROC=on` (`scripts/r9/chain_inproc.py`) | close_inner → soften → lip_place → hush → soften → remix inside the audio process (same argv / env / dirs); lip_place's Whisper = the QA's (large-v3 int8_float16), hush's wav2vec2 = the QA's | chain after the stage m04 16 → 5 s, m01 29 → 15 s | every step's samples identical to the bash chain (the WAV md5 differ only by libsndfile's PEAK-chunk timestamp) | **yes** |
| K | `R9_TTS_PRELOAD=on` | the stage's fp32 Chatterbox loaded on the CPU in a thread while the GPU pipeline runs, moved to the GPU for retake / clause (the pipeline casts its own copy to fp16 in place) | −3.4 / −4.1 s (04 / m04) | pipeline, retake, soften outputs sample-identical | **yes** |
| B1 | `DUB_QA_BEAM=1` | the TTS QA's Whisper greedy instead of beam 5 | −0.5 … −7.7 s (≈ −3 %) | identical on m04 / m01 / 03 / 05, but on 04 one QA verdict flips: coverage 0.8 → 2.0 s, gap 0.8 → 2.6 s, W1, WER 0.054 → 0.082 | no |

Also fixed on the way: the in-process stage now runs with `r8_audio.sh`'s env (`R9_STAGE_ENV`), not the pipeline's. It made no difference to the
outputs, but it is the env the separate process had.

**Result, all 7 videos (8.62 video-min), audio + sep stages:**

| | audio s/vmin | **audio + sep s/vmin** | per video audio + sep (m04 / m01 / 02 / 03 / 05 / 01 / 04) | gate |
|---|---:|---:|---|---|
| baseline (r9 E2E path) | 215.7 | ~229 | – | – |
| r9fast-audio (approved, round 1) | 112.0 | 125.6 | 223 / 114 / 105 / 138 / 144 / 98 / 161 | the ear-approved rows |
| **r9fast-audio round 2 = + O + C + K (default)** | **97.9** | **99.5** | 158 / 91 / 85 / 114 / 110 / 77 / 128 | **0 rows worse than the approved stack**; the envelope rows are the approved ones (6 rows, 7 before) |

`e2e.sh` default (`R9_FAST_AUDIO=2`) = round 2. `R9_SEP_OVERLAP=off`, `R9_CHAIN_INPROC=off` and `R9_TTS_PRELOAD=off` switch the levers off one by one.

**Still above 60–80.** The long clips are at 77–91, the 21-s m04 at 158 (fixed costs). What is left in m01 (130 s):
- the pipeline's TTS stage ~48 s: T3 decode 11, S3Gen 10, QA 12, Qwen 13;
- ASR + diarization + their loads ~20 s;
- retake / clause work ~15 s;
- the in-process chain ~15 s: lip_place's Whisper pass 4, remix 8.

The remaining levers change the output, so each needs its own gate and ear AB:
- a smaller QA / lip_place Whisper (turbo);
- fewer duration-aware candidates;
- one resident worker across videos (the ASR / diarization / Qwen / Chatterbox loads, ~20 s per video).
