# Iteration 3 (2026-10-13 evening): Qwen on the GPU, resident worker, connected sentences

Exclusive GPU (the /mem process stopped). Videos (not in git): `results_videos/oct12_fast/`, `results_videos/final_qwen_gpu/`,
`results_videos/sentence_flow_listen/` (audio-only listening clips: original frames + new dub track, lips NOT synced).

## Speed — the 2026-10-12 pipeline unchanged, made faster

Same flags as the reviewed 2026-10-12 candidate (`--phrase-cut acoustic --gate-merge-gap-s 1.2 --mouth-only-pasteback on`) +
`--qwen-gpu-layers -1` (CUDA build of the same llama-cpp-python, `scripts/setup_llama_cuda.sh`) in the resident worker
(`scripts/pilot/run_worker.py`, models loaded once; 04 ran first = cold).

| s per video-minute | 01 | 02 | 03 | 04 | 05 | median |
|---|---:|---:|---:|---:|---:|---:|
| 2026-10-12 baseline (CPU Qwen, per-run loads, 27 vCPU box) | 306 | 150 | 559 | 508 | 538 | 508 |
| **Oct-12 + Qwen GPU + resident worker** | 144 | 89 | 436 | 494 (cold) | 394 | **394** |

Remaining time on 03 / 04 / 05 is the video stage (LatentSync ~230-290 s/min there); LatentSync speed A/B (DeepCache / DDIM
steps) running. `--video-prep parallel` was measured harmful (RetinaFace on every frame competes with Chatterbox: TTS stage x1.6-2).

## Pilot numbers for the supplementary agreement

* 0.1: llama.cpp 0.3.35 CUDA build runs on sm_120 (compute capability 12.0, no "no kernel image").
* 0.9 addendum (`reports/pilot/0.9_qwen_gpu_addendum.md`): Q8_0 on the GPU 160.9 tok/s vs 5.4 on the CPU; 34/36 outputs identical
  to the CPU Q8_0 of 0.9; GPU outputs are not bit-reproducible across repeats. The Q8_0 choice of 0.9 stands.
* 0.4 with Qwen (`reports/pilot/qwen_gpu_2026-10-13/`): steady residency of all models + Qwen 14.3 GiB; real 03 peak with everything
  resident 31.15 GiB (window batch 2) / 29.4 GiB (batch 1, LatentSync ~1 % slower); production choice: Qwen loaded per job (2 s),
  released before LatentSync -> peak 25-29 GiB.

## Quality: audio A/B with Qwen on the GPU (same pipeline otherwise, five videos)

| sum / mean over 5 | CPU Qwen | **GPU Qwen (q0)** | + fit-up guard | + guard + fit-up 0.85 |
|---|---:|---:|---:|---:|
| WER mean / CER mean | 0.024 / 0.095 | **0.016 / 0.089** | 0.024 / 0.097 | 0.027 / 0.100 |
| bad units / final word missing | 4 / 1 | **2 / 0** | 3 / 0 | 3 / 0 |
| source speech without dub s | 38.3 | **35.2** | 38.0 | 36.7 |
| audio stage s (5 videos) | 2890 | **594** | 606 | 577 |

## Connected sentences (human feedback: "предложения не связаны, размазаны по времени")

* `--sentence-flow-gap G`: inside one sentence, source pauses shorter than G are not reproduced (one connected phrase). Human A/B on the
  listening clips: 1.5 s better than 0, "not yet ideal". Cost: the connected phrase ends earlier than the source
  (03 source-without-dub 10.6 -> 19.4 s at 1.5 s) -> next: fill (fit-up) / mild tempo, not more pauses.
* 04 30-31 s "обрыв before will make life easier": the 31 s sentence was split for TTS after "которое" (widest gap) -> "...application
  which | will make life easier". `sentence_units` now splits at the most balanced clause boundary of the punctuation pass
  ("...приложения | которое облегчит").
* 04 11-12 s "бред": the source enumerates with long pauses ("город Сочи | 28 августа | сегодня"), the translation reordered it ("Sochi
  city today, August 28") so the dub parts land on the wrong source phrases; "Sochi" is mispronounced (Whisper: "Salkis"). Open.
* `--pause-marks qwen` (+ `--long-pause-grow 1`): Qwen marks natural pause points; offline weak-boundary pause seconds 46.8 -> 21.9.
