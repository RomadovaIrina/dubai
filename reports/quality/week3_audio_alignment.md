# Week 3 — temporal alignment of the synthesized speech (summary) (2026-09-29)

Detail reports: `phaseA_e1_placement.md` (E1 placement A/B), `phaseB_burst_aware.md` (burst-aware path, variants), `phaseC_whisperx.md`
(WhisperX A/B). Artifacts: `result_videos/week3_audio/{phaseA_e1,phaseB_burst,phaseC_whisperx,final}/`. Code: `scripts/quality/e1_burst_align.py`
(shared placement), `burst_aware_audio.py` (candidate path), `dub_coverage.py`, `e1_cut_words.py`, `phaseC_whisperx_ab.py`,
`run_clean_pipeline_05.py --alignment burst` (production path; `slot` = frozen baseline, untouched).

## Cause of the silent lip-synced frames
Units = whole whisper segments (4-15 s, 2-6 speech bursts each); one TTS clip per unit, placed at the slot start at natural speed, the
rest of the slot silent (tts/slot 0.40-0.58 on 03/04/05). Result: 38-51 % of the LatentSync frames on 03/04/05 were driven by silence,
423 (04) / 458 (05) / 444 (03) / 46 (01) of them while the original speaker is audible. Not a whisper/word-timestamp problem.

## What was built
1. **Phase A**: existing `e1_burst_align.py` (placement only) confirms the hypothesis (04: 423 -> 229, SyncNet 2.76 -> 3.42) but cuts words
   (7/14 valley cuts inside words on 04, 10/14 on 05, WhisperX-verified) and groups text by duration only -> not shippable.
2. **Phase B**: burst-aware path. Source bursts (Silero VAD in the slot) get the source words by whisper word start; the unit's existing
   Qwen translation is cut by Qwen into exactly N parts as strictly validated JSON (retry with the error fed back, lossless
   proportional fallback; 04: 5/5 units by Qwen, 05: 6/7); Chatterbox per part; each part trimmed to its speech span and fitted into its
   burst by the shared `place_groups` (window to the next burst, atempo cap 1.3, hard cap 1.6, spill into the pause before the next
   unit instead of cutting). Per-unit safety rule: when the burst fit would need > 1.3x on half or more of a unit's groups, the unit
   falls back to the baseline slot rule (concatenated parts, natural speed, uncapped speed-up only if longer than the slot, never cut).
3. **Phase C**: WhisperX installed `--no-deps` (its pins conflict with the frozen torch/ctranslate2/numpy/pyannote) and measured on the
   TTS chunks: faster-whisper hallucinates short synthetic chunks (word count right in 5/19, 1/21), WhisperX forced alignment is right
   on all, but it absorbs leading/trailing silence, so chunk trimming stays on Silero VAD and placement is driven by the source side.
   WhisperX is the tool for TTS word boundaries (cut-word verification, future intra-part splits), not part of the production path.

## Results — dataset, LS-only video stage on the baseline work dirs (candidate path), SyncNet on the full output

| video | policy | silent while speaking (pilot RMS metric) | VAD IoU | orig speech w/o dub s | dub outside orig s | SyncNet (LS-only base / orig) | AV | groups | atempo > 1.25 / > 1.3 / max | overflow / cut | units on slot fallback | Qwen split ok / proportional |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---|
| 04 | baseline | 423 | 0.356 | 15.4 | 10.1 | 2.76 (2.76 / 3.24) | 0 | 5 | 0 / 0 / 1.0 | – / – | – / 5 | – |
| 04 | e1 | 229 | 0.678 | 7.8 | 2.5 | 3.42 (2.76 / 3.24) | 0 | 19 | 1 / 0 / 1.3 | 1 / 0 | – / 5 | – |
| 04 | burst_s | 186 | 0.695 | 5.4 | 5.2 | 3.31 (2.76 / 3.24) | 0 | 19 | 3 / 2 / 1.6 | 2 / 0 | 0 / 5 | 5 / 0 |
| 04 | burst_sfb | 190 | 0.602 | 6.7 | 8.4 | 3.3 (2.76 / 3.24) | 0 | 14 | 1 / 0 / 1.276 | 0 / 0 | 1 / 5 | 5 / 0 |
| 05 | baseline | 458 | 0.348 | 15.3 | 8.3 | 2.61 (2.61 / 1.71) | 0 | 7 | 0 / 0 / 1.0 | – / – | – / 7 | – |
| 05 | e1 | 453 | 0.509 | 11.3 | 4.7 | 2.88 (2.61 / 1.71) | 0 | 20 | 1 / 0 / 1.3 | 1 / 1 | – / 7 | – |
| 05 | burst_s | 305 | 0.699 | 4.9 | 5.0 | 2.93 (2.61 / 1.71) | 0 | 21 | 3 / 2 / 1.6 | 1 / 1 | 0 / 7 | 6 / 1 |
| 05 | burst_sfb | 305 | 0.699 | 4.9 | 5.0 | 2.93 (2.61 / 1.71) | 0 | 21 | 3 / 2 / 1.6 | 1 / 1 | 0 / 7 | 6 / 1 |
| 03 | burst_s | 155 | 0.756 | 7.3 | 9.9 | 6.1 (4.65 / 4.89) | 0 | 33 | 9 / 4 / 1.6 | 7 / 2 | 0 / 9 | 6 / 3 |
| 03 | burst_sfb | 141 | 0.766 | 6.6 | 9.9 | 6.12 (4.65 / 4.89) | 0 | 32 | 8 / 3 / 1.6 | 6 / 1 | 1 / 9 | 6 / 3 |
| 01 | burst_s | 40 | 0.816 | 14.3 | 3.2 | 6.22 (6.84 / 5.36) | 0 | 44 | 15 / 10 / 1.6 | 14 / 2 | 0 / 29 | 11 / 0 |
| 01 | burst_sfb | 29 | 0.848 | 10.9 | 3.6 | 6.61 (6.84 / 5.36) | 0 | 32 | 5 / 2 / 1.6 | 3 / 2 | 9 / 29 | 11 / 0 |

Baseline pilot metric per video: 04 423, 05 458, 03 444, 01 46. Timeline unchanged in every run (slots never move; output duration / frame count validated).

## Final policy: `--alignment burst` = spill on + per-unit slot fallback at 50 % (`burst_sfb`)

| video | silent while speaking | VAD IoU | SyncNet vs LS-only baseline | atempo > 1.3 | cut groups |
|---|---:|---:|---|---:|---:|
| 04 | 190 (baseline 423, 55 % fewer) | 0.602 | 3.3 vs 2.76 (+0.54) | 0 of 14 | 0 |
| 05 | 305 (baseline 458, 33 % fewer) | 0.699 | 2.93 vs 2.61 (+0.32) | 2 of 21 | 1 |
| 03 | 141 (baseline 444, 68 % fewer) | 0.766 | 6.12 vs 4.65 (+1.47) | 3 of 32 | 1 |
| 01 | 29 (baseline 46, 37 % fewer) | 0.848 | 6.61 vs 6.84 (-0.23) | 2 of 32 | 2 |


## E2E — full pipeline from the source video, `--alignment burst --codeformer optimized` (frozen CodeFormer w 1.0 / batch 8 / fp16)

| video | valid | silent while speaking | VAD IoU | SyncNet (LS-only baseline / +CodeFormer baseline) | AV | units on slot fallback | atempo > 1.3 / cut | Qwen split ok / proportional | CodeFormer faces / s | wall s | peak VRAM MiB |
|---|---|---:|---:|---|---:|---:|---|---|---|---:|---:|
| 04 | PASS | 190 | 0.602 | 3.2 (2.76 / 2.67) | 0 | 1 / 5 | 0 / 0 | 5 / 0 (single-burst units 0) | 1090 / 31.114 | 372.892 | 22492 |
| 05 | PASS | 305 | 0.699 | 2.94 (2.61 / 2.5) | 0 | 0 / 7 | 2 / 1 | 6 / 1 (single-burst units 0) | 882 / 25.835 | 336.154 | 22492 |
| 03 | PASS | 141 | 0.766 | 5.88 (4.65 / 4.37) | 0 | 1 / 9 | 3 / 1 | 6 / 3 (single-burst units 0) | 2016 / 58.076 | 778.299 | 23516 |
| 01 | PASS | 29 | 0.848 | 6.45 (6.84 / 6.57) | 0 | 9 / 29 | 2 / 2 | 11 / 0 (single-burst units 18) | 260 / 7.364 | 269.54 | 22498 |

05 was run before the fallback rule existed; the rule does not trigger on 05 (0 units), so its numbers are the final policy's. SyncNet with CodeFormer is 0.1-0.2 below the LS-only value of the same audio (the known CodeFormer cost) and still above the CodeFormer-only baseline on 04/05/03 (3.20 vs 2.67, 2.94 vs 2.50, 5.88 vs 4.37); 01: 6.45 vs 6.57. Outputs: `result_videos/week3_audio/final/`.



## Reading
- The silent-while-speaking frames fall by 33-68 % on every video (dataset: 1371 -> 665 on the pilot metric), the dubbed speech now
  lies on the original speech (VAD IoU 0.35 -> 0.60-0.85), AV offset stays 0 everywhere, and SyncNet rises on the three
  conversational videos (+0.32 / +0.54 / +1.47) because the lip-synced frames now carry speech in the driving audio.
- 01 (narrated, English TTS as long as the Russian) is the case where burst fitting has nothing to gain and can only speed things up:
  pure burst fitting cost 10/44 groups > 1.3x and -0.62 SyncNet; the per-unit slot fallback (9 of 29 units keep the baseline rule)
  brings it to 2/32 groups and -0.23. The remaining SyncNet gap on 01 is explained, not hidden: those 20 units are still burst-fitted
  with mild speed-ups the slot rule did not need.
- Residual "silent while speaking" (141-305 frames) = burst tails after a chunk shorter than its burst (the English part is shorter
  than the Russian burst; Chatterbox chunk length varies). `--fill-slowdown 0.85` recovers ~20 % more (04: 186 -> 147, 05: 305 -> 275)
  at the price of slowed chunks and -0.12 SyncNet on 04; kept as an option, not the default ("no robotic stretching").
- Still open: 1-2 chunks per video are cut at a unit boundary where there is no pause to spill into (05/03: 1, 01: 2), 2-3 groups per
  video run at the 1.6 hard cap, and the Qwen cut follows the English word order, not the Russian one, so a burst sometimes carries the
  neighbouring clause. These are translation-length and TTS-duration issues; the next lever is asking Qwen for parts whose length
  matches the burst (or re-synthesising an over-long part), not finer word timestamps.

## Acceptance
| criterion | status |
|---|---|
| silent-while-speaking -> ~0 | not reached: -55 % / -33 % / -68 % / -37 % (04/05/03/01); residual explained above |
| AV offset = 0 | met on every run |
| SyncNet >= baseline or explained | 04/05/03 above baseline; 01 -0.23, explained (narrated, TTS not shorter than source) |
| no cut words | met (whole translation parts; WhisperX-verified for E1 as the negative control) |
| no robotic stretching | no slow-down in the default policy (fill-slowdown off) |
| atempo > 1.3 only exceptional | 0 / 2 / 3 / 2 groups of 14 / 21 / 32 / 32 (hard-capped 1.6); 1-2 chunks per video cut at unit ends |
| original timeline unchanged | met (slots never move; duration / frame count validated on every output) |
| WhisperX (B2B) | measured A/B, not in the production path; kept as the TTS word-boundary tool (`phaseC_whisperx.md`) |
