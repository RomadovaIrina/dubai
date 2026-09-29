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

## Results — see the tables below (filled from `/tmp/dabai_quality/manifests/*_dub_coverage.json`)
