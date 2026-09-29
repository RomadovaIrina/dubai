# Phase B — burst-aware translation / TTS / placement (04, 05) (2026-09-29)

Path: `scripts/quality/burst_aware_audio.py` on the baseline manifests (audio stages of the frozen pipeline reused up to the unit
translation; nothing re-translated), then `candidate_video_stage.py --codeformer off` (LS-only, comparable with the LS-only baseline),
`quality_manifest.py --syncnet`, `dub_coverage.py`. Same path integrated in `run_clean_pipeline_05.py --alignment burst` (default
`slot` = frozen baseline, byte-identical code path; see the E2E row).

Per whisper unit: source bursts = Silero VAD in the slot; source words assigned to bursts by whisper word START (word ends absorb the
following pause); Qwen2.5-7B Q8_0 cuts the unit's existing translation into exactly N parts as strictly validated JSON (len == N, no empty
part, normalized concatenation == translation word for word; retry with the error fed back; lossless proportional fallback, recorded per
unit); Chatterbox per part (seed per part); every chunk trimmed to its VAD span and placed with the shared `e1_burst_align.place_groups`
(window = burst start .. next burst start, atempo cap 1.3, hard cap 1.6 on collision, slot never moved). Variants:
`burst` (plain), `burst_m06` (source bursts < 0.6 s merged into a neighbour), `burst_s` (**spill**: the last chunk of a unit may run into the
pause before the next unit's first burst and uses the hard cap instead of being cut), `burst_m06s` (both).

Qwen split on 04: 4/5 units valid on the first attempt, 1 unit (2 parts) answered a bare JSON list -> now accepted (was proportional
fallback); on 05: 6/7 units, 1 fallback. No unit lost or gained a word (asserted). Cost on 04: Qwen split 19.7 s CPU (5 calls), TTS 22.5 s
for 19 chunks (vs 5 clips), placement < 1 s.

## Results (silent-while-speaking = LS frames with dub RMS < -50 dB while the original is > -35 dB, the pilot metric; SyncNet on the LS-only output)

| video | variant | silent while speaking | dub-silent LS frames | VAD IoU | orig speech w/o dub s | dub outside orig s | bursts < 50 % covered | SyncNet (orig) | AV | groups | atempo > 1.25 / > 1.3 / max | overflow / cut |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| 04 | baseline | 423 | 496 / 1099 | 0.356 | 15.4 | 10.1 | 9 | 2.76 (3.24) | 0 | 5 | 0 / 0 / 1.0 | – |
| 04 | E1 | 229 | 358 / 982 | 0.678 | 7.8 | 2.5 | 1 | 3.42 | 0 | 19 | 1 / 0 / 1.3 | 1 / 0 |
| 04 | burst | 172 | 294 / 1044 | 0.694 | 5.2 | 5.5 | 1 | 3.27 | 0 | 19 | 4 / 2 / 1.6 | 3 / 1 |
| 04 | burst_m06 | 199 | | 0.697 | 5.4 | 5.1 | 0 | 3.29 | 0 | 18 | 2 / 1 / 1.58 | 0 / 0 |
| 04 | burst_m06s | 195 | | 0.697 | 5.4 | 5.1 | 0 | 3.26 | 0 | 18 | 2 / 1 / 1.58 | 0 / 0 |
| 04 | **burst_s** | **186** | | 0.695 | 5.4 | 5.2 | 0 | **3.31** | 0 | 19 | 3 / 2 / 1.6 | 2 / **0** |
| 05 | baseline | 458 | 458 / 899 | 0.348 | 15.3 | 8.3 | 10 | 2.61 (1.71) | 0 | 7 | 0 / 0 / 1.0 | – |
| 05 | E1 | 453 | 454 / 878 | 0.509 | 11.3 | 4.7 | 6 | 2.88 | 0 | 20 | 1 / 0 / 1.3 | 1 / 1 |
| 05 | burst | 299 | 300 / 852 | 0.732 | 4.7 | 3.8 | 1 | 2.64 | 0 | 21 | 6 / 0 / 1.3 | 4 / **4** |
| 05 | burst_m06 | 371 | | 0.538 | 9.4 | 6.5 | 4 | 2.66 | 0 | 18 | 4 / 0 / 1.3 | 3 / 3 |
| 05 | burst_m06s | 362 | | 0.531 | 7.5 | 7.5 | 4 | 2.97 | 0 | 18 | 2 / 2 / 1.6 | 1 / 1 |
| 05 | **burst_s** | **305** | | 0.699 | 4.9 | 5.0 | 1 | **2.93** | 0 | 21 | 3 / 2 / 1.6 | 1 / 1 |

Timeline: unchanged in every variant (slots never move; output duration / frame count validated by candidate_video_stage.py).
Word cuts: none by construction (chunks are whole translation parts); E1's 11/14 valley cuts inside words are gone.

## Reading
- **Chosen policy: `burst_s`** (spill on, no merge). It halves the silent-while-speaking frames on both videos (04 -56 %, 05 -33 %), doubles
  the VAD IoU (0.36 -> 0.70), brings original speech without dub from 15 s to 5 s per video, keeps AV offset 0 and raises SyncNet
  above the baseline on both (3.31 vs 2.76, 2.93 vs 2.61), with no chunk cut on 04 and one on 05.
- Merging short bursts (`m06`) is a net loss on 05: a merged burst spans the pause, the chunk lands at its start and the second half
  stays silent (IoU 0.73 -> 0.54). Not adopted.
- Spill alone removes the slot-end cuts (05: 4 -> 1) because the last chunk may use the pause before the next unit and the hard cap
  instead of being truncated.
- Residual: 2 chunks per video above 1.3x (hard cap 1.6) and ~5 s of original speech still without dub. Both come from chunk length
  vs burst length: Chatterbox renders one-word or two-word parts long (e.g. "ahead" 2.1 s, "wonderful" 1.0 s for a 0.7 s burst) and
  the English text of some bursts is longer than the Russian burst. This is a TTS-duration / translation-length problem, not a
  word-timestamp precision problem — the placement already knows the burst boundaries to 40 ms from Silero VAD and whisper word
  starts. The semantic quality of the Qwen cut is the other visible weakness (consecutive cutting cannot follow RU/EN word-order
  changes: "как" -> "test how our application").
- The remaining "silent while speaking" frames (186 / 305) are mostly burst tails after a short chunk; filling them needs a slow-down
  of short chunks (`--fill-slowdown`, e.g. 0.85) or longer translations per burst — to be A/B-ed next, after the E2E integration run.

## Not needed for this step
WhisperX: the errors left are in chunk duration and translation split, not in where the source words or the TTS words are; faster-whisper
word starts + Silero VAD already place every chunk on its burst. Phase C will still measure it on the residual (word timestamps of the
synthesized TTS vs faster-whisper) because it is a B2B requirement.

## E2E integration check (`run_clean_pipeline_05.py --alignment burst --burst-spill on`, 04, CodeFormer off)
Full pipeline from the source video (VAD -> ASR -> diarization -> translation + Qwen split while Qwen is loaded -> TTS per part -> shared
placement -> LatentSync): validation PASS, 464x848 / 1349 frames / 53.96 s, wall 333 s. It reproduces the `burst_s` candidate numbers
exactly: silent-while-speaking 186, VAD IoU 0.6945, original speech without dub 5.4 s, SyncNet 3.31, AV offset 0 (Qwen at temperature 0
and per-part TTS seeds make the audio path deterministic). `--alignment slot` (default) leaves the frozen baseline path untouched.
Artifacts: `result_videos/week3_audio/phaseB_burst/` (dubbed tracks of every variant, alignment JSON, burst_s / burst mp4, coverage JSON).
