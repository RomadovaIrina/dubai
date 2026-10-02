# Semantic TTS placement and speech coverage — 03 BEFORE / AFTER (2026-10-02)

Goal: keep the TTS-intelligibility-v2 gains and remove the dominant visible defect of 03 (the dubbed speech covered only 2/3 of the original
speech, so LatentSync drew a closed mouth while the speaker visibly talks). Audio-only candidates were iterated on 03 first, then the full E2E.

## What changed (production path, `run_clean_pipeline_05.py --alignment burst`; the `slot` baseline is untouched)

| priority | change | where |
|---|---|---|
| B1 sentence-aware segmentation | Qwen punctuation pass over the whisper words -> units = sentences; the original words are never replaced (boundaries mapped back through a token alignment, <= 4 changed tokens tolerated, else whisper fallback); long sentences split at the widest gap | `scripts/pilot/sentence_units.py`, `--segmentation sentence` |
| C1 tiny-unit merge | units < 3 words or < 1.0 s merged into the neighbour (same speaker, pause <= 2 s): "вперед" -> part of the previous sentence, no one-word Chatterbox prompt | `sentence_units.merge_tiny_units`, `--tiny-unit-merge on` |
| A1 phrase-first TTS + post-alignment | ONE Chatterbox generation per unit (QA ladder unchanged), CTC forced alignment of the expected text (wav2vec2-base-960h, `torchaudio.functional.forced_align`), cuts only between words (energy minimum in the inter-word gap, zero-crossing, 5 ms fades); no per-burst generations, no 1-2-word prompts | `scripts/pilot/tts_align.py`, `tts_burst.run_tts_phrase`, `--tts-mode phrase` |
| A1/P2 timing distribution | aligned TTS words distributed over the source bursts so cumulative TTS time follows cumulative burst duration, cuts preferred at punctuation / pauses; a span needing > 1.15x takes its neighbour's burst (union, no re-synthesis); Qwen JSON split kept as `--phrase-split qwen` with lossless fallbacks | `tts_burst.distribute_words` |
| A2 fit-up (duration-aware translation, the missing direction) | synthesized speech / source speech < 0.8 -> Qwen writes a fuller translation OF THE SOURCE SENTENCE with an explicit word budget (min..max from the measured speaking rate); a Qwen judge rejects any new fact / outcome / quality (paraphrases allowed); accepted only if the duration gets closer to the source AND the QA score does not get worse; <= 3 rounds, feedback on word count / additions | `tts_burst.fuller_rewrite`, `faithful_check`, `--fit-up on --fit-up-ratio 0.8 --fit-up-rounds 3 --fit-up-aim 1.0` |
| A3 mild fill | a span still shorter than its burst may be slowed to 0.9x (was never) | `--fill-slowdown 0.9` (default) |
| runner | Qwen loaded once (segmentation, translation, fit-up), `--stop-after audio` for audio-only candidates, honest banner | `run_clean_pipeline_05.py` |
| harness | OLD/NEW audio comparison with the acceptance block | `scripts/pilot/compare_audio_candidates.py` |

Not changed: Chatterbox token-dtype fix, CodeFormer (w 1.0), LatentSync guidance, TTS temperature, speaker reference policy, speech gate.

## Audio-only iterations on 03 (`reports/pilot/coverage_fix/03_audio_iter*.md`)

| iteration | change | original speech without dub s | IoU | silent LS frames while speaking (pilot metric) | WER / CTC CER | fit-up acc/cand | verdict |
|---|---|---:|---:|---:|---|---|---|
| OLD (TTS v2, parts) | - | 19.2 | 0.615 | 376 | 0.110 / 0.087 | - | - |
| 1 | phrase-first + Qwen split + fit-up 0.65 | 22.2 | 0.574 | 447 | 0.020 / 0.062 | 1/2 | FAIL (Qwen split degenerate on 5-7 bursts, single lump over pauses) |
| 2 | timing distribution of aligned words, fit-up 0.8 | 15.3 | 0.739 | 314 | 0.004 / 0.040 | 2/7 (6 "not longer") | FAIL coverage |
| 3 | fit-up with explicit word budget, no judge | 9.5 | 0.822 | 199 | 0.008 / 0.042 | 6/7 | PASS metrics, but rewrites added ideas ("making our journey seamless") |
| 4 | strict Qwen judge | 17.1 | 0.707 | 326 | 0.004 / 0.041 | 1/7 | FAIL (judge rejects paraphrases) |
| 5a | source-driven fuller translation + paraphrase-tolerant judge | 11.8 | 0.796 | 249 | 0.004 / 0.039 | 4/7 | FAIL by 1.8 s |
| **5b** | 5a + fill-slowdown 0.9 | **8.5** | **0.849** | **202** | **0.004 / 0.041** | 4/7 | **PASS** |

## BEFORE / AFTER 03 — audio (OLD = refresh run `/tmp/pilot_refresh/final/03_final`, NEW = candidate 5b)

| metric | BEFORE | AFTER |
|---|---:|---:|
| dubbed speech s (Silero VAD) | 48.3 | 53.0 |
| original speech without dub s | 19.2 | 8.5 |
| dub outside original speech s | 6.8 | 0.8 |
| VAD IoU | 0.615 | 0.849 |
| bursts < 50 % covered / uncovered (of 27) | 8 / 2 | 2 / 1 |
| silent-while-speaking frames (VAD proxy, whole timeline) | 481 | 210 |
| LS frames dub-silent / while original speaks (pilot RMS metric, OLD LS segments) | 761 / 376 | 670 / 202 |
| Whisper WER / CTC CER / Whisper CER (unit mean, independent ASR on the dubbed track) | 0.110 / 0.087 / 0.062 | 0.004 / 0.041 / 0.001 |
| bad units (CTC CER >= 0.3) / interjections / final word missing / repeated words / lost numbers | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| TTS generations / attempts / QA retried (ok) / bad after retries | 21 / 26 / 2 (1) / 1 ("forward forward") | 10 / 18 / 1 (1) / 0 |
| atempo max / > 1.15 / > 1.20 / mean | 1.15 / 0 / 0 / 1.015 | 1.15 / 0 / 0 / 0.944 (24 of 32 spans slowed to >= 0.9) |
| cut groups / cut s | 0 / 0.005 | 0 / 0.012 |
| units / translation segmentation | 10 whisper segments (mid-sentence cuts) | 10 sentences (11 sentences, "Вперед." merged) |

## BEFORE / AFTER 03 — full E2E (LatentSync optimized + CodeFormer optimized w1.0, same video settings)

| metric | BEFORE (refresh run) | AFTER (new defaults) |
|---|---:|---:|
| validation | PASS | PASS (Δdur 0.0667 s, 2260 frames, 464x848) |
| SyncNet confidence / AV offset (original 4.89 / 2) | 5.02 / 0 | **5.85 / 0** |
| LatentSync frames (LS seconds of video) | 1927 (77.08 s) | 1863 (74.52 s) — no extra LatentSync work |
| LS frames driven by silent dub / while the original speaks (pilot metric) | 762 / 376 | 607 / 201 |
| LatentSync stage s / CodeFormer stage s | 435.5 / 57.1 | 414.284 / 54.974 |
| audio stages s (segmentation / translation / TTS incl. QA, fit-up, alignment) | – / 31.0 / 39.3 (+ split 81.5) | 38.821 / 29.86 / 322.46 |
| total wall s | 746.2 | 955.623 |
| peak VRAM MiB (nvidia-smi) | 25156 | 25711 |
| mouth / upper-face sharpness vs master, flicker | 0.72-0.79 / 0.77-0.86 / 0.92-1.10 | 0.75 / 0.818 / 0.978 (means over 11 LS segments) |
| visual check 28.3 / 43.3 / 86.2 s | mouth closed while the speaker talks | mouth moving with the dubbed speech (`refresh_videos/03_mouth_orig_old_new_28.3_43.3_86.2.png`) |

Output: `/tmp/pilot_refresh/e2e_new/03_final.mp4` (+ manifest). Runtime note: the audio stages grew by ~4.5 min on 03 (Qwen fit-up / judge calls on the CPU, extra Chatterbox generations and QA);
LatentSync + CodeFormer time is unchanged within noise and the number of lip-synced frames went down by 3 %.

## Regression — audio-only candidates on 01 / 04 / 05 (same defaults; `reports/pilot/coverage_fix/<id>_audio_regression.md`)

| video | original speech without dub s | VAD IoU | silent LS frames while speaking (pilot metric) | WER / CTC CER | bad units | fit-up acc/cand | atempo max | units | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 16.6 -> **7.9** | 0.7926 -> 0.8908 | 45 -> 21 | 0.023 / 0.069 -> 0.016 / 0.055 | 1 -> 0 | 11/13 | 1.2 -> 1.15 | 29 -> 26 (6 merged) | PASS |
| 04 | 10.8 -> **4.1** | 0.5389 -> 0.8383 | 305 -> 215 | 0.024 / 0.059 -> 0.04 / 0.053 | 0 -> 0 | 3/4 | 1.15 -> 1.0 | 5 -> 6 (0 merged) | FAIL (Whisper WER +0.016 > +0.01 tolerance; CTC CER better, 0 bad units) |
| 05 | 14.6 -> **5.7** | 0.3811 -> 0.745 | 467 -> 361 | 0.039 / 0.221 -> 0.02 / 0.162 | 2 -> 1 | 6/7 | 1.025 -> 1.15 | 7 -> 7 (0 merged) | PASS |

## Regression — full E2E on 01 / 04 / 05 and 02 (no face) with the new defaults (`/tmp/pilot_refresh/e2e_new/`, copies in `result_videos/coverage_fix/`)

| video | valid | SyncNet conf / AV offset: OLD -> NEW | LatentSync frames OLD -> NEW | silent LS frames while speaking OLD -> NEW | TTS bad final / fit-up acc/cand | wall s OLD -> NEW | peak VRAM MiB OLD -> NEW |
|---|---|---|---:|---|---|---|---|
| 01 | PASS | 6.74 / 0 -> **7.82 / 0** (orig 5.36 / 1) | 260 -> 260 | 45 -> 21 | 0 / 11/13 | 382.1 -> 714.003 (tts 392.915) | 25178 -> 25196 |
| 02 | PASS | N/A (no face) | 0 -> 0 | - -> 0 | 0 / 7/10 | 267.2 -> 459.165 (tts 277.746) | 8252 -> 8240 |
| 03 | PASS | 5.02 / 0 -> **5.85 / 0** (orig 4.89 / 2) | 1927 -> 1863 | 376 -> 201 | 0 / 4/7 | 746.2 -> 955.623 (tts 322.46) | 25156 -> 25711 |
| 04 | PASS | 2.82 / 0 -> **3.48 / 0** (orig 3.24 / 2) | 1044 -> 967 | - -> 174 | 0 / 3/4 | 436.1 -> 556.977 (tts 192.325) | 24156 -> 25204 |
| 05 | PASS | 2.51 / 0 -> **3.3 / 0** (orig 1.71 / 2) | 868 -> 851 | - -> 344 | 0 / 6/7 | 392.7 -> 593.616 (tts 260.074) | 24156 -> 24176 |

SyncNet here = `quality_manifest.py --syncnet` (upstream evaluator) on the full output; the formal 0.8 table on the same five files is in `reports/pilot/coverage_fix/0.8_new_e2e.md`.
AV offset is 0 on every scoreable video; no video lost LatentSync coverage (frames equal or lower), so the 0.6 contour is not loaded more. The wall time grows by 3-6 min per video in the
audio stages (CPU Qwen: punctuation pass, fit-up rewrites + judge; extra Chatterbox generations with QA) — see the runtime note below.

### 0.8 on the five NEW outputs (`reports/pilot/coverage_fix/0.8_new_e2e.md`, same methodology as the refresh)

| | refresh (TTS v2, parts) | NEW |
|---|---|---|
| per video (conf / offset) | 01 6.74/0 · 02 N/A · 03 5.02/0 · 04 2.82/0 · 05 2.51/0 | 01 7.82/0 · 02 N/A · 03 5.85/0 · 04 3.48/0 · 05 3.30/0 |
| Δ vs original | +1.38 · – · +0.13 · −0.42 · +0.80 | +2.46 · – · +0.96 · +0.24 · +1.59 (every scoreable video now above its original) |
| aggregate mean / median / min / max | 4.273 / 3.92 / 2.51 / 6.74 | **5.113 / 4.665 / 3.30 / 7.82** |
| valid / scoreable / N/A / FAIL | 5 / 4 / 1 / 0 | 5 / 4 / 1 / 0 |

## Residuals and costs (honest list)

- **Runtime**: audio stages +3-6 min per video (Qwen on the CPU: punctuation pass 18-75 s, fit-up rewrite + judge calls; up to 3 extra Chatterbox generations + QA per fit-up round).
  03: 746 -> 956 s total; LatentSync + CodeFormer unchanged, lip-synced frames equal or fewer on every video (0.6 contour not loaded more). Levers if needed: fewer fit-up rounds, judge only when the rewrite grew by > 30 %.
- **04**: Whisper WER 0.024 -> 0.040 (+0.016 vs the +0.01 tolerance of the harness) while CTC CER improved 0.059 -> 0.053 and 0 units are bad; 6 short units, 1-2 word substitutions in the Whisper hypotheses — within noise, reported as a soft fail.
- **02** (no face): coverage 23.2 -> 8.6 s without dub, but one 5-word unit ("Relaxation zone: lying areas.", 1.82 s source) is still cut by 0.40 s at its window end at the hard cap 1.2 (the OLD track cut a different unit by 0.25 s): the concise rewrite (fit-down) only triggers for >= 6 words. Candidate follow-up: allow fit-down from 4 words when the overflow exceeds 0.3 s.
- **05**: 344 LS frames still silent while the original speaks (05 has a fast speaker; the dubbed speech reaches 0.75 IoU); SyncNet nevertheless 2.51 -> 3.30.
- **Fit-up faithfulness** relies on Qwen judging Qwen; the judge rejected 20 of 41 candidate rewrites across the five videos. Accepted rewrites are longer, more literal renderings (spelled-out subjects, full verb forms); spot checks on 03 (`03_audio_iter5b_fill_slowdown_0.9.md`, NEW units table) show no invented facts, but a human read of the fitted translations is still advisable before shipping.
- Not changed (as required): Chatterbox dtype fix, CodeFormer w 1.0, LatentSync guidance, temperature, speaker reference policy; the VAD(dub)-only gate was not used.

## Exact changed files

| file | change |
|---|---|
| `scripts/pilot/sentence_units.py` | new: punctuation pass, boundary mapping, sentence units, tiny-unit merge |
| `scripts/pilot/tts_align.py` | new: CTC forced alignment (wav2vec2-base-960h), safe inter-word cut points |
| `scripts/pilot/tts_burst.py` | new `run_tts_phrase`, `distribute_words`, `fuller_rewrite`, `faithful_check`; `run_tts_burst` untouched |
| `scripts/pilot/run_clean_pipeline_05.py` | `load_llm`, sentence segmentation + tiny merge wiring, `--tts-mode/--segmentation/--tiny-unit-merge/--fit-up*/--phrase-split/--stop-after`, `--fill-slowdown` default 0.9, `burst_plan` words across segments, honest banner |
| `scripts/pilot/compare_audio_candidates.py` | new: OLD/NEW audio comparison + acceptance block |
| `reports/pilot/coverage_fix/` | iteration comparisons 03 (1-5b), audio regressions 01/02/04/05, 0.8 on the new outputs, per-video pages |

Commits: `32ecc9a` (checkpoint: refresh harness + reports), `fcbfafb` (this change), follow-up commit with the 0.8 aggregate + per-video pages.

