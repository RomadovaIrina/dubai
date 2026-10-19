# Agent C, P2: proposed timing policy, residual list for the video side, layer-3 design + GPU request (2026-10-16)

## 1. Proposed policy (runner flags, on top of cand5's audio flags)

```
--alignment burst --phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on      # cand5, unchanged
--sentence-flow-gap 0.2 --spread pauses --fill-slowdown 0.875                                          # new (0.85 only if the listening test accepts it)
```

* `--spread off|centre|pauses` (+ `--spread-ok 0.3 --spread-head-max 0.2 --spread-pause-max 0.25 --spread-pause-total-max 0.35 --spread-min-seg 0.5`)
  is now in `run_clean_pipeline_05.py` (`align_units_burst`, commit ed5195c); default `off` = the old behaviour. `recut_offline.py` calls the
  same code: the P1 tracks of all three finalists on 01 / 03 / 04 / 05 are reproduced sample-exact, the cand5 reference still `--check`
  IDENTICAL.
* Expected effect (offline, four videos, `p1_timing_ab.md`): LatentSync frames inside gaps ≥ 0.35 s 525 → 344 (−34 %), gaps > 0.5 s 21 → 15,
  gaps > 1 s 6 → 3, longest 2.0 → 1.3 s, weak-boundary pauses 4 → 2, WER unchanged; cost: 53 of 100 parts at x0.875–0.9, CTC CER +0.003 / +0.02
  on 04 / 05. The reviewed 03 24.5 s gap is gone (0.5 → 0.1 s); 03 34.6 s stays (0.9 s, layer 3 / video).
* Not yet verified end-to-end: a full runner run (TTS on the GPU) would re-synthesise everything, so the numbers would move with the TTS takes;
  the offline recut is the controlled A/B. A GPU E2E run of the final flags belongs to the next shared-GPU window.

## 2. Residual list for the video side

`reports/quality/audio_timing/residual/` (README inside; also `/workspace/dub/artifacts/audio-timing/residual/`): per video, for `cand5` (the
audio of today's videos), `L1_0.875` (proposal) and `L1_0.85`. Each gap has master-frame indices [f0, f1) at 25 fps, LatentSync frames inside,
position in the source burst, the source words Whisper puts there, the sentence, the dub parts around it, the fit-up history, and whether
layer 3 could remove it.

| video | residual gaps ≥ 0.35 s, cand5: n / s / LS frames | L1 + 0.875 | longest |
|---|---|---|---|
| 01 | 8 / 5.3 / 15 | 5 / 3.0 / 13 | 1.7 → 1.1 s |
| 03 | 12 / 9.4 / 240 | 8 / 5.6 / 143 | 2.0 → 1.3 s |
| 04 | 6 / 4.1 / 106 | 5 / 2.6 / 67 | 1.0 → 0.6 s |
| 05 | 8 / 6.4 / 164 | 7 / 4.8 / 121 | 1.3 → 1.1 s |

All residual gaps lie inside a source speech burst; the common case is a `tail` (the English phrase ended, the Russian continues).

## 3. Layer 3: per-burst literal refit (design)

What the residual looks like (L1 + 0.875, 22 of 25 residual gaps are layer-3 candidates, 14.5 s, 316 LatentSync frames): mostly NOT a missing
idea but a shorter English rendering of the same burst — "будет пользоваться популярностью" → "will be popular with" (0.6 s short), "телефоны
уже позволяют" → "modern phones | already allow this", "работать наша система" → "how our system will work." (1.3 s short), "нам всем тот" →
"bringing us all the desired" (0.9 s short). Today's fit-up rewrites the WHOLE sentence toward a word budget; on exactly these units it was
rejected ("added information: ensuring a seamless journey…", "identical", "not closer") or accepted while a burst stayed short.

Algorithm (offline `refit_offline.py` on a finished run; the same function would later replace / follow fit-up in `run_tts_phrase`):
1. **Targets**: units with a residual gap that has `layer3_could_fill`. Per burst k: its Russian words (burst plan), its English part, its
   shortfall s_k (burst − placed part − 0.3 s) and a word budget n_k = words(part_k) + round(s_k · r), r = this speaker's English words per
   second measured on the unit's own TTS, at most +6 words.
2. **One Qwen call per unit**: the Russian sentence as numbered burst pieces + the current English pieces; rewrite ONLY the pieces marked
   SHORT to n_k words by rendering THEIR OWN Russian words more fully and literally ("пользоваться популярностью" → "enjoy great popularity
   among"), keep the other pieces verbatim, the concatenation must stay one fluent sentence. Strict JSON `{"pieces": [...]}` (len = bursts).
3. **Deterministic guards**: unchanged pieces byte-identical; changed pieces within n_k ± 2 words; numbers / names kept (`missing_numbers`);
   no filler; `appended_clause` per piece; no new weak boundary at a piece end (`LONG_PAUSE_FUNC`, `FIXED_PHRASES`).
4. **Judge**: the existing `faithful_check` (Qwen lists facts the source does not state) on source vs old vs new sentence; one feedback retry.
5. **TTS**: one Chatterbox phrase generation of the new sentence (seed rule unchanged) through the existing QA ladder (Whisper + CTC, gap-QA,
   word-QA); CTC spans; cut positions = the piece boundaries (the acoustic DP may move a cut by one word only if it lands in voiced sound);
   L1 placement (flow 0.2, spread, floor 0.875).
6. **Accept per unit** only if: the unit's residual seconds drop ≥ 30 %, no new gap > 0.5 s, no part above x1.15, unit CTC CER ≤ old + 0.05,
   no new weak-boundary pause, judge passed. Otherwise the old unit stays (lossless fallback).
7. **Measure**: `dub_gaps.py`, `pause_rhythm.py`, `eval_cpu.py`, residual lists, listening clips (same windows + the refit units).

Expected gain: if the refit fills half of each targeted shortfall (fit-up historically moved unit ratios 0.6–0.7 → 0.8–0.9), ~150 of the 316
candidate LatentSync frames; the rest goes to the video side.

## 4. Resources / GPU request

| step | where | estimate (01 / 03 / 04 / 05, ~16 units) |
|---|---|---|
| Qwen rewrite + judge (~50 calls) | **CPU** (server B has no CUDA llama.cpp build; Q8_0 at 6 threads) | ~35–50 min, no GPU |
| Chatterbox re-synthesis, ≤ 3 attempts per sentence + Whisper / CTC QA | **GPU**, fp16, ≤ 6 GB VRAM | ~5–10 min |
| recut / metrics / clips | CPU | ~10 min |

Request: **one GPU window of ~10 min with ≤ 6 GB VRAM** (Chatterbox + QA only), at a time Agent B picks (its LatentSync peak is 22–23 GB, so
this would fit next to its stage-1 work but should not overlap a LatentSync pass). CPU-only alternative without any GPU: Chatterbox on the CPU,
~1.5 h total for the four videos.

## 5. Clean background bed (separate task) — status
`bed_p1.md`: on 03 Spleeter 2stems (MIT; weight licence = repo release, counsel note) is the best candidate (no detectable voice residue,
ambience continuous: −0.1 dB vs HDEMUCS −3.0 dB); UMX-HQ leaks the voice; UMX-SE / room tone clean but dip 2–3 dB under speech. Clips:
`/workspace/dub/outbox/audio-bed-p1/`. Next: 01 / 02 / 04 / 05 and the `background_bed` switch after the listening verdict.
