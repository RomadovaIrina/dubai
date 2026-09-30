# first_res review — TTS intelligibility (analysis only, no pipeline code changed)

Scope: `/workspace/dub/first_res/` contains only **01_e2e_final.mp4** and **03_e2e_final.mp4** (02/04/05 are not there).
Method limit: nothing here is a human listening judgement. "Heard" = output of two independent ASRs that are never given the expected text
(faster-whisper large-v3 free decode, wav2vec2-base-960h greedy CTC = acoustic only). Whisper invents text on 1-2 s fragments, so per-fragment
Whisper numbers are noisy; CTC CER and paired comparisons on the same audio are the reliable signal. Listen to `worst_chunks/` to confirm by ear.

## 0. Lineage reproduction
No manifests / work dirs existed. `scripts/diag/repro_audio.py` re-ran only the audio stages of `run_clean_pipeline_05.py --alignment burst`
(unchanged code, same seed 1247). Reproduced dubbed track vs the audio of the real first_res videos: sample-level correlation **0.9998 (01) / 0.9999 (03)**,
lag 0 -> the reproduced translation / parts / raw TTS / trim / atempo / placement ARE what is in first_res. Regenerating every raw part with
production settings reproduces the runner's raw wav exactly (corr 1.0, 44/44 parts of 01).

## 1. Videos
| file | duration | res / fps | audio | source -> dub | original |
|---|---|---|---|---|---|
| 01_e2e_final.mp4 | 101.68 s | 480x848 / 25 | AAC 24 kHz mono 92 kb/s | ru -> en (Whisper en p=0.989) | test_videos/01.MP4 (101.67 s, 30 fps VFR, AAC 44.1k stereo) |
| 03_e2e_final.mp4 | 90.40 s | 464x848 / 25 | AAC 24 kHz mono 72 kb/s | ru -> en (en p=0.934) | test_videos/03.mp4 (90.33 s, 30 fps) |

## 2. ROOT CAUSE (confirmed by controlled A/B on identical text / seed / reference)
**The production fp16 cast corrupts Chatterbox's speech-token ids.**
`S3Token2Mel.forward` (chatterbox/models/s3gen/s3gen.py:208-212) casts EVERY tensor of `ref_dict` to `self.dtype`, including the int64 `prompt_token`.
In fp16 the prompt tokens become float16, and `torch.concat([prompt_token, token])` in `flow.inference` (flow.py:161) promotes the GENERATED speech-token ids
to float16 too. Integers > 2048 are not representable in fp16 (> 4096 only multiples of 4): 43.8 % of the 6561 ids change, error up to +-2 id units.
Visible symptom in the logs: `ERROR ... 6560.0>6561 out-of-range special tokens found in flow` (fp16(6561) rounds to 6560) - 44x in 01, 2x in 03.
fp32 (upstream default) is exact. `scripts/check_chatterbox.py` validated fp16 only for "not silent / finite", never for intelligibility.

| condition (unit level, 38 units x 3 seeds) | Whisper WER | CTC CER | final word missing | extra words |
|---|---|---|---|---|
| **prod: fp16** | **0.745** | **0.395** | 41 | 1.10 |
| fp32 | 0.094 | 0.089 | 7 | 0.37 |
| **fp16 + int64 ids (fp16fix)** | **0.084** | **0.083** | 8 | 0.39 |
| built-in voice fp32 (ceiling) | 0.045 | 0.052 | 6 | 0.37 |

prod is worse than fp16fix in **38 of 38** units (mean dCER +0.313). fp16fix == fp32 (dCER +0.006) -> the fix keeps fp16 speed.
Per part (77 parts, same text/seed): CTC CER 01: 0.387 -> 0.094, 03: 0.495 -> 0.170; parts with CER>=0.3: 50 -> 7.

## 3. Hypotheses
| | verdict | evidence |
|---|---|---|
| H1 raw Chatterbox bad | **YES, dominant** | 49 of 54 bad chunks; see section 2 |
| H2 VAD trim cuts endings | **mostly no** | clean audio: CER none 0.126 / current 0.131 / +100..300 ms 0.125..0.133 (no gain). 1 severe head cut (01 u21 "Balcony.", 160 ms = "Ba" removed, raw CER 0.00 -> 0.43) |
| H3 atempo hurts | **yes above ~1.3, secondary** | same clean chunk, sweep x1.0/1.15/1.3/1.45/1.6: CER 0.131/0.125/0.148/0.180/0.205; last-3-letters lost 10/9/15/15/19; final word missing 6/7/8/13/15. Safe <=1.15. In first_res the >1.2 bins are 100 % bad but confounded: bad raw is longer, hence needs more speed-up |
| H4 fragments too small | **yes** | fp16fix, 20 multi-part units: one call CER 0.052 vs parts 0.092 (better in 13, worse in 4); parts are 1.3-2x longer (03 u08: 9.6 s vs 4.6 s) -> this inflation is what forces atempo 1.6. 17 of 77 parts are <=2 words, 15 come from `proportional_fallback` |
| H5 fp16 (added) | **the root cause** | section 2 |
| speaker reference | **no** | cleanest 3-6 s word-aligned ref: dCER +0.034, worse in 17 of 38 units. (The prod ref is dense fast Russian, no pauses, but it is not what breaks intelligibility.) |
| temperature 0.6/0.5/0.4 (after fix) | **small** | dCER -0.012 / -0.012 / -0.019 (better in 10-11, worse in 5-10 of 38). Do not change the global default on this; use as retry setting |
| cfg_weight 0 (exploratory) | worse | dCER +0.033, speech 15 % slower |
| comma instead of forced '.' on non-final fragments | worse | non-final parts CER 0.168 -> 0.211 |

## 4. "aaa" / non-lexical vocalizations
Chunks where either ASR hears um/uh/ah/oh/ha/er that is not in the text: **prod 10 / 77, fp16fix 1 / 77, fp32 1 / 77**. Whisper-located in the FINAL first_res audio:
01 35.3-35.6 "Um," (u10_p1 "stem of heat,"), 01 63.6-63.7 "Oh," (u18_p0 "All necessary plumbing,"), 03 10.2-10.4 "Oh," (u00_p4), 03 36.2-36.5 "Um," (u03_p1),
03 48.1-48.3 "Uh" (u04_p3 "you and"), 03 83.8-84.1 "Um," and 84.8-85.3 "um..." (u08_p1 "and will", raw 3.48 s for 2 words).
Spectrograms (`visual/aaa_candidates.png`) show a stationary, harmonic, loud band at the END of several raw chunks after the last word
(03 u08_p1 3.1-3.48 s, 03 u06_p1 0.9-1.2 s, 03 u07_p2 1.1-1.36 s, 01 u21 0.8-1.08 s) - a held vowel before EOS. It exists in RAW (not created by trim/atempo), is
not removed by the VAD trim, and disappears with int64 ids (all but 03 u04_p3). The one remaining case is a 2-word fragment.

## 5. Root-cause counts (54 chunks bad in the final audio: CTC CER >= 0.30 or a stray interjection, of 77)
raw TTS (fp16 ids): 49 | bad chunking (fragment still bad after fix): 1 (03 u02_p4 "what") | VAD trim: 1 (01 u21) | neighbour bleed (context of an adjacent bad chunk, 60 ms apart): 2 | unknown: 1 (01 u15_p1 "9 mm").
Contributing (secondary): atempo >= 1.25 on 6 chunks, bad chunking on 6. voice reference: 0. translation/text: see section 6 (affects correctness, not these counts).
Residual after the numeric fix (raw fixed CER >= 0.30): 7 parts, all 1-3-word fragments: 01 "stem of heat," / "sound insulation." / "0.5%", 03 "what" / "to have" / "you and" / "and will".

## 6. Text / chunking layer (not TTS)
* 03 is punctuation-less monologue: Whisper units are cut mid-sentence ("...пятница 28 | августа...", "...чтобы на | пути..."); the number "28" is lost in translation ("today Friday");
  text reaches TTS lower-case without punctuation; u00/u02/u08 use the proportional fallback -> "what", "and will", "to get to", "you and".
* 01: translation errors come from Whisper mishearing ("стебель" for "степень" -> "stem of heat"; "делке" for "отделке" -> "Prime deal"), "Neometria" is voiced "Blumetria" in prod.
* slot_fallback (23 parts) speeds up with no cap (01 u21-like ratios up to 1.69).

## 7. ASR intelligibility of the REAL first_res audio (unit level, natural sentences)
01: Whisper WER 0.88, CTC CER 0.41 (raw 0.80 / 0.40: post-processing adds ~0.08 WER); 03: WER 0.40, CER 0.42. Worst units: 01 t=43.6 "Interior doors." (heard "I'm going to show you the voice."),
74.6 "Balcony." (trim), 50.8 "Bathroom." ("That's all."), 16.8 "Walls. Wallpaper on a...", 03 t=83.9 "and will" ("Um, well, she, um... What's up?"). Full table: `review/problem_timestamps.csv`.

## 8. Visual (limited)
01 is a full-body shot: face is SMALL_FACE almost everywhere -> pass-through, the original Russian lips remain under English audio (by design; not evaluated further).
03 (talking head): mouth area is smoother / pinker than the original (LatentSync+CodeFormer look), no seam visible in the stills I checked (84.0-85.4 s, 22.4-24.0 s). Sync cannot be judged from stills; a frame-diff map was
uninformative (re-encode noise). Video stage not touched - the audio defect is proven.

## 9. Files
`/workspace/dub/tts_diag/`: `repro/` (reproduced work dirs + manifests), `chunks/<id>/<uNN_pK>/{A_raw_tts,B_after_trim,C_after_atempo,D_final_track,E_first_res,A_prod_fp16,A_fp16fix,A_fp32}.wav`
+ `chunks.csv/json` (all required fields), `worst_chunks/` (20 worst, A/B/C/D/E + FIXED wav + csv/json), `review/problem_timestamps.csv`, `review/chunks_verdict.json`,
`exp/` (regeneration grid + scores.json), `punct/`, `visual/`. Code (new, untracked): `dubai/scripts/diag/*.py`.
