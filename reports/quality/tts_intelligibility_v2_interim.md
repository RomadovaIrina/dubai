# TTS intelligibility v2 — interim E2E results (branch tts-intelligibility-fix)

Fixes (one commit each): fp16 speech-token dtype guard + regression QA (968f33a), natural-phrase chunking (e3d521e), duration-aware fitting + atempo cap 1.15/1.2 (8c241dc),
per-part ASR QA + retry (aa3b267), word-onset protection (553207b). Command per video:
`run_clean_pipeline_05.py --alignment burst --codeformer optimized --video-backend optimized` (LatentSync optimized + CodeFormer optimized, validation PASS on all five).
Media (not in git): `result_videos/tts_intelligibility_v2/<id>/<id>_final.mp4` + manifest, dubbed_24k/16k.wav, intelligibility / tts_qa_retry / alignment / final_quality reports, `summary_row.json`.
"old" for 01/03 = `first_res/*_e2e_final.mp4` (audio reproduced bit-exactly, see first_res_tts_intelligibility_diagnosis.md). Metrics: independent ASR on the AUDIO OF THE FINAL MP4
(Whisper WER, wav2vec2-CTC CER, unit level, vs the dubbed text).

| video | WER old -> new | CER old -> new | bad units old -> new | um/ah units old -> new | missing final word | TTS parts old -> new | retried (success) | max atempo old -> new | parts >1.15 old -> new | cut groups | AV offset | SyncNet old -> new | CodeFormer s | validation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 0.814 -> 0.017 | 0.407 -> 0.068 | 21 -> 1 | 4 -> 0 | 4 -> 0 | 44 -> 32 | 2 (2) | 1.6 -> 1.2 | 9 -> 1 | 0 | 0 | 6.45 -> 6.75 | 9.7 | PASS |
| 03 | 0.413 -> 0.022 | 0.424 -> 0.050 | 8 -> 0 | 6 -> 0 | 1 -> 0 | 33 -> 17 | 1 (1) | 1.6 -> 1.15 | 10 -> 0 | 0 | 0 | 5.88 -> 5.19 | 61.1 | PASS |
| 02 | n/a (no first_res) -> 0.099 | -> 0.122 | -> 1 | -> 0 | -> 0 | -> 18 | 3 (3) | -> 1.15 | -> 0 | 0 | n/a (no face) | n/a | 0 (NO_FACE 2199 frames, all pass-through) | PASS |
| 04 | E2E done, validation PASS; packaging / old baseline pending | | | | | | | | | | | | | PASS |
| 05 | E2E done, validation PASS; packaging / old baseline pending | | | | | | | | | | | | | PASS |

Old problem spots (first_res -> new, independent ASR, `old_vs_new_samples/` wavs in each result folder): 01 @35.3 "Hey, um, what's the matter with the heat?" -> "system of heat, moisture and";
@43.6 "I'm going to show you a tutorial" -> "Interior doors."; @50.8 "First of all, Martha walked out" -> "Bathroom. Matte..."; @63.6 "oh missus ary plemey" -> "All necessary plumbing, including";
@74.6 "How can you go against the ceiling" -> "Balcony, coating the ceiling and walls". 03 @36.2 "um that these are heard" -> "desired result which we wished for to have"; @83.8 "Um, well, she, um..." -> "desire forward";
the number "28" lost in the old translation is now kept (Qwen retry). Remaining on 03: SyncNet -0.69 (5.88 -> 5.19), VAD IoU 0.59 - lip-sync / coverage, accepted per the intelligibility-first priority, not yet analysed.
Open: 01 "stem of heat" / "Prime deal" come from Whisper mishearing the Russian source (translation input), not from TTS.
