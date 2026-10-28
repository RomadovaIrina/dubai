| video | dur s | segments | = r6v1 transcript | no_speech > 0.6 | dropped | trimmed | flagged, kept (VAD speech) | pieces (s) | cuts: t / pause s / dB / kind | cut in speech / in a word | pieces = whole-file (words changed) | rerun = | words changed vs r6v1 |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|
| 01 | 101.679 | 23 | NO | 0 | 0 | 0 | 0 | 0.0-101.679 | None / None / None / no_pause | no / no | yes (0/190) | yes | 1/190 |
| 02 | 110.851 | 11 | NO | 0 | 0 | 0 | 0 | 0.0-62.65, 62.65-110.851 | 62.65 / 5.7 / -48.4 / pause | no / YES | NO (6/137) | yes | 1/137 |
| 03 | 90.324 | 10 | NO | 0 | 0 | 0 | 0 | 0.0-45.3, 45.3-90.324 | 45.3 / 2.0 / -40.5 / pause | no / YES | NO (5/135) | yes | 3/136 |
| 04 | 53.892 | 5 | text yes | 0 | 0 | 0 | 0 | 0.0-53.892 | - | no / no | yes (0/58) | yes | 0/58 |
| 05 | 52.383 | 5 | NO | 0 | 0 | 0 | 0 | 0.0-52.383 | - | no / no | yes (0/63) | yes | 4/62 |

Word changes (whole-file today vs r6v1 | VAD pieces vs whole-file):

- 01: vs r6v1 ['шоу-рум -> шоурум'] | pieces none
- 02: vs r6v1 ['ходим -> хотим'] | pieces ['за банный -> забавный', 'хотим -> ходим', 'холодильничек -> холодильник', 'зона -> зону', 'лежечки -> лежачки']
- 03: vs r6v1 ['ознакомительно -> ознакомительное', 'взимные -> земные', 'все -> '] | pieces ['это  начало -> эта точка начала', ' -> в', ' -> все']
- 04: vs r6v1 none | pieces none
- 05: vs r6v1 ['справится -> справиться', 'оно -> она', 'тестим -> с этим'] | pieces none

Every segment the guard dropped, trimmed or flagged:


## Regression gate vs r7_audio_v1 (r7v1b)

- Recipe rerun on the current code (tag r7v1asr, REUSE=r6v1src): dubbed_24k.wav and final_mix_44k.wav are byte-identical (md5) to r7v1b on all five videos. The r7_audio_v1 build reuses the r6v1 takes and never calls run_asr.
- On a fresh run, the guard returns the whole-file transcript unchanged on all five (max no_speech_prob 0.018), so translation input is unchanged.
- Whole-file ASR is deterministic on this server (a rerun is identical). Versus the r6v1 upload (another server), 0–4 words differ per video (spelling or near-homophones), as listed above.
- ASR time: whole-file 17.5 s for 6.82 video-min = 2.6 s/video-min; with --asr-segments vad 16.2 s = 2.4 s/video-min. The guard costs about 0 ms. The audio stage stays at about 118 s/video-min (14B, plus r7).
- --asr-segments vad stays off by default. faster-whisper already handles any length in 30-s windows, and on 02 / 03 the pieces change 5–6 words in either direction.
