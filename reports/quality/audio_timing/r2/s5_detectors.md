# r2 step 5: foreign sounds, clicks, hot cuts on the delivered tracks

Detectors: `audio_stages.py` (click / hot_edge / extra_voice / overlap / bed_voice / bed_step, traced to the first stage where each appears),
`edge_cuts.py` (hot edges of every TTS span), gap-QA (`--tts-gap-qa on` in the runs; `extra_voice` = TTS energy outside the CTC word spans).
Run on the cand5 and r2 E2E tracks (identical TTS takes) and on the final delivery. `audio_stages.py` now measures the bed AS HEARD (mix −
dub at 44.1 kHz; a 24 kHz subtraction left the dub in at −11 dB and faked "bed_voice" events).

## Events, cand5 → r2 E2E → delivery

| video | clicks in the dub voice | clicks only in the mix | hot edges (audio_stages) | extra_voice | bed_voice (s) | bed steps created by the mix |
|---|---|---|---|---|---|---|
| 01 | 5 → 6 → **0** | 1 (91.8 s) | 0 | 0 | 0 | 0 (all 51 steps are in the music itself) |
| 02 | 0 → 0 → 0 | 1 (5.25 s) | 3 → 3 → 3 | 3 → 3 → 3 | 0.6 → 0 → 0 | 19 → **11** |
| 03 | 0 | 1 (9.51 s) | 1 → 1 → 1 | 0 | 0 → 0.4 → 0.4 (59.3 s, 75 % under the dub) | 9 → **1** |
| 04 | 0 | 0 | 2 → 2 → 2 | 0 | 0 | 5 → **0** |
| 05 | 1 → 1 → **0** | 0 | 6 → 6 → 6 | 1 → 1 → 1 | 0 → 0 → 0.5 (27.3 s: wind noise under dub speech, Silero false positive) | 6 → **2** |

**Clicks (fixed)**, each with its origin: 01 2.71 / 14.32 / 33.68 / 71.06 s and 05 41.99 s are inside the raw Chatterbox take (broadband
vertical stripe in the raw-TTS spectrogram = vocoder click; the same in cand5); 01 42.28 s (**new with r2**) and 59.28 s first appear after
ffmpeg atempo x0.875 (splice in a stop closure). Fix: `declick.py`, runner `--declick on` — the same detector on the assembled dub voice, ±1.5 ms
4 kHz low-pass repair; 7 repaired, 0 left, WER / CER unchanged (`s6`). The 3 mix-only clicks (peak 0.011–0.017, under dub speech) are in
neither the source nor the Spleeter stem nor the 24 kHz voice above threshold: resampling of a sub-threshold voice transient; not repaired.

**Bed steps (fixed)**: 8–10 dB drops of the bed in the 80 ms before a dub phrase where the stem itself moves 3–4 dB = the −6 dB duck attack
("pumping": the ambience audibly dips just before every phrase and swells after it). Runner defaults now −4 dB / 250 ms / 600 ms; bed level
pause − speech ("breathing") 7–8 → 5–6 dB. Whisper WER on the MIX unchanged (01 0.019 → 0.022 = one word, 05 0.053 → 0.053), CTC CER +0.007 / +0.022.

**Hot cuts (not new)**: `edge_cuts.py` hot inner heads / tails cand5 → r2: 01 2/2 → 2/0, 02 8/6 → 7/6, 03 9/4 → 9/4, 04 6/5 → 6/5, 05 12/11 →
12/11 — r2 adds none (flow 0.2 removes some cuts). They are inner cuts of connected TTS speech (2–5-word parts on 04 / 05) that already get
the adaptive 25 ms / 8 ms fades. **extra_voice (not new)**: 02 17.4 / 83.5 / 104.0 s, 05 30.4 s — 80–120 ms of TTS sound before the first
CTC word (onset of the first voiced phoneme; 02 83.5 s checked on the spectrogram: the voiced "And"), identical in cand5.
