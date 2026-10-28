# r7_audio v3a / v3b / v3c on 04 and 05 (WATCH; the user judges by the rendered mouth)

The user's rule: the source mouth must not be open and moving while there is no dub sound.

All three variants sit on r7_audio_v2: gap clean, dub gate (the crackle fix), lip placement 2.0 / 1.5 / 0.5, and Qwen 14B. On 01–03 every variant is r7v2 itself (linked, not rebuilt); the winner gets built on all five.

| variant | what it does | tool |
|---|---|---|
| **v3a** = r7v3 | Re-translates to the source length: the duration-aware Qwen 14B with the same judge, a Chatterbox re-take (normal or deliberate pace), placed at the speaker's pauses, tempo 1.0. | `scripts/r7/retake.py` |
| **v3b** | Aligns the EXISTING take to the Russian bursts. The English is cut between words; a DP maps chunks onto runs of the speaker's own speech pieces (Silero pauses ≥ 0.4 s inside the bursts). Each chunk starts with its piece and is tempo-fitted 0.92–1.08 (ffmpeg atempo, pitch kept) to end with it. No re-translation. | `scripts/r7/burst_align.py` |
| **v3c** | v3a's re-takes, then v3b's aligner on top. | `retake.py` then `burst_align.py` |

DP cost in v3b / v3c:
- face-visible source speech left without dub, counted ×2 when a whole piece gets no dub;
- 0.3 × dub over source silence;
- 0.8 s per cut that isn't a pb1 boundary;
- 0.5 s per one-word chunk;
- 1.0 × |1 − tempo| × seconds.

A lone function word and any overlap with another chunk or the next sentence are refused. A unit changes only when it beats its current placement by ≥ 0.2 s.

## Numbers (04 and 05; r7v2 = the base all three start from)

| video | variant | speaker on screen talks, dub silent (s) | dub speaks, source silent (s) | tempo outside 0.95–1.05 (s) | faithfulness (units < 4) | WER | pause W1 (s) | longest excess gap (s) | pauses inside a phrase | gap voice (s) / words | transients |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| 04 | r7v2 | 5.25 | 8.5 | 4.25 | 5.00 (0) | 0.034 | 0.692 | 4.25 | 0 | 0.06 / 0 | 32 |
| 04 | **v3a** | **2.35** | 6.6 | 4.25 | 5.00 (0) | 0.043 | 0.544 | 4.25 | 0 | 0.02 / 0 | 35 |
| 04 | v3b | 3.9 | **5.3** | 21.9 | 5.00 (0) | 0.045 | **0.485** | **2.0** | 0 | 0.0 / 0 | 32 |
| 04 | v3c | 2.5 | **5.3** | 20.5 | 5.00 (0) | **0.021** | 0.578 | **2.0** | 1 | 0.0 / 0 | 31 |
| 05 | r7v2 | 4.3 | 5.2 | 0 | 4.29 (2) | 0.024 | 0.667 | 2.6 | 0 | 0.0 / 0 | 46 |
| 05 | **v3a** | **1.5** | 4.45 | 0 | **4.71 (0)** | 0.057 | **0.236** | **0.8** | 1 | 0.0 / 0 | 35 |
| 05 | v3b | 1.75 | **3.35** | 22.4 | 4.43 (1) | 0.061 | 0.85 | 3.2 | 0 | 0.06 / 0 | 61 |
| 05 | v3c | 1.75 | 3.4 | 21.3 | 4.71 (0) | 0.068 | 0.679 | 3.2 | 1 | 0.12 / 0 | 42 |

r6v1's transient counts are 37 on 04 and 52 on 05. v3b on 05 is above that (61): the tempo-changed chunks add transients.

**Speed.** On top of the r7 chain, at the full TTS stage from scratch, i.e. about 110.7 s/video-min with 14B plus 8 for r7:
- **v3a:** retake adds 19 s/video-min on average over the five videos (38 on 04 / 05, where most units are re-taken) → about 138.
- **v3b:** the aligner adds about 3 s/video-min (5 s per video: one wav2vec2 load) → about 122.
- **v3c:** both → about 141.

All three are within ~150.

## Reading

- **On the user's key number, v3a wins:** 04 2.35 s and 05 1.5 s, with no tempo change, and it is the only variant that fixes 05's faithfulness. Its cost is one pause inside a phrase on 05.
- **v3b uses no new words, only timing and tempo.** It gives the least dub-over-silence, but 21–22 s per video run outside 0.95–1.05 (inside the allowed 0.92–1.08). It leaves 04 at 3.9 s, because the old English is too short for the drawn-out Russian even at 0.92. On 05 u3 it puts "Well," alone on the first burst and the rest on the second.
- **v3c sits between them on 04 (2.5 s)** and has the best WER there. On 05 it equals v3b on coverage but keeps v3a's faithfulness.

## Handoff

- `handoff/out/F/audio-r7-v3a/`, `-v3b/`, `-v3c/` (61 MB each, READY; `changed_sentences.md` inside).
- Outbox: `<id>_r7_audio_v3{a,b,c}_WATCH_AB_prev-then-new.mp4` (r7v2, then the variant) and `r7_audio_v3{a,b,c}_changed_sentences.md`.
