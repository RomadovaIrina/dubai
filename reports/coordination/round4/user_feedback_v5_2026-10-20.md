# User feedback on e2e_v5 vs r3final_fixed v4 (20.10.2026) — input for the next round

**Fixed (the user confirms): no more lip "explosions" / single-frame flashes.**

## Still wrong (each one must get a GENERIC pipeline fix, not a per-video patch)

1. **Inconsistent pronunciation of proper nouns.** The source says «Сочи Сити» (04); the dub says something like "Solki".
   - Suspects: Chatterbox mispronounces "Sochi"; or Qwen / ASR mangles the name.
   - Generic fix candidates:
     - a named-entity / proper-noun guard: transliteration plus a pronunciation respelling before TTS, like C's "Hamam" fix;
     - back-transcription with Whisper, comparing per named entity, then retrying with respellings;
     - a QA gate: NE mismatch count.
2. **Chin "wave" / second-third chin.** The jaw / chin area deforms and moves like a wave.
   - Suspects:
     - LatentSync regenerates the jaw inside the mask;
     - the r3 mask / face-clip / sanitize colour-texture pass reaches below the lips;
     - the bridge / continuation windows.
   - Generic fix candidates:
     - keep the generated region to mouth + lips;
     - feather back to the source above the jawline;
     - a jaw-contour stability metric (landmark jitter of the chin against the source).
3. **Inconsistent lip movements remain** (generator side): some articulation does not match the speech, and the shape jumps.
4. **Background audio regression in v5.** The original Russian voice bleeds into the dub track. In v4 the bed was better, with no original voice audible.
   - Suspect: the v5 bed remix (D 53cfb22): bed duck −2 dB / 0.4 / 0.8 s, residue duck −6 dB, static gain plus limiter, replacing the earlier −4 dB ducking. Spleeter residue of the Russian voice is now louder.
   - Fix: restore the v4 bed settings, or duck the residue harder during dub speech.
   - Add a metric: Russian-voice residue in the bed under dub speech. Use ASR on the bed or a VAD-on-bed energy check.

## Process lesson
- Validate every change on all videos with one command (D's `r3_e2e.sh` v5 + QA gates).
- Compare against the previous version on the SAME axes before publishing. The v5 bed change shipped a regression that v4 did not have.
