# Round 4, AUDIO agent

Branch `experiment/r4-audio` from `904c5d2`. Server: RTX 5090, yours alone.

## Tasks, in order
1. **Reproduce v4's audio** for 01–05 from 904c5d2, through the one-command audio stage. Compare it with the audio track of
   `/workspace/dub/reference/v4/<id>_v4.mp4`: LUFS, duration, WER, hard cuts, the residue metric (task 2).
2. **Background bed (top priority: the v5 regression the user and Max heard).**
   - In v5 the original Russian voice leaks through the bed, and the construction workers' noise comes to the foreground.
   - Add two metrics to the gate:
     - (a) **original-voice residue** in the final mix under dub speech: VAD or ASR on the bed stem, plus the energy of the bed in
       the speech band where the source talks;
     - (b) **bed level in pauses** vs the source level at the same moments; noise must not jump up in the pauses.
   - Find the best simple bed policy. Compare the v4 settings with each v5 change, one at a time (duck depth / timing, residue duck,
     static gain + limiter). Keep the simplest one that wins on both metrics.
3. **Pauses.** Max: "something weird with the pauses" in v5. Compare the pause-length distribution and the placement against the
   source for v4 and v5 (the v5 sentence flow gap rule of 1.0 s). Keep whichever is closer to the source. No rushed or over-stretched
   speech.
4. **Proper nouns** ("Сочи Сити" → "Solki" on 04). A generic named-entity guard:
   - detect names and places in the source / translation;
   - keep them transliterated, not translated;
   - give TTS a pronunciation respelling;
   - verify with Whisper that the name is heard;
   - retry with an alternative respelling;
   - add a gate metric: named entities misheard.
5. **Voice continuity.** Intonation / pitch / energy jumps between TTS parts of one sentence (`r3_joins.py` measures them). Prefer
   whole-sentence synthesis when the timing allows it. Otherwise condition each part on the previous one and match the level at the
   join.

## Publish
- Only after the gate passes: `outbox/<id>_r4_audio_vN_over-original.mp4` (new audio over the untouched source video), plus
  `outbox/<id>_review.md` and `r4_audio_vN_regression.md`.
- Hand the passing audio to the video agent as `handoff/out/V/audio-r4a-vN/`: the same layout as the earlier audio handoffs,
  `dubbed_16k.wav`, `final_mix_aac.m4a`, manifests, READY. Keep it ≤ 150 MB.
