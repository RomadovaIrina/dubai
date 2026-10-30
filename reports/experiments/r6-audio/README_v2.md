# r6_audio_v2 (WATCH): safe polish of r6_audio_v1 (the audio of record)

**v2 = v1's takes, unchanged (same text, timing and voice) + one remix change:** `--gap-bed-cap-scope edges`. The bed cap (median + 10 dB, only
where the source speaks) now also reaches 150 ms into the dub's edges. Long under-dub stretches are untouched.
**Gate vs r6_audio_v1: PASS on all 5.** Handoff `audio-r6-v2/` has no READY: v1 stays the audio of record until the user prefers v2 by ear.

The three items from the brief:
1. **02 leaked syllable at 12.1 s: fixed.** It was a 100 ms burst of the original voice in the separated bed, overlapping the dub's tail at a 0.3 s
   sentence gap, so the v1 gap mask missed it. Residue exposed -32.7 -> -39.3 dB (pb1: -35.2). 03 residue under dub -24.6 -> -25.9 dB.
   01 / 04 / 05 are unchanged.
   - Rejected: scope "speech" (cap every source-speech frame). It changed the bed dynamics on 05 (pause-onset jump 4.7 -> 6.3 dB, bed speech heard
     0.5 -> 0.8 s).
2. **02 "hesitations": nothing audible to fix.** The five remaining INSIDE rows are not pauses in the takes; the raw TTS takes have no gap there.
   They are boundary pauses that Whisper shifts by one word: "tea | or mineral", "Also, | there are", "bathhouse, | we all", and "hammam | and all"
   (this part also starts with 0.5 s of -55 dB noise). The metric needs a CTC alignment of the final track to stop flagging them; it is not changed here.
3. **05 (slow speaker): not changed.** The gap is missing English words: the dub covers about 40 % of a 3.8 s burst in u02. The only levers are
   added words (the judge rejects them as added facts) or drawl (r6v5: no gain). Both alter the behaviour the user likes, so neither is safe polish.

02 WER .027 -> .044 is ASR decoding variance on an identical dub voice: Whisper switched to unpunctuated output and repeated "paid for that's".
