# DabAI r7 — final dubbed videos (candidate m11)

Files: `<id>_DabAI_r7.mp4` (full quality: the source's resolution and frame rate, H.264 + AAC, faststart) and `<id>_DabAI_r7_phone.mp4`
(the same video under 25 MB for messengers) for the five test videos 01–05. Russian source → English dub with the speaker's own voice,
lips re-rendered to the English where the speaker's face is on screen.

## What the pipeline does
1. **Speech.** Voice activity + Whisper (large-v3) transcription + speaker diarization on the source; the source voice is separated from
   the background (music, street) so the background survives untouched under the dub.
2. **Translation.** Qwen 14B, sentence by sentence, with a back-translation judge (meaning, nothing added or lost); where the speaker is on
   screen and the English would end much earlier than the Russian, the sentence is re-phrased to the source length (r7 v3a) so the mouth
   is not left shut while the person is visibly still talking.
3. **Voice.** Chatterbox TTS cloned from the speaker's own voice, with a QA ladder (every take is transcribed back, names must be heard);
   takes are placed into the speaker's own speech spans and pauses, never time-stretched beyond ±8 %.
4. **Mix.** The dub over the original background at −16 LUFS, with gap cleaning (no leaked Russian in the pauses) and a dub gate
   (no take noise in the pauses — the r7 crackle fix).
5. **Lips.** LatentSync 1.6 renders the mouth to the English audio on a 512-px face crop at 25 fps, only on frames with a visible face
   and only where the dub speaks; a Real-ESRGAN pass restores lip detail; the mouth is composited back at the source frame rate with the
   source's own skin texture, colour-matched, with guards against generator glitches (a lip-texture sanitizer, alignment guards, a
   whole-frame gate that proves nothing outside the mouth changed). Everything outside the mouth region is the original pixel.
6. **Checks.** Each video is scored (lip-sync confidence, lip-flash and face-twitch detectors, seam and sharpness measures, ASR word error
   rate on the final mix, loudness) and gated against the previous version before a human watch.

Speed on this build: about 480–500 s per video-minute for the face stage on a full-face video on one GPU (LatentSync ≈ 75 % of it),
plus ≈ 140 s per video-minute for the audio stage on a second server.

## Known leftovers (r7)
- **Dub-silent spans on a talking face.** Where the English is shorter than the Russian, the mouth is closed while the person is still
  visibly talking (05 ≈ 7 s, 04 ≈ 3 s in total, down from 9 / 5 s in the previous version thanks to the longer takes). Conversely the
  rendered mouth keeps moving a little on the short pauses inside a take.
- **Pause placement.** A re-phrased take can pause inside a phrase (05 once, 04 once).
- **Generator artefacts.** 03 at 37.3 and 37.7 s: a washed-pink lower lip for two frames; 01 at 7.6 s: a one-frame lip pop. Both are the
  lip generator's; the fixes tried cost lip-sync and were not taken.
- **Lip-sync confidence** is lower than on a native recording on the videos with the most re-rendered frames (05, 04); the twitch detector
  still finds a few one-frame lip-texture changes per video (04 the most).
- 02 has no visible face (no lip rendering): the source video with the English mix.

Build: branch `experiment/r7-face`, candidate m11 = audio-r7-v3a + the m8 face stack; details in `reports/experiments/r7-face/STATUS_LOG.md` §11.
