# Round 3, Agent C: VOICE / AUDIO

Branch: `experiment/r3-audio`, created from `origin/experiment/r2-audio` (a1fd764).


## Where you run (user decision): server A, CPU ONLY
You share the server with agent A. A owns the GPU (LatentSync). You run on the CPU (256 cores): always
`export CUDA_VISIBLE_DEVICES=""`.
- Use faster-whisper int8 on CPU, llama.cpp on CPU, and Chatterbox on CPU. Parallelise across units / processes.
- Measure your s/video-min on CPU.
- Your clone: `/workspace/dub/dubai-audio`. Never touch `/workspace/dub/dubai`: that is A's clone.
- The Python environment is the shared one installed by setup (`/workspace/dub/setup.log`, wait for `SETUP_DONE`). Do not reinstall
  packages in a way that changes A's environment. If you need an extra package, install it into a separate venv.
- **Your own root is `/workspace/dub/c/`**: use `/workspace/dub/c/outbox/`, `/workspace/dub/c/handoff/out/<to>/<name>/` and
  `/workspace/dub/c/handoff/in/<from>/`. This replaces the `/workspace/dub/outbox` and `/workspace/dub/handoff` paths in the common
  brief, so that your files and A's files never mix. A still receives your audio through the relay, in
  `/workspace/dub/handoff/in/C/`.
- Inputs, canonical work dirs and common metrics are shared: `/workspace/dub/inputs`, `/workspace/dub/canonical`,
  `/workspace/dub/common`.

## You own
Everything audible:
- TTS intelligibility and voice quality;
- timing, pauses and pace;
- background bed and ducking;
- no extraneous sounds;
- loudness.

## Tasks, in priority order
1. **Regenerate audio-r2 for 01–05** with your round-2 defaults (L1 + floor 0.875, Spleeter 2stems bed, declick, ducking
   −4 dB / 250 ms, layer-3 refit). The round-2 audio was lost with the old servers.
   - Hand it off as soon as it exists, to **A, B and D** (`handoff/out/{A,B,D}/audio-r2/`): final mix, dub-only stem, placements,
     residual-gap JSON.
   - This unblocks the others. Aim to finish it within ~1.5 h, before deep work.
2. **Intelligibility regression guard.** The old problem must not come back (see branch `tts-intelligibility-fix`).
   - Back-transcribe every TTS unit with Whisper.
   - Report WER / CER against the English text, per video and per unit.
   - List the worst 20 units with their wav paths in `outbox/listen/`.
   - Fix the causes: slurring, clipped onsets / offsets, too-fast fit, wrong stress, numbers / abbreviations.
3. **No extraneous sounds.** Detect and fix:
   - clicks and pops at unit joints;
   - breath or noise bursts;
   - TTS artefacts (metallic, repeats, hallucinated words — whisper vs text);
   - Russian voice bleeding through the background bed;
   - pumping from ducking.
   Use numbers (detectors) plus listen clips.
4. **Natural pauses and pace.** Pauses where the source pauses, no rushed or over-slowed units, prosody across unit joints.
   Measure the speech-rate distribution and pause lengths against the source. Keep the residual dub-silent gaps small, and hand
   the gap JSON to A (A handles the mouth in those gaps).
5. **Loudness and mix.** Target integrated loudness (state your LUFS target), true peak ≤ −1 dBTP, dialogue vs bed balance.

## Deliver
- `audio-r3` for 01–05 handed off to A, B, D (same layout as audio-r2), plus listen clips (wav and short mp4 muxed over the
  original video) in `outbox/`.
- A table in `reports/experiments/r3-audio/README.md`: WER / CER, click count, pause stats, LUFS, s/video-min.

Every model must have a clean file licence (Chatterbox MIT, Whisper MIT, Qwen Apache, Spleeter MIT). HDEMUCS weights stay out.
