# Round 3, Agent D: INTEGRATION, QA SCORECARD, SPEED

Branch: `experiment/r3-integration`, created from `origin/clean-base` (326e807). Server: RTX 5090.

## You own
Turning the tracks into ONE runnable E2E candidate, scoring it, and checking the speed norm.

## Tasks, in priority order
1. **E2E runner.** One command runs the whole pipeline on a video:
   Silero VAD → faster-whisper → pyannote → Qwen GGUF → Chatterbox → fit / place → LatentSync → composite → mux.
   - Use the clean defaults.
   - Accept flags or stage overrides for each track's best settings.
   - Merge the round-2 branches step by step (`origin/experiment/r2-audio`, `origin/experiment/r2-lipsync-position`,
     `origin/experiment/r2-composite-position`). Later merge round-3 branches as agents report winners.
   - Resolve conflicts carefully and document every merge in CHANGES.md.
2. **Scorecard script.** It produces one JSON + Markdown table per run, over 01–05:
   - SyncNet (dub-only, mix, AV offset);
   - the v4 jitter metric and the slide metric (`/workspace/dub/common/`);
   - coverage (frames lip-synced);
   - output fps and resolution vs source (spec 9.4);
   - feather px;
   - joint jumps (8.5);
   - Whisper back-transcription WER of the final mix;
   - loudness and true peak;
   - duration match;
   - s/video-min per stage.
   Score cand5 and clean0 first, so there is a reference.
3. **Speed profile.** Per-stage time per video-minute on this 5090 against the 300 s/min norm. List the top three costs and the
   cheap wins that do not change quality: batching, caching models between stages, fewer ffmpeg re-encodes, frame I/O. Do not
   change model settings that the other agents own; propose them instead.
4. **Candidate r3.** When A, B and C hand over their winners (`/workspace/dub/handoff/in/`), assemble candidate r3, run it on all
   five videos, and publish:
   - finals: `outbox/<id>_r3cand_final.mp4`;
   - the scorecard;
   - a side by side CAND5 | R3CAND per video.

## Notes
- **Regression guard:** a candidate that is worse than cand5 / clean0 on any spec item must be flagged loudly in the status.
- Use cand5 canonical work dirs (`/workspace/dub/canonical/`) and C's audio when it arrives. Do not regenerate TTS yourself unless
  you are timing the full pipeline.
