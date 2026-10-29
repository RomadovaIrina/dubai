# DabAI round 8 (29.10.2026): quality leftovers first, then speed

**Repo:** RomadovaIrina/dubai. The servers are reached from the laptop by SSH alias.
- **F, the face:** `ssh r8-f`, Quebec. Clone `/workspace/dub/dubai`, branch `experiment/r8-speed-face` from `origin/experiment/r7-face` (2c87168).
- **A, the audio:** `ssh r8-a`, Czechia. Clone `/workspace/dub/dubai`, branch `experiment/r8-speed-audio` from `origin/experiment/r7-audio` (972348b).

`scripts/setup_dabai.sh`, `download_models.sh`, `setup_mediapipe.sh` and the llama CUDA build run in the background after the clone; the log is `/workspace/dub/setup.log`. Wait for SETUP_DONE.

The HF token is in `/workspace/dub/.env` and `~/.cache/huggingface/token`. Never print it.

Briefs on the server are in `/workspace/dub/briefs/`: `brief_phase2_speed.md`, `brief_r7_tracks.md`, `tz_status.md`, `ML_PLAN.md`.

**Quality reference:** the tag `r7-quality-baseline` (m11; 05 on audio-r7-v3a1). The final videos are in `/workspace/dub/reference/final_r7/<id>_DabAI_r7.mp4`.
- The face history and every metric are in `reports/experiments/r7-face/STATUS_LOG.md` (§11–§13: the baseline, the speed costs, the started phase-2 levers).
- The audio history is in `reports/experiments/r7-audio/`.

The r7 server artefacts (generator raw layers, handoffs) are GONE. Rebuild what you need from the code + `scripts/r7/*`. The audio handoffs from round 7 are on the laptop: `C:\Users\irina\Documents\DabAI_AB_backup\r7_final\{A,F}_handoff_briefs.tar`. The coordinator uploads them on request; the F one contains `handoff/in/audio-r7-v3a1`.

**Decisions so far:**
- Qwen 14B Q5_K_M is approved by the customer (29.10).
- Commercially clean models only: no CodeFormer, InsightFace buffalo_l, RetinaFace R50 or HDEMUCS. LatentSync 1.6 (OpenRAIL++) is OK. Real-ESRGAN (BSD-3) is OK.

**Money (vast bills bandwidth and storage):**
- No download over 5 GB without asking the coordinator.
- No new model families without asking.
- Report disk and bandwidth in every status.

**Common rules:**
- Fixes must be generic: a new test video with a new face will come.
- One-command E2E.
- Every change goes behind a flag and is gated with `scripts/r7/r7_baseline.py gate` against the baseline, plus a WATCH sbs (baseline | new) for the user.
- Commit and push every commit. The deploy keys are being added; until then, commit locally.
- Send a 5-line status every ~45 min.
- POLL your own long jobs: background notifications get lost.

## The user's latest review of final_r7 (29.10)
- **03:** the mouth interior FLICKERS light ↔ dark from frame to frame ("like a little fire in his mouth"). It does not correlate with how wide the mouth is open. The lower part of the face also flickers. Otherwise 03 is "almost very good".
- **04:** strange starts and pauses in the speech: odd pacing.
- **All 5:** "the strange moments are still there". These are the r7 known leftovers: the generator lip artefacts (03 37 s pink lips, the 01 7.6 s pop), open mouth on dub-internal pauses, the remaining closed-mouth-while-source-talks spans.
- The user asked for a commercially clean REPLACEMENT FOR CODEFORMER (face / mouth restoration).
