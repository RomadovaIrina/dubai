# Quality tasks derived from the DabAI spec ("ОПИСАНИЕ, ЭТАПЫ, БЭКЛОГ", Aug 2026) — 2026-10-17

Product: a web service where a blogger uploads a video (≤ 5 min, ≤ 1 GB) and gets it back in another language, in their own
cloned voice, with lips re-synced. Our stack: VAD → ASR (faster-whisper) → diarization (pyannote) → Qwen translation →
Chatterbox TTS → duration fit → LatentSync → (CodeFormer) → composite → subtitles / mux.
Contract facts that bind the quality work:
* Norm: **≤ 300 s of processing per video-minute** (1:5). Steps 9–10 (LatentSync + restoration) eat almost all GPU time.
* We are liable for **seams**: "сборка сегментов — без щелчков на стыках аудио и без рывков на стыках видео" (8.5), and for the
  worker never crashing. Model quality as such ("LatentSync smears on a profile") is not our liability.
* Final MP4: H.264 at the **source resolution and frame rate** (9.4). Any crop or stretch is a contractual defect (5.6).
* Compositing: alpha mask with a **10–15 px feathered edge**; a hard seam is a defect (8.4).
* Only the **active speaker's** lips are changed when two faces are on screen (8.2).
* Face pre-check: **no face → abort + refund; > 30 % of frames with head angle > 30° → voice only, no lip-sync, with a notice** (5.4/5.5).
* Robustness: a model RuntimeError must not crash the worker; a bad frame is replaced by the previous valid one; ≤ 2 % replaced
  = SUCCESS, more = PARTIAL_SUCCESS + refund (11.1–11.4).
* Acceptance: SyncNet confidence / offset on the 5 customer videos (the threshold is still UNDECIDED, 0.8 report), CER per language,
  E2E on 5 videos, reproducible (12.x).

## Gaps found by the coordinator (verified 2026-10-17)

| spec | finding |
|---|---|
| 9.4 frame rate | all 5 sources are 30 fps (01 VFR, avg 25), **all outputs are 25 fps CFR**: the pipeline builds a 25 fps master for LatentSync and muxes it. One source frame in six is dropped → uneven head motion (judder) on the whole frame, on top of the mouth artefacts. |
| 5.4 / 5.5 head angle | no yaw / head-pose estimate and no "> 30 % frames > 30°" pre-check anywhere in `scripts/` |
| 8.3 "убирает мыло" | CodeFormer (NC) is removed in clean0 → the mouth is softer (sharpness 0.71 of the source); the spec's intent (no blur on the face) must now be met with clean means |
| 8.5 video seams | LatentSync's 16-frame windows jump every 0.64 s (60 % of window starts > 1.5x in clean0) + LS↔original switches |
| 8.4 feather | not verified in pixels on the output |
| 8.2 two faces | not tested on the clean MediaPipe stack |
| spec model list | RetinaFace (step 3) and CodeFormer (step 10) are replaced for licence reasons → tell the customer in writing |

## Task list

| prio | spec | task | owner |
|---|---|---|---|
| P0 | 8.5 | Remove the 16-frame window seams and soften LS↔original switches. Generator side (overlapping windows / fusion) vs post side (flow interpolation at boundaries); the same artefact attacked from both sides, compare. | A (generator), B (post) |
| P0 | 9.4 | **Output at the source frame rate.** Composite the 25 fps LatentSync mouth onto the ORIGINAL-fps frames (flow-retimed mouth), so that the face / background is the untouched 30 fps source: no dropped frames, no judder, same duration / frame count as the source. | B (owner), A (manifest / timing support) |
| P0 | 8.3 intent | Clean replacement for "убирает мыло": mouth / lower-face sharpness ≥ 0.95 of the source without texture popping (resid ≈ 1.0), no NC restorer. | B |
| P0 | human feedback | Frozen mouth in dub-silent stretches where the source speaks (03 24–25 s, 35–36 s): option 1 still lower face, option 2 original pass-through with eased transitions → side-by-side for the user. | B |
| P0 | norm 300 s/min | Every change reports video-stage s/video-min. Budget: ≤ clean0 + 10 % unless a speed-up in the same change pays for it. LatentSync speed levers (DeepCache / steps) only with a human check (12 steps already showed twitches). | A (LatentSync cost), both report |
| P1 | 5.4 / 5.5 | Head pose (yaw / pitch) per frame from MediaPipe's transformation matrix; implement the "> 30 % frames > 30°" pre-check (voice only + notice); per-frame gating of large-angle frames to pass-through with eased transitions, if it reduces head-turn artefacts. | A |
| P1 | 8.2 | Two faces: the clean stack must detect both, lip-sync only the active speaker (diarization ↔ face), leave the other untouched. Test on a synthetic two-face clip built from the test videos. | A |
| P1 | 8.4 | Measure and enforce a 10–15 px feather on the output for every composite variant; report the actual px. | B |
| P1 | 11.1–11.3 | New code paths (MediaPipe subprocess, interpolation, any new generator) must not crash the worker: missing landmarks / errors → previous valid frame, counted toward the 2 % rule; keep the OOM retry + validation. | A (face stack, generator), B (composite) |
| P2 | 5.6 / 9.4 | Validation of every output: source resolution, aspect ratio, frame rate, duration, H.264, audio md5 identity for video-only changes. | both |
| P2 | 12.1 / 12.2 | Acceptance metrics on every candidate: SyncNet confidence + offset, CER, common twitch metric v4, side-by-side. | both |
| – | 7.3 (audio) | Root cause of the frozen mouth: the dub is shorter than the source burst and the remainder is filled with a pause. Out of scope for the video tracks; possible third direction after the user decides. | user decision |
