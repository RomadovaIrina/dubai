# DabAI round 9: integrated E2E on 9 new videos (30.10.2026)

**Result: 9 / 9 SUCCESS, no crash.** Every video gives both final files, `<id>_DabAI_r9_demucs.mp4` (htdemucs bed, main) and
`<id>_DabAI_r9_spleeter.mp4`. Both are at the source resolution and frame count, from ONE command per video.

Two videos need the user's eye:
- **m21:** the face is never detected, so it gets the dub without lip-sync.
- **m15:** a hand covers the mouth; SyncNet is 1.95.

Speed baseline (production only, one video at a time per RTX 5090): **1676 s/video-min over all 9**. The face path is 1413 of it. The
~650 reference was measured on 01–05; the target is 300.

Branch `experiment/r9-e2e` (local commits, not pushed). It is the same on r9-a and r9-f, apart from the final REPORT commit, which only r9-a has (the coordinator pulls from there).
- Base: `origin/experiment/r8-speed-face` (c2dbcf6).
- Merges: `origin/experiment/r8-speed-audio` (63569b0) and `origin/experiment/r8-sep` (ef6dc99). Both merged without conflicts.
- Plus the r9 commits listed under "What broke".

## One command
```
bash scripts/r9/e2e.sh <input> <out_dir>            # ID=<short id> optional (default: the file stem)
  -> <id>_DabAI_r9_demucs.mp4  <id>_DabAI_r9_spleeter.mp4  manifest.json  timing.json  logs/
```
The config is final_r8b: the audio-r8-v4 dub, two beds, the m19b face. The stages are:

| stage | what |
|---|---|
| precheck | ffprobe, plus MediaPipe with 3 faces on ~4 frames/s. It sets flags (`no_face`, `two_faces`, `small_face` / `tiny_face`, `high_fps`, `odd_fps`, `variable_fps`, `landscape`, `low_res`, `rotated_source`, `hdr_or_10bit`, `long_video`) and the ERROR cases (unreadable input / no audio). A rotated source is first made upright, in the same display orientation. |
| audio | `scripts/r8/r8_audio.sh` from the raw input: ASR, diarization, Qwen 14B Q5_K_M duration-aware, batched Chatterbox (`DUB_TTS_BATCH=1`), `R8_ONEPROC` retake / clause takes / burst align, lip placement, hush, soften, and the Spleeter bed remix = the **Spleeter** mix |
| sep | htdemucs stems on the GPU (`scripts/r8/sep_stems.py`), then the same dub remixed with the sepH8p flags (`--sep htdemucs --bed-pauses source --bed-noise-back on --src-duck-db -3`) = the **demucs** mix |
| face_gen | LatentSync 1.6 + CodeFormer (no ParseNet, batch 16), with 10 steps where the dub speaks and 20 where it is silent (`r7_e2e.sh`, MediaPipe stack, no InsightFace) |
| face_comp | m19b: CodeFormer detail-only (`CFD=self`), fast layer SR, the tight fallback fb7, no deflicker, and the r7 composite at the source fps (`r7_m7_comp.sh`) |
| mux | the face video rendered ONCE + each mix, stream copy (bit-identical picture within the pair) |
| metrics | NOT production: the r8 gate rows. Production stages run with `METRICS=off` / `SKIP_SCORE=1`, so their times are clean. The whole-frame gate runs in the background (one core). |

Status rules, behaviour on hard cases:
- **no face** → the source picture + the dub, flag `no_face`.
- **face stage fails** → the same, `PARTIAL_SUCCESS`.
- **htdemucs fails** → only the Spleeter file, `PARTIAL_SUCCESS`.
- **no audio / unreadable input / dub fails** → `ERROR`.

`DUB_ALLOW_NC=1` / `DUB_ALLOW_NC_BED=1` are set by the command. This is the user-accepted NC deviation: CodeFormer + htdemucs.

## Per video (final runs)
SyncNet = confidence on dub speech. Pressed lips = closed mouth while the source talks and the dub is silent. Coverage = source speaks and
dub silent (s). Faith = Qwen 14B judge, mean 1–5 (units < 4). WER = Whisper on the final mix vs the dub text. Gap words = Whisper words
in the mix gaps, on the Spleeter / demucs mix. Lip-sync % = the share of the video where LatentSync frames are used.

| id | source | stresses | status / flags | lip-sync % | SyncNet | flashes | fallback % | pressed lips s | mismatch s | frame gate | coverage s | cut words | dead air | faith (<4) | WER | gap words S / D | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m19 | 720x1280 29.6, 18 s | baseline | SUCCESS (odd_fps) | 93 | 6.11 | 4 | 6.4 | 1.55 | 0.27 | PASS | 3.5 | 1 | 1 | 4.67 (0) | 0.0 % | 0 / 0 | 26.5 |
| m01 | 1072x1904 **60**, 86 s | long, 60 fps, hands | SUCCESS (high_fps, long_video) | 100 | **8.74** | 20 | 4.2 | 0.40 | 0.00 | pending¹ | 2.8 | 4 | 5 | 4.69 (0) | 1.4 % | 0 / 0 | 26.9 |
| m04 | 1072x1920 **40 VFR**, 21 s | odd fps | SUCCESS (high / odd / variable fps)² | 98 | 7.77 | 26 | 2.7 | 0.50 | 0.00 | PASS | 1.8 | 3 | 2 | 4.33 (1) | 0.0 % | 0 / 0 | 27.0 |
| m03 | 720x1280 30, 40 s | **two faces** | SUCCESS (two_faces) | 93 | 7.23 | 26 | 5.1 | 0.63 | 0.57 | PASS | 2.6 | 0 | 0 | 4.86 (0) | 0.8 % | 0 / 0 | 27.0 |
| m15 | 720x1280 30, 34 s | hand at chin / mouth | SUCCESS | 95 | **1.95** | 10 | **10.5** | 0.60 | 0.73 | PASS | 2.9 | 3 | 1 | 5.00 (0) | 3.7 % | 0 / 0 | 26.7 |
| m17 | 464x848 30, 43 s | mirror, phone occludes | SUCCESS (low_res) | 96 | 7.93 | 6 | 2.2 | 0.17 | 1.23 | PASS | 0.7 | 3 | 0 | 4.88 (0) | 4.3 % | 0 / 0 | 22.1 |
| m18 | 720x1280 30, 20 s | glasses, dim | SUCCESS | 95 | 7.30 | 7 | 6.9 | 0.00 | 1.03 | PASS | 0.7 | 0 | 1 | 5.00 (0) | 2.0 % | 0 / 0 | 22.1 |
| m21 | **848x464** 30, 33 s | landscape, small face | SUCCESS (landscape, low_res, **no_face**)³ | **0** | – | – | – | – | – | – | 4.2 | 1 | 2 | 5.00 (0) | 3.3 % | 0 / 0 | 27.1 |
| m20 | 464x848 30, 28 s | low-res source | SUCCESS (low_res) | 98 | 6.06 | 17 | 0.2 | 0.10 | 0.60 | PASS | 0.4 | 0 | 5 | 5.00 (0) | 2.1 % | 0 / 0 | 26.3 |

For reference, final_r8b on 01 / 03 / 04 / 05: SyncNet 4.76 / 6.74 / 5.51 / 4.20, flashes 0 / 36 / 8 / 1.

The full machine-readable table is `table.csv` / `table.json` (all the columns, per-stage s/vmin, errors). Each run's `manifest.json` is
in its run dir.

1. **m01 frame gate:** the whole-frame gate (5177 frames at 1072x1904, single core) was still running when this was written. It is
   filled into `manifest.json` when it ends.
2. **m04 is variable-frame-rate:** r_frame_rate 60, average 39.97. The output has the identical 858 frames and the same duration
   (±1 ms), but its timestamps are regularised (average 40.03 fps). For that reason `spec_9_4.fps_equal` is false; there is no A/V drift.
3. **m21:** MediaPipe found no face on any of 143 probed frames (848x464 landscape, a small full-body face, coloured light). The run
   gives the dub on the untouched source picture. It does not crash, but it does not lip-sync either. The face is below what the
   detector (and the generator's `--face-min-h 60` gate) handles.

**Gates over both mixes:**
- 0 gap words everywhere, on both the Spleeter and the htdemucs mix.
- 0 abrupt ends, apart from one on m21.
- Dead air is highest on m01 / m20 (5 each). It comes from the gap clean (`mixdrop` cause), the same class as in r8.
- Cut words: 0–4 per video.

**Face:**
- SyncNet is 6–8.7 on 7 of the 8 videos with a face, above r8b on 01–05: the new videos are larger, frontal faces.
- m15 is the outlier. The hand at the chin / mouth breaks the generator: fallback 10.5 %, SyncNet 1.95.

## Two faces (m03, spec 8.2)
`scripts/r9/twoface_check.py`:
- MediaPipe with 2 faces on every 2nd frame finds two faces:
  - face 0, left, 227 px: the child;
  - face 1, right, 305 px: the mother.
- **Speaker:** face 1. Its mouth moves with the source speech (opening vs Silero VAD r = 0.26, motion std 0.025) more than face 0's
  (r = 0.10, 0.019).
- **What the pipeline re-lipped:** the single-face track it used is on face 1 on 100 % of frames. Face 0 (the child, hand at the mouth)
  is never touched.

So on m03 only the speaker gets lips, as the spec asks. Caveat: the face stack has no explicit active-speaker selection. It follows the
dominant (largest / most frontal) face, and here that is the speaker. A video where the non-speaker dominates would get the wrong face.
The precheck flags `two_faces`, so such videos are visible in the manifest.

## Speed baseline (s/video-min, production only, clean)
One video at a time per GPU. Scoring is excluded. Model loads are included, as in a real one-command run. Measured on r9-a
(Threadripper 7960X, 24 cores) and r9-f (7970X, 32 cores), RTX 5090 each. m19 and m21 were re-run on r9-a; see "What broke".

| video | server | s | audio | a: base (ASR, diar, Qwen, TTS) | a: base TTS | a: model loads | a: retake / clause / align | a: place / hush / remix | sep (htdemucs) | face_gen | fg: LatentSync + CF | fg: own composite (discarded) | face_comp | PRODUCTION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m19 | r9-a | 18 | 328 | 159 | 80 | 34 | 89 | 49 | 28 | 789 | 466 | 321 | 404 | 1557 |
| m01 | r9-f | 86 | 227 | 103 | 78 | 6 | 42 | 20 | 12 | 1498 | 666 | 831 | 1192 | 2936 |
| m04 | r9-f | 22 | 390 | 209 | 117 | 28 | 87 | 46 | 25 | 1320 | 579 | 740 | 1058 | 2803 |
| m03 | r9-a | 40 | 216 | 101 | 58 | 15 | 58 | 28 | 17 | 706 | 460 | 246 | 291 | 1239 |
| m15 | r9-a | 34 | 250 | 144 | 95 | 18 | 48 | 31 | 19 | 661 | 421 | 240 | 305 | 1240 |
| m17 | r9-a | 43 | 182 | 112 | 72 | 14 | 16 | 26 | 16 | 522 | 353 | 169 | 152 | 879 |
| m18 | r9-a | 20 | 245 | 134 | 62 | 30 | 28 | 49 | 25 | 837 | 431 | 404 | 651 | 1764 |
| m21 | r9-a | 33 | 206 | 106 | 58 | 18 | 60 | 31 | 18 |  |  |  |  | 227 |
| m20 | r9-a | 28 | 244 | 125 | 65 | 23 | 48 | 42 | 21 | 576 | 389 | 186 | 203 | 1050 |
| **all 9 (time-weighted)** | | 324 | **239** | **122** | **75** | **17** | **48** | **31** | **18** | **857** | **449** | **407** | **556** | **1676** |
| **the 8 with a face** | | 291 | **243** | **124** | **77** | **17** | **47** | **31** | **18** | **955** | **501** | **454** | **619** | **1842** |
| **the 6 with a face at <= 720x1280, 30 fps** | | 183 | **233** | **124** | **72** | **20** | **44** | **35** | **20** | **657** | **414** | **242** | **298** | **1214** |

What it says:
- **The face is 85 % of the time.**
  - On 1080p / 60 fps sources (m01, m04) both composites run at the source resolution and fps: the generator run's own composite at
    ~740–830 s/vmin and the m19b composite at ~1060–1190.
  - On the six ≤ 720p / 30 fps videos the face is ~955 s/vmin.
- **Cheapest big lever:** `face_gen` includes the generator run's OWN full composite (`r4_video.sh` → `r3_fix30.sh`). The m19b
  composite then throws it away and redoes it from the layers: **407 s/vmin over all 9** (242 on the six 30-fps videos). Skipping it
  changes no output pixel.
- **LatentSync + CodeFormer:** 414–666 s/vmin. It includes the model loads and the MediaPipe passes on the 25 fps master.
- **Audio:** 182–390 s/vmin, 239 time-weighted.
  - Short clips pay the fixed cost: model loads are 17–34, and the Whisper / Qwen / Chatterbox warm-up shows on m19 (18 s) and m04 (21 s).
  - Base TTS is 75; the retake / clause / align pass is 48.
- **Separation:** htdemucs stems are ~1 s/vmin; the whole `sep` stage, with its remix, is 18. Both beds can stay at no real cost.
- **VRAM peak (GB):** audio 27.1, face_comp 18.1, face_gen 15.0, sep 1.4. The audio stage peaks highest (Qwen 14B + Chatterbox +
  Whisper) and needs the GPU alone.
- **Scoring (not production)** costs another 400–1650 s/vmin, mostly CPU: the scorecard, bg-ring, and the whole-frame gate at ~1 s per
  1080p frame.

## What broke and the fixes (all generic, no per-video settings)
| # | symptom | cause | fix (commit on `experiment/r9-e2e`) |
|---|---|---|---|
| 1 | no script could read a new video | 45 scripts hard-coded `/workspace/dub/inputs/test_videos_raw/<id>.*` | the input dir comes from `DUB_INPUTS` (default unchanged); `sep_mix.sh` / `publish_v5.sh` run from the repo they live in, not the `/workspace/dub/sep` worktree (cd01524) |
| 2 | m19 (shake-out): audio failed at once | the pipeline names its work dir after the RESOLVED input (a symlink → `story-…`) | e2e places the input as a real file named `<id>` (hard link, else copy) |
| 3 | **m17 audio ERROR** | `retake.py` wrote no take when no unit needed a re-take (01–05 always had one) | it passes the take through unchanged (4a85f67) |
| 4 | **m20 / m15 / m19 face composite failed** (sporadic; r8-face saw it "under load") | `r8_cf_detail.py` read the output frame rate from a stream whose container was already closed (freed memory). The rate was random; when it was not positive, libx264 refused to open ("The encoder timebase is not set", EINVAL) | the rate is read while the container is open (0fad7e0), plus one retry of cf_detail / layer SR in `r7_m7_comp.sh` (4a85f67). My first diagnosis (thread count, ec335f2) was wrong and is reverted. **Earlier layers that did succeed carry a wrong fps TAG only.** The composite reads them by frame index, so no output is affected; this was likely true in r8 too. |
| 5 | the GPU was idle for 20 min to 2 h per video | the whole-frame gate is single-core (~1 s per 1080p frame; m04 19 min, m01 > 2 h) | it runs in the background after the outputs, and the rows are re-collected (d1d4eed) |
| 6 | faith / pressed lips missing | the faith judge rode inside the production chain, and its result was then overwritten; the pressed-lips helper failed silently inside `r7_baseline freeze` | both run in the metrics stage; `r5_audio.sh` faith follows `METRICS` (25ddd44). Early runs back-filled, measurement only (`scripts/r9/backfill.sh`) |
| 7 | m21 ERROR on r9-f ("moov atom not found") | 2 of the 9 inputs on r9-f (m18, m21) were truncated by the upload; r9-a's were fine | re-uploaded, sha256 verified against the laptop; m21 then re-run on r9-a. The precheck turned the bad file into a clean ERROR, as designed. |
| 8 | fresh-server gaps | not fetched by setup: Qwen 14B Q5_K_M, CodeFormer + ParseNet, Real-ESRGAN x2, htdemucs (quarantine torch home) + demucs deps (submitit, retrying), the Spleeter venv, noisereduce | `/workspace/dub/r9_models.sh` (pre-approved list); worth folding into the setup |

Not fixed (quality tuning is not this round's goal):
- m21: no lip-sync on a small landscape face.
- m15: the hand over the mouth.
- m04: VFR timestamps regularised.
- There is no active-speaker choice for two faces.

## Timing honesty notes
- Every number above is from a run alone on its GPU.
- Exception 1: a 1-core background frame gate of the PREVIOUS video sometimes overlapped the next production run (r9-a from m15 on,
  r9-f m01). On a 24- to 32-core CPU, I expect this to change little.
- Exception 2: m19 and m21 are re-runs on r9-a after the fixes. The counted m19 is the r9-a re-run (m19's first r9-a batch run was a
  PARTIAL_SUCCESS from bug 4). The r9-a shake-out m19 is not counted.
- m01 and m04 are on r9-f, the other 7 on r9-a.

## Outputs, disk, bandwidth
- **Videos:** `/workspace/dub/outbox/r9_e2e/` on each server.
  - r9-a: m03 m15 m17 m18 m19 m20 m21 (+ REPORT).
  - r9-f: m01 m04.
  - Each video: `<id>_DabAI_r9_{demucs,spleeter}.mp4` + `<id>_manifest.json` + `<id>_timing.json`.
- **Disk:**
  - models 30 GB per server (+13.4 GB this round);
  - artifacts 7.0 GB (r9-a) / 9.6 GB (r9-f);
  - free 98 / 96 GB.
- **Bandwidth:**
  - down: ~13.4 GB per server (models);
  - up from the laptop: the 9 inputs (153 MB per server, by the coordinator) + 15 MB re-upload to r9-f + 2 small git bundles.
