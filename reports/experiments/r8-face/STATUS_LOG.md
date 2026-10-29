# r8 face (F) STATUS LOG, 29.10.2026, server r8-f, branch experiment/r8-speed-face

## 1. Baseline rebuilt (m11r8)
Audio: audio-r7-v3a1, from the laptop's r7_handoff/A__audio-r7-v3a1.tar (the r7_final/*_handoff_briefs.tar files are 0 bytes).
RealESRGAN_x2plus.pth was missing on r8-f, so the SR step failed silently and the composite returned rc 1. Fetched it from the official v0.2.1
release; its sha256 matches the pin.
Gate vs r7-quality-baseline: every row ok except 05 joints LF 0.901 -> 0.962. 05 SyncNet 3.47 -> 3.24 is generator run-to-run noise (+-0.2).

## 2. Diagnosis: the mouth flicker and the "explosions" come from the raw LatentSync output
- 03, r8_mouth_flicker.py, inner-mouth frame-to-frame HF luma: source 25 fps master 1.36 -> raw layer 2.47 -> composite 2.59.
  Flicker events 1.1 -> 2.35 /s; corr(luma, opening) 0.28 -> -0.01.
- ESRGAN, the colour match, the texture transfer and the sanitizer add ~0.1 between them.
- 01 0-1.2 s (fast head turn, ~45 px inner mouth): open/shut jumps, washed pink lip blobs (0.28-0.32 s, 0.72 s), a black hole (0.84 s).
- InsightFace is not used: MediaPipe stack, guard_nc().

## 3. Fixes (behind flags, default off)
- r8_layer_deflicker.py (DFK=<label> DFK_ARGS in r7_m7_comp.sh): the inner-mouth tone is smoothed over time and tied to the opening; an
  optional lower-face LF pass.
- r8_fallback.py + fps_composite --fallback (FB=<label> FB_ARGS): a per-frame artefact score vs the source -> held, ramped crossfade to the
  source mouth. Score components: wash, contrast collapse, near-black cavity (vs the source typical cavity), open/shut jerk, window joints.
- Fallback share of LatentSync frames per setting:
  - fb1 (m12safe, tone included): 25-49 %;
  - fb5 (visibly broken frames only): 3-24 %;
  - fb6 (fb5 thresholds, shorter holds; m13b): 1.8-16 %.

## 4. Candidates (WATCH files in /workspace/dub/outbox/)
- m12safe: audio r7, fb1.
- m13: audio-r8-v1, fb5 + dfk1.
- m13b: audio-r8-v1, fb6 + dfk2 (inner mouth only, +-15 %, sigma 1). See m13b_WATCH.md.
- m15: the m13b stack on audio-r8-v4 (in progress).

## 5. Where SyncNet goes (dub-speech, the gate metric)
| arm | 03 | 04 | 05 |
|---|---:|---:|---:|
| m11 baseline | 7.86 | 6.42 | 3.47 |
| r8v1 layers, plain composite | 7.79 | 6.04 | 3.58 |
| + fb5 only | 5.88 | 4.71 | 3.13 |
| + dfk1 only | 6.79 | 5.55 | 2.97 |
| + both (m13) | 5.04 | 4.23 | 2.62 |
| m13b (fb6 + dfk2) | 5.91 | 4.78 | 2.82 |
| m11 audio layers + fb5 + dfk1 (05) | | | 2.35 (plain 3.24) |
- The audio is not the cause: 05 r8v1 plain is 3.58, above the baseline.
- The deflicker (a temporal tone filter) and the fallback each cost ~0.5-1.0.
- On 05 only 3.2 % of frames fall back, yet SyncNet drops 0.45: the flagged frames sit in articulation (dub speech), and SyncNet scores
  exactly those windows.

## 6. Pressed lips on 05 (the gate row) vs the audio agent's source-speaks-dub-silent
- What the face row counts: the OUTPUT mouth closed (< 0.02 IOD) while the SOURCE mouth is open (> 0.05 IOD), inside dub-silent spans
  (energy VAD < -45 dB, >= 0.3 s). It does NOT require the source to be speaking.
- Only ~2.0 s of m13 05's 6.55 s overlap the source speech intervals (m11: ~2.3 of 6.92 s).
- The rest is the source mouth hanging open in pauses and breaths (6.5-7.2, 16.3-17.0, 24.7-25.9, 44.9-45.6 s), where a closed dubbed mouth
  is right.
- Real remaining spans in m13 05 (source speaking, dub silent): 41.07-41.71 (0.63 s), 20.95-21.49 (0.47 s), 46.35-46.59; in m11 also
  31.95-32.22.
- Proposal: gate on source SPEECH (the manifest speech_intervals) instead of the source mouth opening.

## 7. Old NC face stack (user decision relayed by the coordinator: CodeFormer + InsightFace back, licence risk accepted)
Evidence in git:
- cand5 (CodeFormer w 1.0, mouth scope + InsightFace) vs clean0 (MediaPipe, no CodeFormer), same audio, SyncNet dub-only:
  01 8.02 / 7.91, 03 5.73 / 5.75, 04 3.85 / 3.98, 05 3.42 / 3.87 (lipsync-temporal/baseline, r3-integration/scorecard).
- Pilot 0.7: CodeFormer lowers SyncNet at every fidelity weight (04: 3.21 -> 2.85 at w 0.3, 3.17 at w 1.0).
- So it never helped lip-sync; it changes the look (lip / teeth texture).
Restoring it (downloading the weights + unlocking assert_clean_tree under DUB_ALLOW_NC=1) was BLOCKED by this session's permission check. It
needs the user's own go-ahead in the F session. Not done.

## 8. DELIBERATE LICENCE DEVIATION (29.10, the user in the F session: "Yes, restore it"; the licence risk is accepted by the user)
The old NC face stack is restored for experiments, behind DUB_ALLOW_NC=1 only (the defaults stay clean; without the env var assert_clean_tree
still fails on the files below):
- CodeFormer (S-Lab License 1.0, NON-COMMERCIAL): third_party/CodeFormer/weights/CodeFormer/codeformer.pth (sczhou/CodeFormer v0.1.0 release,
  377 MB) + facelib detection_Resnet50_Final.pth (110 MB) + parsing_parsenet.pth (85 MB, CC BY-NC-SA).
- InsightFace buffalo_l (NON-COMMERCIAL research only): third_party/latentsync/checkpoints/auxiliary/models/buffalo_l (deepinsight/insightface
  v0.7 release, 341 MB).
Code: clean_face_stack.assert_clean_tree() lets them through only under DUB_ALLOW_NC=1 (and prints the deviation); candidate_video_stage
--dump-ls-raw loads the 25 fps master lazily (on the InsightFace path the master is written after the hook) + thumbnail fallback match.
m14 = cand5 flags (--codeformer optimized --codeformer-w 1.0 --codeformer-scope mouth --face-stack insightface) on audio-r8-v4 + the m13b
composite (fb6 + dfk2). The dumped layer is CodeFormer-restored (restore_frames runs before the mouth-only composite).

## 9. m14 result
See m14_WATCH.md: the old stack is not a lip-sync win (SyncNet +-0.35 either way), doubles mismatch (03, 05) and pressed lips (03), slightly crisper lips.
Plain composites on audio-r8-v4 (no fallback / deflicker): clean SyncNet 01 6.47, 03 7.90, 05 4.02 (all pass), i.e. the fallback + deflicker cost ~1.5-2 on 03.


## 10. Speed of the CodeFormer face pipeline (user plan: CodeFormer stays in the product)
Profile m16, 03 (1.51 video-min), s/video-min, measured while other jobs ran (inflated):
- generator 453: LatentSync 361 (UNet 225, VAE / conditioning 80), CodeFormer ~41, MediaPipe / prep ~45;
- layer SR ~66, CodeFormer detail ~20, deflicker ~29, fallback + composite ~55;
- face total ~620.

| lever (flag) | arm | effect | gate vs the previous arm |
|---|---|---|---|
| adaptive DDIM steps: 10 on speaking windows, 20 on dub-silent ones (`--steps 10 --steps-silent 20`) | m17 vs m16 | LatentSync 03 361 -> 276, 05 320 -> 264; UNet 225 -> 147 | OK: SyncNet 01 4.05 -> 3.89, 03 5.68 -> 6.11, 05 3.52 -> 3.65; mismatch better; pressed lips +0.2/+0.3 s (tolerance 0.5) |
| drop the ESRGAN layer SR (`SR=""`) | m16nosr vs m16 | -66 s/vm | REJECTED: 05 SyncNet 3.52 -> 3.13 (tiny face); 03 fine (5.75) |
| CodeFormer without ParseNet + batch 16 (`--codeformer-parse off --codeformer-batch 16`) | m18 vs m17 | 37 -> 57 faces/s (03 CodeFormer 58 -> 37 s) | OK (same run, see below) |
| fast layer SR (`SR_EXTRA="--batch 8 --mask down2"`) | m18 vs m17 | 03 SR 90 -> 49 s | OK |
| pre-CodeFormer dump in the generator (`_ls_pre.mp4`, `CFD=self`) | m17 / m18 | no separate clean run needed for the detail transfer | - |

m18 vs m17 (01 / 03 / 05):
- SyncNet 4.06 / 6.10 / 3.76 vs 3.89 / 6.11 / 3.65;
- flashes 0 / 29 / 4 vs 0 / 34 / 3;
- pressed lips 0 / 3.70 / 4.85 vs 0 / 3.70 / 4.88;
- mismatch 0 / 1.87 / 1.67 vs 0 / 1.87 / 1.54;
- whole-frame gate PASS.

m18 face stage on 03 (generator 496 s + SR / detail / deflicker / fallback / composite 222 s) ~ 476 s/vm, under contention; a clean timed
run is pending.

Bug found: r8_teeth.py cached MediaPipe landmarks next to its input. When it was run on /workspace/dub/inputs/test_videos_raw/05.mp4, this
wrote 05.flk_mp.npz into the inputs dir, and r7_frame_gate.sh's 05.* glob then picked it up. The file is moved out and the inputs are
sha256-verified; every 05 frame gate since 14:50 was re-run: PASS.

Next levers:
- fuse detail + SR + deflicker into one decode / encode pass, or use the veryfast preset for intermediates (`--preset`, added): ~-40 s/vm;
- the fix30 caches (FIX30_TD_SHARED, P2_PREP_CACHE);
- CodeFormer torch.compile;
- VAE (75 s/vm);
- CPU / GPU overlap.
