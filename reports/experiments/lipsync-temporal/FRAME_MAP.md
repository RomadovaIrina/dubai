# Source frame rate (spec 9.4): frame mapping and how to composite LatentSync onto the original-fps frames

Owner of the retimed composite: the post-side server. This note gives what the video stage records and how to use it.

## What the video stage records (clean runs, `manifest["frame_map"]`, `scripts/pilot/frame_map.py`)

The video stage runs on a 25 fps CFR master made by `ffmpeg -i SRC -r 25` (`face_aware_latentsync.make_master`). ffmpeg's vsync decides
which source frame each master frame holds (drop / duplicate with a drift tolerance), so the mapping is **measured**: every master frame is
matched to a source frame by grey-thumbnail difference (monotone search). Fields: `source_pts` (s), `master_to_source` (per master frame),
`match_mad`, `lag_s` (source time of the content minus the master timestamp), `source_master_pos` (= source_pts × 25, the fractional
master index of every source frame on the audio clock), `summary`.

| video | source frames (fps) | master frames | dropped / duplicated source frames | lag of master content vs its timestamp: median (min..max) | match error p99 (grey 0-255) |
|---|---|---:|---|---|---:|
| 01 | 2542 (VFR, avg 25.0, r 30) | 2542 | 52 / 18 | -7 ms (-293..+200 ms, VFR gaps) | 0.40 |
| 02 | 3324 (29.98) | 2773 | 551 / 0 | **-60 ms** (-73..0) | 0.41 |
| 03 | 2710 (30) | 2260 | 467 / 17 | **-60 ms** (-107..0) | 0.55 |
| 04 | 1617 (30) | 1349 | 268 / 0 | **-60 ms** (-73..0) | 0.49 |
| 05 | 1567 (29.9) | 1311 | 256 / 0 | **-60 ms** (-73..+7) | 0.57 |

Findings:
1. On every 30 fps source the master shows content **60 ms (1.5 master frames) older than its timestamp**: ffmpeg copies the first source
   frames one-to-one and only starts dropping once its drift tolerance is exceeded (03: master 0-11 = source 0-11). The dubbed audio lives on
   the source clock, so in today's 25 fps output everything except the generated mouth (head, eyes, pass-through frames) runs 60 ms behind
   the audio, and LatentSync generates audio-time-k mouths on faces of time k − 60 ms.
2. Every 6th source frame is dropped (02-05) -> the 5 Hz judder of the whole picture (PHASE1_PLAN.md §2.3: head displacement ×1.5 on one
   transition in five on the 03 master).

## How a composite onto the source frames gets a LatentSync frame per source frame

For source frame j (time t_j = `source_pts[j]` on the audio clock):

1. **Pass-through frames** (outside LS segments, `lipsync.segments`): output = the untouched source frame j. No resampling at all.
2. **LS frames**: the mouth must match the audio at t_j, i.e. fractional master index p = `source_master_pos[j]` = 25·t_j, with
   k = floor(p), u = p − k. LatentSync produced mouths for audio times k/25 and (k+1)/25. Two ways, simplest first:
   * **2D re-registration (no new data needed):** take the LS output frames k and k+1 (full frame, before the composite), cut the lower-face
     region, and warp it from master-frame geometry to source-frame-j geometry with the similarity transform between the stable landmarks
     (eyes + nose) of master frame k (content = source frame `master_to_source[k]`) and of source frame j (MediaPipe on the SOURCE video:
     `clean_face_stack.run_mediapipe(source)`); blend k / k+1 with weight u (u < 0.2 or > 0.8: nearest only; else flow-based interpolation,
     since a linear blend of two different mouth openings ghosts); then the usual feathered mouth-ellipse composite on source frame j, with
     the ellipse from source frame j's landmarks.
   * **Crop-space (cleaner, needs one more dump):** keep LatentSync's aligned 512² outputs per master frame (before its own restore; the
     video stage can dump them on request), interpolate in crop space between k and k+1 (the crop is head-stabilised, so flow there is pure
     articulation), and restore into source frame j with source frame j's own affine (computed from its MediaPipe landmarks with the same
     `MAP106` + anchor calibration as `clean_face_stack.ls_detections`). This removes the 5 Hz judder AND the 60 ms content lag.
3. Edge cases: LS segment boundaries keep the 4-frame crossfade (in source frames: 5); u at a segment end with k+1 outside the segment ->
   nearest frame; VFR sources (01) work the same because everything is driven by `source_pts`.

Alternative on the video-stage side (would need its own A/B, changes the LatentSync input): build the master with
`-vf fps=25:round=near` (nearest source frame per master timestamp) so the master content is not 60 ms late. It keeps the 5 Hz judder, so
the retimed composite above is still the real fix.
