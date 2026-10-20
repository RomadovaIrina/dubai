# Round 4, VIDEO agent (also owns the final merge)

Branch `experiment/r4-video` from `904c5d2`. Server: RTX 5090, yours alone.

## Tasks, in order
1. **Reproduce v4's video** for 01–05 from 904c5d2 with v4's own audio. It must match `/workspace/dub/reference/v4/<id>_v4.mp4`:
   frame count, duration, and the metrics within noise.
2. **Simplify by ablation.** v4 stacks these layers:
   - flow_safe retime 25→30;
   - sanitize-layer;
   - lip sharpen;
   - colour match;
   - source skin texture;
   - face-clip;
   - seam LF ramp;
   - lip guard.

   Remove ONE at a time on all five videos and measure with the gate (+ a short human review sheet). Drop every layer that does not
   clearly help. The goal is the smallest stack that is at least as good as v4.
3. **Render LatentSync directly at the source frame rate**, as an arm. Feed a 30 fps master and map the audio features to 30 fps
   frames (LatentSync's audio windowing assumes 25 fps, so adapt the mel / whisper chunk-per-frame mapping). If quality holds, the
   whole 25→30 retime layer, and the artefacts it caused, disappears. Measure SyncNet, flashes, slide and speed against v4.
4. **"Half the face goes negative" in the first ~10 s of 04 (v5)** — dark / inverted face colour. Find the cause: colour match /
   sanitize-colour / face-clip under sun and glasses. Make sure it cannot happen. A gate metric on the mean colour shift of the pasted
   region vs the source.
5. **Chin / jaw wave** (second / third chin moving like a wave).
   - Keep the generated region to mouth + lips.
   - Return to the source pixels above the jawline with the feather inside.
   - Add a gate metric: chin-contour / jaw-landmark jitter, output vs source.
6. **Lip motion that does not match the speech** (generator).
   - Per-second SyncNet and mouth-open vs audio-energy correlation.
   - Compare guidance 1.5 / 2.0, bridge on / off, and inference steps on the worst seconds.
   - Optional exploratory arm, only if time remains: one alternative lip-sync model whose weights have a clean licence (check the
     licence first and write it in CHANGES.md).

## Publish and merge
- Video-only versions go out with v4's audio, after the gate passes: `outbox/<id>_r4_video_vN.mp4`, plus side by side
  V4 | NEW, `<id>_review.md` and `r4_video_vN_regression.md`.
- When the audio agent hands over a passing audio (`/workspace/dub/handoff/in/A/`), build the merged candidate
  `outbox/<id>_r4_merged_vN.mp4`. The merge passes the same gate against v4 and the previous merged version.
