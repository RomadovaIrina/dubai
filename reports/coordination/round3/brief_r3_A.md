# Round 3, Agent A: LIP-SYNC (generator side)

Branch: `experiment/r3-lipsync`, created from `origin/experiment/r2-lipsync-position` (latest 92de423, which includes the 05
coverage fix). Server: RTX 5090.

## You own
The LatentSync generator and everything that feeds it: face detection and coverage, crop anchors, windows and their joints,
head-pose pre-check, two faces / active speaker, and the dub-silent mouth.

## Keep (round-2 wins, do not regress)
- fp32 AlignRestore;
- flow crop anchors;
- head-locked mask;
- the 05 coverage fix.

## Tasks, in priority order
1. **Reproduce** r2-best on 03 and 05 at 25 fps with the canonical cand5 audio. Record SyncNet (dub-only and final mix, AV
   offset), the slide metric, mouth resid, frozen_mouth, coverage, and s/video-min.
2. **Lip-sync accuracy.** Find where sync is weak, per second, with SyncNet conf/offset and mouth-open vs audio-energy correlation,
   then fix it at the generator. Candidates:
   - audio-feature alignment and offset;
   - guidance and inference steps;
   - window overlap and the continuation context;
   - mouth-region resolution / upscale of the crop.
   Report the per-video and per-second worst cases before and after.
3. **Dub-silent / dub-shorter mouth.** When the dub is silent but the source face still talks, the mouth must look natural:
   closed or relaxed, never frozen inside a talking face, and never snapping at a window start. This is the "24–25 s / 35–36 s
   on 03" class. Work together with C's gap JSON (C hands you `residual gaps` per video). Options:
   - generate a closed or relaxed mouth that keeps jaw motion;
   - blend toward the source mouth with a timed fade;
   - other ideas, with numbers and clips.
4. **Window joints (spec 8.5)** at the generator. Measure jumps at every window start, then reduce them through overlap and
   cross-fade in latent or pixel space. Hand the joint list to B, because B attacks the same joints in the composite.
5. **Coverage and robustness (spec 8.2, 5.4 / 5.5, 11).** Profile and turned heads, two faces (only the active speaker),
   occlusions, the 2 % rule. 05 is the hard case.

## Audio
C regenerates and improves the audio (`audio-r3`) and hands it to you in `/workspace/dub/handoff/in/C/`. Until it arrives, use
canonical cand5. When it arrives, render 01–05 with it and hand the generator outputs (25 fps lip renders + per-frame face meta
the composite needs) to **B and D** through `handoff/out/B/` and `handoff/out/D/`.

## Deliver
- Clips in `outbox/`: ORIGINAL | CAND5 | R2BEST | R3 side by side, normal and half speed, plus a 2× mouth crop on the worst
  seconds.
- A table in `reports/experiments/r3-lipsync/README.md`.

You have the heaviest GPU work, so keep runs short (worst windows first, full videos only for winners).
