# Round 3, Agent B: FACE STABILITY & COMPOSITE (post side)

Branch: `experiment/r3-face`, created from `origin/experiment/r2-composite-position` (6341e2e, FIX30). Also merge
`origin/experiment/r2-lipsync-position` (A's round-2 generator) so you can render your own inputs. Server: RTX 5090.

## You own
Everything after the generator:
- pasting the mouth / lower face back at the source frame rate and resolution (spec 9.4);
- the mask and its feather (spec 8.4);
- joints in the final video (spec 8.5);
- colour, light and sharpness match of the pasted region;
- the final mux.

The user's complaints that you own: "the lower face lags or slides relative to the upper face" and "the face jitters / twitches",
most visible on camera shake.

## Tasks, in priority order
1. **Finish FIX30.** Run source-fps compositing on all of 01–05. It was partial on 05 and on 03 60–68 s last round: find out why
   and fix it.
   - Prove there is no slide or lag on the shakiest windows (`/workspace/dub/common/shaky_windows.json`) using the slide metric
     and the v4 metric. Report residual px mean / p95 and the lowerface jerk against cand5 and r2.
   - Compare with the native 25 fps final.
2. **Jitter / twitch hunt in the final video.** Rank seconds by the v4 metric plus a temporal flicker measure on the pasted region
   (luma / colour flicker, edge crawl at the mask border). Look at the worst 10 seconds per video and fix the causes:
   - mask-border crawl;
   - colour flicker between windows;
   - sub-pixel resampling;
   - mask motion vs face motion.
3. **Joints (spec 8.5)** in the composite: detect and smooth them (temporal cross-fade of the pasted region at window starts,
   optical-flow-guided blend). A attacks the same joints in the generator: compare numbers and keep what wins.
4. **Sharpness without forbidden restorers.** CodeFormer stays out.
   - Try source-texture transfer, unsharp / frequency matching to the source mouth area, or an allowed restorer. GFPGAN is
     Apache-2.0 code: check that its *weights* licence is clean before using it and write the result to CHANGES.md.
   - The mouth must not look blurrier than the rest of the face, and must not flicker.
5. **Feather 10–15 px** verified on every video (report px).

## Inputs
- Until A hands you `audio-r3` lip renders (`/workspace/dub/handoff/in/A/`), render your own inputs with A's round-2 generator code
  on canonical cand5 audio.
- Hand your composite module and its best flags to **D** (via `handoff/out/D/` + your branch).

## Deliver
- Clips in `outbox/`: ORIGINAL | CAND5 | R3 side by side, normal and half speed, plus a 2× lower-face crop on the shakiest windows.
- A table in `reports/experiments/r3-face/README.md`.
