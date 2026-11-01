# r10 FACE: what the eye sees, and where it is born (01.11.2026)

The user rejected F1 by eye ("Не лучше"), despite its metric gains. This note explains why. The clips are on r10-q in `/workspace/dub/outbox/r10_face/diag/`:
- `<id>_diagN_<t>s_{1x,0.5x}.mp4`: the mouth + chin crop at about 2x, four panels **SOURCE | RAW LatentSync | r9 + fix | F1**. RAW is the pre-CodeFormer layer at the nearest master frame.
- `<id>_diagN_sheet_frames*.jpg`: 12 consecutive frames, one row per panel.

The windows are the worst 4 s of the r9 + fix final, by lip-brightness pops + chin slide:
- m03 at 2.6 s and 13.4 s;
- m18 at 15.0 s and 3.8 s.

## What changes from frame to frame (plain words)
1. **Lip shape and lip definition, in the RAW generator output.** The source holds one steady mouth: a smile with the teeth showing (m03), or a calm open mouth (m18). The raw layer changes the mouth every frame or two:
   - **m03 2.6 s:** teeth on → mouth open dark → lips closed, no teeth;
   - **m18 3.8 s (frames 224–227):** the lips go soft and pale, the lip line almost disappears, then a sharp "o" pops in.
   - Part of this is the dub's articulation (it must differ from the source). The visible defect is the *definition* jumping between frames: sharp → mushy → sharp, teeth there → gone.
2. **The lip outline is drawn darker and redder than the source in the final**: the CodeFormer detail + the composite's lip sharpening. That high-contrast outline makes every shape jump above much more visible. The raw layer is softer than the final.
3. **The chin.** The re-rendered lower face is the generator's: smoother skin, no chin fold. It moves with the dub's jaw, while the jaw OUTLINE and the neck stay the source's (the composite keeps the source beyond 0.38 IOD from the lips). So the inside of the chin moves against a static outline, which reads as sliding / elastic. On m18 a chin crease line appears for 2 frames (225–226) and vanishes.
4. **The paste seam** itself is not visible in these windows.

## Where it is born (layer by layer, same metric on every layer; ratio to the source)
| video | lip-ring brightness pops: raw LS / + CF / + detail / + SR / final r9fix | chin flow wander (IOD/frame): raw LS / final |
|---|---|---|
| m03 | 2.14 / 2.28 / 2.15 / 2.08 / 1.87 | 0.0117 / 0.0049 |
| m18 | 2.08 / 2.23 / 2.11 / 2.10 / 2.15 | 0.0071 / 0.0070 |
| m15 | 2.28 / 2.48 / 2.34 / 2.27 / 1.90 | 0.0110 / 0.0069 |
| m19 | 2.60 / 2.76 / 2.60 / 2.54 / 2.27 | 0.0056 / 0.0041 |
| old03 | 1.91 / 1.88 / 1.91 / 1.87 / 1.81 | 0.0040 / 0.0018 |
| old05 | 0.92 / 0.95 / 0.93 / 0.95 / 1.08 | 0.0068 / 0.0044 |

- **Both artefacts are already in the RAW LatentSync output**, before CodeFormer, SR and the composite.
  - CodeFormer adds a little (+0.1–0.2).
  - The composite removes a little; it halves the chin wander via its region cap.
  - Nothing downstream creates them.
- A composite-side filter (F1) can damp brightness and smooth registration. It cannot make the generator's mouth shape / definition consistent from frame to frame, and that is what the eye follows.

## Proposal: generator-side levers (files owned by SPEED; for hand-off)
In order of expected value / cost. Each one should be checked on the same metrics (raw-layer lip-ring pops, chin wander) + SyncNet + the eye on these 4 windows.

| # | lever | what it changes | expected effect | cost / risk |
|---|---|---|---|---|
| 1 | **Shared initial noise across frames.** The same Gaussian noise for every frame of a window and across windows, instead of independent per-frame noise. | Per-frame texture / definition differences that come from the noise realisation | Fewer "sharp → mushy" lip-definition jumps; texture consistent frame to frame | No GPU cost. Must verify that lip-sync is unchanged (articulation comes from audio + reference, not noise). I did not read the generator code: if the pipeline already shares the noise, skip this row. |
| 2 | **Overlapping windows with latent blending.** For example 16-frame windows with a 4-frame overlap, latents cross-faded during denoising. | The jumps at window joints (every 16 frames); the composite's C2 interpolation is only a patch | Smoother transitions at the joints; fewer 1-frame pops | +25 % UNet calls |
| 3 | **More denoising steps where the dub speaks** (10 → 20), or DeepCache off in speech. | Under-converged frames (mushy lips) | Lip definition more stable | ~2x generator time in speech: conflicts with the 300 s/vmin target |
| 4 | **Temporal smoothing of the predicted x0 / latents between adjacent frames inside the denoising loop.** An EMA over ±1 frame of the x0 prediction in the last steps. | Frame-to-frame latent noise | Fewer flicker pops; can soften fast articulation (must be gated on SyncNet) | ~0 cost |
| 5 | **Generation mask: keep the source jaw / chin.** A mask that stops above the chin (mouth + lower lip + a band) instead of LatentSync's lower half of the crop. | The re-rendered chin that moves against the source jaw outline | Chin slide largely gone (the chin stays source) | The model was trained on the lower-half mask: risk to the mouth quality / sync at the mask edge; needs a test |
| 6 | **Reference frame per window, not per frame** (one calm source frame per 16-frame window). | The per-frame reference that changes the generated texture with the source mouth | More consistent texture within a window | r9 saw the fixed / shift references cost pressed lips + mismatch; this is a middle ground |
| 7 | Higher mouth-crop resolution | Detail resolution | Not the cause of the flicker (raw at 512 already flickers) | Low priority |

Composite-side, outside the "same kind" of tweaks, one cheap thing for the eye: soften the drawn lip outline. `r9_lipcolour.py` exists; the source-like contrast was 1.06x vs 1.73x on m18. A sharper outline amplifies any generator jump. Worth one A/B with the user only if the generator levers are delayed.
