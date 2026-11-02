# r11 FACE: per-speaker LoRA fine-tune of LatentSync 1.6 (CLOSED, flags off)

Idea (user-approved 01.11): fine-tune the generator on the person's own video with the source audio (self-reconstruction), then generate with the
dub, so that the lips / teeth / colour become hers and the frame-to-frame lip flicker drops.

## Code (all default off)
- `scripts/r11/ls_lora.py`: a minimal LoRA, no peft.
  - Adapters on `attentions.*` + `motion_modules.*` to_q / to_k / to_v / to_out.0 (288 for `all`); `self` = spatial attn1 only (64).
  - Merged into the base weights at load, so inference costs nothing extra. `set_enabled()` switches the adapters off to get the frozen-base teacher.
- `scripts/r11/ft_speaker.py --work <generator work dir> --out lora.pt`:
  - Inputs: the 25 fps master, LatentSync's own affine crops (MediaPipe detections replayed), the fixed mask, whisper chunks of the SOURCE audio.
  - Loss: mixed-noise epsilon, ×4 on the regenerated region; the reference comes from another window.
  - Data split: the first 80 % trains, the last 20 % validates at t = 100 / 400 / 700.
  - Flags:
    - `--targets all|self`;
    - `--open-bias k` (sample windows with p ~ the mouth opening^k);
    - `--distill w` (shape-keeping term, see below; `--distill-pool 4`).
- `scripts/quality/ls_stable.py`: `LS_LORA=<pt>` and `LS_LORA_STRENGTH=<s>` merge an adapter at StableRunner init.
- Cost (m18, 20 s, rank 16, 300 steps, RTX 5090):
  - 230-290 s training, 8.8 GB peak VRAM, ~690-810 GPU-s per video-minute;
  - with `--distill`: ~555-585 s, ~1660-1750 GPU-s per video-minute.

## Results (gate = r10 gate_cmp vs r9+fix of the same run; raw lip ring = face_metrics on the generator layer before CodeFormer)
| arm | SyncNet speech (base) | Δ | raw lip ring (base) | strong flashes (base) | verdict |
|---|---|---|---|---|---|
| m18 all, strength 1.0 | 4.61 (7.28*) | −2.7 | 1.33 (2.13) | — | FAIL: talks with a closed mouth |
| m18 all, strength 0.5 | 6.62 (7.74) | −1.13 | — | 5 (8) | FAIL |
| m18 all, strength 0.3 | 7.18 (7.74) | −0.57 | — | 5 (8) | FAIL |
| m18 self + open-bias 2 (r11-s) | 5.45 (7.74) | −2.28 | 1.20 (2.13) | 1 (8) | FAIL |
| m18 all + distill 10 | 7.33 (7.74) | −0.42 | 1.97 (2.13) | 5 (8) | FAIL (near) |
| m18 self + distill 10 | 7.50 (7.74) | −0.24 | 2.07 (2.13) | 7 (8) | FAIL (occluder ring 0 → 1); sync holds but no look / flicker gain |
| m03 all, strength 0.5 | 7.38 (7.65) | −0.27 | 1.84 (2.22) | 12 (13) | PASS; look gain small, no user clip |

\* First m18 arm, scored on the earlier gate base.

## Why it fails
- Self-reconstruction of a speaker whose mouth is mostly closed (m18) teaches the UNet to fill the mouth from the reference / masked context instead of following the audio.
- This happens regardless of which layers train: self-attention only was the worst arm.
- Lowering the strength interpolates back to the base model.
- The shape-keeping distillation (LoRA vs the frozen base on the same window with a FOREIGN audio window, avg-pooled ×4, regenerated region only) recovers most of the sync (4.61 → 7.33). But it keeps most of the flicker too (raw ring 1.97 vs 2.13).
- So her look and sync trade off along a single curve. No point on it gave both the −0.3 sync target and a visible lip gain.
- On a wide-open speaker (m03) the plain recipe holds sync, but the visible gain is small.
- Stopped 01.11 on the user's direction change (KeySync as the generator, quality work moves to the composite).

## Licence note
- LatentSync code is Apache-2.0; the weights are Open RAIL++-M. A per-speaker adapter is a derivative, so the use restrictions flow down.
- The adapters were internal only, never shipped.
