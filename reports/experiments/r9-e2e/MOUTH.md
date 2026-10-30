# r9: mouth rendering on the new videos (30.10.2026)

The user, on m03 (both mixes): "the sound is fine on both, but the lip-sync seems to go to hell". They describe three things:
- the mouth opens unnaturally;
- the teeth look drawn;
- the lips stick together and unstick;

plus one more: "the mother's mouth slid onto the son's ear".

**Conclusions:**
- No timing problem.
- Three rendering causes, two of them our own composite.
- One generic composite-only fix set (`fx`) that improves SyncNet on all three test videos.
- The LatentSync reference change is not a clear win.

## 1. What it is not
- **Timing:** the SyncNet best offset is 0 in all 12 speech windows of 3 s on m03 (no drift); speech conf 3.8–9.4.
- **The face track:** SyncNet scores the mother (track conf 6.66; the child 0.15). The generator, its layer and the composite stay on her in 100 % of frames, and her landmark track never jumps (≤ 0.055 IOD per frame; it matches an independent 2-face pass).
- **Occlusion:** outside her lips the output equals the source on every frame (95th-percentile luma difference 6–9). The child's ear is never overwritten. This check (`scripts/r9/occl_scan.py`) is now a gate row anyway.

## 2. Causes (m03 traced frame by frame through raw LatentSync → +CodeFormer → +detail / SR → final)
| user's words | cause | layer | evidence |
|---|---|---|---|
| lips stick / unstick | the m19b **tight fallback fb7** swaps a flagged generated mouth for the SOURCE mouth (another shape) for ONE frame (weight 0.2 → 1.0 → 0.2); the swap itself is the snap. Also 1–2-frame opening spikes in raw LatentSync. | composite / fallback (+ raw LatentSync) | 4 of the 5 composite-made events on m03 sit exactly on fb7 peaks. Over the 8 face videos, composite / fallback makes 27 final events, raw LatentSync 16, CodeFormer 3. |
| drawn teeth | **CodeFormer** detail makes the inside of the mouth 1.5x sharper and +13 luma brighter than her real teeth; the detail transfer keeps 1.3x, and the composite re-brightens. | CodeFormer → composite | m03, open-mouth frames: fine-detail ratio to the source 0.60 (raw) / 1.51 (CF) / 1.32 (detail + SR) / 1.17 (final); brightness −4 / +13 / +3 / +10 luma |
| mouth slid onto the son's ear | **raw LatentSync** draws a doubled / smeared mouth, shifted sideways, for ~6–7 frames while she turns fast with the child pushing in (18.9–19.2 s), and again with red crack lines along the jaw (20.3 s). fb7 flags only 1 of those frames. | raw LatentSync | lateral offset of the layer's lip colour (Lab a*) from her lips: 0.07–0.11 IOD there; the video's 99th percentile is 0.045 |
| mouth copies Russian (sync feels off) | LatentSync's reference = the current unmasked source frame (`--ref-mode same`), so it re-articulates her Russian mouth | raw LatentSync | output opening vs her mouth r = 0.46, vs the dub 0.26 |

Artefacts per video (shipped r9, final output; classes from `scripts/r9/artefacts.py`):

| video / arm | LS frames | a stick / unstick | b amplitude | c teeth | d slide | origin of the final events |
|---|---|---|---|---|---|---|
| m19 r9 | 415 | 0 (0 fr) | 0 (0 fr) | 0 (0 fr) | 0 (0 fr) |  |
| m01 r9 | 2156 | 4 (5 fr) | 0 (0 fr) | 16 (32 fr) | 1 (2 fr) | raw LatentSync 12, composite / fallback 7, detail transfer / SR 1, CodeFormer 1 |
| m04 r9 | 524 | 2 (2 fr) | 0 (0 fr) | 2 (4 fr) | 0 (0 fr) | composite / fallback 4 |
| m03 r9 | 944 | 6 (9 fr) | 0 (0 fr) | 0 (0 fr) | 2 (2 fr) | composite / fallback 6, raw LatentSync 2 |
| m15 r9 | 795 | 2 (4 fr) | 0 (0 fr) | 0 (0 fr) | 2 (6 fr) | composite / fallback 4 |
| m17 r9 | 1040 | 1 (1 fr) | 1 (1 fr) | 0 (0 fr) | 1 (1 fr) | composite / fallback 1, raw LatentSync 1, CodeFormer 1 |
| m18 r9 | 475 | 3 (3 fr) | 1 (2 fr) | 0 (0 fr) | 2 (2 fr) | composite / fallback 5, raw LatentSync 1 |
| m20 r9 | 679 | 0 (0 fr) | 0 (0 fr) | 1 (1 fr) | 0 (0 fr) | CodeFormer 1 |
| **all** | | | | | | composite / fallback 27, raw LatentSync 16, CodeFormer 3, detail transfer / SR 1 |

## 3. Fixes (all generic and opt-in; the defaults are unchanged)
- **Layer repair** (`scripts/r9/r9_layer_repair.py`, `REP=1` in `r7_m7_comp.sh`):
  - Flags: the fallback's own flags, 1–2-frame opening spikes, landmark slides, and a lip-colour detector for sideways slides (lateral only: a different mouth SHAPE moves the colour centre vertically).
  - Runs of ≤ 3 frames are re-drawn by flow interpolation between the neighbouring good generated frames.
  - Longer runs get the source mouth over the whole run with Gaussian ramps (σ 2 frames). There are no 1-frame swaps any more.
- **Teeth:** `r8_cf_detail.py --inner-alpha 0.4` sets the CodeFormer detail weight inside the inner-lip polygon. m03 teeth sharpness goes 1.17x → 0.97x of hers. (`--sharpen-lips-max 0` changed nothing.)
- **Reference:** `--ref-mode fixed / shift16` plus a new `--audio-lead` (in 50 Hz whisper steps). Without the source copy, the generator answers ~0.5 frame late, and lead 0.5 brings the offset back to 0.
- **One command:** `E2E_FX=1` (repair + teeth) and `E2E_REF=<mode>:<lead>` in `scripts/r9/e2e.sh`.

## 4. The WATCH panels (full videos in `outbox/r9_e2e/watch_mouth/`: r9 shipped | r9 + fix | ref shift16 + fix | ref fixed + fix)
| video | panel | SyncNet (dub speech) | flashes | mouth / dub mismatch s | pressed lips s | a stick / unstick | b amplitude | c teeth | d slide | source fallback frames |
|---|---|---|---|---|---|---|---|---|---|---|
| m03 | r9 shipped | 7.225 | 26 | 0.57 | 0.63 | 6 | 0 | 0 | 2 | 39 |
| m03 | r9 + fix | 7.558 | 19 | 0.3 | 0.67 | 0 | 0 | 0 | 1 | 43 |
| m03 | ref shift16 + fix | 7.414 | 31 | 0.63 | 1.23 | 2 | 1 | 0 | 2 | 36 |
| m03 | ref fixed + fix | 7.251 | 32 | 1.27 | 2.64 | 2 | 0 | 0 | 2 | 36 |
| m19 | r9 shipped | 6.114 | 4 | 0.27 | 1.55 | 0 | 0 | 0 | 0 | 22 |
| m19 | r9 + fix | 6.755 | 3 | 0.54 | 1.62 | 0 | 0 | 0 | 0 | 12 |
| m19 | ref shift16 + fix | 6.275 | 9 | 0.27 | 1.66 | 0 | 0 | 0 | 0 | 23 |
| m19 | ref fixed + fix | 5.508 | 9 | 2.97 | 0.91 | 0 | 0 | 0 | 3 | 83 |
| m15 | r9 shipped | 1.947 | 10 | 0.73 | 0.6 | 2 | 0 | 0 | 2 | 68 |
| m15 | r9 + fix | 6.973 | 12 | 1.3 | 0.93 | 4 | 0 | 0 | 0 | 41 |
| m15 | ref shift16 + fix | 6.007 | 11 | 1.1 | 0.87 | 1 | 0 | 0 | 0 | 106 |
| m15 | ref fixed + fix | 5.553 | 20 | 1.67 | 0.73 | 1 | 0 | 0 | 1 | 183 |

Reading:
- **r9 + fix** (shipped generator + fx):
  - SyncNet goes up on all three; m15 jumps 1.95 → 6.97, because its shipped output kept swapping to the source mouth on 68 frames.
  - Flashes are equal or lower; m03's stick / unstick goes 6 → 0 and both slides are gone.
  - Watch-outs: the mouth / dub mismatch rises on m19 / m15, and m15 keeps 4 single-frame stick / unstick events.
- **The new references** cut the Russian-mouth copy (m03 0.46 → 0.26), but they cost pressed lips (fixed: 2.64 s on m03), mismatch (fixed: 2.97 s on m19), flashes, and much more fb7 fallback (fb7 was tuned on the shipped generator).
- **Recommendation:** r9 + fix. The user's eye decides on the reference.

## 5. IMG_6185 (158 s, 720x1280, 20 fps, lipstick near the face, burned-in Russian subtitles that remain)
- **Config:** run with r9 + fix: SUCCESS, both mixes, lip-sync on 3934 / 3954 master frames.
- **Repair:** 150 frames re-drawn, 10 on the source fallback (fb7 alone: 111).
- **First attempt:** the generator was OOM-killed at 44 GB. It kept every LatentSync frame twice in RAM, ~22 GB at 158 s. Fix 599c0ee: those frames go to disk; anonymous memory then stayed at 20–29 GB.
- **Production:** 46 min.
