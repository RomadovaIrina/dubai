# r13 KeySync: handoff (paused 2026-11-03, the user's decision)

Branch `experiment/r13-keysync`. Scripts are in `scripts/keysync/`; gate logs and colour-check JSONs are in `reports/experiments/r13-keysync/colorfix/`.
The KeySync code changes are in `scripts/keysync/keysync_5e2503b_sm120_r13.patch` (a diff of the keysync repo at 5e2503b, incl. the r12 changes).
The server r13-k is destroyed after this push.

## Finals (management set, current default)

`outbox/r13_ks/final_s8/` (downloaded by the coordinator):

| file | KeySync | post | SyncNet speech at0 | strong flashes | jerk (src) |
|---|---|---|---|---|---|
| m18_s8cfhf_KeySync.mp4 | f_s8 | cfhf + α 0.8 | 7.652 | 3 | 15.5 % (11.0 %) |
| m03_s8cfhf_KeySync.mp4 | f_s8 | cfhf + α 0.8 | 7.302 | 7 | 10.9 % (16.1 %) |
| old03_s8cfhf_KeySync.mp4 | f_s8 | cfhf, no smoothing | 7.434 | 27 | 24.0 % (24.3 %) |
| old03_s8cfts_KeySync.mp4 | f_s8 | cfhf + α 0.8 | see the r13_s8final.log tail | | 17.9 % |

- The earlier 10-step finals with cfhf are in `outbox/r13_ks/colorfix/`.
  - m18: 7.404 / 4
  - m03: 7.264 / 5
  - old03 cfhf: 7.353 / 28
  - old03 cfts: 6.995 / 20
- The f_s8 finals are no worse on the gate.
- The m03 global SyncNet reads av_offset 6 on every m03 version (10-step and s8, v2 and cfhf). The speech-at0 number is fine, but m03 sync has never been eye-checked for this.

**OPEN, the user's choice:** old03 with smoothing or without.
- cfts (α 0.8): fewer flashes and smoother motion (10-step: 28 → 20 flashes, jerk 22 → 16 %). On 10-step it cost ~0.36 SyncNet vs cfhf.
- cfhf (no smoothing): what the viewer saw, colour + detail fixed.
- 0.5x clip: `colorfix/old03_cfhf_vs_cfts_crop_28.5s_halfspeed.mp4`

## Speed default: f_s8 + KS_LMK_CACHE ≈ 490 s/vmin (1×5090)

- **Config:** `ks_fast.py --cfg _s8 --vae-half --compile unet`, KS_DC=2:0 (DeepCache on both models), KS_LMK_CACHE=<dir>.
- **Landmark cache:** in prod, landmarks are computed during TTS / separation. The output is identical; −7..8 %.
- **Measured (warm, s/vmin):**
  - m18: 496
  - m03: 485
  - old03: 496 on the second pass (~531 for f_s8 warm without the cache)
- **Stage split, m18 at 8 steps:** interpolation sampling 132 s of 166 (80 %), VAE decode interp 29 s, keyframes 14 s.
- ~430 s/vmin is where KeySync ends without training / distillation. The 300 target is not reachable with inference tricks.

### Levers tried

| lever | result |
|---|---|
| 10 steps (authors) | ~1035 s/vmin, the r12 baseline |
| f_s8 (8 steps) | ~530, pass |
| f_s6 (6 steps, both models) | ~470, FAIL: m18 SyncNet 7.10, flashes 15 |
| VAE decoder channels_last (f_s8dec) | pass, but no faster (~550): dead |
| FP8 (torchao) | dead on this stack, see below |
| SageAttention 1.0.6 (KS_SAGE=1) | −1 % only; m03 SyncNet 7.108, 17 flashes: FAIL, flag off |
| interpolation 7 steps, keyframes 8 (k8i7) | −11..12 % (m18 440 / m03 427); m18 / m03 sync OK, but old03 flashes 24 vs 19: FAIL |
| interpolation 6 steps, keyframes 8 (k8i6) | ~420–430; flashes m18 13 / m03 18: FAIL |
| chunk overlap reduction | no lever: the segments overlap by 1 frame of 14 (the ends are the keyframes); overlap 0 = skips every other segment |
| DeepCache on keyframes only | skipped: already on (KS_DC is global), and keyframes are 8 % of the time |
| CFG batching | already batched (3 rows per UNet call) |
| landmarks off the critical path | −7..8 %, pass: in the default |

**FP8 details:**
- PerTensor on fp32 weights: 0 layers quantised.
- bf16 cast + PerRow: needs bf16 activations → KS_AC_BF16 autocast, plus an input-cast pre-hook.
- Then: "Rowwise scaling is not currently supported" on sm_120 with torch 2.7.1.
- PerTensor + bf16: stuck in compile, killed.

## Quality stack (all behind flags; ks_final.sh env)

- **KS_CMS=1:** BT.709 colour-managed ffmpeg I/O (`ks_cms.py`). ffmpeg 6.1 ignored the tags and lost ~1.7 L per YUV↔RGB round trip; the box was 6–9 L darker.
- **ks_layer --end-fade 12:** cross-fades KeySync → master before KeySync's last frame. KeySync outputs 8 frames fewer, which caused a hard colour cut at the end.
- **ks_fix:**
  - --skin-cm 12 --cm-robust 1: skin colour match, running median.
  - --guard-fade 10: the explosion-guard swap cross-fades over 10 frames instead of 3.
  - --guard 3 --chin tex: as before.
- **KS_HF=3 KS_SHARP=0.3** (ks_paste.py): outside the dilated lower face, the source's high frequencies replace KeySync's. Unsharp 0.3 on KeySync's own part.
  - Sharpness ratio vs src: ~0.5 → 0.84–0.99.
- **KS_TSMOOTH=0.8:** the r12 lag fix.
- **Result on old03:** colour jumps in the upper face 3 → 0, mean L −8.7 → +0.5, sharpness up/low/bg 0.47/0.54/0.39 → 0.84/0.83/0.91.
- **Tools:** `ks_colorcheck.py` = per-region Lab drift, colour jumps and the Laplacian sharpness ratio vs the source. Panels: `r13_panel.sh`, `r13_cfhf.sh`.

### Failed quality fixes (flags stay off)

- **ks_fix --flash-fix 4** (single-frame mouth flash repair at 25 fps):
  - Repaired 235 frames (10 %) on old03; SyncNet 7.353 → 6.51; flashes 28 → 27. FAIL.
  - The strong flashes are mostly NOT 25 fps layer spikes; they appear in the 25→30 fps paste.
- **ks_fix --lip-chroma 0.7** (the pale upper lip on old03): the measured lip chroma delta is small (a* +1.35). No measurable change. The "vanishing" lip is shape / edge, not colour.
- What does cut flashes is the α 0.8 smoothing (old03 29 → ~20).

## NEXT (when KeySync resumes)

1. The user's verdict on old03 cfhf vs cfts and on the colour / detail fix overall (the eye check is pending).
2. Flashes at the 25→30 fps paste: try motion-compensated or blend resampling in ks_paste instead of the nearest frame, and measure r3_flash before / after on the paste only.
3. The old03 upper lip: shape-level (the layer's upper-lip landmarks vs the source's; a lip-line edge transfer), not chroma.
4. Eye-check m03 sync (the global SyncNet offset 6 on every version).
5. Speed below ~430 needs training: step distillation (LCM / consistency) of the interpolation UNet, or a smaller interpolation model. Out of scope for inference tricks.
