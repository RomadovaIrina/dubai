# r16 K2 (mouth): report, 07.11.2026

Server r16-k2. Branch experiment/r16-mouth (not pushed). Deliverables in /workspace/dub/outbox/r16_k2/{full,clips}; work dir /workspace/dub/r16k2.
All variants start from the r15 raw fast2 renders (HANDOFF_K/paste_inputs) and the plain paste (ks_plain.py). The mouth fix runs on the raw 512 crop before the paste (`scripts/keysync/r16/k2_mouth.py`).
The full outputs carry the accepted `final_mix_aac.m4a` unchanged (packet md5 checked; ks_plain itself re-encodes the AAC, so the outbox files were remuxed).

## Setup
- TRT on driver 580 works: kf 58.2 dB at 377 ms (ref 58.3 dB / 417 ms), interp student 58.4 dB at 254 ms (ref 281 ms).
- k_setup.sh now uses the r15 patch (it still referenced r13).

## 1. Black hole: share of open-mouth frames with a near-black, toothless cavity (k_cavity.py)
| variant | m03 | old03 | m18 | SyncNet speech@0 m03 / old03 / m18 |
|---|---|---|---|---|
| source | 3.1 % | 0.4 % | 6.7 % | – |
| plain (current) | 16.1 % | 10.8 % | 35.2 % | 8.636 / 7.681 / 7.989 |
| r15 --cavity 0.8 (lift) | 7.3 % | 5.2 % | – | – |
| **a: source teeth bank 0.6 + teeth detail** | **1.0 %** | **0.3 %** | 12.3 % | 8.508 / 7.667 / 7.803 (−0.13 / −0.01 / −0.19) |
| a': bank 0.85 | 0.0 % | 0.0 % | 0.8 % | 8.365 / 7.589 / 7.741 (−0.27 / −0.09 / −0.25) |
| b: 8 interp steps on wide-open frames | 15.5 % | 8.5 % | 38.5 % | 8.468 / 7.594 / 7.983 |
| c: synthetic interior + upper-teeth hint | 3.9 % | 4.0 % | 8.2 % | 8.626 / 7.675 / 7.901 (≈ 0) |

- **a (bank):** per generated frame, the nearest source mouth by inner-lip shape, with hysteresis. Its interior is affine-warped into the generated inner lip.
  - Only on frames where KeySync drew a black hole (darkness gate), and only as a lighten-only blend, so teeth KeySync already drew are kept.
  - Real teeth instead of the black hole at m03 25.3–25.5 s. The teeth are a bit grey and contrasty.
  - Costs SyncNet −0.1..−0.2 (bank 0.85: up to −0.27, which is within the r10 gate's −0.3 tolerance).
- **b (steps): more steps do not cure the black hole on m03.** Full renders: 2 / 4 / 8 steps = 16.3 / 15.2 / 17.2 % (old03 11.3 / 6.9 / 5.7; m18 42 / 36 / 30). prodKS (8 AYS + CFG) was 20 %.
  - Splicing 8 steps on wide-open frames only would cost ~202 / 156 / 187 face-GPU-s/vmin (m03 / old03 / m18; 75 % of m03's interp chunks contain a wide-open frame). **Rejected.**
- **c (synth):** metric-good and SyncNet-neutral, but by eye it looks like a grey-brown fill with a faint teeth strip. Less convincing than a.

## 2. #6: m03 0:03 smeared mouth
- Not a paste or mask problem: the child's finger is far from mama's mouth, and the mouth-box sharpness of the final is 0.8–1.1× the source.
- What reads as smeared is KeySync's teeth: one flat white bar with no separations, while the source shows separate teeth.
- `--detail 0.8` (bank mode) adds the matched source interior's luma high frequencies on every open frame. By eye it is a subtle improvement, not a cure; it is included in variant a.

## 3. #11 / #12 on KeySync plain (k2_lipmetrics.py; same definitions as r9 artefacts (a) / r14 tone_flicker)
- **Sticky lips (#12): not a problem.**
  - Stick/unstick events per minute: m18 6.0, m03 4.4, old03 2.0.
  - The source's own 1–2-frame spikes at the same threshold: 12.0 / 5.9 / 2.7 per minute, so KeySync is below the speaker's own level.
- **Lip flicker (#11): no every-other-frame shimmer** (12.5 Hz share 0.014–0.026).
  - Lips mean |dF−dS|: 4.5 / 3.6 / 5.0 grey levels (m18 / m03 / old03). r15 prodKS through the tsmooth+paste chain was 3.6 / 2.7 / 4.4, on another grid, so not strictly comparable.
  - Plain has no tsmooth, so it is ~25 % higher. Worth an eye check, but not an alarm.
  - Variants a and c lower it slightly (m03 3.55 → 3.11 / 3.26).

## 4. #7: stubble
- `--tex 2` on old03: `full/old03_keysync_fast2_plainfx_tex2.mp4`.
- Clips: plain | tex2 | source at 3–7 s and 23–25 s (0.5×). By eye the chin and cheek texture is back; the black-cavity metric is unchanged.

## Files
- full/: `<v>_keysync_fast2_{plain,a_bank,b_steps8,c_synth}.mp4` and `old03_keysync_fast2_plainfx_tex2.mp4`. Time burnt in, original audio.
- clips/: `m03_mouth_4panel_…_23-27s` and `_0-4s`, `m18_…_2-6s`, `old03_…_37-40s`, `old03_stubble_…_3-7s` and `_23-25s` (0.5×, time burnt in).

## Next (if a wins by eye)
- Tone the bank teeth down (they are a bit grey / contrasty): match the bank interior's L / chroma to KeySync's own teeth tone.
- Smooth the bank pick over time (110 switches on 283 touched m03 frames, held by hysteresis; check for teeth jumping in motion).
- Add tsmooth α 0.8 before the plain paste for #11.

## Round 2 (Irina, m03 ~25 s: "все еще немножко черновато, но хотя бы есть зубы"): interior tone
- `k2_mouth.py --tone T`: per open frame, the generated inner-mouth luminance histogram is mapped to the source's open-mouth frames.
  - Lighten only, so teeth that are already brighter keep their tone, and the chroma is pulled toward the source interior.
  - Applied on top of the bank (strength 0.6 + detail 0.8).
- Black-cavity share: 0 % on all three videos. Mean interior L on m03: plain 25.6, bank 27.7, tone 0.7: 29.3, tone 1.0: 30.1.

| SyncNet speech@0 | m03 | old03 | m18 |
|---|---|---|---|
| plain | 8.636 | 7.681 | 7.989 |
| bank | 8.508 | 7.667 | 7.803 |
| bank + tone 0.7 | 8.422 (−0.21) | 7.598 (−0.08) | 7.721 (−0.27) |
| bank + tone 1.0 | 8.371 (−0.27) | 7.553 (−0.13) | 7.680 (−0.31) |

- The drop comes from lifting the black-hole frames themselves (tone only on dark-gated frames gives the same −0.28 on m03): SyncNet treats a dark open mouth as a strong cue. A lighter interior trades directly against SyncNet.
  - Tone 1.0 is the closest to the source interior by eye (sheet m03 24.9–25.9 s).
  - Tone 0.7 stays inside −0.3 on all three videos.
- Files:
  - `clips/<v>_teeth_tone_4panel_plain_bank_tone07_tone10_*_half_speed.mp4` (m03 23–27 and 0–4 s, m18 2–6 s, old03 37–40 s);
  - `full/<v>_keysync_fast2_a_bank_tone{07,10}.mp4`.
  - All yuv420p with +faststart; the full files carry the original audio (packet md5 checked).

## Round 3 (Irina on the round-1 panel: "b лучше на маме, a и c хуже"): (a), (c) and the tone round dropped; gated steps
- **Why (a) looked worse** (m03 25.1–25.6 s frame by frame): on the open "O" the bank filled the whole cavity with a grille of teeth, i.e. an open jaw with clenched teeth.
  - The source interior was affine-stretched to the taller dub opening, so the teeth were stretched, the gaps read as bars, and the colour was brown.
  - It was not flicker: inner-mouth zigzag plain 13.6 = bank 13.6, b8 12.8, source 8.1.
- **`ks_gate.py`** (hook: `KS_GATE="thr=0.12,cap=0.25,hi=8,lo=2"` in ks_fast.py):
  - The gate: per decoded keyframe, the dark-pixel share inside the source mouth box (landmark cache). On m03 this correlates 0.83 with the generated opening, and threshold 0.12 ≈ opening > 0.22.
  - It is capped at the darkest 25 % of keyframes per keyframe batch; a segment is "hi" if either of its keyframes is.
  - The sampler runs the 8-step (or 4-step) AYS grid. Lo segments are evaluated only at s0 / s4 (= the plain 2-step student), and 28-row UNet chunks with no active row are skipped.
  - About 21–23 % of segments run hi.

| warm face GPU-s/vmin m03 / old03 / m18 | SyncNet speech@0 m03 / old03 / m18 (plain tex2 = 8.616 / 7.717 / 7.986) |
|---|---|
| 2 steps (plain): 137 / 126 / 134 | – |
| b8 full (8 steps spliced on open > 0.18; Irina liked it): est. 212 / 165 / 200 | 8.488 / 7.630 / 7.977 |
| **b8 gate: 161 / 130 / 176** | 8.494 / 7.813 / 8.117 |
| **b4 gate: 135 / 128 / 144** | 8.464 / 7.784 / 8.216 |

- Raw black-cavity share stays roughly where it was (m03 16.5 / 14.3 %): steps don't cure the black hole. What Irina liked in b is the steadier and smaller mouth (the inner-mouth zigzag drops).
- Files:
  - `clips/{m03,m18}_steps_4panel_plain_b8full_b8gate_b4gate_*_half_speed.mp4` (m03 23–27 and 0–4 s, m18 2–6 s);
  - `full/<v>_keysync_fast2_{plain_tex2,b8_full,b8_gate,b4_gate}.mp4`.
  - All with tex2 (now the ks_plain default), the original audio, yuv420p with +faststart.

## Round 4: generator levers for the black hole (b8 gate accepted by Irina as the default)
Window tests (m03 22–28 s, m18 2–6 s; the window clip = [reference frame] + window, KS_REF_IDX=0, first output frame dropped). Black-cavity share of open frames (small samples: ~80 / ~35 open frames):

| arm | m03 | m18 | note |
|---|---|---|---|
| base (identity = frame 0) | 13.8 % | 22.9 % | |
| **ref_teeth: identity = a source frame with visible upper teeth (m03 #84 = 3.36 s, m18 #238 = 9.52 s)** | **8.1 %** | **7.9 %** | teeth share 0.028 → 0.047 (m03); free |
| ref #25 / #887 (m03), #76 / #7 (m18) | 9.4 / 12.6 % | 23 / 27 % | |
| keyframe audio guidance 3 / 7 | 11.4 / 13.8 % | 31 / 15 % | noise |
| keyframe ref guidance 1 / 3 | 12.5 / 10.1 % | 24 / 12 % | rr3 over-sharpens (m18 dark cheek spots, harsh contrast) |
| keyframes 12 / 16 steps (all keyframes) | 13.6 / 12.2 % | 22 / 19 % | +3 / +6 s/vmin keyframe stage; little effect |
| keyframes without DeepCache | 13.3 % | 22.9 % | |

With the b8 gate on top:

| arm | black-cavity m03 / m18 | SyncNet conf at best offset m03 / m18 |
|---|---|---|
| base | 20 % / 15 % | 7.70 / 6.30 |
| ref_teeth | 8.1 % / 9.1 % | 7.95 / 7.60 |
| ref_teeth + rr3 | 9.0 % / 16 % | 7.76 / 7.30 |
| rr3 | 16 % / 11 % | 7.45 / 6.88 |

- SyncNet at offset 0 is not usable on these windows (the window cut leaves a −2 offset for all arms alike).
- With ref_teeth the gate flags fewer segments (16 vs 40 %), so it is cheaper.
- Hooks:
  - `KS_REF_IDX` (dubbing_pipeline_raw.py: random_cond_idx; scripts/keysync/r16/keysync_r16_refidx.patch);
  - `KS_KF_AR` / `KS_KF_RR` (ks_fast.py: keyframe guider ratios).
- Picking the frame automatically: a slightly open mouth (opening 0.08–0.25), visible upper teeth, sharp, good light; it needs an eye check per speaker (m03 #84 has the child's finger by his own mouth, which is harmless).
- Panels:
  - `clips/m03_keyframe_ref_b8gate_base_refteeth_refteeth-rr3_rr3_23-27s_half_speed.mp4`;
  - `clips/m18_…_2-6s_half_speed.mp4`.
  - Raw 512 KeySync windows with the final-mix audio; at m03 25.56 s ref_teeth shows upper teeth where base has the black "O".

## Round 5: ref_teeth full length, the parted-lips side effect, pause arms, the tracked crop
**ref_teeth, full length** (`k2_refpick.py` auto-pick; m03 #46 = 1.84 s, m18 #242 = 9.68 s, old03 #2002 = 80.08 s; contact sheets in `outbox/r16_k2/refpick/`):
- Black-cavity share: m03 16.5 → 8.5 %, m18 39 → 25 %, old03 4.5 → 5.9 %.
- Side effect: lips rest parted in dub pauses (closed in silence m03 59 → 22 %, old03 47 → 21 %; source 51 / 50 %).
- A less-open teeth reference (open target 0.10) doesn't fix it (m03 12 %, old03 17 %), and on m18 it makes the black cavity worse. Any reference with visible teeth has parted lips.

**Pause arms** (Irina's rule: no KeySync ↔ source fade on a visible face). Pause = a dub-silent run ≥ 6 frames (> 35 dB below peak), eased over 4 frames.
- (a) `ks_plain --lip-keep` (k2_pause.py weights): the source lips (hull + 0.10 IOD) show through the paste mask, only where the source mouth is closed too.
- (b) `k2_pause.py --mode warp`: KeySync's own lips closed by a per-column vertical warp (the upper lip band stretched down, the lower lip band up, 0.35-mouth-width skin margin); no source pixels at all.

| closed in dub pauses | source | ref_teeth | a | b |
|---|---|---|---|---|
| m03 | 53 % | 27 % | 52 % | 53 % |
| old03 | 50 % | 23 % | 49 % | 44 % |

- Black cavity unchanged.
- By eye at m03 35.2 s: ref_teeth = open with teeth; a = unchanged there (the source mouth is not closed on that frame); b = naturally closed lips.
- Panels: `clips/{m03,old03}_pause_3panel_refteeth_a-srclips_b-lipwarp_*_half_speed.mp4` (m03 33–36.5 and 37.5–40.5 s, old03 54.5–58 and 81.5–84.5 s).

**Tracked crop** (Q's q05 / q06: the static box makes the face 0.27–0.36 of KeySync's input):
- `k2_prep_track.py` gives a constant-side square box (1.8 face_h + 2 × the smoothing residual) whose centre follows the gaussian-smoothed face track (sigma 6 frames); `ks_plain` takes per-frame boxes (`xs` / `ys` in box.json). The static path is bit-identical (regression max diff 0).
- Face share in the input: q05 0.355 → 0.44, q06 0.27 → 0.52. m03 / m18 / old03 keep the static box (already 0.49–0.72). Policy: track only when the static face share is < 0.45.
- q05 (driven by its own audio for the A/B): the static box gives a teal blob over the mouth and a white neck slab in every shot; the tracked box gives a normal talking face (the beard is thinner, KeySync's known beard weakness).
  - Panel: `clips/q05_crop_source_static_tracked_{2-6,28-32}s_half_speed.mp4`.
  - Still broken: the 20 s cut to a product shot with a small picture-in-picture of him, in every crop. Needs shot handling: no KeySync on shots where the face is small / inset.
- q06: VHS, near-profile, heavy gestures; MediaPipe finds a face in only 19 % of frames (27–38 s). KeySync's own detector found none in the static crop (crash: "No valid landmarks found"). The tracked crop is rendering.

## Round 6: full finals = ref_teeth + b8 gate + K1 chain (k1/r16-paste 5c07cf7) + pause arm b
Script: `scripts/keysync/r16/k2_k1chain.sh <v> <ref>` (run dir /workspace/dub/k1run with K1's tools).
- The chain: the pad60 crop (no duplicated seam frame) + the 40 ms delayed, silence-padded dub, then ECC, tsmooth gate:0.5, snap, guard, ks_plain (lfm 6, upper-src, keep-nose, align 2, tex 2).
- Optionally `k2_pause.py` warp on the guarded layer.

| video (ref) | SyncNet speech@0 base / + pause b | black-cavity raw | guard replaced | last 30 frames |
|---|---|---|---|---|
| m03 (auto #46) | 8.71 / 8.73 | 8.0 % | 0 | clean (2.3×) |
| **m03 (manual #84)** | 8.60 / 8.62 | 6.7 % | 1 (24.96 s) | clean |
| m18 (#242) | 7.68 / 7.67 | – | 0 | clean (2.3×) |
| old03 (#2002), shift −1 | 7.56 / 7.56 | – | 23 (42–85 s) | clean (2.8×) |

- **The fixed 40 ms delay is wrong for old03** (its raw render was already at offset 0, so the final came out +1 frame late: SyncNet 4.63). Set shift = −(raw offset vs the ORIGINAL dub) per video.
- m03 at 25.3–25.5 s: the auto-pick (#46, a smile) still draws a dark "O"; #84 (3.36 s, more open with upper teeth) shows the teeth. The picker is a candidate generator; a human picks from the contact sheet.
- Files:
  - `full/<v>_keysync_r16_refteeth_b8gate_k1chain{,_pauseB}.mp4` and `full/m03_keysync_r16_ref84_b8gate_k1chain{,_pauseB}.mp4`;
  - `clips/<v>_final_k1chain_refteeth_pauseB_source_*` and `clips/m03_final_k1chain_ref46_ref84_ref84pauseB_source_23-27s_half_speed.mp4`.
- q06: not KeySync-able (KeySync's detector finds no face even in the tracked crop); needs a no-face → skip / LatentSync rule.
