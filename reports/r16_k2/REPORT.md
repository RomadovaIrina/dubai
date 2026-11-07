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
