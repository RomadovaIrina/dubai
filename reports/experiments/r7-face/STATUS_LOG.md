# r7 F (face + merged E2E): status log

## 1. 05 25-26 s "rectangle with the face pasted over the trees" = the SOURCE's own picture-in-picture inset (28.10, 09:30)
Test video 05 is a phone dual-camera recording: the main camera films the park, a selfie inset with the speaker (and the inset's own
background) sits in the top-right corner for the WHOLE clip (MediaPipe: face in 1511/1567 frames, always at x 330-396 of 464).
The published m5 05 WATCH differs from the source only at the mouth inside the inset: whole-frame gate PASS, worst frame 0.000 % of the
frame changed outside the mouth region. Evidence for the user: `outbox/r7_05_pip_check/05_src_vs_m5_24.5-26.5s_wholeframe.jpg` (SRC | M5 | diff,
11 frames) and `05_src_vs_m5_24-27s_sbs.mp4`.
Generic guards kept anyway (the new test video's block 9 pans away from the face):
- `fps_composite.py --paste-guard 0.5`: the layer is pasted only when the CURRENT source frame holds the face at the paste position (grey NCC
  source vs registered + colour-matched layer on the paste region outside the lips >= 0.5), otherwise pass-through (counted); `--max-hold 2`.
- `scripts/r7/r7_frame_gate.py` / `.sh`: per frame, % of the frame changed (|out - src| > 24 grey) outside the source lips hull + 0.6 IOD;
  PASS = no frame > 0.2 %. Top-12 whole-frame sheets. Part of `r7_e2e.sh`.

### Whole-frame gate on the published versions (all 5 videos; 01 = small distant face, 02 = no face at all)
| version | id | verdict | worst frame | worst % | frames > 0.2 % | face frames |
|---|---|---|---|---|---|---|
| m2 | 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 260/2542 |
| m2 | 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0/3324 |
| m2 | 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 2710/2710 |
| m2 | 04 | PASS | 111 (3.7 s) | 0.001 | 0 | 1617/1617 |
| m2 | 05 | PASS | 1204 (40.13 s) | 0.0 | 0 | 1511/1567 |
| m3 | 03 | PASS | 2639 (87.97 s) | 0.0 | 0 | 2710/2710 |
| m3 | 04 | PASS | 111 (3.7 s) | 0.001 | 0 | 1617/1617 |
| m3 | 05 | PASS | 1197 (39.9 s) | 0.0 | 0 | 1511/1567 |
| m5 | 05 | PASS | 1502 (50.07 s) | 0.0 | 0 | 1511/1567 |
| r4_video_v1 | 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 260/2542 |
| r4_video_v1 | 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0/3324 |
| r4_video_v1 | 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 2710/2710 |
| r4_video_v1 | 04 | PASS | 164 (5.47 s) | 0.001 | 0 | 1617/1617 |
| r4_video_v1 | 05 | PASS | 1309 (43.63 s) | 0.0 | 0 | 1511/1567 |

## 2. 03 ~7 s seam / blur: metric `scripts/r7/r7_seam.py` (baseline m3 03)
seam (|grad out| - |grad src| on the footprint boundary band) mean -0.35 p95 0.32; colour step mean 2.97 p95 3.84;
skin blur (|Laplacian| out / src inside the footprint, outside the lips) mean 0.721 p05 0.576 -> the regenerated skin is ~28 % softer than
the source everywhere, worst frames ~0.47 (15.6 s, 34.1 s, 62.8 s). On the fast frames (head > 0.3 IOD/s, 77 fr): blur 0.707, seam -0.41.
At 7 s itself (frames 195-245) blur 0.6-0.8, no seam spike: the user's "seam" is most likely the softness step at the footprint edge.

## 3. Paste guard proven on a synthetic "camera looks away" (28.10, 10:40)
`scripts/r7/r7_guard_synth.sh 03 m6 600 720`: the source's lower face is covered by a background patch from the same frame for frames
600-719 while the landmarks / layer still say "face" (the only way to exercise the guard: a real face loss already passes through as NaN
landmarks). Changed pixels per covered frame (|out - src| > 24): **without the guard ~17 000 px** (a face pasted on the trees = the user's
05 report), **with `--paste-guard 0.5` 0 px** (frames 600-710), the frames outside the window untouched. On the real videos the guard flags
0 frames on 03 / 01 / 02 and 1 on 05 (frame 705: the tiny inset face under motion blur, NCC 0.49, a false positive) -> `--guard-min-run 3`:
pass-through only inside runs of >= 3 flagged frames; 05 keeps its paste there. m6 05 twitch 20.0 vs m5 19.0 (generator non-determinism,
not the guard: the guard changed nothing on 05).
Gate blind spot noted: `r7_frame_gate.py` excludes the mouth region from the SOURCE landmarks, so on this synthetic cover (landmarks still
present) it cannot see the paste; on a real pan the landmarks are NaN and the whole frame counts. Fine for the pipeline, not for this test:
the test measures the raw changed-pixel count instead.

## 4. The soapy lower face: where the softness is (28.10, 10:40)
`scripts/r7/r7_sharp_probe.py` (|Laplacian| out / src on the changed pixels, by distance from the lips in mouth widths; control = untouched
lower-face pixels): m6 03 lips 0.735, 0-0.15 mw 0.775, 0.15-0.3 0.759, 0.3-0.5 0.795, control 1.000 (the encode loses nothing).
The raw LatentSync layer itself: lips 0.56 (its own sharpen chain brings it to 0.73). LatentSync 1.6 already runs at 512 with a 420x560 crop,
so 03's ~300 px face is NOT downscaled: the softness is the diffusion VAE's output. Composite-only arms on 03 (r7_comp_arm.sh; twitch m5 = 47.65):
| arm | lips | 0-0.15 | 0.15-0.3 | 0.3-0.5 | seam blur mean | twitch |
|---|---|---|---|---|---|---|
| m6 | 0.735 | 0.775 | 0.759 | 0.795 | 0.758 | 47.64 |
| --lip-texture on | 0.741 | 0.778 | 0.754 | 0.760 | 0.756 | 48.67 |
| --lip-dilate 0.06 | 0.729 | 0.782 | 0.778 | 0.880 | 0.758 | 48.02 |
| --sharpen-lips-max 1.2 | 0.78 | 0.83 | 0.74 | 0.77 | 0.756 | 46.27 |
| --mouth-region 0.30 | 0.73 | 0.77 | 0.77 | 0.76 | 0.757 | 47.71 |
| --tex-tau 30 | 0.72 | 0.786 | 0.764 | 0.808 | 0.789 | 49.51 |
| --tex-tau 1000 | 0.73 | 0.80 | 0.84 | 0.85 | 0.819 | 52.05 |
| --skin-texture off | - | - | - | - | 0.693 | 47.72 |
| **--tex-tau 1000 --tex-sigma 2.5** | - | - | - | - | **0.843** | **45.72** |
The lips never move with composite flags (0.72-0.78): the detail has to come back on the layer -> `r7_layer_sr.py` (Real-ESRGAN x2, BSD-3,
models/realesrgan/RealESRGAN_x2plus.pth 64 MB, sha256 49fafd45...266abb; RRDBNet re-implemented in plain torch, `weights_only` load, no
facexlib / basicsr / detector) on the layer's lips region (hull + 0.5 mw, feathered); 2258 frames in ~2 min on the 5090. Arms `sr`, `sr_tex` pending.

### ESRGAN-on-the-layer arms (03, 28.10 11:00; sharpness bands lips / 0-0.15 / 0.15-0.3 / 0.3-0.5 mw; twitch m5 47.65)
| arm | lips | 0-0.15 | 0.15-0.3 | 0.3-0.5 | twitch | note |
|---|---|---|---|---|---|---|
| m6 | 0.735 | 0.775 | 0.759 | 0.795 | 47.64 | |
| sr (ESRGAN layer) | 0.771 | 0.761 | 0.684 | 0.771 | 48.38 | the sharpen-auto loop renormalises most of the gain away |
| sr_tex (+ --tex-tau 1000 --tex-sigma 2.5) | 0.764 | 0.772 | 0.813 | 0.895 | 51.17 | visibly crisper lips + stubble (closeups) |
| noauto (--sharpen-auto 0) | 0.587 | 0.683 | 0.730 | 0.749 | 50.05 | the raw layer after the composite, no unsharp |
| sr_noauto | 0.673 | 0.709 | 0.660 | 0.744 | 54.73 | ESRGAN alone: +0.09 on the lips, but per-frame detail flickers (twitch +4.7) |
| sr_tex_noauto | 0.675 | 0.708 | 0.833 | 0.822 | 51.75 | |
| tauinf_s25 (no SR) | 0.723 | 0.781 | 0.897 | 0.920 | 45.72 | skin ring from the source texture: twitch BETTER than m5 |
Reading: the source-texture transfer at tau 1000 / sigma 2.5 is a clean win for the skin ring (keep); ESRGAN gives the lips +0.04-0.09 but
flickers frame to frame -> next: temporally stabilise the SR detail (flow-aligned average of the added high frequencies over +-1 layer frame)
or lower its strength, judged on twitch + a half-speed review.

### SR arms, continued (11:30)
| arm | lips | 0.15-0.3 | 0.3-0.5 | twitch | |
|---|---|---|---|---|---|
| sr06_tex (ESRGAN strength 0.6 + tex) | 0.758 | 0.864 | 0.855 | 48.99 | |
| srt1_tex (ESRGAN temporal +-1 + tex) | **0.809** | 0.801 | 0.872 | 55.79 | best lips; the twitch rise is the METRIC: its top events are the SAME dub articulations as m6's (37.8 / 42.5 / 47.8 s, same scores +-0.1), plus the mouth opening at 5.83 s (frames 174-176, present in m6 at 1.0, scored 3.2 here) - the lips channel measures flow magnitude on the lips, and crisper lips give larger flow. No new visual defect in the 9-frame strip. |
Consequence: for sharpness arms the twitch acceptance must use the skin channels (skin_motion / skin_app = the paste jumping) and the
half-speed review, not the lips channel. r5_twitch.py gets a per-channel sum in the table.

## 5. m6 WATCH published (11:00) and m7 candidate (11:40)
m6 WATCH: outbox/<id>_r7_m6_WATCH*.mp4 + r7_m6_WATCH.md; gate vs m5 PASS on all 5, whole-frame gate PASS, twitch = m5 (skin sums identical).
m7 = m6 layers + `r7_layer_sr.py --temporal 1` (ESRGAN on the layer's lips, detail averaged over +-1 frame) + `--tex-tau 1000 --tex-sigma 2.5`,
`scripts/r7/r7_m7_comp.sh` (no generator rerun; +18-80 s/video-min). All 5: whole-frame gate PASS.
| video | lips m6 -> m7 | ring 0.15-0.3 m6 -> m7 | twitch m5 / m6 / m7 | skin_sum m5 / m6 / m7 |
|---|---|---|---|---|
| 03 | 0.735 -> 0.809 | 0.759 -> 0.801 | 47.65 / 47.64 / 51.77 | 0 / 0 / 0 |
| 01 | 1.06 -> 1.02 (tiny face, already crisp) | 0.84 -> 0.77 | 12.12 / 12.12 / 9.59 | 1.24 / 1.24 / 1.28 |
| 05 | 0.726 -> 0.921 | 0.809 -> 0.892 | 18.99 / 20.01 / 20.05 | 5.56 / 5.56 / 4.58 |
| 04 | 0.701 -> 0.770 | 0.917 -> 0.980 | 50.49 / 51.51 / 53.85 | 0 / 0 / 0 |
The lips-channel twitch rise on 03 / 04 is the same dub articulations scored higher because crisper lips give larger flow (section 4);
the skin channels (the paste jumping = the user's "twitch") are equal or better everywhere. Human check: the half-speed top-20 clips in
outbox/r7_m7_twitch/.

## 6. m8 (m7 stack + audio-r7-v1) and a cracked-lip defect found and fixed (12:25)
m8raw = LatentSync on the audio agent's audio-r7-v1 (all 5, 54-490 s/video-min), then the m7 look (r7_m7_comp.sh). First pass: 05 twitch
10.5 (m7 20.1), flashes 7 (24); 03 49.4; 01 14.1; 04 62.0 with 10 new lips events. The 04 review strip at 31.9 s showed a REAL defect:
frame 956 has a 'cracked' lower lip (thin bright lines). Window A/Bs (fps_composite --only-frames 945:970): the crack is in the sanitizer's
texture fix (gone only with --sanitize-layer 0; present with tau 12, without SR, without unsharp). Cause: for every flagged frame on 04 the
neighbours' flow interpolation has confidence 0 (frames 2-4 apart differ in the mouth), so `interp_safe` falls back to "the nearer frame
warped by that (wrong) flow", and the warp tears the lip line into cracks; the r6 ring guard cannot see it (the ring is aligned).
Fix (9b8f594, generic): at interpolation confidence < --sanitize-min-conf (0.3) the fix frame is the nearer good frame moved by a SIMILARITY
fitted on the lips + eyes / nose landmarks (a similarity cannot tear); a per-pixel blend of dense / rigid (--sanitize-tear 16) doubled the
lip line on 948 / 954, so the default is rigid everywhere (--sanitize-tear 0). On 04 all 124 sanitized frames take this path; the real
flash at 985 is still fixed. m8 is being re-composited on all 5 with it; m7 (published) still carries the old sanitizer.

## 7. "Lip snap" class (01 7.6 s / 04 39.4 s): what it is, the m9 arm, verdict

Measured on the m8 raw LatentSync layers (all 4 videos with LS frames): pixel single-frame outliers on the lips crop (min |f_k - f_k±1| / |f_k-1 - f_k+1| >= 2): **0**; lips-shape second difference on the layer landmarks p99.9 < 0.04 IOD everywhere, no spike at the two frames. There is no single-frame snap on the layers. The 01 7.6 s event is LatentSync's own **2-frame pursed bilabial** (layer frames 189–190, mouth opening 0.003 IOD) — and the sanitizer flagged 189 as a "texture collapse" (collapse ratio 0.77 vs the 0.83 threshold) and gave it frame 188's lip line (texture mode = own LF + donor HF) while 190 kept its bulge: a flat lip → bulge pop in the output. The donor's HF correlates −0.11 with the frame's own HF on the lip box; the same pattern at the 0.3 s bilabial (frames 7–8) and on 04 f985 (ncc 0.28).

Generic guard added to `fps_composite.py`: `--sanitize-mild R --sanitize-hf-ncc C` — a flagged frame whose collapse ratio ≥ R (its own lip line exists) is fixed only when the donated HF correlates ≥ C with its own on the lip box; else the frame is kept as drawn. Default off (m8 unchanged). m9 = m8 + `--sanitize-mild 0.6 --sanitize-hf-ncc 0.3`: kept 9/44 (01), 0/359 (03), 4/132 (05), 7/147 (04) sanitized frames.

| video | twitch m8 → m9 | skin_sum | SyncNet | flash strong | whole-frame gate |
|---|---|---|---|---|---|
| 03 | 49.65 → 49.65 | 0 → 0 | 6.65 → 6.60 | 57 → 51 | PASS |
| 01 | 14.24 → **25.06** | 2.15 → 0 | 6.82 → 6.82 | 2 → 4 | PASS |
| 05 | 16.55 → 15.06 | 2.88 → 2.88 | 2.63 → 2.76 | 7 → 7 | PASS |
| 02 | 0 → 0 | - | - | - | PASS |
| 04 | 39.62 → 41.46 | 1.41 → 1.46 | 4.15 → 4.14 | 17 → 17 | PASS |

Verdict: the pop is gone on 01 (sheet `artifacts/r7f/crack956/01_m8_vs_m9_bilabials.jpg`: m9 shows the generator's pursed lips for 2–3 frames instead of flat→bulge), skin_sum 2.15 → 0, but the lips channel scores the sustained pursing against the open source mouth as four new events (0.4, 0.8, 7.6, 9.6 s: 25.06 vs 14.24) and flash 2 → 4. Under the rule "twitch at or below the previous candidate" m9 does NOT pass on 01; it is neutral-to-better on 03/05, slightly worse on 04 (+1.8). The pursed look itself is LatentSync's rendition of p/b/m on this audio; a composite cannot make it a natural closure. Not published; m8 stays the candidate. Guard committed, default off.
