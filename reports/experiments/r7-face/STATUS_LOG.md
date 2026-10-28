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

## 8. User review of m8: pressed lips (05 24–25 s), black mouth, 03 37 s, 03 amplitude — measurements and arms

**Pressed / "glued" lips where the dub is silent and the source talks.** The raw LatentSync layer already has it (sheet `artifacts/r7f/mouth_probe/05_24s_layer_raw.jpg`), m6 too: it is the generator's rendering of silence, not the sanitizer or a hold path. Tried at the source: −42 dBFS pink room tone in the silent spans of the conditioning wav (05 re-run, `m9rt`) → the layer is unchanged (closed fraction 0.53 → 0.58, same look). Tried in the composite: `--mouth-region-closed 1.2` (the generator's closed jaw pasted with its lips while the source holds the mouth open, ramped; m9a on 05/04) → no visible change, twitch 05 16.6 → 18.0, 04 39.6 → 46.0. Measured (`scripts/r7/r7_mouth_probe.py`, output landmarks, dub-silent closed frames where the source talks): mouth height 0.99 and lip thickness 0.98 of the source's own resting closed mouth, lip-line dip depth 0.56 vs 0.62 (NOT a darker crease), lip L*/a*/saturation equal to the source lips (05: a 145.1 vs 145.3, S 113 vs 100). Geometrically and in colour the closed mouth IS a resting mouth; the "about to throw up" impression is the talking source face (cheeks, folds, jaw muscles) around a mouth that is shut. Nothing in the composite can relax that; the remedies are shorter spans (audio agent) or pass-through there (the mouth talks while the dub is silent, the m3 rule reversed) — a user decision.

**"Black mouth inside" (user: new in m8).** Inner-lip-polygon luma on open-mouth frames (each variant's own landmarks): 05 src 56 / m5 60 / m6 65 / m7 60 / m8 62; 04 47 / 49 / 49 / 53 / 54; 03 69 / – / – / 68 / 68; 01 45 / 38 / 38 / 38 / 41. Teeth fraction equal (≤ 1–6 %). m8 is not darker than m6 or the source on any video; the dark toothless cavity is LatentSync's and is in m5 already (sheets `artifacts/r7f/mouth_probe/<id>_open_mouth_sheet.jpg`). No fix applied.

**03 37–38 s.** Generator pink-lip collapse on layer frames 934–936 (37.4 s) and 946–948 (37.8 s); the sanitizer (texture mode: own LF + donor HF) keeps the washed colour. Arms on 03: m9b `--sanitize-colour on` → still washed, SyncNet 6.65 → 6.26; m9c `--sanitize-mode replace` → both spots clean (sheet `03_37s_m8_vs_m9c.jpg`), flash-strong 57 → 45, twitch 49.65 → 46.89, SyncNet 6.65 → 5.52; m9d hybrid (replace only below ratio 0.7) → worse (52.1 / 58 / 5.77).

**03 amplitude.** Mouth opening / IOD on dub-speech frames vs the source over its whole face track: p50 0.074 vs 0.055, p90 0.107 vs 0.106, p98 0.124 vs 0.127 — m8's 03 opens as much as the source does; the "barely moves" impression is not amplitude.

m9 (built, pending the user's call) = m8 + `--sanitize-mild 0.6 --sanitize-hf-ncc 0.3` (01 7.6 s pop) + `--sanitize-mode replace` (03 37 s); costs: 01 twitch 14.2 → 25.1 (lips channel on the sustained pursing), 03 SyncNet −1.1.

## 9. m8b published, m10 (restricted replace mode) fails the gate, audio v3a / v3b / v3c on the face (05, 04)

**m8b WATCH** (outbox `<id>_r7_m8b_WATCH.mp4` + `_sbs.mp4`, left m8 | right m8b, order 05 04 03 01 02; `r7_m8b_WATCH.md`): the m8 face stack
unchanged on audio-r7-v2, LatentSync re-run on v2 for all 5. Gate vs m8: PASS 04 03 01 02; 05 FAIL on one row only, mouth / dub mismatch
1.47 -> 2.17 s (49.2 s talking mouth on a dub pause, 40.5 s closed mouth on dub speech, 0.3-0.4 s each; the generator on the new placement).
Twitch m8 -> m8b: 05 16.6 -> 17.8, 04 39.6 -> 36.0, 03 49.7 -> 41.7, 01 14.2 -> 14.8; SyncNet dub-speech equal within 0.1. Whole-frame gate PASS.

**m10 = m8b + replace mode restricted to colour-collapsed frames.** New flag `--sanitize-replace-luma Y` (with `--sanitize-replace-below X`):
a flagged frame is replaced by the whole flow interpolation only when its collapse ratio < X AND its lip luma > Y x the neighbours' median
(the washed-pink blob is BRIGHTER: 03 v2 layer frames 931-933 / 942-943 ratio 0.34-0.46, luma x1.3-1.5; closures and blur are not). With
X 0.5, Y 1.25 it selects 60 of 364 flagged frames on 03 (sheet `artifacts/r7f/mouth_probe/colour_collapsed_sel.jpg`: all washed blobs),
11 on 04, 4 on 05, 0 on 01. Result (`gate_m10_vs_m8b.md`): 03 SyncNet dub-speech 7.86 -> 7.30 FAIL, twitch 41.7 -> 47.4, flash 44 -> 47;
04 flash 9 -> 15 FAIL, twitch 36.0 -> 41.6; 05 twitch 17.8 -> 19.9; 01 unchanged. Even 60 whole-interpolated frames (2.7 % of 03) cost
SyncNet, because the interpolation removes the lip motion of a frame that is still articulating under the washed colour. Per the
coordinator's rule (restrict it or keep m8): **m8 / m8b stays the default stack; replace mode stays default-off** (flags kept, documented).

**Audio v3 variants on the face** (05 and 04, the m8 stack; `m8rawv3{a,b,c}` generator runs 416-470 s per video = 340-380 s/video-min,
composites `m8v3{a,b,c}`): outbox `<id>_r7_audio_v3_variants_WATCH.mp4` (a row of four, v2 | v3a | v3b | v3c, labelled, four titled audio
tracks) + `r7_audio_v3_variants_WATCH.md` (tables). New metric `scripts/r7/r7_closed_mouth.py`: closed output mouth (< 0.02 IOD) while
the source talks (> 0.05 IOD), total and dub-silent subset. v2 -> v3a / v3b / v3c: 05 8.7 -> 6.9 / 7.7 / 5.9 s dub-silent closed,
04 4.7 -> 3.1 / 2.2 / 2.6 s; SyncNet 05 3.91 -> 3.58 / 4.33 / 3.72, 04 6.18 -> 6.42 / 6.11 / 6.28; twitch 05 17.8 -> 15.6 / 9.8 / 12.5,
04 36.0 -> 38.8 / 37.9 / 41.2; flash 05 5 -> 6 / 6 / 8, 04 9 -> 16 / 11 / 12. All three reduce the pressed-lips seconds; v3c best on 05,
v3b / v3c on 04; the longer takes add "open mouth while the dub is silent" (internal pauses, 05 5.0 -> 6.5-8.0 s). User's eye decides.

## 10. m11 = m8 stack + audio-r7-v3a on all 5 (the user's pick), WATCH sbs m8b | m11

05 / 04 = the m8v3a renders; 03 / 01 / 02 = the m8b renders (v3a audio bit-identical to v2 there, verified by cmp). Gate vs m8b
(`gate_m11_vs_m8b.md`): 03 / 01 / 02 PASS (identical); 05 FAIL on SyncNet dub-speech 3.91 -> 3.58, window joints 0.89 -> 0.95 / LF 0.84 -> 0.90
and the speed row (500 vs 421 s/video-min, still under 600); 04 FAIL on lip flash strong 9 -> 16. Twitch 05 17.8 -> 15.6, 04 36.0 -> 38.8;
whole-frame gate PASS all 5. These are the generator's response to the re-taken units (05 units 2-4, 04 unit 2), not a composite change.
Outbox: `<id>_r7_m11_WATCH.mp4` + `_sbs.mp4` (m8b | m11), `r7_m11_WATCH.md`, `r7_m11_twitch/`, `r7_m11_frame_gate/`.
**02 update (audio-r7-v3a-all5):** the all-5 handoff differs from v2 only on 02 (one re-taken sentence; 01/03 identical to v2, 04/05 identical
to v3a — cmp). 02 has no face frames, so its m11 output is the source + the new mix (r7_e2e, 90 s, no generator); WATCH + sbs regenerated
for all 5, gate rows unchanged (02 PASS, no sync / flash).

## 11. m11 = the accepted candidate (user: 04 "excellent", 05 no longer "vomiting"; pause placement forgiven for now)

**What m11 is.** Audio: audio-r7-v3a (r7 audio v2 = r6 takes + lip placement 2.0 / 1.5 / 0.5 s + gap clean + dub gate, plus duration-aware
re-translation / re-take of the units where the speaker is on screen and the dub ended early; 02 on v3a-all5). Face: the m8 stack =
LatentSync 1.6 at 512 / 25 fps on the dub, `--ls-split-dub-silence 1.0`, then `scripts/r7/r7_layer_sr.py` (Real-ESRGAN x2 on the layer's lips,
temporal ±1) and `fps_composite.py` at the source fps with: sim_ring registration (upper ring), edge 12 px, colour match upper, source skin
texture at full confidence (`--tex-tau 1000 --tex-sigma 2.5`), lip guard, flow_safe retime, face clip, texture-mode sanitizer 0.83 with the
alignment guard and the rigid fallback (`--sanitize-min-conf 0.3 --sanitize-tear 0`), sharpen-auto 0.95 / lips max 0.8, mouth region
0.38, paste guard 0.5. Default-off and unused: `--sanitize-mild`, `--sanitize-replace-below/-luma`, `--mouth-region-closed`.
Known leftovers: 03 37.3 / 37.7 s washed-pink lips (generator; every fix costs SyncNet), 01 7.6 s one-frame lip pop, the generator's open
mouth on dub-internal pauses (05 7.5 s, 04 4.1 s), the pressed look on the remaining dub-silent spans (05 6.9 s, 04 3.1 s).

**Speed on m11 (s per video-minute, face stage only; this server: one GPU, 24 cores).** Wall per stage from the manifests / score.json:

| video (face frames) | LatentSync 25 fps (gen + load + prep) | layer SR (Real-ESRGAN) | composite (fps_composite + fix30 wrapper) | mediapipe (source + layer) | face stage total |
|---|---:|---:|---:|---:|---:|
| 05 (1511 / 1567) | 373 | ~80 | ~45 | 31 | 500 |
| 04 (1617 / 1617) | 374 | ~95 | ~80 | 48 | 482 |
| 03 (2710 / 2710) | 372 | ~75 | ~50 | 26 | 480 |
| 01 (260 / 2542) | 65 | ~15 | ~18 | 12 | 95 |
| 02 (0 / 3324) | 22 (no generator) | 0 | 11 | 10 | 33 |

(The SR / composite split is from the 05 run: 70 s of the 108 s "layer SR + composite" is Real-ESRGAN; fps_composite itself is ~33 s =
sanitizer 8.6, composite 9.2, encode 5.8, boundaries 4.3; the fix30 wrapper adds ~40-60 s per video for the frame map, landmarks and
checks.) The budget is 600 s/video-min: a full-face video sits at ~480-500, i.e. ~100 s of headroom; the audio stage (~140 s/video-min
with 14B + retake, audio agent) runs on the other server.

**Top-3 speed costs for phase 2:**
1. **LatentSync itself: ~370 s/video-min, 75 % of the face stage.** Fixed per generated frame (≈ 0.25 s/frame at 512, batch 1); the only
   levers are fewer generated frames (the dub-silence split already skips the silent stretches: 01 pays 65 for 260 frames) and the
   generator's own batching / fp16 / torch.compile, untried in r7.
2. **Real-ESRGAN layer SR with temporal ±1: ~75-95 s/video-min (16-20 %).** Three passes per frame (temporal ±1); a single pass would be
   ~30, and the lips-only crop is already used. Can run concurrently with the generator of the next video (CPU/GPU overlap, not done).
3. **The composite wrapper (fix30 + fps_composite): ~45-80 s/video-min (10-16 %)**, dominated by the per-frame landmarks / frame map
   (fix30 40-60 s) rather than fps_composite (33 s on 05, of which the sanitizer 8.6 and encode 5.8). Source mediapipe (face_stack 20-30)
   is computed again by fix30; caching the source landmarks (they exist in artifacts/r7f/lm) saves ~20 s/video-min.
Everything else (speech gate, prep, assembly, score) is < 15 s/video-min together.

## 12. 05 on audio-r7-v3a1, the final set, the quality baseline tag

audio-r7-v3a1 differs from v3a only on 05 and only in the dub track at 31.62-31.98 s (the junk ducked; 01-04 bit-identical, cmp). The old
05 already had the mouth shut there (opening 0.004-0.015 IOD at 31.6-32.0 s), so the change is the mix; 05 was nevertheless re-rendered
end to end on v3a1 (generator 363 s + layer SR + composite 91 s + score) so that the candidate is exactly reproducible from its handoff:
twitch 15.6 -> 16.4, SyncNet dub-speech 3.58 -> 3.47, flash 6 -> 6, joints 0.953 -> 0.945, whole-frame PASS; within the baseline
tolerances. Old 05 kept in `artifacts/r4m/m11/prev_05_v3a/`.
Final set: `outbox/final_r7/<id>_DabAI_r7.mp4` (24-56 MB) + `<id>_DabAI_r7_phone.mp4` (23-24 MB) + README.md (`scripts/r7/r7_final_set.sh`).
Quality baseline for phase 2: `reports/experiments/r7-face/quality_baseline.json` (frozen from m11 with 05 on v3a1; `r7_baseline.py freeze`),
gate = `r7_baseline.py gate --tag <arm> --audio <handoff>`: a metric fails when worse than the baseline beyond its tolerance (twitch +15 %,
skin +1, SyncNet -0.3, flash +3, whole-frame worst <= 0.2 % / 0 frames over, pressed lips +0.5 s, mismatch +0.5 s, joints +0.05, mouth
sharpness -0.03, chin flicker / jaw jitter +0.1, ring colour +1 dL, spec pass, speed <= 600 s/video-min) — plus the user WATCH. Git tag
`r7-quality-baseline` = this commit.
