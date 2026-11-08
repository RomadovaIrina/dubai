# r16 Q: metric trust table v2 (08.11)

What each metric can be trusted for after the step-0 audit fixes. Golden numbers: `golden/<id>.eval.json`.

## Golden baseline (eval_final.sh, speaker track, full length, original dub)
| clip | speech@0 | noise ± | seconds scored | av offset | note |
|---|---|---|---|---|---|
| m03 | 8.676 | 0.051 | 34 | 0 | route: child excluded (child conf_at0 −0.10) |
| m18 | 7.927 | 0.009 | 18 | 0 | CPU − GPU = 0.006 |
| old03 | 7.551 | 0.033 | 53 | 0 | |
| q02 | 1.805 | 0.075 | 3 | (−7 whole clip) | only 3 on-screen speech seconds; 23 off-screen narrator seconds excluded |
Suggested pass rule for the golden test: e2e speech@0 ≥ golden − max(0.15, 3 × noise), with status OK.

## Trust table
| metric | v1 status (audit) | v2 status | use it for |
|---|---|---|---|
| **eval_final.sh speech@0** (full length, pasted final, speaker track, original dub, on-screen seconds per route) | not existing | **USABLE.** Noise ±0.01–0.05 on 18–53 s clips (re-encode / 8 px crop); CPU = GPU (Δ 0.006); face-lost check | A/B of finals, golden test. A difference < 2 × noise is not real. Not for < ~10 scored seconds (q02: 3 s, ±0.075) |
| sync_perframe global conf | multi-face bug (the mean over tracks) | fixed (speaker tracks), but whole-clip / best offset | info only. eval_final's speech@0 is the decision number |
| m03 numbers before 726427a | understated ~2× (child track) | superseded | do not compare old m03 SyncNet with new |
| L's l_agg / l_eval sync (best offset, raw 512 crop, 6 s windows) | not comparable | unchanged (L's) | trends inside L's runs only; ±0.3–0.5 noise per window |
| k_gate.sh speech@0 | drops face-less seconds, multi-face bug | replaced by eval_final.sh | — |
| r16q_detect black | absolute (flagged the source's own dark mouths) | relative to the source (q02 119 → 53 frames) | spot finding; thresholds still from 3 videos, no false-positive rate yet |
| r16q_detect explode | no source reference | snaps the source also has are removed (q02 16 → 9) | spot finding |
| r16q_detect cache | stale det.npz reused | keyed by both videos' fingerprint + t0/t1 | — |
| r16q_detect slide | included the chin (moves with the jaw) | 08.11: upper + cheeks only; the chin is its own slide_ch. old03 e2e +255 slide frames were ALL chin = a wider jaw opening | spot finding |
| r16q_detect texture / num_faces / edge | audit #11 open | unchanged | spot finding only; texture flags low-light noise |
| **r16q_spring.py** (K1 spring3 + Q floor / CI) | not existing | **USABLE for KeySync vs LatentSync / vs golden on one clip** (beards: KeySync 1.8x floor, CI clear); NOT for ranking clips against each other; see SPRING.md | the beard bounce |
| k_lag_final (eye anchors) | ~0 by design | r16q_klag.py: lower-face contour (+ eye anchors as a ~0 sanity check). Untested | after one validation run |
| k2_lipmetrics share_12.5Hz | abs() before the FFT → blind | r16q_k2lip.py: signed series, longest valid run. Untested | after one validation run |
| l_boundary ratio | argmax phase → biased up | fix proposed to L (fixed phase 13k); L's file | after L applies it |
| beard score | 2.0 threshold, thin margin | unchanged | routing default; a low-light guard is still missing |
| SyncNet in general | rewards a dark open mouth | unchanged | never the only judge of black-hole fixes; Irina's eye decides |

## Detectors on the eye-ACCEPTED golden finals (= the false-positive floor), golden/det_new_<id>.json
| clip | black abs → rel | black share ours / src | explode (shape abs → rel, px) | slide | texture | edge | flicker | swap |
|---|---|---|---|---|---|---|---|---|
| old03 | 15 → 15 | 0.036 / 0.002 | 2 → 2, 3 | 111 | 736 | 14 | 254 | 0 |
| m03 | 42 → 40 | 0.104 / 0.027 | 3 → 3, 0 | 40 | 0 | 22 | 17 | 5 |
| q02 (KeySync, day 1) | 119 → 53 | — / 0.719 | 16 → 9 | | | | | |

Reading:
- The source-relative change removes flags only where the source itself is dark or snapping (q02). On the golden clips the sources are bright, so black is nearly unchanged.
- The golden finals still raise black 15–40 frames, slide 40–111 frames, texture 736 (old03 low light) and flicker 254 (old03). These are false positives on finals Irina accepted.
- So a detector count is only meaningful RELATIVE to the golden one for the same clip. Never use an absolute "0 flags" target.
- texture / flicker on old03 are not usable as gates.
