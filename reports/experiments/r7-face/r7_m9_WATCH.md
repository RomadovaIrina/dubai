# r7 m9: WATCH candidate

m9 = m8b + sanitizer fixes (mild guard: 01 7.6 s pop; replace mode: 03 37.4/37.8 s pink lips); m8b = the m8 face stack on audio-r7-v2 (crackle fix, wider lip placement). Both on audio-r7-v2; PREV (left) = m8b. Order to watch: 05, 04, 03, 01, 02. Listen/watch list at the end.

Files: `<id>_r7_m9_WATCH.mp4`, `<id>_r7_m9_WATCH_sbs.mp4` (M8B | M9). Gate verdict per video below (rows FAIL / better).

## 05: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 3.510 / 3.911 **FAIL** |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.630 / 2.170 better |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.029 vs 0.030 px; 0:27.5-0:34.2: 0.041 vs 0.041 px
Lip flash candidates: 7 vs 5.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.17 s (v4-repro) -> 0.63 s; worst: 0:40.5 closed mouth, dub speaks (0.4 s), 0:33.9 talking mouth, dub silent (0.3 s)

## 04: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 5.320 / 6.183 **FAIL** |
| v4 resid mouth | 0.945 / 1.010 better |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.108 vs 0.101 px; 0:39.0-0:47.6: 0.117 vs 0.122 px
Lip flash candidates: 11 vs 9.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.73 s (v4-repro) -> 0.4 s; worst: 0:51.5 closed mouth, dub speaks (0.4 s)

## 03: gate FAIL (2 rows)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 6.471 / 7.861 **FAIL** |
| v4 resid mouth | 1.043 / 1.165 better |
| window joints LF ratio | 0.906 / 0.837 **FAIL** |
| lip flash strong | 32 / 44 better |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.070 vs 0.070 px; 1:00.0-1:08.0: 0.058 vs 0.061 px; 0:21.6-0:37.6: 0.052 vs 0.055 px
Lip flash candidates: 32 vs 44.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.77 s (v4-repro) -> 0.77 s; worst: 1:08.5 closed mouth, dub speaks (0.4 s), 1:01.5 closed mouth, dub speaks (0.4 s)

## 01: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 6.105 / 6.654 **FAIL** |
| v4 resid mouth | 1.040 / 1.137 better |
| window joints LF ratio | 0.853 / 0.923 better |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.084 vs 0.082 px
Lip flash candidates: 3 vs 4.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m9_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 05 | PASS | 1336 (44.53 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 04 | PASS | 58 (1.93 s) | 0.0 | 0 | 0.0 | 1617/1617 |
| 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m9_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 14.78 | 14.46 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 41.73 | 49.73 |
| 04 | - | 36.01 | 53.71 |
| 05 | - | 17.83 | 15.01 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 2.41 | 1.3 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 1.36 | 2.54 |
| 05 | - | 4.34 | 4.1 |

**n_events** (lower = better)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 8 | 8 |
| 02 | - | 0 | 0 |
| 03 | - | 34 | 37 |
| 04 | - | 22 | 28 |
| 05 | - | 14 | 12 |

**n_events_ge2** (lower = better)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 2 | 2 |
| 02 | - | 0 | 0 |
| 03 | - | 1 | 3 |
| 04 | - | 3 | 5 |
| 05 | - | 0 | 0 |

**top20_mean** (lower = better)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 1.847 | 1.808 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.355 | 1.574 |
| 04 | - | 1.696 | 2.25 |
| 05 | - | 1.274 | 1.251 |

**p99** (lower = better)

| video | v4 | m8b | m9 |
|---|---|---|---|
| 01 | - | 0.669 | 0.751 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.068 | 1.126 |
| 04 | - | 1.345 | 1.448 |
| 05 | - | 1.061 | 0.98 |

**top events** (t s: score channel cause head-speed)
- m8b 01: 7.6s 4.3 lips inside_ls h0.98; 1.2s 2.8 lips inside_ls h1.65; 4.4s 1.6 lips inside_ls h1.48; 8.6s 1.5 lips inside_ls h0.45; 0.1s 1.3 skin_motion inside_ls h1.44
- m8b 02: 
- m8b 03: 42.5s 2.2 lips inside_ls h0.4; 48.1s 1.5 lips inside_ls h0.19; 74.7s 1.5 lips inside_ls h0.05; 17.8s 1.5 lips inside_ls h0.08; 77.2s 1.4 lips inside_ls h0.14
- m8b 04: 32.7s 4.2 lips inside_ls h2.52; 9.0s 3.9 lips inside_ls h1.19; 33.3s 2.7 lips inside_ls h2.85; 41.8s 1.7 lips inside_ls h1.26; 36.9s 1.7 lips inside_ls h1.17
- m8b 05: 28.9s 1.6 skin_motion inside_ls h1.15; 24.7s 1.6 skin_motion seg_end h1.68; 5.1s 1.5 lips window_joint h1.34; 14.0s 1.4 lips inside_ls h0.85; 38.8s 1.4 lips inside_ls h1.74
- m9 01: 7.6s 3.9 lips window_joint h1.3; 1.2s 2.8 lips inside_ls h1.65; 4.4s 1.6 lips inside_ls h1.48; 8.6s 1.5 lips inside_ls h0.45; 0.1s 1.3 skin_motion inside_ls h1.44
- m9 02: 
- m9 03: 28.6s 3.4 lips inside_ls h0.05; 55.2s 2.3 lips inside_ls h0.08; 42.5s 2.0 lips inside_ls h0.4; 74.7s 1.9 lips inside_ls h0.05; 77.2s 1.8 lips inside_ls h0.14
- m9 04: 37.8s 6.3 lips inside_ls h1.57; 39.3s 5.7 lips window_joint h2.22; 2.8s 4.5 lips inside_ls h1.99; 9.0s 3.5 lips inside_ls h1.4; 33.3s 2.1 lips inside_ls h2.85
- m9 05: 28.9s 1.6 skin_motion inside_ls h1.15; 5.1s 1.5 lips window_joint h1.34; 24.7s 1.4 skin_motion seg_end h1.68; 38.8s 1.3 lips inside_ls h1.74; 14.0s 1.3 lips inside_ls h0.85

### m9 vs m8b (r7 F) — both on audio-r7-v2
| video | m8b twitch | m9 twitch | m8b skin_sum | m9 skin_sum | SyncNet m8b / m9 | flash strong m8b / m9 | whole-frame gate |
|---|---|---|---|---|---|---|---|
| 05 | 17.83 | 15.01 | 4.34 | 4.10 | 2.61 / 2.35 | 5 / 7 | PASS / PASS |
| 04 | 36.01 | 53.71 | 1.36 | 2.54 | 4.28 / 3.81 | 9 / 11 | PASS / PASS |
| 03 | 41.73 | 49.73 | 0 | 0 | 6.62 / 5.48 | 44 / 32 | PASS / PASS |
| 01 | 14.78 | 14.46 | 2.41 | 1.30 | 6.82 / 6.27 | 4 / 3 | PASS / PASS |
| 02 | 0 | 0 | 0 | 0 | - | - | PASS / PASS |

**m8b** = the m8 face stack (m6 guards + Real-ESRGAN temporal lips + full-confidence source skin texture, texture-mode sanitizer) on the
audio agent's **audio-r7-v2** (crackle fix; wider lip placement: on-screen dub silence 05 7.7 → 4.3 s, 04 6.4 → 5.25 s). LatentSync re-run
on v2 for all 5.
**m9** = m8b + two sanitizer changes: (a) `--sanitize-mild 0.6 --sanitize-hf-ncc 0.3` — a flagged frame that still has its own lip line is
not given a misaligned neighbour's lip line (the 01 7.6 s flat→bulge pop); (b) `--sanitize-mode replace` — a flagged (texture-collapsed)
frame becomes the neighbours' flow interpolation whole, not only its high frequencies (the 03 37.4 / 37.8 s washed-pink lips kept their
colour in texture mode). Cost of (b): SyncNet −0.3…−1.1 on every video (interpolated frames carry less lip motion) and twitch +18 on 04 /
+8 on 03 (lips channel). The user judges visually; m8b is the fallback.

**Listen / watch list (m8b | m9 sbs, left = m8b):**
- 05 24.0–25.5 s, 04 1.5–5.5 s / 28–29 s / 51–52 s: the dub-silent-while-the-source-talks spans shortened by audio v2 (both sides; the
  remaining closed-mouth look is LatentSync's rendering of silence — the user chose longer dub takes, coming as audio-r7-v3).
- 03 37.4 s and 37.8 s: washed-pink lower lip (m8b) → clean lips (m9).
- 01 7.6 s: flat lip line → bulge pop (m8b) → the generator's own 2-frame bilabial (m9).
- 04 38.2 s: the earlier crack (fixed since m8, both sides clean).
- Whole videos: crackles gone (audio v2) on both sides.

Pressed lips / black mouth measurements (STATUS_LOG §8): closed-mouth height 0.99 and lip thickness 0.98 of the source's resting mouth,
mouth width 0.605 vs 0.611 IOD (no pursing), lip colour equal; inner-mouth luma m8 ≥ m6 ≥ source on 03/04/05 — neither is changed by the
face stack, so nothing in m9 targets them.
