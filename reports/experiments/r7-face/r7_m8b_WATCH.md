# r7 m8b: WATCH candidate

m8b = the m8 face stack (unchanged) on the audio agent's audio-r7-v2 (crackle fix, wider lip placement); LatentSync re-run on v2 for all 5. Left m8 (audio v1) | right m8b (audio v2).

Files: `<id>_r7_m8b_WATCH.mp4`, `<id>_r7_m8b_WATCH_sbs.mp4` (M8 | M8B). Gate verdict per video below (rows FAIL / better).

## 05: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.170 / 1.470 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.030 vs 0.031 px; 0:27.5-0:34.2: 0.041 vs 0.039 px
Lip flash candidates: 5 vs 7.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.47 s (v4-repro) -> 2.17 s; worst: 0:49.2 talking mouth, dub silent (0.4 s), 0:40.5 closed mouth, dub speaks (0.4 s), 0:49.7 talking mouth, dub silent (0.3 s), 0:04.2 talking mouth, dub silent (0.3 s)

## 04: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.730 / 2.530 better |
| lip flash strong | 9 / 17 better |
| video s / video-min | 438.9 / 526.1 better (over limit) |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.101 vs 0.102 px; 0:39.0-0:47.6: 0.122 vs 0.120 px
Lip flash candidates: 9 vs 17.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.53 s (v4-repro) -> 0.73 s; worst: 0:51.5 closed mouth, dub speaks (0.4 s), 0:01.8 talking mouth, dub silent (0.3 s)

## 03: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 0.770 / 1.570 better |
| lip flash strong | 44 / 57 better |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.070 vs 0.069 px; 1:00.0-1:08.0: 0.061 vs 0.066 px; 0:21.6-0:37.6: 0.055 vs 0.053 px
Lip flash candidates: 44 vs 57.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.57 s (v4-repro) -> 0.77 s; worst: 1:08.5 closed mouth, dub speaks (0.4 s), 1:01.5 closed mouth, dub speaks (0.4 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.082 vs 0.078 px
Lip flash candidates: 4 vs 2.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m8b_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 05 | PASS | 1336 (44.53 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 04 | PASS | 1434 (47.8 s) | 0.0 | 0 | 0.0 | 1617/1617 |
| 03 | PASS | 2494 (83.13 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m8b_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 14.24 | 14.78 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 49.65 | 41.73 |
| 04 | - | 39.62 | 36.01 |
| 05 | - | 16.55 | 17.83 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 2.15 | 2.41 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 1.41 | 1.36 |
| 05 | - | 2.88 | 4.34 |

**n_events** (lower = better)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 8 | 8 |
| 02 | - | 0 | 0 |
| 03 | - | 37 | 34 |
| 04 | - | 25 | 22 |
| 05 | - | 13 | 14 |

**n_events_ge2** (lower = better)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 2 | 2 |
| 02 | - | 0 | 0 |
| 03 | - | 3 | 1 |
| 04 | - | 4 | 3 |
| 05 | - | 0 | 0 |

**top20_mean** (lower = better)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 1.78 | 1.847 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.581 | 1.355 |
| 04 | - | 1.72 | 1.696 |
| 05 | - | 1.273 | 1.274 |

**p99** (lower = better)

| video | v4 | m8 | m8b |
|---|---|---|---|
| 01 | - | 0.627 | 0.669 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.104 | 1.068 |
| 04 | - | 1.378 | 1.345 |
| 05 | - | 1.001 | 1.061 |

**top events** (t s: score channel cause head-speed)
- m8 01: 7.6s 4.4 lips inside_ls h0.98; 1.2s 2.7 lips inside_ls h1.65; 4.4s 1.4 lips inside_ls h1.48; 8.6s 1.4 lips inside_ls h0.45; 0.4s 1.2 lips inside_ls h1.36
- m8 02: 
- m8 03: 73.7s 4.1 lips inside_ls h0.07; 77.0s 2.8 lips inside_ls h0.07; 42.5s 2.0 lips inside_ls h0.4; 19.3s 1.6 lips inside_ls h0.04; 48.1s 1.6 lips inside_ls h0.19
- m8 04: 9.0s 4.5 lips inside_ls h1.19; 32.7s 3.1 lips inside_ls h2.52; 44.8s 2.3 lips window_joint h1.37; 33.3s 2.0 lips inside_ls h2.85; 36.9s 1.8 lips inside_ls h1.17
- m8 05: 28.9s 1.8 skin_motion window_joint h1.15; 5.0s 1.6 lips inside_ls h2.18; 37.0s 1.4 lips inside_ls h1.41; 38.3s 1.4 lips inside_ls h1.14; 40.3s 1.3 lips inside_ls h1.4
- m8b 01: 7.6s 4.3 lips inside_ls h0.98; 1.2s 2.8 lips inside_ls h1.65; 4.4s 1.6 lips inside_ls h1.48; 8.6s 1.5 lips inside_ls h0.45; 0.1s 1.3 skin_motion inside_ls h1.44
- m8b 02: 
- m8b 03: 42.5s 2.2 lips inside_ls h0.4; 48.1s 1.5 lips inside_ls h0.19; 74.7s 1.5 lips inside_ls h0.05; 17.8s 1.5 lips inside_ls h0.08; 77.2s 1.4 lips inside_ls h0.14
- m8b 04: 32.7s 4.2 lips inside_ls h2.52; 9.0s 3.9 lips inside_ls h1.19; 33.3s 2.7 lips inside_ls h2.85; 41.8s 1.7 lips inside_ls h1.26; 36.9s 1.7 lips inside_ls h1.17
- m8b 05: 28.9s 1.6 skin_motion inside_ls h1.15; 24.7s 1.6 skin_motion seg_end h1.68; 5.1s 1.5 lips window_joint h1.34; 14.0s 1.4 lips inside_ls h0.85; 38.8s 1.4 lips inside_ls h1.74

### m8b (r7 F): the m8 face stack, unchanged, on audio-r7-v2
Left m8 (audio-r7-v1) | right m8b (audio-r7-v2: crackle fix = dub gate, wider lip placement 2.0 / 1.5 / 0.5 s, gap clean). LatentSync re-run
on v2 for all 5; the face stack (m6 guards + Real-ESRGAN temporal lips + full-confidence source skin texture + texture-mode sanitizer with the
rigid fallback) is the same code and flags as m8. Twitch and SyncNet differences are the generator's response to the new audio, not a
composite change (whole-frame gate PASS on all 5).
Listen / watch: 04 whole video (the v1 crackles are gone); 05 24-25.5 s and 04 1.5-5.5 / 28-29 / 51-52 s (the dub-silent-while-the-source-
talks spans, shortened by v2; the user chose longer takes = audio v3, rendered next on 05 / 04); 05 49.2 s and 40.5 s (the gate's
mouth / dub mismatch rise 1.47 -> 2.17 s: talking mouth on a dub pause, closed mouth on dub speech, each 0.3-0.4 s).
