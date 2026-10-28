# r7 m11: WATCH candidate

m11 = the m8 face stack (unchanged) + the audio agent's audio-r7-v3a (longer dub takes where the speaker is on screen; the user's pick among v3a/v3b/v3c; 02 on audio-r7-v3a-all5). Left m8b (audio v2) | right m11 (audio v3a).

Files: `<id>_r7_m11_WATCH.mp4`, `<id>_r7_m11_WATCH_sbs.mp4` (M8B | M11). Gate verdict per video below (rows FAIL / better).

## 05: gate FAIL (3 rows)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 3.576 / 3.911 **FAIL** |
| window joints ratio | 0.953 / 0.892 **FAIL** |
| window joints LF ratio | 0.895 / 0.842 **FAIL** |
| video s / video-min | 500.2 / 420.6 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.031 vs 0.030 px; 0:27.5-0:34.2: 0.038 vs 0.041 px
Lip flash candidates: 6 vs 5.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.17 s (v4-repro) -> 1.84 s; worst: 0:04.2 talking mouth, dub silent (0.4 s), 0:49.2 talking mouth, dub silent (0.4 s), 0:40.5 closed mouth, dub speaks (0.4 s), 0:18.8 talking mouth, dub silent (0.3 s)

## 04: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| lip flash strong | 16 / 9 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.107 vs 0.101 px; 0:39.0-0:47.6: 0.120 vs 0.122 px
Lip flash candidates: 16 vs 9.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.73 s (v4-repro) -> 0.73 s; worst: 0:51.5 closed mouth, dub speaks (0.4 s), 0:01.8 talking mouth, dub silent (0.3 s)

## 03: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.070 vs 0.070 px; 1:00.0-1:08.0: 0.061 vs 0.061 px; 0:21.6-0:37.6: 0.055 vs 0.055 px
Lip flash candidates: 44 vs 44.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.77 s (v4-repro) -> 0.77 s; worst: 1:08.5 closed mouth, dub speaks (0.4 s), 1:01.5 closed mouth, dub speaks (0.4 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.082 vs 0.082 px
Lip flash candidates: 4 vs 4.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m11_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 05 | PASS | 1336 (44.53 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 04 | PASS | 58 (1.93 s) | 0.0 | 0 | 0.0 | 1617/1617 |
| 03 | PASS | 2494 (83.13 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m11_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 14.78 | 14.78 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 41.73 | 41.73 |
| 04 | - | 36.01 | 38.83 |
| 05 | - | 17.83 | 15.59 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 2.41 | 2.41 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 1.36 | 5.74 |
| 05 | - | 4.34 | 2.45 |

**n_events** (lower = better)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 8 | 8 |
| 02 | - | 0 | 0 |
| 03 | - | 34 | 34 |
| 04 | - | 22 | 26 |
| 05 | - | 14 | 12 |

**n_events_ge2** (lower = better)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 2 | 2 |
| 02 | - | 0 | 0 |
| 03 | - | 1 | 1 |
| 04 | - | 3 | 1 |
| 05 | - | 0 | 1 |

**top20_mean** (lower = better)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 1.847 | 1.847 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.355 | 1.355 |
| 04 | - | 1.696 | 1.628 |
| 05 | - | 1.274 | 1.299 |

**p99** (lower = better)

| video | v4 | m8b | m11 |
|---|---|---|---|
| 01 | - | 0.669 | 0.669 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.068 | 1.068 |
| 04 | - | 1.345 | 1.383 |
| 05 | - | 1.061 | 1.002 |

**top events** (t s: score channel cause head-speed)
- m8b 01: 7.6s 4.3 lips inside_ls h0.98; 1.2s 2.8 lips inside_ls h1.65; 4.4s 1.6 lips inside_ls h1.48; 8.6s 1.5 lips inside_ls h0.45; 0.1s 1.3 skin_motion inside_ls h1.44
- m8b 02: 
- m8b 03: 42.5s 2.2 lips inside_ls h0.4; 48.1s 1.5 lips inside_ls h0.19; 74.7s 1.5 lips inside_ls h0.05; 17.8s 1.5 lips inside_ls h0.08; 77.2s 1.4 lips inside_ls h0.14
- m8b 04: 32.7s 4.2 lips inside_ls h2.52; 9.0s 3.9 lips inside_ls h1.19; 33.3s 2.7 lips inside_ls h2.85; 41.8s 1.7 lips inside_ls h1.26; 36.9s 1.7 lips inside_ls h1.17
- m8b 05: 28.9s 1.6 skin_motion inside_ls h1.15; 24.7s 1.6 skin_motion seg_end h1.68; 5.1s 1.5 lips window_joint h1.34; 14.0s 1.4 lips inside_ls h0.85; 38.8s 1.4 lips inside_ls h1.74
- m11 01: 7.6s 4.3 lips inside_ls h0.98; 1.2s 2.8 lips inside_ls h1.65; 4.4s 1.6 lips inside_ls h1.48; 8.6s 1.5 lips inside_ls h0.45; 0.1s 1.3 skin_motion inside_ls h1.44
- m11 02: 
- m11 03: 42.5s 2.2 lips inside_ls h0.4; 48.1s 1.5 lips inside_ls h0.19; 74.7s 1.5 lips inside_ls h0.05; 17.8s 1.5 lips inside_ls h0.08; 77.2s 1.4 lips inside_ls h0.14
- m11 04: 9.0s 4.1 lips inside_ls h1.19; 46.6s 1.8 lips inside_ls h2.34; 36.5s 1.8 lips window_joint h2.08; 41.8s 1.8 lips window_joint h1.26; 35.1s 1.7 skin_motion inside_ls h1.68
- m11 05: 28.9s 2.4 skin_motion seg_end h1.15; 5.0s 1.4 lips inside_ls h1.34; 38.8s 1.3 lips inside_ls h1.74; 5.6s 1.3 lips inside_ls h1.56; 38.3s 1.3 lips inside_ls h1.14

### m11 (r7 F) = the m8 face stack + audio-r7-v3a on all 5
Left m8b (audio-r7-v2) | right m11 (audio-r7-v3a: v2 + re-translated / re-taken units where the speaker is on screen and the dub ended
early; `retake.py`, tempo 1.0). The face stack is unchanged (m6 guards + Real-ESRGAN temporal lips + full-confidence source skin texture +
texture-mode sanitizer with the rigid fallback). 05 and 04 are the m8v3a renders from the variants WATCH (LatentSync re-run on v3a);
03 and 01 are the m8b renders, because the v3a audio there is bit-identical to v2 (no unit passed the re-take bar) — same files, same
numbers; 02 (no face frames) is the source + the audio-r7-v3a-all5 mix, one re-taken sentence. Changed sentences: 04 unit 2; 05 units 2, 3, 4; 02 one unit.
Closed mouth while the source talks, dub-silent subset (r7_closed_mouth.py): 05 8.7 -> 6.9 s, 04 4.7 -> 3.1 s. Open mouth while the dub is
silent rises (05 5.0 -> 7.5 s, 04 1.9 -> 4.1 s: the longer takes' internal pauses, during which the generator keeps the mouth moving).
Listen / watch (05 | 04): 05 24-25.5 s (the pressed-lips span, now mostly dubbed), 05 units 2-4 (~10-30 s) for the new English; 04 1.5-5.5 s
(unit 2, re-taken) and 28-29 / 51-52 s (unchanged spans); 04 flash count 9 -> 16 (the re-take's consonant closures, lips channel).
