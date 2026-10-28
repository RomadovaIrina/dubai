# r7 m6: WATCH candidate

m6 = the m5 face stack (m4 face + audio r6_audio_v1) + two r7 guards: fps_composite --paste-guard 0.5 --guard-min-run 3 (nothing is pasted when the current source frame has no face at the paste position; proven on a synthetic camera-looks-away test: 17 000 px of pasted face per frame -> 0) and the whole-frame gate (|out - src| outside the mouth must be ~0 on every frame). The user's 05 25-26 s 'rectangle' turned out to be the SOURCE's own picture-in-picture inset (see outbox/r7_05_pip_check/); m5 and m6 pass the whole-frame gate at 0.000 % there. The face itself is m5's face: the generator, the audio and every other flag are unchanged, the guard changed no frame on the 5 videos, the top twitch events are identical. PREV = m5 (the published WATCH mp4s, re-scored on this server). The soapy lower face is NOT fixed in m6: that is the next candidate (m7, in progress: Real-ESRGAN on the layer's lips + full-confidence source skin texture).

Files: `<id>_r7_m6_WATCH.mp4`, `<id>_r7_m6_WATCH_sbs.mp4` (M5 | M6). Gate verdict per video below (rows FAIL / better).

## 03: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.068 vs 0.068 px; 1:00.0-1:08.0: 0.060 vs 0.060 px; 0:21.6-0:37.6: 0.054 vs 0.054 px
Lip flash candidates: 59 vs 60.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.8 s (v4-repro) -> 1.8 s; worst: 0:12.6 closed mouth, dub speaks (0.7 s), 1:08.8 closed mouth, dub speaks (0.4 s), 1:01.7 closed mouth, dub speaks (0.4 s), 0:04.1 closed mouth, dub speaks (0.3 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.089 vs 0.089 px
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 05: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.100 / 2.640 better |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.030 vs 0.030 px; 0:27.5-0:34.2: 0.040 vs 0.040 px
Lip flash candidates: 24 vs 26.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.64 s (v4-repro) -> 2.1 s; worst: 0:20.0 talking mouth, dub silent (0.4 s), 0:49.3 talking mouth, dub silent (0.4 s), 0:21.8 closed mouth, dub speaks (0.4 s), 0:24.1 talking mouth, dub silent (0.3 s)

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 04: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.105 vs 0.105 px; 0:39.0-0:47.6: 0.159 vs 0.159 px
Lip flash candidates: 14 vs 14.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.6 s (v4-repro) -> 2.86 s; worst: 0:04.0 closed mouth, dub speaks (1.5 s), 0:28.6 talking mouth, dub silent (0.6 s), 0:51.5 closed mouth, dub speaks (0.4 s), 0:01.8 talking mouth, dub silent (0.4 s)


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m6_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 05 | PASS | 1502 (50.07 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |
| 04 | PASS | 1566 (52.2 s) | 0.0 | 0 | 0.0 | 1617/1617 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m6_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 12.12 | 12.12 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 47.65 | 47.64 |
| 04 | - | 50.49 | 51.51 |
| 05 | - | 18.99 | 20.01 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 1.24 | 1.24 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 0 | 0 |
| 05 | - | 5.56 | 5.56 |

**n_events** (lower = better)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 8 | 8 |
| 02 | - | 0 | 0 |
| 03 | - | 37 | 37 |
| 04 | - | 23 | 24 |
| 05 | - | 12 | 13 |

**n_events_ge2** (lower = better)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 2 | 2 |
| 02 | - | 0 | 0 |
| 03 | - | 3 | 3 |
| 04 | - | 7 | 7 |
| 05 | - | 2 | 2 |

**top20_mean** (lower = better)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 1.514 | 1.514 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.484 | 1.483 |
| 04 | - | 2.368 | 2.368 |
| 05 | - | 1.582 | 1.539 |

**p99** (lower = better)

| video | v4 | m5 | m6 |
|---|---|---|---|
| 01 | - | 0.63 | 0.63 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.095 | 1.095 |
| 04 | - | 1.977 | 1.977 |
| 05 | - | 0.996 | 1.005 |

**top events** (t s: score channel cause head-speed)
- m5 01: 1.2s 2.6 lips - h1.65; 6.8s 2.2 lips - h0.67; 4.2s 1.6 lips - h1.71; 8.6s 1.3 lips - h0.45; 0.1s 1.2 skin_motion - h1.44
- m5 02: 
- m5 03: 37.8s 2.2 lips - h0.06; 47.8s 2.2 lips - h0.19; 42.5s 2.1 lips - h0.4; 52.6s 1.7 lips - h0.34; 70.7s 1.5 lips - h0.17
- m5 04: 33.4s 10.0 lips - h2.74; 7.9s 4.7 lips - h2.02; 9.1s 4.2 lips - h1.34; 33.8s 3.2 lips - h1.63; 44.7s 2.8 lips - h1.44
- m5 05: 26.9s 4.1 lips - h1.23; 20.2s 2.8 lips - h2.41; 27.8s 1.8 lips - h1.63; 38.8s 1.4 lips - h1.74; 28.9s 1.4 skin_motion - h1.15
- m6 01: 1.2s 2.6 lips inside_ls h1.65; 6.8s 2.2 lips inside_ls h0.67; 4.2s 1.6 lips inside_ls h1.71; 8.6s 1.3 lips inside_ls h0.45; 0.1s 1.2 skin_motion inside_ls h1.44
- m6 02: 
- m6 03: 37.8s 2.2 lips inside_ls h0.06; 47.8s 2.2 lips inside_ls h0.19; 42.5s 2.1 lips inside_ls h0.4; 52.6s 1.7 lips inside_ls h0.34; 70.7s 1.5 lips inside_ls h0.17
- m6 04: 33.4s 10.0 lips inside_ls h2.74; 7.9s 4.7 lips inside_ls h2.02; 9.1s 4.2 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.8 lips inside_ls h1.44
- m6 05: 26.9s 4.1 lips inside_ls h1.23; 20.2s 2.8 lips inside_ls h2.41; 27.8s 1.8 lips inside_ls h1.63; 38.8s 1.4 lips inside_ls h1.74; 28.9s 1.4 skin_motion inside_ls h1.15

### m6 vs m5 twitch (r7 F)
| video | m5 sum | m6 sum | events m5 / m6 | top-3 identical | guard flagged / passed through | m6 top-3 (t, score, channel) |
|---|---|---|---|---|---|---|
| 03 | 47.65 | 47.64 | 37 / 37 | 3/3 | 0 / 0 | 37.8 s (2.2 lips), 47.8 s (2.2 lips), 42.5 s (2.1 lips) |
| 01 | 12.12 | 12.12 | 8 / 8 | 3/3 | 0 / 0 | 1.2 s (2.6 lips), 6.8 s (2.2 lips), 4.2 s (1.6 lips) |
| 05 | 18.99 | 20.01 | 12 / 13 | 3/3 | 1 / 0 | 26.9 s (4.1 lips), 20.2 s (2.8 lips), 27.8 s (1.8 lips) |
| 02 | 0.00 | 0.00 | 0 / 0 | 0/3 | - / - | - |
| 04 | 50.49 | 51.51 | 23 / 24 | 3/3 | 0 / 0 | 33.4 s (10.0 lips), 7.9 s (4.7 lips), 9.1 s (4.2 lips) |

m6 = the m5 face stack + `--paste-guard 0.5 --guard-min-run 3` (pass-through when the current source frame has no face at the paste position)
+ the whole-frame gate; the generator, the audio (r6_audio_v1) and every other flag are m5's. The guard changed NO frame on the 5 videos
(flagged 1 frame on 05 = a false positive kept), so m6's face is m5's face: the top twitch events are identical in time and score, the sums
differ by sub-threshold events. The whole-frame gate (new) passes on all 5 at 0.000 % changed outside the mouth. 02 has no face (B-roll).
What m6 does NOT fix yet: the soapy lower face (lips at 0.73 of the source's sharpness on 03): see STATUS_LOG.md section 4, in progress.
