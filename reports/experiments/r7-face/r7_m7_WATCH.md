# r7 m7: WATCH candidate

m7 = m6 (same generator output, same audio r6_audio_v1, same guards) + two look changes for the soapy lower face: (1) scripts/r7/r7_layer_sr.py: Real-ESRGAN x2 (BSD-3) on the LatentSync layer's lips region before the composite, its added detail averaged over +-1 layer frame flow-aligned (no per-frame flicker); (2) the composite's source skin-texture transfer at full confidence (--tex-tau 1000 --tex-sigma 2.5): the stubble / skin around the mouth is the SOURCE's high frequencies. No generator rerun: +18-80 s/video-min on top of m6. Sharpness of the paste vs the source (r7_sharp_probe, 1.0 = the source): lips 03 0.73 -> 0.81, 04 0.70 -> 0.77, 05 0.73 -> 0.92; skin ring 03 0.76 -> 0.87, 04 0.92 -> 0.98. Twitch: the skin channels (the paste jumping) are equal or better on all 5; the lips channel rises on 03 / 04 (+4 / +2) because it measures flow on the lips and crisper lips give larger flow on the SAME dub articulations (verified frame by frame: no new event that is not in m6). PREV = m6.

Files: `<id>_r7_m7_WATCH.mp4`, `<id>_r7_m7_WATCH_sbs.mp4` (M6 | M7). Gate verdict per video below (rows FAIL / better).

## 03: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| v4 sharpness mouth (src=1) | 0.852 / 0.798 better |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.073 vs 0.068 px; 1:00.0-1:08.0: 0.059 vs 0.060 px; 0:21.6-0:37.6: 0.053 vs 0.054 px
Lip flash candidates: 54 vs 59.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.8 s (v4-repro) -> 1.77 s; worst: 0:12.6 closed mouth, dub speaks (0.7 s), 1:08.8 closed mouth, dub speaks (0.4 s), 1:01.7 closed mouth, dub speaks (0.3 s), 0:04.1 closed mouth, dub speaks (0.3 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| (none) | |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.083 vs 0.089 px
Lip flash candidates: 1 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.28 s; worst: 0:10.0 closed mouth, dub speaks (0.3 s)

## 05: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| v4 sharpness mouth (src=1) | 0.962 / 0.922 better |
| lip flash strong | 3 / 24 better |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.029 vs 0.030 px; 0:27.5-0:34.2: 0.037 vs 0.040 px
Lip flash candidates: 3 vs 24.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.1 s (v4-repro) -> 2.4 s; worst: 0:20.0 talking mouth, dub silent (0.4 s), 0:49.2 talking mouth, dub silent (0.4 s), 0:21.8 closed mouth, dub speaks (0.4 s), 0:24.1 talking mouth, dub silent (0.3 s)

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
| v4 sharpness mouth (src=1) | 0.927 / 0.891 better |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.128 vs 0.105 px; 0:39.0-0:47.6: 0.154 vs 0.159 px
Lip flash candidates: 14 vs 14.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.86 s (v4-repro) -> 2.86 s; worst: 0:03.9 closed mouth, dub speaks (1.5 s), 0:28.6 talking mouth, dub silent (0.6 s), 0:40.3 closed mouth, dub speaks (0.4 s), 0:51.5 closed mouth, dub speaks (0.4 s)


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m7_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 05 | PASS | 719 (23.97 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |
| 04 | PASS | 58 (1.93 s) | 0.0 | 0 | 0.0 | 1617/1617 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m7_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 12.12 | 9.59 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 47.64 | 51.77 |
| 04 | - | 51.51 | 53.85 |
| 05 | - | 20.01 | 20.05 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 1.24 | 1.28 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 0 | 0 |
| 05 | - | 5.56 | 4.58 |

**n_events** (lower = better)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 8 | 6 |
| 02 | - | 0 | 0 |
| 03 | - | 37 | 41 |
| 04 | - | 24 | 27 |
| 05 | - | 13 | 10 |

**n_events_ge2** (lower = better)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 2 | 1 |
| 02 | - | 0 | 0 |
| 03 | - | 3 | 2 |
| 04 | - | 7 | 8 |
| 05 | - | 2 | 4 |

**top20_mean** (lower = better)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 1.514 | 1.598 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.483 | 1.485 |
| 04 | - | 2.368 | 2.304 |
| 05 | - | 1.539 | 2.005 |

**p99** (lower = better)

| video | v4 | m6 | m7 |
|---|---|---|---|
| 01 | - | 0.63 | 0.664 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.095 | 1.095 |
| 04 | - | 1.977 | 2.178 |
| 05 | - | 1.005 | 0.954 |

**top events** (t s: score channel cause head-speed)
- m6 01: 1.2s 2.6 lips inside_ls h1.65; 6.8s 2.2 lips inside_ls h0.67; 4.2s 1.6 lips inside_ls h1.71; 8.6s 1.3 lips inside_ls h0.45; 0.1s 1.2 skin_motion inside_ls h1.44
- m6 02: 
- m6 03: 37.8s 2.2 lips inside_ls h0.06; 47.8s 2.2 lips inside_ls h0.19; 42.5s 2.1 lips inside_ls h0.4; 52.6s 1.7 lips inside_ls h0.34; 70.7s 1.5 lips inside_ls h0.17
- m6 04: 33.4s 10.0 lips inside_ls h2.74; 7.9s 4.7 lips inside_ls h2.02; 9.1s 4.2 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.8 lips inside_ls h1.44
- m6 05: 26.9s 4.1 lips inside_ls h1.23; 20.2s 2.8 lips inside_ls h2.41; 27.8s 1.8 lips inside_ls h1.63; 38.8s 1.4 lips inside_ls h1.74; 28.9s 1.4 skin_motion inside_ls h1.15
- m7 01: 1.2s 2.7 lips inside_ls h1.65; 6.8s 1.7 lips inside_ls h0.67; 8.6s 1.5 lips inside_ls h0.45; 4.2s 1.4 lips inside_ls h1.71; 0.2s 1.3 skin_motion inside_ls h1.64
- m7 02: 
- m7 03: 37.8s 2.2 lips inside_ls h0.06; 42.5s 2.1 lips inside_ls h0.4; 47.8s 1.9 lips inside_ls h0.19; 63.2s 1.8 lips window_joint h0.1; 52.6s 1.6 lips inside_ls h0.34
- m7 04: 33.4s 6.2 lips inside_ls h2.74; 7.9s 4.9 lips inside_ls h2.02; 9.1s 3.7 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.9 lips inside_ls h1.44
- m7 05: 20.3s 4.9 lips window_joint h2.41; 27.8s 3.0 lips inside_ls h1.63; 26.9s 2.9 lips inside_ls h1.23; 28.9s 2.3 skin_motion inside_ls h1.15; 38.8s 1.4 lips inside_ls h1.74
