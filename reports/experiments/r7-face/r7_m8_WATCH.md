# r7 m8: WATCH candidate

m8 = the m7 face stack (m6 paste guards + Real-ESRGAN temporal lips + full-confidence source skin texture) on the audio agent's audio-r7-v1 (r6v1 takes + lip placement + gap clean): LatentSync re-run on the new audio for all 5, plus the sanitizer crack fix (04 38.2 s). PREV = m7 (published, old audio). Order to watch: 03, 01, 05, then 02, 04.

Files: `<id>_r7_m8_WATCH.mp4`, `<id>_r7_m8_WATCH_sbs.mp4` (M7 | M8). Gate verdict per video below (rows FAIL / better).

## 03: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| window joints LF ratio | 0.843 / 0.802 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.069 vs 0.073 px; 1:00.0-1:08.0: 0.066 vs 0.059 px; 0:21.6-0:37.6: 0.053 vs 0.053 px
Lip flash candidates: 57 vs 54.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.77 s (v4-repro) -> 1.57 s; worst: 1:12.8 closed mouth, dub speaks (0.5 s), 1:08.6 closed mouth, dub speaks (0.4 s), 1:01.7 closed mouth, dub speaks (0.3 s), 0:12.5 closed mouth, dub speaks (0.3 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 6.732 / 6.411 better |

Slide (lower face vs the head, fixed shaky windows): 0:00.0-0:10.4: 0.078 vs 0.083 px
Lip flash candidates: 2 vs 1.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.28 s (v4-repro) -> 0.0 s; worst: 

## 05: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.470 / 2.400 better |
| window joints ratio | 0.912 / 0.966 better |
| window joints LF ratio | 0.839 / 0.947 better |
| lip flash strong | 7 / 3 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.031 vs 0.029 px; 0:27.5-0:34.2: 0.039 vs 0.037 px
Lip flash candidates: 7 vs 3.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.4 s (v4-repro) -> 1.47 s; worst: 0:19.9 talking mouth, dub silent (0.4 s), 0:49.3 talking mouth, dub silent (0.4 s), 0:40.5 closed mouth, dub speaks (0.4 s), 0:33.9 talking mouth, dub silent (0.3 s)

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| video s / video-min | 37.1 / 54.0 better |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 04: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| slide nose-base px | 0.111 / 0.141 better |
| lip flash strong | 17 / 14 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.102 vs 0.128 px; 0:39.0-0:47.6: 0.120 vs 0.154 px
Lip flash candidates: 17 vs 14.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.86 s (v4-repro) -> 2.53 s; worst: 0:04.0 closed mouth, dub speaks (1.5 s), 0:51.5 closed mouth, dub speaks (0.4 s), 0:01.8 talking mouth, dub silent (0.4 s), 0:28.6 talking mouth, dub silent (0.3 s)


## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in `r7_m8_frame_gate/`)
| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |
|---|---|---|---|---|---|---|
| 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 0.0 | 2710/2710 |
| 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 0.0 | 260/2542 |
| 05 | PASS | 1407 (46.9 s) | 0.0 | 0 | 0.0 | 1511/1567 |
| 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0.0 | 0/3324 |
| 04 | PASS | 1434 (47.8 s) | 0.0 | 0 | 0.0 | 1617/1617 |

## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in `r7_m8_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 9.59 | 14.24 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 51.77 | 49.65 |
| 04 | - | 53.85 | 39.62 |
| 05 | - | 20.05 | 16.55 |

**skin_sum** (lower = better; skin_motion + skin_app events only, not the lips channel)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 1.28 | 2.15 |
| 02 | - | 0 | 0 |
| 03 | - | 0 | 0 |
| 04 | - | 0 | 1.41 |
| 05 | - | 4.58 | 2.88 |

**n_events** (lower = better)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 6 | 8 |
| 02 | - | 0 | 0 |
| 03 | - | 41 | 37 |
| 04 | - | 27 | 25 |
| 05 | - | 10 | 13 |

**n_events_ge2** (lower = better)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 1 | 2 |
| 02 | - | 0 | 0 |
| 03 | - | 2 | 3 |
| 04 | - | 8 | 4 |
| 05 | - | 4 | 0 |

**top20_mean** (lower = better)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 1.598 | 1.78 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.485 | 1.581 |
| 04 | - | 2.304 | 1.72 |
| 05 | - | 2.005 | 1.273 |

**p99** (lower = better)

| video | v4 | m7 | m8 |
|---|---|---|---|
| 01 | - | 0.664 | 0.627 |
| 02 | - | 0.0 | 0.0 |
| 03 | - | 1.095 | 1.104 |
| 04 | - | 2.178 | 1.378 |
| 05 | - | 0.954 | 1.001 |

**top events** (t s: score channel cause head-speed)
- m7 01: 1.2s 2.7 lips inside_ls h1.65; 6.8s 1.7 lips inside_ls h0.67; 8.6s 1.5 lips inside_ls h0.45; 4.2s 1.4 lips inside_ls h1.71; 0.2s 1.3 skin_motion inside_ls h1.64
- m7 02: 
- m7 03: 37.8s 2.2 lips inside_ls h0.06; 42.5s 2.1 lips inside_ls h0.4; 47.8s 1.9 lips inside_ls h0.19; 63.2s 1.8 lips window_joint h0.1; 52.6s 1.6 lips inside_ls h0.34
- m7 04: 33.4s 6.2 lips inside_ls h2.74; 7.9s 4.9 lips inside_ls h2.02; 9.1s 3.7 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.9 lips inside_ls h1.44
- m7 05: 20.3s 4.9 lips window_joint h2.41; 27.8s 3.0 lips inside_ls h1.63; 26.9s 2.9 lips inside_ls h1.23; 28.9s 2.3 skin_motion inside_ls h1.15; 38.8s 1.4 lips inside_ls h1.74
- m8 01: 7.6s 4.4 lips inside_ls h0.98; 1.2s 2.7 lips inside_ls h1.65; 4.4s 1.4 lips inside_ls h1.48; 8.6s 1.4 lips inside_ls h0.45; 0.4s 1.2 lips inside_ls h1.36
- m8 02: 
- m8 03: 73.7s 4.1 lips inside_ls h0.07; 77.0s 2.8 lips inside_ls h0.07; 42.5s 2.0 lips inside_ls h0.4; 19.3s 1.6 lips inside_ls h0.04; 48.1s 1.6 lips inside_ls h0.19
- m8 04: 9.0s 4.5 lips inside_ls h1.19; 32.7s 3.1 lips inside_ls h2.52; 44.8s 2.3 lips window_joint h1.37; 33.3s 2.0 lips inside_ls h2.85; 36.9s 1.8 lips inside_ls h1.17
- m8 05: 28.9s 1.8 skin_motion window_joint h1.15; 5.0s 1.6 lips inside_ls h2.18; 37.0s 1.4 lips inside_ls h1.41; 38.3s 1.4 lips inside_ls h1.14; 40.3s 1.3 lips inside_ls h1.4

### m8 vs m7 twitch (r7 F)
| video | m7 sum | m8 sum | m7 skin_sum | m8 skin_sum | events m7 / m8 | SyncNet m7 / m8 | m8 top-3 (t, score, channel) |
|---|---|---|---|---|---|---|---|
| 03 | 51.77 | 49.65 | 0 | 0 | 37 / 37 | 6.63 / 6.65 | 73.7 s (4.1 lips), 77.0 s (2.8 lips), 42.5 s (2.0 lips) |
| 01 | 9.59 | 14.24 | 1.28 | 2.15 | 5 / 8 | 6.52 / 6.82 | 7.6 s (4.4 lips), 1.2 s (2.7 lips), 4.4 s (1.4 lips) |
| 05 | 20.05 | 16.55 | 4.58 | 2.88 | 15 / 13 | 2.64 / 2.63 | 28.9 s (1.8 skin_motion), 5.0 s (1.6 lips), 37.0 s (1.4 lips) |
| 02 | 0.00 | 0.00 | 0 | 0 | 0 / 0 | - | - |
| 04 | 53.85 | 39.62 | 0 | 1.41 | 25 / 25 | 4.25 / 4.15 | 9.0 s (4.5 lips), 32.7 s (3.1 lips), 44.8 s (2.3 lips) |

m8 = the m7 face stack (m6 guards + Real-ESRGAN temporal lips + full-confidence source skin texture) on the audio agent's **audio-r7-v1**
(r6v1 takes + lip placement + gap clean): LatentSync re-run on the new audio for all 5, dub-speech spans re-derived by VAD on the new
dubbed track. The twitch sums are not frame-aligned to m7 (different dub audio = different mouth motion), so the events move; the sums are
a per-video total. Whole-frame gate PASS on all 5 (0 frames > 0.2 % changed outside the mouth).

**Crack fix (in m8, not in the published m7).** 04 frame 956 (38.2 s): the texture sanitizer's low-confidence fallback (the nearer good
frame warped by a dense flow that was wrong on the lip line) tore the lower lip into cracks. m8's composite uses the new rigid fallback
(`--sanitize-min-conf 0.3 --sanitize-tear 0`: the nearer good frame moved by a lip+rigid-landmark similarity). Verified on 04 frames
954/956/958 (clean closed lips in m8; crack in m7-style composite) — crop sheet `artifacts/r7f/crack956/04_m7_vs_m8_final_954-985.jpg`.
Rigid fallback frames: 03 127, 01 110, 05 124, 04 14. Frame 985 (39.4 s, a real LatentSync lip flash) is still a visible one-frame lip
texture change in both m7 and m8.

**01 (9.59 → 14.24, skin_sum 1.28 → 2.15):** one new event, 7.6 s (4.4 lips) = the whole rise. On the new audio LatentSync closes the mouth for 3–4 frames (layer mouth opening 0.003–0.027 IOD at frames 188–191, a bilabial in the new placement) between open frames; in m7 the sanitizer happened to flag 189–191 and interpolated them from the open neighbours, in m8 only 189 is flagged so the generator's closure stays (one pursed frame at 190). Generator articulation on the new audio, not a composite / paste regression (same 260 LS frames, whole-frame gate 0.005 % worst). Strip: `artifacts/r7f/crack956/01_7.6s_src_m7_m8.jpg`.
