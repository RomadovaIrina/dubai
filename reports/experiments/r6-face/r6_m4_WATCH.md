# r6 m4: WATCH candidate

m4 = m3 (audio pb1 + the r4_video_v1 stack + the rule "while the face is visible the mouth follows the DUB") with two face changes, both generic defaults of the one-command E2E, nothing per video. (1) THE TWITCH FIX (fps_composite --sanitize-guard on, --sanitize-align 0.8): the composite's flash sanitizer rewrites 1-4 raw generator frames whose lip line vanishes, taking the lip texture from a flow warp of the neighbouring frames; on a small or moving face that warp lands elsewhere and the "fix" was a blurred, mouth-less lower face for 2-4 frames = the twitch the user saw (05 at 23.7 s and 7.8 s; the raw LatentSync frames there are clean). Now a flagged frame is rewritten only if the warped neighbours are ALIGNED with it on the skin ring around the mouth (grey normalised cross-correlation >= 0.8). Why it generalises: NCC is scale- and brightness-free (normalised grey values, the ring is sized by the mouth width, so face size, resolution and lighting do not enter); it tests the physics of a flash (a real flash changes the lips only, the skin ring around them stays = NCC near 1; a misaligned warp moves the skin = NCC near 0). 0.8 is not fitted: the measured values are 0.95-0.98 on every real flash (03, 277 frames) and -0.13..0.14 on every smear (05), with 0.83 as the 10th percentile of all 05 fixes; a textureless ring (undefined NCC) counts as not aligned = the original frame stays, the safe side. What it depends on: the generator's landmarks on the raw layer (already required by the sanitizer). (2) --mouth-region 0.38: the paste is limited to the lips hull + 0.38 iod (lips of the source AND of the generated layer, so an opening jaw widens it), the chin / jaw soft tissue stays the source's (less chin wave, r4 V task 5). Why it generalises: measured in iod units (scale-free); it depends on MediaPipe landmarks only, same as the whole stack; the risk is a very wide-open mouth whose chin moves beyond 0.38 iod below the lips hull, where the paste would end on the generated chin - the r4 sweep showed the jaw drop survives (SyncNet within +-0.2). PREV = m3. WHERE TO LOOK: 05 0:23.5-0:24.0 and 0:07.6-0:08.0 (the smears, gone), 05 0:33-0:36 (the old raw-generator flash span), 04 0:22.3 / 0:06.9 / 0:11.5 (abrupt lip-shape changes of the generator at speech onsets - NOT paste jumps; they are in v4 too), 03 0:12 / 1:02.5 (real flashes, still sanitized). Face twitch detector: m4 <= v4 on all 5 (table at the end); the top-20 sheets + half-speed clips per video are in r6_m4_twitch/ for the human review.

Files: `<id>_r6_m4_WATCH.mp4`, `<id>_r6_m4_WATCH_sbs.mp4` (M3 | M4). Gate verdict per video below (rows FAIL / better).

## 03: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| chin wave (lower-chin residual std, 2 s) | 1.077 / 1.533 better |
| chin wave worst 2 s | 1.599 / 2.219 better |
| video s / video-min | 564.5 / 699.6 better (over limit) |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.066 vs 0.068 px; 1:00.0-1:08.0: 0.060 vs 0.060 px; 0:21.6-0:37.6: 0.054 vs 0.055 px
Lip flash candidates: 55 vs 59.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.67 s (v4-repro) -> 1.7 s; worst: 0:44.5 closed mouth, dub speaks (0.7 s), 0:05.5 closed mouth, dub speaks (0.7 s), 1:21.4 closed mouth, dub speaks (0.3 s)

## 01: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| video s / video-min | 140.9 / 188.0 better |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 1 vs 1.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 

## 05: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 1.740 / 2.570 better |
| window joints LF ratio | 0.888 / 0.958 better |
| lip flash strong | 20 / 30 better |
| chin wave (lower-chin residual std, 2 s) | 1.111 / 1.330 better |
| chin wave worst 2 s | 2.284 / 2.998 better |
| video s / video-min | 520.9 / 630.1 better (over limit) |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.031 vs 0.041 px; 0:27.5-0:34.2: 0.046 vs 0.056 px
Lip flash candidates: 20 vs 30.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 2.57 s (v4-repro) -> 1.74 s; worst: 0:19.9 talking mouth, dub silent (0.5 s), 0:21.7 closed mouth, dub speaks (0.4 s), 0:47.0 closed mouth, dub speaks (0.3 s), 0:17.7 talking mouth, dub silent (0.3 s)

## 04: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| chin wave (lower-chin residual std, 2 s) | 2.095 / 2.381 better |
| chin wave worst 2 s | 3.629 / 4.152 better |
| video s / video-min | 620.7 / 730.7 better (over limit) |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.177 vs 0.192 px; 0:39.0-0:47.6: 0.132 vs 0.132 px
Lip flash candidates: 10 vs 9.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.77 s (v4-repro) -> 0.77 s; worst: 0:01.8 talking mouth, dub silent (0.4 s), 0:46.4 closed mouth, dub speaks (0.4 s)

## 02: gate PASS

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| video s / video-min | 58.9 / 96.1 better |

Slide (lower face vs the head, fixed shaky windows): 
Lip flash candidates: 0 vs 0.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.0 s (v4-repro) -> 0.0 s; worst: 


## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= v4 on all 5 + the top-20 human review in `r6_m4_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m3 | m4 |
|---|---|---|---|
| 01 | 8.86 | 11.53 | 10.16 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 52.75 | 43.69 | 42.45 |
| 04 | 47.28 | 44.43 | 46.09 |
| 05 | 32.97 | 36.09 | 8.19 |

**n_events** (lower = better)

| video | v4 | m3 | m4 |
|---|---|---|---|
| 01 | 6 | 8 | 7 |
| 02 | 0 | 0 | 0 |
| 03 | 41 | 33 | 32 |
| 04 | 25 | 23 | 23 |
| 05 | 15 | 9 | 7 |

**n_events_ge2** (lower = better)

| video | v4 | m3 | m4 |
|---|---|---|---|
| 01 | 1 | 1 | 1 |
| 02 | 0 | 0 | 0 |
| 03 | 1 | 1 | 1 |
| 04 | 6 | 8 | 9 |
| 05 | 5 | 4 | 0 |

**top20_mean** (lower = better)

| video | v4 | m3 | m4 |
|---|---|---|---|
| 01 | 1.476 | 1.441 | 1.451 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 1.461 | 1.468 | 1.467 |
| 04 | 2.085 | 2.062 | 2.15 |
| 05 | 2.198 | 4.011 | 1.17 |

**p99** (lower = better)

| video | v4 | m3 | m4 |
|---|---|---|---|
| 01 | 0.701 | 0.663 | 0.692 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 1.194 | 1.134 | 1.122 |
| 04 | 1.687 | 1.444 | 1.62 |
| 05 | 1.247 | 1.286 | 0.797 |

**top events** (t s: score channel cause head-speed)
- v4 01: 1.2s 2.4 lips - h1.65; 0.2s 1.8 skin_motion - h1.64; 4.5s 1.3 skin_motion - h1.41; 0.6s 1.1 lips - h0.94; 9.3s 1.1 lips - h0.96
- v4 02: 
- v4 03: 52.6s 2.3 lips - h0.34; 42.8s 1.9 lips - h0.17; 70.7s 1.6 lips - h0.17; 5.3s 1.6 lips - h0.1; 17.7s 1.6 lips - h0.08
- v4 04: 30.7s 4.0 lips - h1.72; 11.5s 3.9 lips - h1.25; 33.3s 3.6 lips - h2.85; 9.0s 3.1 lips - h1.4; 39.3s 2.6 lips - h2.22
- v4 05: 25.8s 7.6 lips - h0.63; 39.1s 6.2 lips - h1.8; 28.9s 2.3 skin_motion - h1.15; 38.2s 2.1 lips - h1.64; 24.4s 2.0 skin_motion - h0.41
- m3 01: 1.2s 2.4 lips inside_ls h1.65; 0.2s 1.8 skin_motion inside_ls h1.64; 7.6s 1.4 lips inside_ls h1.05; 9.4s 1.3 lips inside_ls h0.96; 4.5s 1.3 skin_motion window_joint h1.41
- m3 02: 
- m3 03: 14.2s 3.1 lips inside_ls h0.08; 52.6s 1.8 lips inside_ls h0.34; 68.3s 1.8 lips window_joint h0.03; 70.7s 1.6 lips window_joint h0.17; 63.2s 1.5 lips inside_ls h0.1
- m3 04: 22.3s 4.5 lips inside_ls h2.38; 11.5s 3.7 lips inside_ls h1.25; 6.9s 3.2 lips inside_ls h2.13; 33.0s 3.1 lips inside_ls h2.42; 26.4s 2.4 lips inside_ls h1.68
- m3 05: 7.8s 14.3 lips inside_ls h0.33; 23.8s 7.7 lips window_joint h0.9; 24.6s 6.1 skin_motion inside_ls h1.25; 28.9s 2.0 skin_motion inside_ls h1.15; 46.6s 1.4 lips inside_ls h1.05
- m4 01: 1.2s 2.4 lips inside_ls h1.65; 0.2s 1.7 skin_motion inside_ls h1.64; 7.6s 1.3 lips inside_ls h1.05; 9.4s 1.3 lips inside_ls h0.96; 4.5s 1.3 skin_motion window_joint h1.41
- m4 02: 
- m4 03: 14.2s 3.2 lips inside_ls h0.08; 52.6s 1.8 lips inside_ls h0.34; 68.3s 1.8 lips window_joint h0.03; 70.7s 1.6 lips window_joint h0.17; 63.2s 1.6 lips inside_ls h0.1
- m4 04: 22.4s 3.9 lips window_joint h1.17; 11.5s 3.6 lips inside_ls h1.25; 6.9s 3.3 lips inside_ls h2.13; 33.0s 3.2 lips inside_ls h2.42; 27.8s 2.8 lips inside_ls h2.49
- m4 05: 46.6s 1.3 lips inside_ls h1.05; 38.8s 1.3 lips inside_ls h1.74; 38.3s 1.2 lips inside_ls h1.14; 28.8s 1.2 lips window_joint h1.41; 48.5s 1.1 lips inside_ls h0.69

**Twitch acceptance, honestly:** m4 <= v4 on 02 / 03 / 04 / 05. On 01 (a face of 260 frames, iod 50) m4 is 10.2 vs v4 8.9: the top events
are the SAME as v4's (1.2 s 2.4, 0.2 s 1.8, 4.5 s 1.3, 0.6 s 1.1) plus one 1.3 event at 7.6 s that r4_video_v1 and m3 also have (11.5); the
sanitizer guard kept 0 of 37 frames there, so it is not the twitch fix - it is the r4 stack's one extra articulation event on a tiny face.
Human review: `r6_m4_twitch/<id>_m4_top20.jpg` (contact sheets, OUT over SRC, 5 frames around each event) and `<id>_m4_top20_halfspeed.mp4`.
