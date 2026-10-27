# r6 m5: WATCH candidate

m5 = the m4 face stack (r4_video_v1 + the rule "while the face is visible the mouth follows the DUB" + the r6 twitch fix: sanitize alignment guard, + --mouth-region 0.38) on the AUDIO OF RECORD r6_audio_v1 (the audio agent's duration-aware translation, the user's choice by ear), one command (scripts/r6/r6_e2e.sh), the same flags for every video, generator unchanged (G's recommendation: guidance 2.0, bridge on). PREV = m4 (same face stack on audio pb1), so this sbs isolates what the NEW AUDIO does to the face: the dub speaks at different moments, so the LatentSync segments start / end elsewhere, the generator sees different speech, and the lip shapes at speech onsets / offsets change; the face paste itself is identical. WHAT TO WATCH IN THE FACE: (a) no smeared / mouth-less lower face anywhere (the m3 twitch; the guard kept 15 of 122 sanitizer fixes on 05, 05 0:23.5-0:24.0 and 0:07.6-0:08.0 stay clean); (b) the moments the detector ranks highest are the generator closing or opening the mouth within 2 frames at the new speech boundaries (05 0:26.9 and 0:20.2, both checked on frame grids: articulation, not a paste jump); (c) mouth-follows-dub: a closed mouth wherever the new dub is silent while the source talks. The twitch table at the end compares v4 / m4 / m5; the top-20 sheets + half-speed clips are in r6_m5_twitch/.

Files: `<id>_r6_m5_WATCH.mp4`, `<id>_r6_m5_WATCH_sbs.mp4` (M4 | M5). Gate verdict per video below (rows FAIL / better).

## 03: gate FAIL (1 row)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| v4 resid mouth | 1.191 / 1.113 **FAIL** |
| window joints ratio | 0.887 / 0.944 better |
| window joints LF ratio | 0.794 / 0.862 better |
| video s / video-min | 748.0 / 564.5 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:46.0-0:56.4: 0.068 vs 0.066 px; 1:00.0-1:08.0: 0.060 vs 0.060 px; 0:21.6-0:37.6: 0.054 vs 0.054 px
Lip flash candidates: 60 vs 55.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.7 s (v4-repro) -> 1.8 s; worst: 0:12.6 closed mouth, dub speaks (0.7 s), 1:08.8 closed mouth, dub speaks (0.4 s), 1:01.7 closed mouth, dub speaks (0.4 s), 0:04.1 closed mouth, dub speaks (0.3 s)

## 05: gate FAIL (4 rows)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 4.057 / 5.079 **FAIL** |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.640 / 1.740 **FAIL** |
| window joints LF ratio | 0.942 / 0.888 **FAIL** |
| lip flash strong | 26 / 20 **FAIL** |
| chin wave (lower-chin residual std, 2 s) | 0.983 / 1.111 better |
| video s / video-min | 1048.1 / 520.9 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:42.0-0:49.4: 0.030 vs 0.031 px; 0:27.5-0:34.2: 0.040 vs 0.046 px
Lip flash candidates: 26 vs 20.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 1.74 s (v4-repro) -> 2.64 s; worst: 0:20.0 talking mouth, dub silent (0.4 s), 0:49.3 talking mouth, dub silent (0.4 s), 0:21.8 closed mouth, dub speaks (0.4 s), 0:24.1 talking mouth, dub silent (0.3 s)

## 04: gate FAIL (3 rows)

| gate row (FAIL / better) | NEW / PREV |
|---|---|
| SyncNet on dub-speech seconds (conf at 0) | 6.215 / 5.819 better |
| mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks) | 2.600 / 0.770 **FAIL** |
| slide nose-base px | 0.132 / 0.154 better |
| window joints ratio | 0.909 / 0.967 better |
| window joints LF ratio | 0.865 / 0.963 better |
| lip flash strong | 14 / 10 **FAIL** |
| chin wave worst 2 s | 4.195 / 3.629 **FAIL** |
| video s / video-min | 745.3 / 620.7 **FAIL** |

Slide (lower face vs the head, fixed shaky windows): 0:25.0-0:31.6: 0.105 vs 0.177 px; 0:39.0-0:47.6: 0.159 vs 0.132 px
Lip flash candidates: 14 vs 10.
Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): 0.77 s (v4-repro) -> 2.6 s; worst: 0:03.9 closed mouth, dub speaks (1.5 s), 0:51.5 closed mouth, dub speaks (0.4 s), 0:01.8 talking mouth, dub silent (0.4 s), 0:28.6 talking mouth, dub silent (0.3 s)


## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= v4 on all 5 + the top-20 human review in `r6_m5_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 8.86 | 10.16 | 12.12 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 52.75 | 42.45 | 47.65 |
| 04 | 47.28 | 46.09 | 50.49 |
| 05 | 32.97 | 8.19 | 18.99 |

**n_events** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 6 | 7 | 8 |
| 02 | 0 | 0 | 0 |
| 03 | 41 | 32 | 37 |
| 04 | 25 | 23 | 23 |
| 05 | 15 | 7 | 12 |

**n_events_ge2** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 1 | 1 | 2 |
| 02 | 0 | 0 | 0 |
| 03 | 1 | 1 | 3 |
| 04 | 6 | 9 | 7 |
| 05 | 5 | 0 | 2 |

**top20_mean** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 1.476 | 1.451 | 1.514 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 1.461 | 1.467 | 1.484 |
| 04 | 2.085 | 2.15 | 2.368 |
| 05 | 2.198 | 1.17 | 1.582 |

**p99** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 0.701 | 0.692 | 0.63 |
| 02 | 0.0 | 0.0 | 0.0 |
| 03 | 1.194 | 1.122 | 1.095 |
| 04 | 1.687 | 1.62 | 1.977 |
| 05 | 1.247 | 0.797 | 0.996 |

**top events** (t s: score channel cause head-speed)
- v4 01: 1.2s 2.4 lips - h1.65; 0.2s 1.8 skin_motion - h1.64; 4.5s 1.3 skin_motion - h1.41; 0.6s 1.1 lips - h0.94; 9.3s 1.1 lips - h0.96
- v4 02: 
- v4 03: 52.6s 2.3 lips - h0.34; 42.8s 1.9 lips - h0.17; 70.7s 1.6 lips - h0.17; 5.3s 1.6 lips - h0.1; 17.7s 1.6 lips - h0.08
- v4 04: 30.7s 4.0 lips - h1.72; 11.5s 3.9 lips - h1.25; 33.3s 3.6 lips - h2.85; 9.0s 3.1 lips - h1.4; 39.3s 2.6 lips - h2.22
- v4 05: 25.8s 7.6 lips - h0.63; 39.1s 6.2 lips - h1.8; 28.9s 2.3 skin_motion - h1.15; 38.2s 2.1 lips - h1.64; 24.4s 2.0 skin_motion - h0.41
- m4 01: 1.2s 2.4 lips inside_ls h1.65; 0.2s 1.7 skin_motion inside_ls h1.64; 7.6s 1.3 lips inside_ls h1.05; 9.4s 1.3 lips inside_ls h0.96; 4.5s 1.3 skin_motion window_joint h1.41
- m4 02: 
- m4 03: 14.2s 3.2 lips inside_ls h0.08; 52.6s 1.8 lips inside_ls h0.34; 68.3s 1.8 lips window_joint h0.03; 70.7s 1.6 lips window_joint h0.17; 63.2s 1.6 lips inside_ls h0.1
- m4 04: 22.4s 3.9 lips window_joint h1.17; 11.5s 3.6 lips inside_ls h1.25; 6.9s 3.3 lips inside_ls h2.13; 33.0s 3.2 lips inside_ls h2.42; 27.8s 2.8 lips inside_ls h2.49
- m4 05: 46.6s 1.3 lips inside_ls h1.05; 38.8s 1.3 lips inside_ls h1.74; 38.3s 1.2 lips inside_ls h1.14; 28.8s 1.2 lips window_joint h1.41; 48.5s 1.1 lips inside_ls h0.69
- m5 01: 1.2s 2.6 lips inside_ls h1.65; 6.8s 2.2 lips inside_ls h0.67; 4.2s 1.6 lips inside_ls h1.71; 8.6s 1.3 lips inside_ls h0.45; 0.1s 1.2 skin_motion inside_ls h1.44
- m5 02: 
- m5 03: 37.8s 2.2 lips inside_ls h0.06; 47.8s 2.2 lips inside_ls h0.19; 42.5s 2.1 lips inside_ls h0.4; 52.6s 1.7 lips inside_ls h0.34; 70.7s 1.5 lips inside_ls h0.17
- m5 04: 33.4s 10.0 lips inside_ls h2.74; 7.9s 4.7 lips inside_ls h2.02; 9.1s 4.2 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.8 lips inside_ls h1.44
- m5 05: 26.9s 4.1 lips inside_ls h1.23; 20.2s 2.8 lips inside_ls h2.41; 27.8s 1.8 lips inside_ls h1.63; 38.8s 1.4 lips inside_ls h1.74; 28.9s 1.4 skin_motion inside_ls h1.15

**Twitch, honestly:** with the new audio the generator produces different lip motion, so the detector's "lips" channel (mouth-shape snaps at
speech onsets / offsets, present in v4 too) moves: 05 19.0 (m4 8.2, v4 33.0), 04 50.5 (m4 46.1, v4 47.3). On 05 the two strongest events
(0:26.9, 0:20.2) are a mouth closing within 2 frames, checked on frame grids - no paste smear anywhere (the twitch fix holds: 15 of 122
sanitizer fixes kept back by the alignment guard). 04 is above v4 by 3 points on this channel; 04 is the user's lowest-priority video and its
events are the same kind (lip-shape changes during fast head turns, head speed > 2 iod/s). Human review: `r6_m5_twitch/<id>_m5_top20.jpg` +
`<id>_m5_top20_halfspeed.mp4`.
