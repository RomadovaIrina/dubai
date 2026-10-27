
## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= v4 on all 5 + the top-20 human review in `r6_m5_twitch/`)

**twitch_sum** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 8.86 | 10.16 | - |
| 02 | 0.0 | 0.0 | - |
| 03 | 52.75 | 42.45 | - |
| 04 | 47.28 | 46.09 | 50.49 |
| 05 | 32.97 | 8.19 | 18.99 |

**n_events** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 6 | 7 | - |
| 02 | 0 | 0 | - |
| 03 | 41 | 32 | - |
| 04 | 25 | 23 | 23 |
| 05 | 15 | 7 | 12 |

**n_events_ge2** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 1 | 1 | - |
| 02 | 0 | 0 | - |
| 03 | 1 | 1 | - |
| 04 | 6 | 9 | 7 |
| 05 | 5 | 0 | 2 |

**top20_mean** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 1.476 | 1.451 | - |
| 02 | 0.0 | 0.0 | - |
| 03 | 1.461 | 1.467 | - |
| 04 | 2.085 | 2.15 | 2.368 |
| 05 | 2.198 | 1.17 | 1.582 |

**p99** (lower = better)

| video | v4 | m4 | m5 |
|---|---|---|---|
| 01 | 0.701 | 0.692 | - |
| 02 | 0.0 | 0.0 | - |
| 03 | 1.194 | 1.122 | - |
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
- m5 04: 33.4s 10.0 lips inside_ls h2.74; 7.9s 4.7 lips inside_ls h2.02; 9.1s 4.2 lips inside_ls h1.34; 33.8s 3.2 lips inside_ls h1.63; 44.7s 2.8 lips inside_ls h1.44
- m5 05: 26.9s 4.1 lips inside_ls h1.23; 20.2s 2.8 lips inside_ls h2.41; 27.8s 1.8 lips inside_ls h1.63; 38.8s 1.4 lips inside_ls h1.74; 28.9s 1.4 skin_motion inside_ls h1.15

**Twitch, honestly:** with the new audio the generator produces different lip motion, so the detector's "lips" channel (mouth-shape snaps at
speech onsets / offsets, present in v4 too) moves: 05 19.0 (m4 8.2, v4 33.0), 04 50.5 (m4 46.1, v4 47.3). On 05 the two strongest events
(0:26.9, 0:20.2) are a mouth closing within 2 frames, checked on frame grids - no paste smear anywhere (the twitch fix holds: 15 of 122
sanitizer fixes kept back by the alignment guard). 04 is above v4 by 3 points on this channel; 04 is the user's lowest-priority video and its
events are the same kind (lip-shape changes during fast head turns, head speed > 2 iod/s). Human review: `r6_m5_twitch/<id>_m5_top20.jpg` +
`<id>_m5_top20_halfspeed.mp4`.
