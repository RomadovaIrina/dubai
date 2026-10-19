# 03: the two human-marked twitch windows (24-25 s, 35-36 s) — what happens there

Source of the marks: the user watched baseline (cand5) 03 and flagged 24-25 s (frames ~600-625) and 35-36 s (~875-900) as the strongest twitching.
Evidence: face-stabilised lower-face crops source / cand5 / C2T at full resolution, per-frame difference heat maps, the run manifest
(dub placements, source VAD, LS segments), `mouth_jitter_hr.py` v4 per-second table (`metric_validation/03_cand5_v4.json`).

## What is on screen

| window | dub (placed TTS) | source speaker | LatentSync mouth in cand5 | LS window boundaries | LS <-> original switch |
|---|---|---|---|---|---|
| 24-25 s | "the start" 23.90-24.40 s, then **silent until 25.50 s** | speaks until 25.0 s (teeth visible at frames 612-614) | closed through 598-603 while "the" is spoken, **snaps open exactly at the window boundary 604**, wide at 606, shut again at 610, then **frozen shut** while the source keeps talking (610-625) | 604, 620 | none (LS segment 76-1114) |
| 35-36 s | "...the desired" ends 34.22 s, **silent until 35.90 s** ("result that we wished for") | speaks until 35.5 s with large jaw drops (teeth at 875-879) | **frozen shut** for the whole gap; lips pink and still | 876, 892 | none |

Reading: both windows are **dub-silent stretches inside the speech gate while the source speaker still talks**. LatentSync, driven by
silence, holds the mouth closed and still, but the face around and under the feathered paste mask is the SOURCE, which keeps articulating
(jaw, cheeks, nasolabial folds, chin outline). A still mouth inside a talking face reads as facial twitching ("мимика"); at the speech
onsets the mouth snaps open at a 16-frame window boundary (604). The audio report already counts such frames ("LS frames dub-silent while
the original speaks": 251 on 03), but no face metric connected them to the visible defect.

Side note (not the cause): the source is a steady 30 fps phone video with hand-held motion; the 30 -> 25 fps master drops one frame in six,
so head motion arrives in uneven steps in the source and the output alike.

## Why the metric missed it, and the extension

`resid` (texture popping) and `d1` rank these seconds average (24 s: 38 / 90, 35 s: 77 / 90): the cand5 mouth is *calm* there — too calm.
`mouth_jitter_hr.py` v4 adds, per second with ranks:
* **frozen_mouth** = (lower-face motion / mouth motion) of the output over the same ratio of the source; > 1 = the mouth is stiller than the
  face around it, relative to the source. 03: **34 s rank 8, 35 s rank 9, 24 s rank 22** of 90; 8 of the top-12 seconds are dub-silent
  stretches with the source speaking (37, 50, 66, 49, 76, 34, 35); correlation with the dub-silent-while-source-speaks fraction: 0.43.
* **shear / shear95** = spatial gradient of the head-relative flow in the lower-face ring (a seam where pasted and source skin move
  differently); 03 top seconds 78, 81, 8.
* **max_d1_jump** = the largest single-frame mouth change of the second / median (snaps).
Controls: identity = 1.000 for every new value, per second.

## What this track can do about it (clean, post-process / transitions)
1. In dub-silent stretches where the source speaks, make the whole lower face coherent: grow the paste mask (MediaPipe face contour below
   the nose) so the jaw / cheeks come from the still LatentSync frame, with the source's skin texture transferred onto that geometry —
   a still face instead of a still mouth inside a talking face.
2. Alternative for the human A/B: pass the original through in those stretches (lips move without voice), with eased transitions.
3. C2 boundary interpolation for the onset snaps (604).
The root cause (the dub is shorter than the source speech there) is on the audio side; the coordinator decides whether to raise it.
