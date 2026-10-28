# r7 F (face + merged E2E): status log

## 1. 05 25-26 s "rectangle with the face pasted over the trees" = the SOURCE's own picture-in-picture inset (28.10, 09:30)
Test video 05 is a phone dual-camera recording: the main camera films the park, a selfie inset with the speaker (and the inset's own
background) sits in the top-right corner for the WHOLE clip (MediaPipe: face in 1511/1567 frames, always at x 330-396 of 464).
The published m5 05 WATCH differs from the source only at the mouth inside the inset: whole-frame gate PASS, worst frame 0.000 % of the
frame changed outside the mouth region. Evidence for the user: `outbox/r7_05_pip_check/05_src_vs_m5_24.5-26.5s_wholeframe.jpg` (SRC | M5 | diff,
11 frames) and `05_src_vs_m5_24-27s_sbs.mp4`.
Generic guards kept anyway (the new test video's block 9 pans away from the face):
- `fps_composite.py --paste-guard 0.5`: the layer is pasted only when the CURRENT source frame holds the face at the paste position (grey NCC
  source vs registered + colour-matched layer on the paste region outside the lips >= 0.5), otherwise pass-through (counted); `--max-hold 2`.
- `scripts/r7/r7_frame_gate.py` / `.sh`: per frame, % of the frame changed (|out - src| > 24 grey) outside the source lips hull + 0.6 IOD;
  PASS = no frame > 0.2 %. Top-12 whole-frame sheets. Part of `r7_e2e.sh`.

### Whole-frame gate on the published versions (all 5 videos; 01 = small distant face, 02 = no face at all)
| version | id | verdict | worst frame | worst % | frames > 0.2 % | face frames |
|---|---|---|---|---|---|---|
| m2 | 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 260/2542 |
| m2 | 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0/3324 |
| m2 | 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 2710/2710 |
| m2 | 04 | PASS | 111 (3.7 s) | 0.001 | 0 | 1617/1617 |
| m2 | 05 | PASS | 1204 (40.13 s) | 0.0 | 0 | 1511/1567 |
| m3 | 03 | PASS | 2639 (87.97 s) | 0.0 | 0 | 2710/2710 |
| m3 | 04 | PASS | 111 (3.7 s) | 0.001 | 0 | 1617/1617 |
| m3 | 05 | PASS | 1197 (39.9 s) | 0.0 | 0 | 1511/1567 |
| m5 | 05 | PASS | 1502 (50.07 s) | 0.0 | 0 | 1511/1567 |
| r4_video_v1 | 01 | PASS | 2523 (84.1 s) | 0.005 | 0 | 260/2542 |
| r4_video_v1 | 02 | PASS | 3288 (109.6 s) | 0.0 | 0 | 0/3324 |
| r4_video_v1 | 03 | PASS | 2001 (66.7 s) | 0.0 | 0 | 2710/2710 |
| r4_video_v1 | 04 | PASS | 164 (5.47 s) | 0.001 | 0 | 1617/1617 |
| r4_video_v1 | 05 | PASS | 1309 (43.63 s) | 0.0 | 0 | 1511/1567 |

## 2. 03 ~7 s seam / blur: metric `scripts/r7/r7_seam.py` (baseline m3 03)
seam (|grad out| - |grad src| on the footprint boundary band) mean -0.35 p95 0.32; colour step mean 2.97 p95 3.84;
skin blur (|Laplacian| out / src inside the footprint, outside the lips) mean 0.721 p05 0.576 -> the regenerated skin is ~28 % softer than
the source everywhere, worst frames ~0.47 (15.6 s, 34.1 s, 62.8 s). On the fast frames (head > 0.3 IOD/s, 77 fr): blur 0.707, seam -0.41.
At 7 s itself (frames 195-245) blur 0.6-0.8, no seam spike: the user's "seam" is most likely the softness step at the footprint edge.
