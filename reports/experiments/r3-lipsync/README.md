# Round 3, track A — lip-sync (generator side): results

Audio: every row says which. Round-3 deliverables use track C audio-r2 / audio-r3 (Spleeter bed); cand5 audio = reference only.
Full arm table (all metrics): `table_arms.md`. Changelog: `CHANGES.md`. Metric definitions: `scripts/quality/r3_eval.py`, `r3_table.py`.

## 1. Reproduction (task 1) — full videos, 25 fps, r2best vs clean0

| video | audio | arm | SyncNet dub / mix | per-s conf (dub-speech s) mean / p10 | lip-dub r | mouth vel | nose slide | joint jump med / >1.5 | LS s/video-min |
|---|---|---|---|---|---|---|---|---|---|
| 03 | cand5 | r2best | 5.75 / 5.51 | 7.18 / 5.47 | 0.45 | 0.71 | 0.079 | 1.67 / 0.69 | 304 |
| 03 | audio-r2 | clean0 | 5.79 / 5.55 | 7.14 / 4.93 | 0.46 | 0.79 | 0.156 | 1.56 / 0.53 | – |
| 03 | audio-r2 | r2best | 5.79 / 5.54 | 7.12 / 5.11 | 0.46 | 0.72 | 0.079 | 1.65 / 0.63 | 359* |
| 05 | cand5 | r2best | 3.70 / 3.21 | 4.65 / 3.95 | 0.20 | 0.87 | 0.058 | 1.56 / 0.53 | – |
| 05 | audio-r2 | clean0 | 3.30 / 2.76 | 3.89 / 2.58 | 0.24 | 1.01 | 0.229 | – | – |
| 05 | audio-r2 | r2best | 3.05 / 2.58 | 3.71 / 2.66 | 0.21 | 0.86 | 0.058 | 1.50 / 0.50 | 251* |

Coverage (audio-r2): 01 260, 03 2030, 04 1074, 05 1025 LatentSync frames; all PASS, spec-11 status SUCCESS. *GPU shared with track C
(LatentSync +~40 % when C's jobs run) and an oversubscribed CPU. Source joint jump at the same frames: 0.96 / 0.17 (03), 0.88 / 0.19 (05).

**D's scorecard gap (05 −0.86, 04 −0.34 vs clean0) on the SAME audio:** 05 −0.25 (3.05 vs 3.30), 03 ±0.00. Most of the gap is the audio
difference. The rest is round 2's known effect: clean0's crop follows detector noise, which shows up as extra "mouth" motion SyncNet reads
(05 mouth vel 1.01 vs 0.86, slide 0.229 vs 0.058); round 2 showed it vanishes on head-stabilised clips. Guidance 2.0 (below) gains more than that.

## 2. Lip-sync accuracy (task 2) — 03 window 40-58 s, audio-r2, every arm = r2best + one change

| arm | SyncNet dub | per-s conf mean / p10 | lip-dub r | mouth vel / resid | joint jump med / >1.5 | nose slide | verdict |
|---|---|---|---|---|---|---|---|
| base (r2best) | 5.11 | 6.77 / 5.87 | 0.45 | 0.72 / 1.03 | 1.54 / 0.59 | 0.086 | – |
| fixed reference (calm closed mouth, t=5.7 s) | 5.68 | 7.60 / 6.04 | 0.49 | 0.65 / 1.07 | 1.81 / 0.68 | 0.132 | sync up, slide / joints worse |
| reference = same face 16 frames later | 5.58 | 7.42 / 6.30 | 0.49 | 0.68 / 1.06 | 1.75 / 0.73 | 0.111 | same trade, smaller |
| reference 32 frames later | 5.63 | – | 0.52 | 0.68 / 1.07 | 1.91 / 0.73 | 0.110 | same |
| **guidance 2.0** | **5.66** | – | 0.50 | **0.78** / 1.17 | **1.48 / 0.46** | 0.087 | **winner: sync + articulation, joints better, slide equal** |
| guidance 2.5 | 5.94 | – | 0.56 | 0.85 / 1.36 | 1.44 / 0.46 | 0.096 | more sync, texture change 1.36 (human look needed) |
| **bridge 8x4 (spec 8.5)** | 5.19 | – | 0.46 | 0.71 / 0.98 | **1.07 / 0.23** | 0.085 | **winner: joints at source level** |
| ref 16 later + bridge 8x4 | 5.63 | – | 0.49 | 0.66 / 1.00 | 1.29 / 0.36 | 0.111 | bridge repairs the joints, slide stays |

Reading: the generator under-articulates (mouth vel 0.72 of the source). A reference from another time removes the source-mouth copy (+0.5
SyncNet) but the generated lower face then follows the reference's pose a little (slide up uniformly over all 16 window phases, not at the
seams). Higher guidance raises sync AND articulation without moving the face. Bridges fix the joints independently. -> R3 candidate =
r2best + guidance 2.0 + bridge 8x4 (+ the 400-frame LatentSync call cap), rendered on audio-r3.

## 3. Dub-silent mouth (task 3) — 03 window 21.6-37.6 s, audio-r2

| arm | silent-second frozen_mouth (v4) | SyncNet dub | lip-dub r |
|---|---|---|---|
| base | 1.44 | 5.73 | 0.55 |
| silence mix 0.3 / 0.5 / 0.7 | 1.44 / 1.41 / 1.36 | 5.72 / 5.68 / 5.71 | 0.52 / 0.49 / 0.48 |

The conditioning mix toward the source speech barely moves the frozen-mouth index on this window (only one fully dub-silent second inside
LatentSync with audio-r2; C's audio-r2 cut 03's residual gaps to 4.6 s). Not adopted; the composite-side options (track B: source pass-through /
still lower face with eased ramps) remain the main lever. Needs a window with more residual gaps (05) before closing.

## 4. Spec items

**5.4 / 5.5 head pose** (`pose_5_4.md`): pre-check (> 30 % of face frames |yaw| > 30 deg) fires on no video; 05 6.4 % (max 62.5 deg, runs at
4.0-4.8 / 12.3-13.8 / 19.4-20.3 s), 01-04 0 %, 02 no face. Per-frame gate 30 deg on 05 (audio-r2): SyncNet 3.05 -> 3.21, lip-dub r 0.21 -> 0.13,
joints unchanged -> neutral, not adopted.

**11 robustness** (04, audio-r2, fault injection, worker never crashed, all outputs valid):

| fault | status | held landmark frames | LatentSync-failed frames | degraded | output valid |
|---|---|---|---|---|---|
| LatentSync RuntimeError on one call | PARTIAL_SUCCESS | 0 | 37 | 3.45 % | True |
| 2 % landmarks dropped + three 8-frame gaps | PARTIAL_SUCCESS | 25 | 0 | 2.35 % | True |
| 10 % landmarks dropped (3-frame runs) | PARTIAL_SUCCESS | 114 | 0 | 10.71 % | True |

The 2 % rule counts held frames and failed segments; long gaps split the segment (pass-through, not counted).

**8.5 joints**: see §2 (bridge 8x4: share of joints > 1.5x 0.59 -> 0.23, source 0.23). LatentSync call cap 400 frames (D's VRAM audit): the cap
falls on a 16-frame window seam; measured on 03 (capped vs uncapped, audio-r3) — pending.

**8.2 two faces**: synthetic 03 | 04 clip, speaker left / right — pending (first attempt failed on an env var, re-queued).
