# r16 K2 step 0: making e2e reproduce the accepted look (base: experiment/r16-int 66d6041)

## 1. refpick (k2_refpick.py)
- The face-position term is added on top of int's yaw / roll / light terms: −8·max(0, centre distance / face height − 0.08).
- **`--face-lm lm_<tag>.npz --box box.json`.** A frame is a candidate only if:
  - this face is present in the master track (finite landmarks), AND
  - the crop landmarks are that face (their centre lies within 0.25 face heights of the master track projected through the box).
  - This fixes audit orch. 10: the reference can no longer be another person or an empty crop. ks_face.sh passes both arguments.
- **The hard-coded /workspace/dub/dubai path is removed.** Without `--lm`, mp_landmarks.py is found next to the repo and `$DABAI_ROOT` is used.
- **Line endings:** kept CRLF like int's copy, so the diff shows only the real change.
- **Picks:**

| video | before | after |
|---|---|---|
| m03 | 531 | 531 |
| old03 | 1518 | 1518 |
| m18 | 158 | 158 |
| q02, tracked crop | 655 | 655 (the frame in the accepted golden) |
| q02, static crop | — | 755 |

## 2. ks_fast.py / ks_plain.py
Not touched; int's copies stay.

## 3. Multi-shot (ks_shots.py + ks_shots.sh, hooked into ks_face.sh; KS_SHOTS=auto by default, `off` = one run)
- **detect** runs after ks_face's master and landmarks:
  - **Hard cuts:** HSV-histogram distance > 0.35, more than 4× the neighbour median, and luma change > 0.10.
  - **Faceless gaps** over 2 s (= the bridge limit) between face runs; the bound goes in the middle of the gap. This catches fades and dissolves through a title card.
  - **Per shot, KeySync runs only from the first to the last face frame.** KeySync crashes on a clip that starts faceless: "No valid landmarks found" (q03 shot 2).
  - **Fade frames at those edges stay source.** A transition zone = at least 3 frames with d > 0.08 within ±5 frames. Static talking heads never form one (≤ 2 isolated frames per clip).
- **ks_shots.sh** runs ks_face.sh once per face segment:
  - each segment gets its own plan, crop with track smoothing, ref, bridge, KeySync job, ECC / tsmooth / snap / guard, and paste with lfm / align / colour match. Nothing crosses a cut (audit orch. 5, paste 12, prep 4).
  - Inputs are reused: a lossless cut of the master and the sliced MediaPipe npz (no second landmark pass).
  - Route and manifest are cut and shifted to each segment.
  - Longest segment first; the first successful delay probe sets KS_DELAY for the rest.
  - KS_HELD_FD is handed over after the first GPU section.
  - Source segments are encoded exactly like ks_plain's output, then joined with concat `-c:v copy`.
  - The final mix is stream-copied (md5 of the audio packets is identical; no `-shortest`).
- **Single-segment videos** (m03, old03, m18, q01, q02) take the unchanged path.
- **q03:**

| range | treatment |
|---|---|
| 0–22.76 s | KeySync, ref 349 |
| 22.76–30.40 s | source (title fade, title card, dissolve) |
| 30.40–37.72 s | KeySync, ref 60 |
| 37.72–45.00 s | source (fresco cutaway) |

  - 1125/1125 frames, ~7 min wall including two model loads.
  - Int's single run (KS_SHOTS=off) crashes on q03: ks_tsmooth hits NaN landmarks over the title card (audit item 10).
- **SyncNet speech@0, full video, vs the original dub:**

| render | speech@0 |
|---|---|
| multi-shot, KS_DELAY probe (+40 ms) | 3.97, av_offset +1 |
| delay 0 + int's auto shift | 5.69 (shot B pasted at +1) |
| **delay 0 + shift 0** | **6.03** |
| final with fade trimming | 5.83 (the fade frames are source mouth vs dub, by design) |
| hand-made r16 K2 | 6.39 |

  - On the lip-synced frames only: shot A 6.82 vs 7.23; shot B (30.4–37.72 s) 5.23 vs 5.36.
- **Still open:**
  - a dissolve directly from face A to face B (no faceless gap) is not split;
  - the source → KeySync switch at a fade end is a hard cut on a visible face (q03 30.4 s), per Irina's no-fade rule. Needs her eye.

## Finding for T (int's delay / shift chain, audit orch. 3 / 4)
- On q03 the probe chose +40 ms, which made the full video +1 frame late (3.97).
- Int's auto paste shift moved shot B the wrong way: +1 went in, +1 came out.
- KS_DELAY=0 KS_PASTE_SHIFT=0 was best here. The hand-made renders use −(raw offset vs the original dub) per video.

## 4. Golden
outbox/r16_k2/golden/q02_keysync_r16_route_track.mp4 (md5 32fe415a…) + README.md (the full recipe).
