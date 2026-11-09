# Irina's 09.11 defects as automatic checks (Q, r16 day 3)

`scripts/quality/r16q_defects.py --src SOURCE --ours FINAL --out X.json [--npz det.npz]`, CPU only (MediaPipe on the 512 face crop, reusing r16q_detect's geometry cache). About 2–3 min per 45 s video on 1 core; 11 videos run in parallel. `--from-series X.series.npz` re-summarises without decoding.

Calibrated on the 11 e2e videos of 08–09.11 against ФИДБЕК_0911 (`defects_calib.py DIR` → CALIB.md).

**Agreement: tail 8/8, outside 9/9, chroma 7/7, occlusion 7/7** (long5: LONG5_RESULT). Small set: 2–6 videos per class, thresholds sit in the gap between them, so treat the margins below as the real confidence.

## The checks

| check | what it measures | flag rule | bad (Irina) | fine | margin |
|---|---|---|---|---|---|
| **tail** | Local jump at the end: max over the last 1 s (excluding the final frame) of a 4-frame mean of `res_out` (Lab residual ours−source on the lower face outside the mouth) and `jaw_dev` (ours' jaw landmarks vs the source's, % IOD), divided by the median over the 2 s before. | ratio ≥ 1.5 | m03 2.50 (res_out), m18 2.09, q01 2.14, q01_subs 1.92 (jaw) | q04 1.03, old03 1.13 | 1.13 → 1.92 |
| **outside** (mask outside the face) | Temporal wobble of the residual in rings 0.05–0.45 IOD outside the face oval (left / right at cheek level and below; neck under the jaw), divided by the same wobble inside the cheeks. Untouched source pixels don't wobble; regenerated ones do. | any band p95 ≥ 1.2 | q03 neck 2.58, q05 neck 2.03, old03 right 1.29 / neck 1.21 | max 1.11 (q01_subs neck); m03 1.07, m18 0.97, q01 1.02, q04 0.77 | 1.11 → 1.21 (thin) |
| **chroma** (blue darks) | At ours' dark lower-face pixels (L* < 30): source b* − ours b* (> 0 = ours bluer) and the share with b* < −4. | per frame blue > 4 and share > 0.25; flag if ≥ 2 % of frames | long5 (pending) | all others 0.0 % | m03 5.5 % (Irina: "shadow on the cheek", no verdict) |
| **occlusion / turns** (where we should keep the original) | (a) **painted over an object**: ours' jaw landmarks leave the source's, 1-s rolling median > max(10 % IOD, 4× the video median); (b) **turns**: \|yaw\| > 0.45 for ≥ 0.5 s; (c) colour occluder vs the frame's cheek skin, relative to the video's own median (beards are constant). | any run | q05: paint 11.1–13.2 s (the cup) + turns 11.8–13.3 / 17.2; q02: turns 8.6–30.4 s | m03, q01, m18, old03, q04: none | |

## Seam jitter at the jaw (09.11 evening, Irina: "the neck↔face transition jitters in ALL videos")

One definition, agreed with K1:
- **Band:** K1's jaw line (MediaPipe 58…152…288) ±0.08 IOD, drawn per frame on the source landmarks.
- **`seam.excess` (primary):** the frame-to-frame change of the residual (final − source) in the band, minus the same in a forehead ring that is never pasted (the codec floor). Units: L* per frame. Head motion cancels.
- **`seam.k1_ratio`:** K1's raw definition, |ΔV| of the final / |ΔV| of the source. It carries head motion and barely separates the arms, so it is info only.
- **Output:** median, p95, worst seconds, and a per-second series. The codec floor itself is 0.25–0.5.

**Baseline, the 11 delivered e2e** (excess median / p95, worst seconds):

| video | median | p95 | worst s |
|---|---|---|---|
| q03_e2e_alpha_full | 2.56 | 5.21 | 18, 16, 19, 21 |
| q05_e2e_full | 1.99 | 5.88 | 13, 4, 10, 43 |
| f0811_q02 | 1.61 | 6.02 | 35, 34, 36, 33 (the hand) |
| full_old03 | 1.41 | 2.89 | 37, 80, 64, 31 |
| f0811_q01 | 1.39 | 3.30 | 18, 12, 15, 13 |
| f0811_m03 | 1.35 | 3.62 | 29, 11, 40, 27 |
| f0811_q01_subs | 1.32 | 3.19 | 18, 12, 15, 13 |
| full_m18 | 1.05 | 3.10 | 20, 18, 6, 13 |
| f0811_q04 | 0.81 | 2.24 | 10, 19, 30, 24 |
| f0811_q06 | 0 (no KeySync) | | |
| long5 | LONG5_SEAM | | |

**K1's mask arms** (r16-k1 k1c, each against its own KeySync master; excess median / p95):

| clip | now | tight | tightp (parse) |
|---|---|---|---|
| q03a | 2.59 / 5.11 | 0.33 / 0.58 | 0.33 / 0.55 |
| old03 | 1.49 / 3.00 | 0.11 / 0.43 | 0.11 / 0.40 |
| q01 | 1.35 / 3.16 | 0.24 / 0.59 | 0.24 / 0.54 |
| q04 | 0.86 / 2.47 | 0.26 / 0.84 | 0.24 / 0.76 |
| l5w | 1.30 / 3.61 | 0.18 / 0.72 | 0.17 / 0.60 |
| m03 | 1.33 / 3.59 | **1.28 / 3.54** | **1.28 / 3.54** |

- The tight mask cuts seam jitter by 80–90 % on 5 of 6 clips.
- **m03 is unchanged** (only the outer half of the band dropped, 1.12 → 0.78; the inner half stays 1.74). Asked K1 whether the two-face path bypasses the tight mask.
- Suggested pass bar for the fix: median ≤ 0.35 (near the codec floor).
- On tight arms the outside-wobble flag can still fire (q03a 1.34, old03_tightp 1.23): it is a ratio against the inside cheeks, which tight also calms. For mask arms, read the seam excess.

## What it does not do yet (honest)

- **The q02 hand at 34.8–38.6 s is not detected as such.** q02 flags only through the turn runs. A skin-coloured hand passes the colour cue, and the generator doesn't move the jaw landmarks there. Needs face parsing (MIT) or a mouth-visibility cue; not done.
- **The old03 outside margin is thin** (1.21–1.29 vs 1.11 on a fine video). The 1-s runs don't localise it to 64–66 s; the p95 over the video does flag it. Re-check after K1's tight mask: it should drop below 1.0.
- Thresholds are fitted on these 11 videos only. A new clip set (K1's tight-mask renders, tonight's re-render) is the first real test.
- The tail rule needs ≥ 3 s of face before the end; q03 (no face at the end) and q06 (no KeySync) are n/a.

## Per video (CALIB.md has the full table)

| video | tail | outside | chroma | occlusion / turn |
|---|---|---|---|---|
| f0811_m03 | **FLAG** 2.50 @39.6 s | – | 5.5 % (shadow?) | – |
| full_m18 | **FLAG** 2.09 @20.0 s | – | – | – |
| f0811_q01 | **FLAG** 2.14 @44.6 s | – | – | – |
| f0811_q01_subs | **FLAG** 1.92 @44.7 s | – | – | – |
| f0811_q04 | – | – | – | – |
| full_old03 | – | **FLAG** right 1.29 / neck 1.21 | – | – |
| q03_e2e_alpha_full | n/a | **FLAG** neck 2.58 (0–23 s, 29–32 s) | – | – |
| q05_e2e_full | – | **FLAG** neck 2.03, left 1.51 (11.6–15.5 s) | – | **FLAG** paint 11.1–13.2, turns |
| f0811_q02 | (1.33 / 2.56, no verdict) | – | – | **FLAG** turns |
| f0811_q06 | n/a (no KeySync) | – | – | (profile turns; source kept) |
| long5 | LONG5_ROW |

## How to use it in the pipeline / on new renders

- Per final: `r16q_defects.py --src <job input> --ours <final>`. JSON fields: `tail.flag`, `outside.flag`, `chroma.share`, `occlude.{paint_runs, turn_runs, occl_runs}`.
- The tail and outside flags are regressions to block on. The occlusion / turn runs are the frame ranges where Irina's rules say keep the original (T / K1 can feed them to the gate).
- Run files on r16-k2: `/workspace/dub/q_tmp/{run_defects.sh, resum.sh, defects/}`.
- long5's source was rebuilt by the coordinator (T's recipe), not the original file: see the long5 note.
