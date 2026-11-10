# Per-segment best-variant selector: prototype (Q, day 4)

- **Code:** r16-c:/workspace/dub/k1/q_sel/select_compose.py
- **Composites:** /workspace/dub/k1/q_sel/{m03,old03}_sel3.mp4, with cut lists `*.cuts.txt` and plans `*.plan.json` next to them.

## Rule

1. **Score every frame.** For each variant, badness = the max of Q's calibrated checks, each scaled so 1.0 = its threshold:
   - ghost chin
   - black hole (relative to the source)
   - jaw-seam wobble
   - soft mouth
   - lip shape
   - tail-like break (jump vs the 2 s before)
2. **Build segments.** Take the frames where the default variant is ≥ 1. Pad by 0.2 s, merge gaps under 1 s, and use at least 0.6 s. Never switch back within the last 1 s of the video, because a jump there makes a new tail.
3. **Pick a variant per segment.** Take the one with the lowest badness on the flagged frames, if it is ≤ 0.8× the default and loses ≤ 1.5 of K1's per-second speech@0. The source is a candidate only when the whole segment is in dub silence.
4. **Crossfade.** Variants blend in and out with 0.32 s crossfades.

## Results (A2 / A8b = K1's arms; A9 = the new lf-warp arm, rendered about 10:10)

| video | variant | switched | ghost s (p95) | black fr | seam | tail | lip-thin runs |
|---|---|---|---|---|---|---|---|
| m03 | A2 | – | 0.4 (0.12) | 25 | 1.15 | 3.03 | 0 |
| m03 | A8b | – | 0.0 (0.05) | 22 | **0.13** | 2.03 | 1 |
| m03 | A9 | – | 0.0 (0.04) | 21 | 0.92 | 3.36 | 1 |
| m03 | **sel3 = A2 + A8b on 5 segments** | 10.1 s | 0.0 (0.08) | 25 | 0.87 | **1.84** | 1 |
| m03 | A8b as the default + A2 / A9 | 1.7 s | (estimate: no gain, 2.5 → 2.8 flagged s) | | | | |
| old03 | A2 | – | 12.2 (0.95) | 2 | 1.04 | 1.28 | 0 |
| old03 | A8b | – | 12.2 (2.69) | 4 | **0.07** | 1.11 | 0 |
| old03 | A9 | – | **33.7 (3.76)** | 4 | 1.27 | 1.15 | 0 |
| old03 | **sel3 = A2 + A8b on 4 segments** | 25.9 s | **9.2 (1.12)** | 5 | 0.64 | **1.02** | 0 |

old03 has soft-mouth runs (30) in every variant, so the selector cannot fix that; it comes from the generator. The m03 "sel" from v1 / v2 used K1's side-test A8 (feather 0.08), so v3 replaces it.

## Reading

- **m03: the selector buys little over A8b alone.** sel3 has the best tail (1.84 vs A8b's 2.03), but A8b alone has a much lower seam (0.13 vs 0.87). With A8b as the default, only 1.7 s would switch. **m03 → just A8b.**
- **old03: the selector helps.** It has the fewest ghost seconds (9.2 vs 12.2) and the best tail (1.02), at seam 0.64. A8b alone has the lowest seam but the ghost p95 is high (2.69: worse moments).
- **A9 never wins a segment.** On old03 its ghost is 33.7 s, the worst. The lf-warp source lower face makes chin-edge structure, so by eye this is a K1 question.
- **Speech@0** on the composites: K1 runs eval_final; K1's numbers will be added here.
- **Not fixed:** the composites are 1–3 frames short at the end (40.47 / 90.30 s vs 40.50 / 90.32 s) because the decoder's 25 fps grid drops the last frames.
- **Recommendation:** the selector is worth shipping only where no single variant wins (old03-like videos). Default A8b, and switch only for a ghost or tail-break segment.

## Speech@0 (K1 eval_final, full length)

| video | A2 | A8b | A9 | A9n | sel3 (A2 base) | sel3b (A8b base) |
|---|---|---|---|---|---|---|
| m03 | 8.251 | 7.678 | 7.809 | 7.815 | 8.140 (−0.11) | 7.652 |
| old03 | 7.766 | 7.107 | 7.252 | 7.189 | 7.564 (−0.20) | 7.341 |

A9n (tight zero ring) halves old03's ghost (p95 2.49, 21 runs) but stays far above A2 (0.95, 15 runs), so lf-warp is not a fix for old03's chin.

## Decision (Irina, 10.11)

She judged the clips "на мужике всё ок, на маме нет", so the selector ships only on old03, as **sel4**:

- A2 everywhere;
- A8b on 47.4–49.4 s (ghost; sync −0.07);
- the source in the silent first 2.0 s and last 1.6 s.

The v4 rule is a sync guard of −0.3 per piece, with pieces ≤ 3 s. T builds it on the final old03 layer with `apply_cuts.py` + `old03_sel4.cuts.json`. m03 stays A2 + tail fade.

## Clips

`clips/` holds 0.5x "A2 | selected | source" clips from v1. The m03 v1 clips include K1's side-test A8 at 3.0–4.2 and 29.0–30.2 s; v3 uses A8b there.
