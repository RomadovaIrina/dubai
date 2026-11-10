# Paste ablation on the same KeySync layers (Q, 10.11)

Question (Irina): did the jitter / ghost "cleanup" kill faces that were fine on 08.11?

**Answer from this run: no.** On the same layers, every flagged spot is already there with the old oval paste, and the cleanup steps are neutral or better on every check. What they cost is lip sync (about −0.5 speech@0 in total) and, on q01, more ghost chin. Irina's eye decides on the clips in `clips\`.

## How it was run

- CPU re-pastes of today's layers through T's `r16d_repost.sh`, on private copies of the job dirs (r16-a: m03, m18, old03, q01; r16-b: q04, long5).
- Each arm = the previous one + one change:
  - P0: oval paste, all extras off (08.11-style; only the seam weight differs, per T)
  - P1: + tight mask (no parsing, no cm-shadow)
  - P2: + face parsing
  - P3: P2 + cm-shadow (side arm; today's finals have it off)
  - P4: P2 + occlusion / turn guard
  - P5: + mouth-opening cap 1.0
  - P6: + pause lip-closing
  - today: P6 + tail fade (old03 today = sel4b)
- "08.11 file" = the delivered 08.11 final scored the same way. Its layer and dub differ, so compare it only loosely.
- long5 has P0, P2, P4 only (one re-paste takes 40–105 min); its "today" score failed and is re-running.

## Columns

- black_fr / black_share: black-hole frames, share of open-mouth frames
- ghost_s: seconds of ghost second chin
- seam: jaw-band jitter (codec floor 0.25–0.5)
- edgeL / edgeR: NEW, jitter along the cheek edge from ear to jaw corner, image-left / image-right (image-right = the speaker's left cheek); edgeR_p95 = its 95th percentile
- stL / stR / stJ: NEW, static difference from the source in the side / jaw bands; dLJ: signed lightness shift in the jaw band (negative = darker); darkJ: share of frames with the jaw band > 4 L darker
- out_l / out_r / out_n: wobble outside the face (left, right, neck) relative to inside
- tail: end-of-video jump (≥ 1.5 = broken)
- soft / thin / fat: soft-mouth and lip-shape stretches
- moments: tail-like breaks with rank ≥ 3
- speech0: speech@0 (T's eval, same dub for all arms of a video)

## r16-a

| video | arm | black_fr | black_share | ghost_s | seam | edgeL | edgeR | edgeR_p95 | stL | stR | stJ | dLJ | darkJ | out_l | out_r | out_n | tail | soft | thin | fat | moments | speech0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m03 | 08.11 file | 27 | 0.071 | 0 | 1.364 | 0.723 | 0.589 | 1.489 | 10.462 | 6.429 | 15.22 | -4.413 | 0.556 | 1.05 | 0.9 | 0.78 | 2.49 | 0 | 0 | 0 | 0 |  |
| m03 | P0 oval | 37 | 0.099 | 0 | 1.317 | 0.632 | 0.604 | 1.524 | 9.94 | 6.729 | 14.893 | -4.441 | 0.594 | 0.98 | 0.87 | 0.78 | 2.16 | 0 | 0 | 0 | 0 | 7.943 |
| m03 | P1 +tight | 28 | 0.079 | 0 | 1.157 | 0.202 | 0.236 | 0.684 | 4.276 | 4.048 | 12.335 | -3.684 | 0.436 | 1.05 | 0.84 | 0.79 | 2.71 | 0 | 0 | 0 | 2 | 7.849 |
| m03 | P2 +parse | 29 | 0.081 | 0 | 1.16 | 0.201 | 0.235 | 0.678 | 4.276 | 4.046 | 12.322 | -3.686 | 0.438 | 1.05 | 0.84 | 0.79 | 2.71 | 0 | 0 | 0 | 1 | 7.836 |
| m03 | P3 +cm-shadow (side) | 46 | 0.126 | 0 | 1.211 | 0.209 | 0.245 | 0.708 | 4.467 | 4.644 | 14.006 | -5.456 | 0.747 | 1.07 | 0.83 | 0.78 | 3.29 | 0 | 0 | 0 | 2 | 7.829 |
| m03 | P4 P2+occl | 32 | 0.09 | 0 | 1.114 | 0.249 | 0.248 | 0.676 | 4.883 | 4.338 | 11.831 | -4.218 | 0.563 | 1.08 | 0.89 | 0.83 | 2.67 | 0 | 0 | 0 | 2 | 7.791 |
| m03 | P5 +cap | 18 | 0.052 | 0 | 1.105 | 0.251 | 0.251 | 0.671 | 4.88 | 4.356 | 11.913 | -4.251 | 0.569 | 1.09 | 0.88 | 0.83 | 2.68 | 0 | 0 | 0 | 2 | 7.617 |
| m03 | P6 +pause | 17 | 0.053 | 0 | 1.112 | 0.246 | 0.246 | 0.673 | 4.909 | 4.321 | 11.905 | -4.246 | 0.568 | 1.09 | 0.89 | 0.83 | 2.61 | 0 | 0 | 0 | 2 | 7.623 |
| m03 | today (+tail) | 20 | 0.063 | 0 | 1.02 | 0.211 | 0.209 | 0.591 | 5.313 | 4.681 | 11.867 | -4.534 | 0.611 | 1.28 | 1.09 | 1.03 | 1.44 | 0 | 0 | 0 | 1 |  |
| old03 | 08.11 file | 0 | 0.0 | 7.4 | 1.428 | 0.301 | 0.573 | 1.261 | 6.844 | 9.787 | 12.027 | 3.059 | 0.0 | 0.78 | 1.25 | 1.2 | 1.14 | 30 | 0 | 4 | 18 |  |
| old03 | P1 +tight | 8 | 0.016 | 15.1 | 1.251 | 0.05 | 0.24 | 0.562 | 3.115 | 5.341 | 9.084 | 0.883 | 0.0 | 0.62 | 0.98 | 0.92 | 1.13 | 30 | 0 | 1 | 27 | 6.814 |
| old03 | P2 +parse | 8 | 0.017 | 14.9 | 1.248 | 0.05 | 0.24 | 0.561 | 3.112 | 5.343 | 9.08 | 0.884 | 0.0 | 0.62 | 0.98 | 0.92 | 1.13 | 30 | 0 | 1 | 28 | 6.827 |
| old03 | P3 +cm-shadow (side) | 28 | 0.054 | 13.8 | 1.252 | 0.035 | 0.244 | 0.599 | 3.161 | 5.438 | 9.286 | 0.615 | 0.0 | 0.63 | 1.03 | 0.94 | 1.16 | 27 | 0 | 1 | 27 | 6.989 |
| old03 | P4 P2+occl | 6 | 0.012 | 11.1 | 0.945 | 0.066 | 0.243 | 0.562 | 3.56 | 5.558 | 7.787 | 0.496 | 0.0 | 0.64 | 0.96 | 0.74 | 1.14 | 30 | 0 | 1 | 21 | 6.728 |
| old03 | P6 +pause | 4 | 0.01 | 12.8 | 0.948 | 0.063 | 0.235 | 0.548 | 3.535 | 5.502 | 7.826 | 0.491 | 0.0 | 0.63 | 0.93 | 0.72 | 1.06 | 30 | 0 | 1 | 20 | 6.394 |
| old03 | today (+tail) | 5 | 0.013 | 11.6 | 0.832 | 0.017 | 0.195 | 0.505 | 4.595 | 6.538 | 7.906 | -0.588 | 0.0 | 0.8 | 1.16 | 0.92 | 0.57 | 26 | 0 | 1 | 20 |  |
| m18 | 08.11 file | 14 | 0.088 | 6.3 | 1.053 | 0.928 | 0.594 | 1.383 | 5.321 | 5.178 | 6.127 | 0.421 | 0.012 | 0.8 | 0.92 | 0.97 | 2.09 | 0 | 1 | 0 | 7 |  |
| m18 | P0 oval | 50 | 0.299 | 6.8 | 1.163 | 0.985 | 0.652 | 1.531 | 5.97 | 6.157 | 6.965 | 0.789 | 0.012 | 0.72 | 0.83 | 0.95 | 2.14 | 0 | 1 | 0 | 6 | 7.424 |
| m18 | P1 +tight | 16 | 0.099 | 1.3 | 0.921 | 0.322 | 0.228 | 0.513 | 2.53 | 2.61 | 5.371 | 0.537 | 0.0 | 0.78 | 0.89 | 0.86 | 2.24 | 0 | 1 | 0 | 3 | 7.388 |
| m18 | P2 +parse | 15 | 0.092 | 1.2 | 0.923 | 0.329 | 0.225 | 0.523 | 2.552 | 2.626 | 5.37 | 0.547 | 0.0 | 0.78 | 0.9 | 0.87 | 2.18 | 0 | 1 | 0 | 4 | 7.359 |
| m18 | P3 +cm-shadow (side) | 90 | 0.489 | 0.3 | 1.037 | 0.336 | 0.27 | 0.672 | 2.663 | 3.368 | 6.504 | -0.834 | 0.028 | 0.75 | 0.88 | 0.88 | 2.11 | 0 | 1 | 0 | 3 | 7.15 |
| m18 | P4 P2+occl | 21 | 0.126 | 1.0 | 0.781 | 0.344 | 0.261 | 0.523 | 3.016 | 3.157 | 5.008 | -0.092 | 0.004 | 0.77 | 0.92 | 0.81 | 2.21 | 0 | 1 | 0 | 3 | 7.382 |
| m18 | P5 +cap | 3 | 0.035 | 1.2 | 0.79 | 0.344 | 0.26 | 0.534 | 3.021 | 3.245 | 5.167 | -0.097 | 0.004 | 0.78 | 0.9 | 0.81 | 2.4 | 0 | 1 | 0 | 4 | 7.113 |
| m18 | P6 +pause | 3 | 0.037 | 1.1 | 0.799 | 0.345 | 0.256 | 0.533 | 3.021 | 3.25 | 5.182 | -0.098 | 0.004 | 0.77 | 0.89 | 0.79 | 2.29 | 0 | 1 | 0 | 3 | 7.108 |
| m18 | today (+tail) | 3 | 0.037 | 1.1 | 0.799 | 0.345 | 0.256 | 0.533 | 3.021 | 3.25 | 5.182 | -0.098 | 0.004 | 0.77 | 0.89 | 0.79 | 2.29 | 0 | 1 | 0 | 3 |  |
| q01 | 08.11 file | 6 | 0.009 | 1.9 | 1.395 | 0.569 | 0.431 | 2.154 | 7.831 | 8.369 | 11.448 | 1.6 | 0.0 | 0.75 | 0.87 | 1.01 | 2.02 | 0 | 0 | 1 | 4 |  |
| q01 | P0 oval | 2 | 0.003 | 1.9 | 1.392 | 0.568 | 0.424 | 2.172 | 7.701 | 8.476 | 11.285 | 1.564 | 0.0 | 0.76 | 0.88 | 1.01 | 2.35 | 0 | 0 | 1 | 5 | 7.109 |
| q01 | P1 +tight | 1 | 0.002 | 2.8 | 1.272 | 0.32 | 0.235 | 0.94 | 6.699 | 7.291 | 10.135 | 1.36 | 0.0 | 0.65 | 0.78 | 0.94 | 2.28 | 0 | 0 | 1 | 8 | 6.924 |
| q01 | P2 +parse | 1 | 0.002 | 3.0 | 1.274 | 0.319 | 0.236 | 0.94 | 6.699 | 7.291 | 10.136 | 1.36 | 0.0 | 0.65 | 0.78 | 0.94 | 2.28 | 0 | 0 | 1 | 8 | 6.933 |
| q01 | P3 +cm-shadow (side) | 26 | 0.04 | 0 | 1.356 | 0.321 | 0.245 | 0.934 | 6.705 | 7.357 | 11.184 | -1.156 | 0.016 | 0.65 | 0.77 | 0.96 | 2.26 | 0 | 0 | 1 | 2 | 7.231 |
| q01 | P4 P2+occl | 3 | 0.005 | 5.0 | 1.224 | 0.369 | 0.278 | 1.01 | 6.138 | 6.716 | 9.183 | 0.626 | 0.0 | 0.62 | 0.72 | 0.85 | 2.27 | 0 | 0 | 1 | 15 | 6.967 |
| q01 | P5 +cap | 1 | 0.002 | 5.1 | 1.219 | 0.372 | 0.274 | 1.006 | 6.185 | 6.748 | 9.231 | 0.629 | 0.0 | 0.62 | 0.71 | 0.84 | 2.0 | 0 | 0 | 2 | 16 | 6.696 |
| q01 | P6 +pause | 1 | 0.002 | 6.4 | 1.227 | 0.372 | 0.277 | 1.001 | 6.212 | 6.731 | 9.254 | 0.634 | 0.0 | 0.62 | 0.72 | 0.84 | 1.88 | 0 | 0 | 2 | 13 | 6.705 |
| q01 | today (+tail) | 1 | 0.002 | 6.4 | 1.223 | 0.373 | 0.28 | 0.983 | 6.207 | 6.741 | 9.209 | 0.632 | 0.0 | 0.61 | 0.71 | 0.82 | 1.96 | 0 | 0 | 2 | 12 |  |

## r16-b

| video | arm | black_fr | black_share | ghost_s | seam | edgeL | edgeR | edgeR_p95 | stL | stR | stJ | dLJ | darkJ | out_l | out_r | out_n | tail | soft | thin | fat | moments | speech0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q04 | P0 oval | 108 | 0.85 | 0 | 0.803 | 0.391 | 0.073 | 0.288 | 6.138 | 6.408 | 7.668 | -0.202 | 0.0 | 0.72 | 0.52 | 0.62 | 1.07 | 0 | 0 | 1 | 0 | 4.758 |
| q04 | P1 +tight | 90 | 0.826 | 0 | 0.801 | 0.24 | 0.008 | 0.167 | 5.493 | 5.44 | 6.954 | 0.248 | 0.0 | 0.7 | 0.5 | 0.6 | 1.09 | 0 | 0 | 1 | 0 | 4.699 |
| q04 | P2 +parse | 90 | 0.826 | 0 | 0.801 | 0.24 | 0.008 | 0.167 | 5.493 | 5.44 | 6.954 | 0.248 | 0.0 | 0.7 | 0.5 | 0.6 | 1.09 | 0 | 0 | 1 | 0 | 4.699 |
| q04 | P3 +cm-shadow (side) | 139 | 0.921 | 0 | 0.771 | 0.226 | 0.015 | 0.184 | 5.543 | 6.563 | 10.377 | -2.988 | 0.26 | 0.69 | 0.49 | 0.62 | 1.3 | 0 | 0 | 1 | 0 | 4.196 |
| q04 | P4 P2+occl | 96 | 0.821 | 0 | 0.799 | 0.306 | 0.04 | 0.209 | 5.214 | 5.014 | 7.12 | -0.306 | 0.0 | 0.63 | 0.46 | 0.53 | 1.02 | 0 | 0 | 1 | 0 | 4.718 |
| q04 | P5 +cap | 89 | 0.802 | 0 | 0.789 | 0.305 | 0.039 | 0.215 | 5.21 | 5.069 | 7.327 | -0.351 | 0.0 | 0.66 | 0.46 | 0.52 | 1.34 | 0 | 0 | 1 | 0 | 4.558 |
| q04 | P6 +pause | 78 | 0.804 | 0 | 0.824 | 0.309 | 0.043 | 0.217 | 5.22 | 5.097 | 7.354 | -0.353 | 0.0 | 0.64 | 0.46 | 0.53 | 1.27 | 0 | 0 | 1 | 0 | 4.597 |
| q04 | today (+tail) | 77 | 0.77 | 0 | 0.791 | 0.275 | 0.008 | 0.165 | 4.531 | 3.966 | 7.208 | -0.896 | 0.0 | 0.67 | 0.46 | 0.54 | 0.4 | 0 | 0 | 1 | 0 |  |
| long5 | P0 oval | 94 | 0.037 | 2.8 | 0.941 | 0.559 | 0.564 | 2.659 | 9.053 | 7.33 | 10.853 | 1.364 | 0.012 | 1.87 | 2.32 | 1.78 | 1.09 | 13 | 2 | 0 | 50 | 4.117 |
| long5 | P2 +parse | 64 | 0.025 | 8.3 | 0.886 | 0.442 | 0.481 | 2.532 | 7.489 | 6.123 | 9.757 | 1.405 | 0.003 | 1.91 | 2.23 | 1.75 | 1.13 | 15 | 2 | 0 | 50 | 5.141 |
| long5 | P4 P2+occl | 82 | 0.032 | 8.2 | 0.791 | 0.416 | 0.459 | 2.498 | 7.545 | 6.23 | 9.36 | 0.935 | 0.005 | 1.92 | 2.26 | 1.76 | 1.07 | 14 | 2 | 0 | 45 | 5.066 |

## The three flagged spots

| spot | what the arms show | cause |
|---|---|---|
| m03, left cheek edge at the hair (0:19–0:23) | Cheek-edge jitter: 08.11 file 1.22, oval 1.20, tight 0.68, today 0.80. Static difference 7.4 / 9.3 / 4.1 / 5.2. | Already in the oval paste and stronger there; the tight mask halves it. Not caused by the cleanup. |
| old03 2–3 s, glued blurry lips | Mouth opening 2.1–3.5 s: source 0.009; 08.11 file 0.005; without pause-closing 0.003; with it 0.003. Sharpness 0.66 in all four. | KeySync itself closes and blurs the mouth in the silence before the first phrase. Pause-closing changes nothing there. |
| long5 ~0:50, abrupt blend | One-frame jump of the lower-face difference at 48.6 s (9.6 → 12.8) in oval, tight + parsing and + occlusion guard alike. | In the generated layer, not the paste. The 08.11 file has a smaller one at 53.6 s (its dub differs). |
| long5 0:00, dark smudge at the jaw by the hood | Left-edge lightness shift 0–38 s: 08.11 file +3.6, oval +3.4, tight + parsing +2.1, + occlusion +1.6; darkest 5 % about −2 to −2.8 in all. | No arm stands out; my check does not isolate this smudge. Needs the eye on clip c1. |

## What each change does (same layers)

| change | helps | hurts |
|---|---|---|
| tight mask | Cheek-edge jitter −50…−70 % on every video; seam down; m18 ghost 6.8 → 1.3 s and black frames 50 → 16 | speech@0 −0.04…−0.19; old03 ghost seconds stay high (generator smear) |
| parsing | Nothing measurable (q04 output byte-identical; others ±0.01) | Nothing measurable |
| cm-shadow | q01 ghost 3.0 → 0 s | Black frames up on every video (m18 15 → 90, q04 90 → 139, old03 8 → 28, q01 1 → 26); jaw band darker |
| occlusion / turn guard | Seam down on old03 (1.25 → 0.95) and m18 | q01 ghost 3.0 → 5.0 s and breaks 8 → 15 |
| cap 1.0 | Black frames: m03 32 → 18, m18 21 → 3 | speech@0 −0.17…−0.27 (m03, m18, q01); visibly smaller mouth opening (long5 0:50) |
| pause closing | Nothing measurable | old03 speech@0 −0.33; q01 ghost 5.1 → 6.4 s |
| tail fade | Tail: m03 2.6 → 1.4, old03 1.1 → 0.6, q04 1.3 → 0.4 | m18 and q01 tails unchanged (2.3 / 2.0) |

## Proposed conditional rules

1. **Tight mask: always.** It is the one step that clearly improves the cheek and jaw edges, and Irina already preferred it on beards.
2. **cm-shadow: off.** Exception only if a ghost-chin check fires on the tight paste and black frames stay near the source (q01-like).
3. **Parsing: only when hair, a hand or an object crosses the face mask** (parse classes inside the tight mask); otherwise it is a no-op that costs time.
4. **Occlusion / turn guard: only on frames where an occluder or a turn is detected** (jaw landmarks off the source, |yaw| > 0.45), not as a global switch: it raised ghost chin on q01.
5. **Cap: only where the black-hole detector fires on the uncapped paste** (m03, m18 here), not globally: it costs 0.2–0.3 sync and narrows the mouth.
6. **Pause closing: off** until K2's fix (close only to the source's lip gap; skip silence before the first and after the last phrase). It bought nothing here.
7. **Tail fade: on.**
8. **Not fixable by paste rules:** old03's closed, blurry mouth in silence and soft mouth overall; long5's jump at 48.6 s; old03's lip smear on the chin. These are in the generated layer.

A "proposed" render (rules 1–7 applied per video) does not exist yet; the clips show the arms closest to it.

## Clips (0.5x, zoomed on the face)

- `a1_m03_19-23_cheek_hair`, `a2_m03_8-12_cheek_hair`: 08.11 | oval | tight (no parsing) | today | source
- `b1_old03_1.5-5_glued_lips`: 08.11 | oval | cap, no pause-closing | today | source
- `c1_long5_0-6_jaw_hood`, `c2_long5_47-53_abrupt`: 08.11 | oval | tight + parsing | today | source
