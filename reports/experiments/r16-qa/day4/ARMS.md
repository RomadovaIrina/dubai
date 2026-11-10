# Arm scores, day 4 (Q), 10.11 ~09:20 UTC

Metrics are triage only; Irina's eye decides.

**Metrics**

- **ghost s:** seconds in ghost-chin runs, from Q's detector in r16q_defects.py: a second jaw-parallel edge below the jaw, final minus source. Lower is better.
- **K1 ghost:** K1's own ghost measure, ghost_mean (gs/*.json).
- **black:** flagged frames and black share from r16q_detect (source share in brackets).
- **seam:** excess median from the jaw-band wobble. The codec floor is 0.25–0.5.
- **tail:** the local end ratio. 1.5 or more is a broken tail.
- **speech@0:** from K1 / K2 eval (SyncNet).

Sources: each box's master_25fps, which is the source resampled to 25 fps.

## K1: ghost-chin mask arms (r16-c, /workspace/dub/k1/gh/arms)

Arms:

- **oval:** the old default mask.
- **tc:** tight + chin, about the same as now_1011 batch b.
- **A1:** tc plus neck-follow.
- **A2:** A1 plus chin-down.
- **A3:** tight, no chin.

| video | arm | ghost s (Q) | ghost p95 (Q) | K1 ghost | black fr / share | seam | tail | speech@0 |
|---|---|---|---|---|---|---|---|---|
| m03 | oval | 0.0 | 0.08 | 1.86 | 28 / 0.075 | 1.32 | 2.46 | 8.33 |
| m03 | tc | 1.2 | 0.20 | 1.47 | 22 / 0.060 | 1.15 | 3.02 | 8.23 |
| m03 | A1 | 0.4 | 0.14 | 1.45 | 23 / 0.063 | 1.11 | 3.00 | 8.25 |
| m03 | A2 | 0.4 | 0.12 | 1.47 | 25 / 0.069 | 1.15 | 3.03 | 8.25 |
| m03 | **A3** | **0.0** | **0.04** | **0.26** | 27 / 0.075 | **0.12** | 2.47 | 7.79 |
| q01 | oval | 12.2 | 1.04 | 2.57 | 7 / 0.011 | 1.34 | 2.25 | 5.37 |
| q01 | tc | 2.4 | 1.34 | 1.31 | 6 / 0.009 | 0.92 | 2.11 | 5.23 |
| q01 | A1 | 3.2 | 0.97 | 1.34 | 7 / 0.011 | 0.97 | 1.95 | 5.21 |
| q01 | A2 | 7.6 | 1.47 | 1.68 | 6 / 0.009 | 1.09 | 2.32 | 5.26 |
| q01 | **A3** | **0.3** | **0.52** | **0.56** | 5 / 0.008 | **0.25** | 1.39 | 5.21 |
| q04 | oval | 0.0 | 0.09 | 1.60 | 98 / 0.79 (dark src) | 0.86 | 1.08 | 4.96 |
| q04 | tc | 0.0 | 0.11 | 1.02 | 79 / 0.71 | 0.72 | 1.05 | 4.89 |
| q04 | A1 | 0.0 | 0.10 | 1.05 | 79 / 0.74 | 0.72 | 1.06 | 4.88 |
| q04 | A2 | 0.0 | 0.10 | 1.20 | 84 / 0.76 | 0.77 | 1.07 | 4.84 |
| q04 | **A3** | 0.0 | 0.10 | **0.33** | 83 / 0.68 | **0.26** | 0.99 | 4.67 |
| old03 | oval | 13.0 | 1.33 | 2.17 | 1 / 0.002 | 1.49 | 1.29 | 7.81 |
| old03 | tc | 18.8 | 2.59 | 0.95 | 3 / 0.006 | 0.80 | 1.24 | 7.65 |
| old03 | A1 | 21.6 | 2.91 | 1.26 | 2 / 0.004 | 0.93 | 1.25 | 7.63 |
| old03 | A2 | 12.2 | 0.95 | 1.58 | 2 / 0.004 | 1.04 | 1.28 | 7.77 |
| old03 | A3 | 16.3 | 2.70 | **0.50** | 4 / 0.008 | **0.11** | 1.16 | 7.38 |
| m18 | oval | 7.1 | 1.20 | 0.52 | 34 / 0.21 | 1.06 | 1.35 | 7.89 |

**Reading (K1)**

- **A3 (tight, no chin) wins on m03 / q01 / q04.** On both ghost measures and on seam it is the lowest: seam 0.11–0.26, about at the codec floor.
- **The cost of A3 is speech@0, −0.2 to −0.5:** m03 8.23 → 7.79, old03 7.65 → 7.38, q04 4.89 → 4.67. The tight mask clips some jaw motion. Irina should check by eye whether lip sync still reads.
- **On old03 my ghost does not drop with any mask arm** (13–22 s). K1's measure drops (0.95 → 0.50). By eye, old03's "ghost" is a pink smear of the lower lip on the chin, which is the generator, not the chin feather. So the mask can't remove it; it belongs to K2 (mouth) or L.
- **A1 / A2 (neck-follow, chin-down) don't help.** On q01, A2 is worse than tc.
- **The m03 tail is still 2.5–3.0 in every arm.** That is the wide face at 37.9–39.9 s, which comes from the generator or the tail, not the mask (routed to T).
- **m18:** only oval exists so far.

## K2: black-hole arms (r16-b, /workspace/dub/k2/bh/old03)

Arms:

- **now:** ref 0 + gate, oval paste.
- **T:** auto ref with teeth.
- **C:** opening cap.
- **P:** pause closure.
- **CP, TCP:** combos.

| video | arm | black fr / share (Q) | K2 black share of open frames | ghost s (Q) | seam | tail | speech@0 (K2) |
|---|---|---|---|---|---|---|---|
| old03 | now | 11 / 0.027 (src 0.002) | 0.094 | 9.3 | 1.03 | 1.20 | 4.70 |
| old03 | T | 9 / 0.012 | 0.065 | 7.0 | 0.91 | 0.98 | 4.66 |
| old03 | **C** | **0 / 0.000** | 0.075 | 8.5 | 1.02 | 1.24 | 4.65 |
| old03 | P | pending | 0.100 | | | | 4.74 |
| old03 | CP | pending | 0.072 | | | | 4.64 |
| old03 | TCP | pending | 0.063 | | | | 4.66 |

**Reading (K2)**

- The opening cap C removes every flagged black frame on old03, and T halves the share. Speech@0 is about flat (−0.05).
- C and T measure different things: my detector flags dark frames when the mouth is open; K2's share is over all open frames. Both say T / C / TCP beat now.
- The speech@0 scales differ between boxes (K1 eval 7.6 vs K2 sync_perframe 4.7 on old03), so compare arms only within the same box.
- P / CP / TCP and m18 / m03 / q04 / q02 are scoring automatically on r16-b (`/workspace/dub/q_tmp/q/out/k2/SUMMARY.txt`).
