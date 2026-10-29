# r8 F: m13b = m13 with a tighter fallback + milder deflicker (recommended candidate; WATCH order 01 03 05 04 02)

Same audio-r8-v1 LatentSync layers as m13. Fallback fb6 = fb5 thresholds (same broken frames caught: washed pink blob, lip contrast collapse,
near-black cavity, open/shut jump) but held for less time (no dilation, merge gap 2 frames, ramp sigma 1.2). Deflicker dfk2 = inner mouth only,
+-15 % clamp, sigma 1 frame (no lower-face pass). sbs: left m11 = final_r7, right m13b.

| video | LS frames falling back | whole-frame gate | flashes m11 -> m13b | SyncNet m11 / m13 / m13b | pressed lips s | mismatch s |
|---|---:|---|---|---|---|---|
| 01 | 16.2 % | PASS | 4 -> 0 | 6.65 / 3.63 / 4.35 | 0 | 0 |
| 03 | 6.1 % | PASS | 44 -> 32 | 7.86 / 5.04 / 5.91 | 3.2 -> 2.2 | 0.77 -> 1.67 |
| 05 | 1.8 % | PASS | 6 -> 1 | 3.47 / 2.62 / 2.82 | 6.95 -> 6.38 | 1.84 -> 2.24 |
| 04 | 4.9 % | PASS | 16 -> 4 | 6.42 / 4.23 / 4.78 | 3.13 -> 1.97 | 0.73 -> 1.33 |
| 02 | (no face) | PASS | 0 | - | 0 | 0 |

03 inner-mouth flicker (frame-to-frame HF luma): source 1.21, m11 2.52, m13 1.50, m13b 1.72; events/s 0.63 / 4.07 / 1.22 / 1.72.
Ablation (03 / 04 SyncNet): audio-r8-v1 alone 7.79 / 6.04; + fallback fb5 5.88 / 4.71; + deflicker dfk1 6.79 / 5.55; both 5.04 / 4.23.
