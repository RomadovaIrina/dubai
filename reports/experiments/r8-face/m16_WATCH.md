# m16 = CodeFormer DETAIL on the clean LatentSync mouth (audio-r8-v4; 01 03 05; NC CodeFormer, user decision 29.10)

LatentSync is seeded and deterministic: with the MediaPipe face stack the layer before CodeFormer equals m15's layer (full-frame difference
0.12 grey levels). m16 keeps that clean layer's low frequencies (mouth shape, opening, motion) and adds only CodeFormer's high-frequency
detail (lip line, teeth, texture):
- the detail is (CF - clean) minus its Gaussian blur (sigma 0.12 x mouth width), inside a feathered mouth mask;
- it is averaged over frames t-1..t+1 after flow alignment, so the per-frame CodeFormer does not flicker.
Then the m15 composite: fallback fb6 + inner-mouth deflicker dfk2. CodeFormer runs on the MediaPipe alignment (InsightFace is not used).

Files (comparison files <= 25 MB):
- <id>_m15_vs_m16_sbs.mp4 (left m15 clean | right m16).
- <id>_m15_vs_m16_zoom_sbs.mp4 (03 2x, 01 / 05 ~3x).
- <id>_r8_m16_WATCH.mp4.

## Per video: m15 / m14 / m16
| | SyncNet | flashes | pressed lips s | mismatch s | flicker HF | teeth share / black-hole % |
|---|---|---|---|---|---|---|
| 01 | 4.35 / 3.56 / **4.05** | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 3.28 / 2.97 / 3.15 | 0 / 0 / 0 ; 50.6 / 44.3 / 44.0 |
| 03 | 5.82 / 5.87 / **5.68** | 29 / 32 / 36 | 2.03 / 4.17 / 3.47 | 1.70 / 2.63 / 1.97 | 1.67 / 1.87 / 1.78 | 0.003 / 0.008 / 0.005 ; 13.6 / 11.2 / 10.7 |
| 05 | 3.36 / 3.73 / **3.52** | 4 / 1 / 3 | 4.88 / 4.85 / 4.61 | 0.97 / 2.50 / 1.94 | 3.65 / 3.55 / 3.79 | 0.013 / 0.014 / 0.016 ; 79.6 / 69.4 / 77.9 |

## Where the m14 lip-sync loss came from
Generator only (no fallback / deflicker), SyncNet 01 / 03 / 05 and mismatch 03 / 05:
| stack | SyncNet 01 / 03 / 05 | mismatch 03 / 05 s | pressed lips 03 s |
|---|---|---|---|
| clean (m15plain) | 6.47 / 7.90 / 4.02 | 1.67 / 1.20 | 2.03 |
| CF + InsightFace (m14plain) | 6.81 / 7.55 / 4.39 | 3.37 / 3.21 | 4.40 |
| CF on MediaPipe (m16plain) | 6.53 / 7.59 / 3.96 | 1.27 / 2.04 | 4.13 |
| CF detail only (m16d) | 6.62 / 7.74 / 4.19 | 1.70 / 0.27 | 3.70 |

Reading:
1. InsightFace alignment caused most of m14's mismatch: on MediaPipe, 03 mismatch is 1.27 (m14 3.37).
2. Full CodeFormer still changes the mouth shape: 05 mismatch 2.04.
3. Detail-only keeps the clean shape: SyncNet within +-0.17 of clean, mismatch equal or better.
4. Pressed lips 03 stays higher with any CodeFormer (3.5-4.4 vs 2.0), even detail-only, whose low frequencies are the clean ones. This is
   most likely the sharper lip line changing where MediaPipe puts the inner lip (a landmark measurement effect), not a pressed shape: it
   needs a look at 03.
5. m16 full arm vs m15: SyncNet 01 -0.30, 03 -0.14, 05 +0.16; mismatch 05 1.94 vs 0.97 (the fallback picks other frames on the CodeFormer
   layer).
