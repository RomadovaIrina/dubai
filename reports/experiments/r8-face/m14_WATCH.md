# r8 F: m14 = the OLD NC face stack (cand5) on audio-r8-v4 + the m13b composite (WATCH; licence deviation, user decision 29.10)

What m14 is:
- LatentSync 1.6 with CodeFormer (w 1.0, mouth scope) and InsightFace buffalo_l detection / landmarks: the cand5 flags, run with
  DUB_ALLOW_NC=1. Both models are NON-COMMERCIAL; the user accepted that risk.
- Then the m13b composite: fallback fb6 + inner-mouth deflicker dfk2.

Files:
- <id>_r8_m14_WATCH.mp4.
- Side-by-sides (left | right): _vs_m13b_sbs (m13b on audio-r8-v1 | m14), _vs_m15_sbs (m15 = the same composite on the clean stack, SAME
  audio-r8-v4 | m14: the fair comparison), _vs_m11_sbs.

## m14 vs m15 (same audio, same composite)
| video | LS frames falling back m15 / m14 | SyncNet m15 / m14 | flashes m15 / m14 | pressed lips s m15 / m14 | mismatch s m15 / m14 |
|---|---|---|---|---|---|
| 01 | 16.2 / 20.0 % | 4.35 / 3.56 | 0 / 1 | 0 / 0 | 0 / 0 |
| 03 | 7.4 / 5.5 % | 5.82 / 5.87 | 29 / 32 | 2.03 / 4.17 | 1.70 / 2.63 |
| 05 | 2.0 / 3.4 % | 3.36 / 3.73 | 4 / 1 | 4.88 / 4.85 | 0.97 / 2.50 |
| 04 | 4.1 / 3.2 % | 5.39 / 5.55 | 2 / 4 | 2.43 / 2.73 | 1.03 / 0.73 |

## Generator only (plain composite: no fallback, no deflicker), clean vs old stack, same audio
| video | SyncNet clean / CF+IF | mismatch s clean / CF+IF | pressed lips s clean / CF+IF |
|---|---|---|---|
| 01 | 6.47 / 6.81 | 0 / 0 | 0 / 0 |
| 03 | 7.90 / 7.55 | 1.67 / 3.37 | 2.03 / 4.40 |
| 05 | 4.02 / 4.39 | 1.20 / 3.21 | 5.31 / 5.11 |

03 inner-mouth flicker (HF luma): m15 1.67, m14 1.87 (source 1.21, m11 2.52).

Look (contact sheet 03 20-21.6 s): m14 lips are slightly crisper (a clearer lip line, more teeth detail). The difference is subtle at this
face size.

Verdict:
- The old stack is NOT a lip-sync win: SyncNet moves by +-0.35 per video, either way.
- It doubles the mouth / dub mismatch (03, 05) and the pressed-lips time (03).
- It changes the look slightly.
- InsightFace also gates fewer frames into LatentSync (05: 951 vs 1106 LS frames).
