# r16b K2: black hole, plain vs K2 (refpick auto + b8 gate) through e2e ks_face.sh (experiment/r16-mouth 3d66d9e)

Same chain for both arms (delay 0, paste shift 0, LoRA r16in/lora/lora_pick.pt, TRT rebuilt).
- **plain** = KS_REF=0, no gate.
- **K2** = KS_REF=auto + KS_GATE=thr=0.12,cap=0.25,hi=8,lo=2.

Cavity: k_cavity.py on the full pasted output with MediaPipe landmarks. Each cell is open frames (gap/width > 0.15) / black-hole share (open frames with mean L < 15 and < 2 % teeth).

| video | speech@0 plain → K2 | source | plain | K2 | note |
|---|---|---|---|---|---|
| q01 | 6.85 → 7.31 | 738 / 2.2 % | 661 / 2.0 % | 312 / 2.2 % | no black hole; K2 opens the mouth half as often |
| q02 | (SyncNet 0, unscorable) | 398 / 82 % | 321 / 52 % | 98 / 0 % | dark source mouth; K2 removes black but opens much less (= the amplitude note) |
| q03 | 5.49 → 6.16 | 151 / 0 % | 138 / 0.7 % | 252 / 0.4 % | no black hole |
| q04 | 5.19 → 5.47 | 107 / 77 % | 75 / 85 % | 68 / 97 % (L 9.0) | dark beard mouth; K2 darker (pick #731, no teeth to copy) |
| q05 | failed both | | | | ks_tsmooth NaN landmarks on head turns (audit item 10, K1 / T file) |
| q06 | kept source | | | | no-face rule (route skips it); BUT ks_shots split it into 18 pieces (MediaPipe misses the profile face, so it saw faceless gaps) |
| m18 | 8.15 → 7.20 | 30 / 6.7 % | 125 / 24 % | 103 / 30 % | golden (K1 clean5) 157 / 24 %; K2 does NOT help here in e2e (day 1 crop metric said 23 → 8 %) |

## Conclusions
- refpick + gate is not a general black-hole fix in e2e. It helps q02, is neutral on q01 / q03 (SyncNet up), and is worse on q04 and m18. On m18 SyncNet also drops by ~1.
- **Real black-hole cases:**
  - m18: 24–30 % vs source 7 %;
  - q02 / q04: the source is dark too (82 % / 77 %), so judge relative to the source.
- **K2 opens the mouth less often** (q01 312 vs 661, q02 98 vs 321). It is a plausible side effect of the slightly-open teeth reference. Needs Irina's eye.

## Next
1. Find why m18 differs from day 1. Same pick #158; changed since day 1:
   - int ks_fast (4th ref column);
   - lora_pick instead of lcm_r1 fp16;
   - the int static crop;
   - the metric now runs on the full frame.
2. Use the teeth reference only when the pick really shows teeth (teeth_up above a threshold); otherwise ref 0 (q04).
3. ks_shots gap rule: split a faceless gap only if the picture changes (q03 title card yes, q06 detector misses no).

## Update: third arm r0g = ref 0 + b8 gate (the new e2e default after Irina's m18 verdict)

| video | speech@0 plain / r0g / K2 | black share plain / r0g / K2 | open frames plain / r0g / K2 |
|---|---|---|---|
| q01 | 6.85 / 6.91 / 7.31 | 2.0 / 2.5 / 2.2 % | 661 / 653 / 312 |
| q02 | (unscorable) | 52 / 36 / 0 % | 321 / 475 / 98 |
| q03 | 5.49 / 5.43 / 6.16 | 0.7 / 0.7 / 0.4 % | 138 / 140 / 252 |
| q04 | 5.19 / 5.23 / 5.47 | 85 / 82 / 97 % | 75 / 76 / 68 |
| m18 | 8.15 / 8.09 / 7.20 | 24 / 21 / 30 % | 125 / 164 / 103 |

- **The gate alone (r0g vs plain):**
  - slightly less black: q02 52 → 36 %, m18 24 → 21 %, q04 85 → 82 %;
  - the mouth opens as often or more (q02, m18);
  - speech@0 is within ±0.06 of no gate in these runs (delay 0, shift 0).
- **The "opens less" effect is the refpick, not the gate:** K2 has half the open frames on q01 and a third on q02.
- This supports ref 0 + gate as the default. Refpick is the option to keep off (worse on m18 / q04).
