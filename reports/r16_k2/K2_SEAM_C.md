# K2: seam variant (c) = tight mask WITHOUT the generated chin, and when to allow it (10.11)

**What (c) is:**
- a pure RE-PASTE of K1's "now" KeySync layers (/workspace/dub/k1/seam/w_<id>; today's default ref 0 + gate);
- K1's base args, plus `--mask tight` and no `--mask-chin` (k1/r16-paste2 371339e);
- only the paste differs from K1's "now" and (a), so the 4-panels stack.

**Finals:** r16-k2:/workspace/dub/k2/seamc/<id>_c.mp4 for q03a, old03, m03, q04, q01.
- Seam score: K1 runs Q's r16q_defects seam excess on all arms, for one table.
- Clips: K1 builds the 4-panels (now | (a) | (c) | source).

## Sync cost (Q's eval_final speech@0, full clip)

| clip | now | (c) | change |
|---|---|---|---|
| q03a | 4.99 | 4.65 | −0.34 (−6.8 %) |
| old03 | 7.78 | 7.35 | −0.43 (−5.5 %) |
| m03 | 8.33 | 7.79 | −0.54 (−6.5 %) |
| q04 | 4.97 | 4.66 | −0.31 (−6.2 %) |
| q01 | 5.42 | 5.19 | −0.24 (−4.4 %) |

(c) costs a steady 4–7 % of speech@0 on every clip. Under "no sync loss beyond noise" no clip passes as a whole.

## Where the loss is (per second, sync_main.json conf_at0, (c) − now)
- **The loss is spread evenly over the speech seconds.** Spearman(Δ, generated jaw excess = raw lip gap − source lip gap) is −0.11 … +0.29.
- **High-excess (> 0.05) and low-excess seconds lose about the same:**

| clip | high-excess seconds | low-excess seconds |
|---|---|---|
| q03a | −0.30 | −0.38 |
| old03 | −0.40 | −0.45 |
| m03 | −0.27 | −0.63 |
| q04 | −0.47 | −0.23 |
| q01 | −0.23 | −0.24 |

- So SyncNet reads the generated chin all through the speech, not only on big jaw drops.
- **A jaw-amplitude rule cannot isolate the loss:** keeping the chin only where excess > 0.03 still loses −0.08 … −0.43 while (c) covers only 40–71 % of the seconds.
- The per-video jaw excess (atlas) doesn't predict the cost either: m03 has the smallest excess (0.033) and the biggest cost.

## Proposed automatic rules
1. **Pause rule (no sync cost by construction; recommended as the default).**
   - (c) inside dub pauses (dub-only track < peak − 30 dB for ≥ 0.5 s, the face present), with a 4-frame crossfade to the chin-kept mask at the edges.
   - The chin kept during speech (tight + chin, or K1's (a) neck-follow).
   - The seam goes to the floor in the pauses, where the jaw isn't needed.
   - **Coverage** (dub-only tracks): q02 42 %, q05 42 %, old03 35 %, q06 28 % (no KeySync anyway), q04 23 %, q03 17 %, m03 16 %, q01 13 %, m18 10 %.
2. **Margin rule (whole video, if Irina accepts a sync trade).**
   - (c) everywhere only when speech@0(now) × 0.93 ≥ the video's floor (the floor = Q's pass rule, e.g. m03 ≥ 8.52).
   - On today's numbers none of the 5 clips passes (m03 now 8.33 is already below its floor).
   - So in practice this means "never (c) everywhere" unless the floor is relaxed.
3. **During speech, the seam fix has to come from (a)** (the source neck follows the generated jaw), not from removing the chin.

## Files
- /workspace/dub/k2/seamc/: <id>_c.mp4, ev/<id>_{now,c}/ (eval.json, sync_main.json), c_rule.py (the per-second analysis), run.out.
