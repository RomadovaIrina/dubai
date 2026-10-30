# DabAI r9: management set, three face variants (31.10.2026)

Each video comes in 3 face variants x 2 background beds. The dub (voice, timing, translation) is identical across all six files of a video;
only the mouth rendering differs. Files: `<id>_<variant>_<bed>.mp4`, full quality, source resolution / fps.

| variant | what it is |
|---|---|
| **r9fix** | the r9 generator + the composite fix: (1) short mouth glitches (the old fallback's 1-frame swaps to the original mouth, 1-2-frame opening jumps, a mouth that slides / doubles) are re-drawn from the neighbouring good frames; longer broken runs ease to the original mouth instead of snapping; (2) CodeFormer's detail inside the mouth at 0.4, so the teeth are no sharper than the real ones |
| **fxlite** | r9fix without the smoothing of 1-2-frame opening jumps (only real breakages are repaired), in case that smoothing hides real consonant articulation |
| **fxstab** | r9fix + a mouth-detail stabiliser: inside the mouth each frame keeps its own shape and motion but takes the median fine detail of its +-2 neighbours, so CodeFormer's frame-to-frame texture pops ('explosions') are outvoted |

Beds: `demucs` = htdemucs background (main), `spleeter` = Spleeter background.

## Numbers
SyncNet = lip-sync confidence on the dub speech (higher = better). Flashes = 1-frame mouth texture / brightness / colour jumps vs the source (strong in brackets; lower = better).
Twitch = whole-face twitch detector vs the source (events >= 2 in brackets; lower = better). Mismatch = seconds where the mouth and the dub disagree (open mouth on silence / closed on speech).

| video | variant | SyncNet | flashes (strong) | twitch (strong) | mismatch s |
|---|---|---|---|---|---|
| m03 | r9 (before, for reference) | 7.22 | 85 (26) | 35.7 (3) | 0.57 |
| m03 | r9fix | 7.56 | 77 (19) | 23.7 (1) | 0.30 |
| m03 | fxlite | 7.59 | 76 (23) | 26.6 (1) | 0.57 |
| m03 | fxstab | 7.39 | 63 (12) | 22.0 (1) | 0.30 |
| m18 | r9 (before, for reference) | 7.30 | 56 (7) | 11.5 (0) | 1.03 |
| m18 | r9fix | 7.83 | 63 (8) | 9.2 (0) | 1.73 |
| m18 | fxlite | 7.94 | 67 (8) | 8.3 (0) | 1.77 |
| m18 | fxstab | 7.60 | 28 (7) | 13.2 (1) | 2.50 |
| old03 | r9fix | 7.22 | 200 (37) | 33.9 (1) | 1.80 |
| old03 | fxlite | 7.22 | 200 (40) | 31.8 (1) | 1.80 |
| old03 | fxstab | 7.00 | 180 (33) | 39.8 (1) | 3.60 |
| old05 | r9fix | 3.96 | 32 (13) | 19.8 (2) | 1.50 |
| old05 | fxlite | 3.98 | 34 (13) | 20.8 (2) | 0.94 |
| old05 | fxstab | 3.54 | 22 (5) | 8.6 (1) | 3.44 |

Notes: m03: mother + child (two faces), 40 s; m18: older woman, glasses, dim light, 20 s; old 03: the man (test set), 90 s; old 05: the man (test set), 52 s. The old 03 / 05 are re-dubbed from scratch with the current pipeline (final_r8b audio config, r9 generator).

**For comparison with what management saw before (final_r8b, old dub):** 03 SyncNet 6.74, strong flashes 36; 05 SyncNet 4.20, strong flashes 1.
- old03 is better on SyncNet (7.22) at a similar strong-flash count (33–40).
- old05 is WORSE than r8b on both SyncNet (3.5–4.0 vs 4.20) and strong flashes (5–13 vs 1). Its face is small and side-on, the case the generator handles worst, and the new dub places speech differently.

**Per-variant trade-off:**
- **fxstab** has the fewest flashes everywhere (m18 halved; old05 13 → 5 strong). Its mouth / dub mismatch is the highest (2.5–3.6 s).
- **r9fix and fxlite** are almost identical.
