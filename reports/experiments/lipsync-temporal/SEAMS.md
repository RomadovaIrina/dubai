# LatentSync 16-frame window seams on the clean base (clean0) — generator-side A/B (in progress)

Budget (contract): video stage <= clean0 + 10 % (norm 300 s per video-minute). Arms: `scripts/quality/ls_window_ab.py` on 03 window B
(21.6-37.6 s, same 16-frame phase as the full run, contains the human-marked 24-25 s and 35-36 s) and window A (40-58 s: switches, head turn);
everything = clean0 (MediaPipe stack, no restorer, mouth-only composite, cand5 audio, same gate) except the window scheme.
Score: `scripts/quality/seam_table.py` on the common metric v4 per-frame series at the seams each arm RECORDED (`ls_seams`).
Seam jump = mouth d1 at the seam / median d1 of the 8 surrounding non-seam frames; the reference column = the source at the same frames.

| arm | what | LatentSync s (cost) |
|---|---|---|
| C0 | clean0: independent 16-frame windows | 83.8 B / 79.3 A |
| W24 | 24-frame windows (motion-module maximum), no overlap | 78.6 A (-1 %); B re-running (first attempt OOMed in my loop, fixed) |
| K24o2 / K16o2 / K16o4 | sequential windows, the first 2 / 2 / 4 frames pinned to the previous window's result at every step (RePaint-style continuation) | +12 % / +15-16 % / +28-29 % |

Full table: `seams_table_v1.md`. Key numbers (B / A):

| arm | seam jump median (source) | share of seams > 1.5x (source) | SyncNet dub-only | mouth resid / twitch | human-marked resid / twitch (B) |
|---|---|---|---|---|---|
| C0 | 1.71 / 1.42 (0.83 / 1.02) | 67 % / 41 % (8 / 14 %) | 5.64 / 5.89 | 0.99 / 1.08 · 1.04 / 1.02 | 0.94 / 1.12 |
| W24 | – / 1.83 | – / 64 % | – / 6.19 | – · 1.05 / 1.02 | – |
| K24o2 | 1.66 / 1.66 | 53 % / 63 % | 6.09 / 6.13 | 0.98 / 1.08 · 1.05 / 1.03 | 0.89 / 1.12 |
| K16o2 | 1.62 / 1.25 | 64 % / 38 % | 5.84 / 6.15 | 1.00 / 1.07 · 1.05 / 1.03 | 0.89 / 1.11 |
| K16o4 | 1.51 / 1.36 | 53 % / 40 % | 5.80 / 6.12 | 1.00 / 1.06 · 1.06 / 1.02 | 0.92 / 1.13 |

Reading (so far):
* **No cheap generator-side variant removes the seams consistently.** Pinning the overlap to the previous window (continuation) lowers the
  jumps a little on one window and not on the other; with ~20 seams per window the differences are within noise. Longer windows (W24) give
  fewer but larger jumps (A: 14 seams, 64 % > 1.5x vs 22 seams, 41 %).
* Nothing else moves: mouth texture (resid ~1.0), twitch index, sharpness, upper face (1.02) are the same in every arm; SyncNet equal or
  slightly higher (+0.2-0.45 dub-only, AV offset 0 everywhere).
* Hard pinning (RePaint-style) is known to harmonise poorly at the boundary; the per-step fusion arm (stride 8, no DeepCache, over budget,
  a quality ceiling) is queued to decide whether a budget fusion (only near seams / only late steps) is worth building.

Side-by-side zoom clips (ORIGINAL | CLEAN0 | (W24) | K24o2 | K16o4): `/workspace/dub/artifacts/lipsync-temporal/sbs/03_seams_*_zoom.mp4`.

## Update before the pause (2026-10-16 ~23:20 UTC)

| arm (window B) | LatentSync cost | seam jump median (source 0.83) | seams > 1.5x (source 8 %) | mouth resid / twitch | human-marked resid / twitch | SyncNet dub-only |
|---|---|---|---|---|---|---|
| clean0 | – | 1.71 | 67 % | 0.99 / 1.08 | 0.94 / 1.12 | 5.64 |
| per-step fusion, stride 8, no DeepCache (ceiling) | +322 % | 0.96 | 8 % | 0.89 / 1.05 | 0.82 / 1.11 | 5.88 |
| 24f + 1-frame continuation | +7 % | 1.71 | 59 % | 0.99 / 1.08 | 0.92 / 1.11 | 6.18 |
| **bridge 8 frames x last 4 steps** | **+10 %** | **1.28** | **29 %** | **0.94 / 1.05** | **0.89 / 1.07** | 5.62 |

* Per-step fusion removes the seam completely (seam numbers at the source level) -> the seam IS a window-independence artefact of the
  generator, fixable on the generator side.
* The budget bridge (`--bridge-len 8 --bridge-steps 4`, scripts/quality/ls_stable.py) gets about half of the way at exactly the +10 % budget.
  Not yet run: window A, bridges 16x4 / 16x8, full 03, human look. Full table: `seams_table_v2.md`.
