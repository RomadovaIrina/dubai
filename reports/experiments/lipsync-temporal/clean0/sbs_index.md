# Track A side-by-side clips (server A)

Panels: ORIGINAL (25 fps master, the frames the pipeline sees) | CAND5 baseline (rendered 2026-10-16, reference only) | CLEAN0
(`--face-stack mediapipe --codeformer off`, LatentSync 1.6, same cand5 audio: decoded audio md5 identical). Same frames, H.264 crf 18,
audio = the final mix. `_zoom` = 2x crop of the lower face (box 136,404,376,604 in source pixels).

| clip | time | why picked | common metric (mouth, video/reference) cand5 -> clean0 |
|---|---|---|---|
| 03_clean0_vs_cand5_human24-25s_{full,zoom}.mp4 | 22.5-27.5 s | **human mark** (strongest twitching 24-25 s): dub "the start" fills 0.5 s of a 1.1 s source burst (mouth snaps open / shut while the source keeps talking), LatentSync window seams at 24.16 s and 24.80 s, head turn starts 24.5 s | frames 600-625 + 875-900: unexplained appearance change (resid) 1.24 -> 0.96, twitch index 1.22 -> 1.14, pixel acceleration 0.97 -> 0.74, sharpness 1.00 -> 0.71; window-start jump x1.39 -> x1.66 (3/3 seams > 1.5x) |
| 03_clean0_vs_cand5_human35-36s_{full,zoom}.mp4 | 33.5-38.5 s | **human mark** (35-36 s): dub silent 34.22-35.50 s while the source still talks (generated mouth stays shut, cheeks / jaw around it move), window seam at 35.68 s | (same pair of ranges as above) |
| 03_clean0_vs_cand5_switch44.56-headturn47.8_{full,zoom}.mp4 | 43.0-50.0 s | LS -> original 44.56 s, original -> LS 46.04 s, worst head-turn jitter 47.8 s | whole 03: resid 1.29 -> 1.02, twitch 1.064 -> 1.057 |
| 03_clean0_vs_cand5_switches82.44-83.64_{full,zoom}.mp4 | 80.0-86.0 s | switches 82.44 / 83.64 s (2026-10-12 complaint "face glitches near the end") | whole 03: upper face 1.02 both |
