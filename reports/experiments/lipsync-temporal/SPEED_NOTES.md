# Notes for the later speed phase (recorded 2026-10-16, not optimised yet — quality first)

## MediaPipe face-stack pass: 44 s per video-minute on CPU (04 E2E: 40 s for 53.9 s; 03: 55 s for 90.3 s)

| option | crop stability (LatentSync anchors after ±2 smoothing, 03 master) | anchor error vs every frame | expected cost |
|---|---|---|---|
| every frame (today) | accel 0.352 / p95 0.833 px | – | 44 s/video-min (CPU) |
| landmarks every 2nd frame, linear interpolation | 0.347 / 0.833 px | mean 0.028 px, p95 0.167 px | ~22 s/video-min |
| every 3rd frame | 0.344 / 0.667 px | mean 0.059 px, p95 0.175 px | ~15 s/video-min |
| every 4th frame | 0.343 / 0.667 px | mean 0.107 px, p95 0.333 px | ~11 s/video-min |
| GPU delegate (`BaseOptions.Delegate.GPU`) | – | – | initialises in this container (EGL available); per-frame speed not measured yet |
| run it in parallel with the audio stage | unchanged | 0 | off the critical path (depends only on the master) |

Reading: subsampling every 2nd-3rd frame keeps the crop as stable as today (interpolation is smoother than detector noise) with
sub-pixel anchor error on 03; check on 05 (head turns up to 62 deg) before adopting. The cheapest zero-risk lever is running the pass in
parallel with the audio stage, like `--video-prep parallel` did for the old detectors.

## Seam fixes: cost
24-frame windows +0-1 %; 1-frame continuation on 24-frame windows ~ +4 % (pending); 2-frame continuation +12 %; per-step fusion needs
DeepCache off (x2) — see SEAMS.md.
