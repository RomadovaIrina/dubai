# LatentSync window-batch sweep (test_videos/04.mp4)

Source: 53.90 s. Baseline = upstream sequential pipeline, DeepCache, fp16, 20 steps, seed 1247.  
Largest SAFE batch (PASS, valid output, no OOM): **4** · fastest measured batch: **4** · OOM at: [16]

| label | batch | DeepCache | compile | status | cold s | warm s | s/video-min | speedup | proc peak MiB | torch reserved MiB | frames | dur s | PSNR vs baseline (mean/min) |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | True | none | PASS | 348.041 | 348.502 | 387.943 | 1.0 | 20878 | 19656.0 | 1349 | 53.96 | - |
| accel_b1 | 1 | True | none | PASS | 348.69 | 348.407 | 387.837 | 1.0 | 20536 | 19658.0 | 1349 | 53.96 | 99.0/99.0 |
| accel_b16 | 16 | True | none | OOM OOM | - | - | - | - | - | 29392.0 | - | - | - |
| accel_b2 | 2 | True | none | PASS | 341.134 | 341.103 | 379.706 | 1.022 | 25788 | 24910.0 | 1349 | 53.96 | 40.85/35.65 |
| accel_b4 | 4 | True | none | PASS | 338.616 | 338.822 | 377.167 | 1.029 | 30780 | 29902.0 | 1349 | 53.96 | 41.09/35.64 |
| accel_b8 | 8 | True | none | FAIL | - | - | - | - | - | - | - | - | - |
