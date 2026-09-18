# DeepCache cache_interval A/B — LatentSync 1.6 on test_videos/04.mp4 (2026-09-18)

Setup: own audio, whole video (no speech gate), window batch 2, fp16, 20 DDIM steps, guidance 1.5, seed 1247, RTX 5090.

| setting | LatentSync s | UNet s | VAE+cond s | SyncNet conf | AV offset | PSNR vs interval 3 (mean/min/p05) |
|---|---:|---:|---:|---:|---:|---|
| cache_interval 3 (previous) | 352.0 | 255.8 | 69.1 | 3.23 | 0 | - |
| cache_interval 5 | 281.5 | 186.1 | 69.3 | 3.21 | 0 | 44.19 / 39.46 / 41.43 |
| original 04 (reference) | - | - | - | 3.24 | 2 | - |

Speedup: LatentSync x1.251, UNet x1.374. Decision: cache_interval 5 adopted as the optimized-backend default (run_clean_pipeline_05.py OPTIMIZED_DEFAULTS); --deepcache-interval 3 restores the previous setting.
