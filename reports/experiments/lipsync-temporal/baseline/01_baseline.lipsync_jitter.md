# lipsync_jitter 01_baseline frames 0-2542 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_diff | mouth_flicker | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 229 | 1.274 | 1.086 | 1.166 | 1.048 | 1.24 | 1.013 | 1.008 | 2.123 |
| S2_switch | 10 | 0.992 | 0.841 | 1.022 | 1.232 | 1.023 | 1.024 | 1.029 | 1.465 |
| S3_head_turn | 175 | 1.247 | 1.077 | 1.154 | 1.037 | 1.274 | 1.01 | 1.007 | 2.132 |
| S4_closed_out | 6 | 1.494 | 1.26 | 1.357 | 0.952 | 1.173 | 1.003 | 0.994 | 2.091 |
| S4_closed_master | 0 | – | – | – | – | – | – | – | – |
| ls_core | 255 | 1.257 | 1.088 | 1.179 | 1.054 | 1.195 | 1.011 | 1.007 | 2.128 |
| orig_core | 2277 | – | – | – | – | – | – | – | – |

window test (LatentSync 16-frame windows): mouth_diff step x1.467 (out 1.31, master 0.893, n=15), mouth_warp step x1.416 (out 1.437, master 1.015, n=15), mouth_flicker step x1.094 (out 1.157, master 1.058, n=15), lip_jitter step x0.829 (out 0.975, master 1.175, n=15), upper_diff step x0.998 (out 0.847, master 0.848, n=15)
