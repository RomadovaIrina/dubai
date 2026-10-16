# lipsync_jitter 05_baseline frames 0-1311 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_diff | mouth_flicker | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 346 | 1.099 | 0.903 | 0.919 | 0.933 | 0.981 | 0.997 | 1.0 | 1.289 |
| S2_switch | 100 | 1.053 | 0.976 | 0.98 | 0.936 | 0.875 | 1.002 | 1.003 | 1.048 |
| S3_head_turn | 844 | 1.103 | 0.892 | 0.915 | 0.94 | 0.977 | 1.002 | 1.001 | 1.318 |
| S4_closed_out | 172 | 1.087 | 0.885 | 0.921 | 0.938 | 1.018 | 1.003 | 1.002 | 1.285 |
| S4_closed_master | 21 | 1.226 | 0.962 | 1.055 | 0.895 | 1.079 | 1.01 | 1.006 | 1.238 |
| ls_core | 880 | 1.108 | 0.889 | 0.915 | 0.944 | 0.97 | 1.002 | 1.001 | 1.319 |
| orig_core | 331 | 1.009 | 1.003 | 1.005 | 0.985 | 1.081 | 1.005 | 1.002 | 1.006 |

window test (LatentSync 16-frame windows): mouth_diff step x1.378 (out 1.438, master 1.043, n=54), mouth_warp step x1.604 (out 1.595, master 0.994, n=54), mouth_flicker step x1.251 (out 1.185, master 0.947, n=54), lip_jitter step x1.005 (out 0.944, master 0.939, n=54), upper_diff step x1.0 (out 1.079, master 1.079, n=54)
