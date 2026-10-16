# lipsync_jitter 04_baseline frames 0-1349 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_diff | mouth_flicker | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 471 | 1.257 | 0.857 | 0.833 | 0.825 | 0.779 | 1.007 | 1.002 | 1.032 |
| S2_switch | 120 | 1.133 | 0.934 | 0.922 | 0.942 | 1.384 | 1.007 | 1.003 | 0.891 |
| S3_head_turn | 1002 | 1.232 | 0.823 | 0.812 | 0.805 | 0.717 | 1.008 | 1.002 | 1.051 |
| S4_closed_out | 838 | 1.255 | 0.823 | 0.813 | 0.812 | 0.697 | 1.008 | 1.002 | 1.055 |
| S4_closed_master | 638 | 1.282 | 0.833 | 0.827 | 0.818 | 0.99 | 1.009 | 1.002 | 1.032 |
| ls_core | 1014 | 1.231 | 0.823 | 0.812 | 0.805 | 0.717 | 1.008 | 1.002 | 1.054 |
| orig_core | 215 | 1.023 | 1.004 | 1.005 | 1.016 | 0.844 | 1.009 | 1.003 | 1.022 |

window test (LatentSync 16-frame windows): mouth_diff step x1.308 (out 1.325, master 1.014, n=61), mouth_warp step x1.404 (out 1.414, master 1.007, n=61), mouth_flicker step x1.17 (out 1.178, master 1.006, n=61), lip_jitter step x1.138 (out 1.029, master 0.905, n=61), upper_diff step x1.0 (out 1.011, master 1.011, n=61)
