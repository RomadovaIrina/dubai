# lipsync_jitter 03_baseline frames 0-2260 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_diff | mouth_flicker | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 1174 | 1.161 | 0.876 | 0.836 | 0.893 | 0.762 | 1.024 | 1.017 | 0.784 |
| S2_switch | 80 | 1.093 | 1.008 | 1.0 | 1.108 | 0.854 | 1.021 | 1.014 | 0.845 |
| S3_head_turn | 178 | 1.203 | 0.929 | 0.91 | 0.913 | 0.624 | 1.015 | 1.008 | 0.808 |
| S4_closed_out | 953 | 1.085 | 0.849 | 0.857 | 0.828 | 0.444 | 1.022 | 1.015 | 0.786 |
| S4_closed_master | 112 | 1.204 | 1.029 | 1.066 | 1.019 | 0.697 | 1.024 | 1.017 | 0.778 |
| ls_core | 1985 | 1.127 | 0.862 | 0.842 | 0.868 | 0.668 | 1.024 | 1.017 | 0.784 |
| orig_core | 195 | 1.018 | 1.014 | 1.015 | 1.0 | 1.014 | 1.014 | 1.013 | 1.01 |

window test (LatentSync 16-frame windows): mouth_diff step x1.513 (out 1.46, master 0.965, n=123), mouth_warp step x1.454 (out 1.426, master 0.98, n=123), mouth_flicker step x1.281 (out 1.261, master 0.984, n=123), lip_jitter step x1.085 (out 1.054, master 0.971, n=123), upper_diff step x1.0 (out 0.968, master 0.969, n=123)
