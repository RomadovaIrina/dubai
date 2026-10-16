# lipsync_jitter 03_cand5 frames 0-2260 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_pix_hf | mouth_diff | mouth_flicker | lip_hf | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 1174 | 1.354 | 0.821 | 0.887 | 0.838 | 0.777 | 0.788 | 0.713 | 1.025 | 1.016 | 0.834 |
| S2_switch | 80 | 1.183 | 0.992 | 1.021 | 1.007 | 0.962 | 1.025 | 1.148 | 1.017 | 1.013 | 0.849 |
| S3_head_turn | 175 | 1.402 | 0.895 | 0.962 | 0.918 | 0.816 | 0.827 | 0.706 | 1.007 | 1.006 | 0.864 |
| S4_closed_out | 1234 | 1.275 | 0.827 | 0.859 | 0.849 | 0.782 | 0.786 | 0.561 | 1.022 | 1.014 | 0.844 |
| S4_closed_master | 733 | 1.342 | 0.888 | 0.927 | 0.916 | 0.81 | 0.809 | 0.592 | 1.019 | 1.013 | 0.882 |
| ls_core | 1985 | 1.309 | 0.825 | 0.875 | 0.846 | 0.775 | 0.783 | 0.64 | 1.024 | 1.015 | 0.835 |
| orig_core | 195 | 1.018 | 1.013 | 1.013 | 1.014 | 1.018 | 0.995 | 0.979 | 1.019 | 1.012 | 1.016 |

lip trajectory energy share 6-12.5 Hz (LS runs >= 64 frames): {'lip_hf_band_fraction_out': 0.1059, 'lip_hf_band_fraction_master': 0.0957, 'ratio': 1.107}

window test (LatentSync 16-frame windows): mouth_diff step x1.508 (out 1.472, master 0.977, n=123), mouth_warp step x1.491 (out 1.48, master 0.993, n=123), mouth_flicker step x1.287 (out 1.282, master 0.995, n=123), mouth_pix_hf step x1.259 (out 1.248, master 0.992, n=123), lip_hf step x1.25 (out 1.227, master 0.981, n=123), lip_jitter step x1.264 (out 1.238, master 0.979, n=123), upper_diff step x1.001 (out 0.948, master 0.947, n=123)
