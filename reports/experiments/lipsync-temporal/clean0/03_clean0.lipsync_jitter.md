# lipsync_jitter 03_clean0 frames 0-2260 (output / master ratios; 1.0 = as calm as the source)

| situation | frames | mouth_warp | mouth_pix_hf | mouth_diff | mouth_flicker | lip_hf | lip_jitter | open_jitter | upper_warp | upper_diff | mouth_sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_speech | 1174 | 1.12 | 0.712 | 0.807 | 0.721 | 0.743 | 0.759 | 0.724 | 1.025 | 1.016 | 0.408 |
| S2_switch | 80 | 1.014 | 0.904 | 0.939 | 0.917 | 0.956 | 1.039 | 0.788 | 1.016 | 1.012 | 0.799 |
| S3_head_turn | 175 | 1.109 | 0.75 | 0.854 | 0.761 | 0.778 | 0.799 | 0.724 | 1.006 | 1.006 | 0.431 |
| S4_closed_out | 939 | 1.023 | 0.736 | 0.78 | 0.75 | 0.786 | 0.797 | 0.643 | 1.022 | 1.014 | 0.436 |
| S4_closed_master | 733 | 1.064 | 0.76 | 0.821 | 0.778 | 0.806 | 0.818 | 0.668 | 1.022 | 1.014 | 0.452 |
| ls_core | 1985 | 1.066 | 0.715 | 0.788 | 0.727 | 0.756 | 0.771 | 0.668 | 1.026 | 1.016 | 0.428 |
| orig_core | 195 | 1.021 | 1.016 | 1.015 | 1.017 | 1.011 | 0.997 | 0.917 | 1.02 | 1.013 | 1.014 |

lip trajectory energy share 6-12.5 Hz (LS runs >= 64 frames): {'lip_hf_band_fraction_out': 0.1014, 'lip_hf_band_fraction_master': 0.0957, 'ratio': 1.06}

window test (LatentSync 16-frame windows): mouth_diff step x1.647 (out 1.608, master 0.977, n=123), mouth_warp step x1.657 (out 1.645, master 0.993, n=123), mouth_flicker step x1.428 (out 1.421, master 0.995, n=123), mouth_pix_hf step x1.376 (out 1.365, master 0.992, n=123), lip_hf step x1.371 (out 1.346, master 0.981, n=123), lip_jitter step x1.412 (out 1.382, master 0.979, n=123), upper_diff step x1.0 (out 0.947, master 0.947, n=123)
