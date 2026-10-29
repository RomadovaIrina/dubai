# gate m13b vs quality baseline m11

| video | metric | baseline | new | tol | verdict |
|---|---|---:|---:|---|---|
| 01 | sync_speech | 6.654 | 4.353 | -0.3 | **FAIL** |
| 01 | flash_strong | 4 | 0 | +3 | ok |
| 01 | frame_gate_worst_pct | 0.005 | 0.005 | +0.2 | ok |
| 01 | frame_gate_over | 0 | 0 | +0 | ok |
| 01 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | joints_resid | 0.926 | 0.911 | +0.05 | ok |
| 01 | joints_lowfreq | 0.923 | 0.865 | +0.05 | ok |
| 01 | sharp_mouth | 0.946 | 0.953 | -0.03 | ok |
| 01 | chin_flicker | 0.904 | 0.914 | +0.1 | ok |
| 01 | jaw_jitter | 1.006 | 1.003 | +0.1 | ok |
| 01 | colour_worst_1s_dL | 1.11 | 1.15 | +1.0 | ok |
| 01 | spec_pass | 1 | 1 | -0 | ok |
| 01 | s_per_video_min | 94.5 | 113.8 | <= 600 | ok |
| 03 | sync_speech | 7.861 | 5.91 | -0.3 | **FAIL** |
| 03 | flash_strong | 44 | 32 | +3 | ok |
| 03 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 03 | frame_gate_over | 0 | 0 | +0 | ok |
| 03 | pressed_lips_s | 3.2 | 2.23 | +0.5 | ok |
| 03 | mouth_audio_mismatch_s | 0.77 | 1.67 | +0.5 | **FAIL** |
| 03 | joints_resid | 0.909 | 0.913 | +0.05 | ok |
| 03 | joints_lowfreq | 0.837 | 0.86 | +0.05 | ok |
| 03 | sharp_mouth | 0.833 | 0.848 | -0.03 | ok |
| 03 | chin_flicker | 0.887 | 0.89 | +0.1 | ok |
| 03 | jaw_jitter | 0.958 | 0.963 | +0.1 | ok |
| 03 | colour_worst_1s_dL | 1.33 | 1.15 | +1.0 | ok |
| 03 | spec_pass | 1 | 1 | -0 | ok |
| 03 | s_per_video_min | 480.0 | 481.6 | <= 600 | ok |
| 05 | sync_speech | 3.472 | 2.82 | -0.3 | **FAIL** |
| 05 | flash_strong | 6 | 1 | +3 | ok |
| 05 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 05 | frame_gate_over | 0 | 0 | +0 | ok |
| 05 | pressed_lips_s | 6.95 | 6.38 | +0.5 | ok |
| 05 | mouth_audio_mismatch_s | 1.84 | 2.24 | +0.5 | ok |
| 05 | joints_resid | 0.945 | 0.976 | +0.05 | ok |
| 05 | joints_lowfreq | 0.901 | 0.998 | +0.05 | **FAIL** |
| 05 | sharp_mouth | 0.959 | 0.967 | -0.03 | ok |
| 05 | chin_flicker | 0.939 | 0.937 | +0.1 | ok |
| 05 | jaw_jitter | 0.988 | 1.003 | +0.1 | ok |
| 05 | colour_worst_1s_dL | 1.65 | 1.5 | +1.0 | ok |
| 05 | spec_pass | 1 | 1 | -0 | ok |
| 05 | s_per_video_min | 414.4 | 457.2 | <= 600 | ok |
| 04 | sync_speech | 6.421 | 4.781 | -0.3 | **FAIL** |
| 04 | flash_strong | 16 | 4 | +3 | ok |
| 04 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 04 | frame_gate_over | 0 | 0 | +0 | ok |
| 04 | pressed_lips_s | 3.13 | 1.97 | +0.5 | ok |
| 04 | mouth_audio_mismatch_s | 0.73 | 1.33 | +0.5 | **FAIL** |
| 04 | joints_resid | 0.926 | 0.919 | +0.05 | ok |
| 04 | joints_lowfreq | 0.901 | 0.886 | +0.05 | ok |
| 04 | sharp_mouth | 0.927 | 0.925 | -0.03 | ok |
| 04 | chin_flicker | 0.952 | 0.948 | +0.1 | ok |
| 04 | jaw_jitter | 0.976 | 0.997 | +0.1 | ok |
| 04 | colour_worst_1s_dL | 1.41 | 1.2 | +1.0 | ok |
| 04 | spec_pass | 1 | 1 | -0 | ok |
| 04 | s_per_video_min | 482.2 | 469.1 | <= 600 | ok |
| 02 | flash_strong | 0 | 0 | +3 | ok |
| 02 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 02 | frame_gate_over | 0 | 0 | +0 | ok |
| 02 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 02 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 02 | colour_worst_1s_dL | 0 | 0 | +1.0 | ok |
| 02 | spec_pass | 1 | 1 | -0 | ok |
| 02 | s_per_video_min | 32.9 | 47.9 | <= 600 | ok |

## Verdict: **FAIL**
- 01 sync_speech: 4.353 vs baseline 6.654
- 03 sync_speech: 5.91 vs baseline 7.861
- 03 mouth_audio_mismatch_s: 1.67 vs baseline 0.77
- 05 sync_speech: 2.82 vs baseline 3.472
- 05 joints_lowfreq: 0.998 vs baseline 0.901
- 04 sync_speech: 4.781 vs baseline 6.421
- 04 mouth_audio_mismatch_s: 1.33 vs baseline 0.73
