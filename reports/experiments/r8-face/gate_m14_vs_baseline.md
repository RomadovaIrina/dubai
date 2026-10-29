# gate m14 vs quality baseline m11

| video | metric | baseline | new | tol | verdict |
|---|---|---:|---:|---|---|
| 01 | sync_speech | 6.654 | 3.564 | -0.3 | **FAIL** |
| 01 | flash_strong | 4 | 1 | +3 | ok |
| 01 | frame_gate_worst_pct | 0.005 | 0.005 | +0.2 | ok |
| 01 | frame_gate_over | 0 | 0 | +0 | ok |
| 01 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | joints_resid | 0.926 | 0.902 | +0.05 | ok |
| 01 | joints_lowfreq | 0.923 | 0.831 | +0.05 | ok |
| 01 | sharp_mouth | 0.946 | 1.031 | -0.03 | ok |
| 01 | chin_flicker | 0.904 | 0.927 | +0.1 | ok |
| 01 | jaw_jitter | 1.006 | 0.999 | +0.1 | ok |
| 01 | colour_worst_1s_dL | 1.11 | 0.97 | +1.0 | ok |
| 01 | spec_pass | 1 | 1 | -0 | ok |
| 01 | s_per_video_min | 94.5 | 108.7 | <= 600 | ok |
| 03 | sync_speech | 7.861 | 5.873 | -0.3 | **FAIL** |
| 03 | flash_strong | 44 | 32 | +3 | ok |
| 03 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 03 | frame_gate_over | 0 | 0 | +0 | ok |
| 03 | pressed_lips_s | 3.2 | 4.17 | +0.5 | **FAIL** |
| 03 | mouth_audio_mismatch_s | 0.77 | 2.63 | +0.5 | **FAIL** |
| 03 | joints_resid | 0.909 | 0.888 | +0.05 | ok |
| 03 | joints_lowfreq | 0.837 | 0.843 | +0.05 | ok |
| 03 | sharp_mouth | 0.833 | 0.862 | -0.03 | ok |
| 03 | chin_flicker | 0.887 | 0.913 | +0.1 | ok |
| 03 | jaw_jitter | 0.958 | 0.963 | +0.1 | ok |
| 03 | colour_worst_1s_dL | 1.33 | 1.16 | +1.0 | ok |
| 03 | spec_pass | 1 | 1 | -0 | ok |
| 03 | s_per_video_min | 480.0 | 539.3 | <= 600 | ok |
| 05 | sync_speech | 3.472 | 3.732 | -0.3 | ok |
| 05 | flash_strong | 6 | 1 | +3 | ok |
| 05 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 05 | frame_gate_over | 0 | 0 | +0 | ok |
| 05 | pressed_lips_s | 6.95 | 4.85 | +0.5 | ok |
| 05 | mouth_audio_mismatch_s | 1.84 | 2.5 | +0.5 | **FAIL** |
| 05 | joints_resid | 0.945 | 0.98 | +0.05 | ok |
| 05 | joints_lowfreq | 0.901 | 1.009 | +0.05 | **FAIL** |
| 05 | sharp_mouth | 0.959 | 1.015 | -0.03 | ok |
| 05 | chin_flicker | 0.939 | 0.96 | +0.1 | ok |
| 05 | jaw_jitter | 0.988 | 1.006 | +0.1 | ok |
| 05 | colour_worst_1s_dL | 1.65 | 1.33 | +1.0 | ok |
| 05 | spec_pass | 1 | 1 | -0 | ok |
| 05 | s_per_video_min | 414.4 | 484.7 | <= 600 | ok |
| 04 | sync_speech | 6.421 | 5.549 | -0.3 | **FAIL** |
| 04 | flash_strong | 16 | 4 | +3 | ok |
| 04 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 04 | frame_gate_over | 0 | 0 | +0 | ok |
| 04 | pressed_lips_s | 3.13 | 2.73 | +0.5 | ok |
| 04 | mouth_audio_mismatch_s | 0.73 | 0.73 | +0.5 | ok |
| 04 | joints_resid | 0.926 | 0.932 | +0.05 | ok |
| 04 | joints_lowfreq | 0.901 | 0.917 | +0.05 | ok |
| 04 | sharp_mouth | 0.927 | 0.962 | -0.03 | ok |
| 04 | chin_flicker | 0.952 | 0.977 | +0.1 | ok |
| 04 | jaw_jitter | 0.976 | 0.989 | +0.1 | ok |
| 04 | colour_worst_1s_dL | 1.41 | 1.19 | +1.0 | ok |
| 04 | spec_pass | 1 | 1 | -0 | ok |
| 04 | s_per_video_min | 482.2 | 477.0 | <= 600 | ok |
| 02 | flash_strong | 0 | 0 | +3 | ok |
| 02 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 02 | frame_gate_over | 0 | 0 | +0 | ok |
| 02 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 02 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 02 | colour_worst_1s_dL | 0 | 0 | +1.0 | ok |
| 02 | spec_pass | 1 | 1 | -0 | ok |
| 02 | s_per_video_min | 32.9 | 46.7 | <= 600 | ok |

## Verdict: **FAIL**
- 01 sync_speech: 3.564 vs baseline 6.654
- 03 sync_speech: 5.873 vs baseline 7.861
- 03 pressed_lips_s: 4.17 vs baseline 3.2
- 03 mouth_audio_mismatch_s: 2.63 vs baseline 0.77
- 05 mouth_audio_mismatch_s: 2.5 vs baseline 1.84
- 05 joints_lowfreq: 1.009 vs baseline 0.901
- 04 sync_speech: 5.549 vs baseline 6.421
