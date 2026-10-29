# gate m19b vs quality baseline m11

| video | metric | baseline | new | tol | verdict |
|---|---|---:|---:|---|---|
| 01 | sync_speech | 6.654 | 4.761 | -0.3 | **FAIL** |
| 01 | flash_strong | 4 | 0 | +3 | ok |
| 01 | frame_gate_worst_pct | 0.005 | 0.005 | +0.2 | ok |
| 01 | frame_gate_over | 0 | 0 | +0 | ok |
| 01 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | joints_resid | 0.926 | 0.929 | +0.05 | ok |
| 01 | joints_lowfreq | 0.923 | 0.932 | +0.05 | ok |
| 01 | sharp_mouth | 0.946 | 1.023 | -0.03 | ok |
| 01 | chin_flicker | 0.904 | 0.917 | +0.1 | ok |
| 01 | jaw_jitter | 1.006 | 0.974 | +0.1 | ok |
| 01 | colour_worst_1s_dL | 1.11 | 1.26 | +1.0 | ok |
| 01 | spec_pass | 1 | 1 | -0 | ok |
| 01 | s_per_video_min | 94.5 | 97.6 | <= 600 | ok |
| 03 | sync_speech | 7.861 | 6.742 | -0.3 | **FAIL** |
| 03 | flash_strong | 44 | 36 | +3 | ok |
| 03 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 03 | frame_gate_over | 0 | 0 | +0 | ok |
| 03 | pressed_lips_s | 3.2 | 3.87 | +0.5 | **FAIL** |
| 03 | mouth_audio_mismatch_s | 0.77 | 1.83 | +0.5 | **FAIL** |
| 03 | joints_resid | 0.909 | 0.89 | +0.05 | ok |
| 03 | joints_lowfreq | 0.837 | 0.8 | +0.05 | ok |
| 03 | sharp_mouth | 0.833 | 0.857 | -0.03 | ok |
| 03 | chin_flicker | 0.887 | 0.921 | +0.1 | ok |
| 03 | jaw_jitter | 0.958 | 0.935 | +0.1 | ok |
| 03 | colour_worst_1s_dL | 1.33 | 1.3 | +1.0 | ok |
| 03 | spec_pass | 1 | 1 | -0 | ok |
| 03 | s_per_video_min | 480.0 | 427.9 | <= 600 | ok |
| 05 | sync_speech | 3.472 | 4.196 | -0.3 | ok |
| 05 | flash_strong | 6 | 1 | +3 | ok |
| 05 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 05 | frame_gate_over | 0 | 0 | +0 | ok |
| 05 | pressed_lips_s | 6.95 | 5.08 | +0.5 | ok |
| 05 | mouth_audio_mismatch_s | 1.84 | 2.54 | +0.5 | **FAIL** |
| 05 | joints_resid | 0.945 | 0.982 | +0.05 | ok |
| 05 | joints_lowfreq | 0.901 | 1.036 | +0.05 | **FAIL** |
| 05 | sharp_mouth | 0.959 | 1.016 | -0.03 | ok |
| 05 | chin_flicker | 0.939 | 0.953 | +0.1 | ok |
| 05 | jaw_jitter | 0.988 | 1.029 | +0.1 | ok |
| 05 | colour_worst_1s_dL | 1.65 | 1.47 | +1.0 | ok |
| 05 | spec_pass | 1 | 1 | -0 | ok |
| 05 | s_per_video_min | 414.4 | 437.8 | <= 600 | ok |
| 04 | sync_speech | 6.421 | 5.505 | -0.3 | **FAIL** |
| 04 | flash_strong | 16 | 8 | +3 | ok |
| 04 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 04 | frame_gate_over | 0 | 0 | +0 | ok |
| 04 | pressed_lips_s | 3.13 | 2.9 | +0.5 | ok |
| 04 | mouth_audio_mismatch_s | 0.73 | 1.43 | +0.5 | **FAIL** |
| 04 | joints_resid | 0.926 | 0.943 | +0.05 | ok |
| 04 | joints_lowfreq | 0.901 | 0.913 | +0.05 | ok |
| 04 | sharp_mouth | 0.927 | 0.944 | -0.03 | ok |
| 04 | chin_flicker | 0.952 | 0.971 | +0.1 | ok |
| 04 | jaw_jitter | 0.976 | 0.984 | +0.1 | ok |
| 04 | colour_worst_1s_dL | 1.41 | 1.39 | +1.0 | ok |
| 04 | spec_pass | 1 | 1 | -0 | ok |
| 04 | s_per_video_min | 482.2 | 409.9 | <= 600 | ok |

## Verdict: **FAIL**
- 01 sync_speech: 4.761 vs baseline 6.654
- 03 sync_speech: 6.742 vs baseline 7.861
- 03 pressed_lips_s: 3.87 vs baseline 3.2
- 03 mouth_audio_mismatch_s: 1.83 vs baseline 0.77
- 05 mouth_audio_mismatch_s: 2.54 vs baseline 1.84
- 05 joints_lowfreq: 1.036 vs baseline 0.901
- 04 sync_speech: 5.505 vs baseline 6.421
- 04 mouth_audio_mismatch_s: 1.43 vs baseline 0.73
