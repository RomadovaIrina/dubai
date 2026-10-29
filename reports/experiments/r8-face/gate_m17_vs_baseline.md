# gate m17 vs quality baseline m11

| video | metric | baseline | new | tol | verdict |
|---|---|---:|---:|---|---|
| 01 | sync_speech | 6.654 | 3.886 | -0.3 | **FAIL** |
| 01 | flash_strong | 4 | 0 | +3 | ok |
| 01 | frame_gate_worst_pct | 0.005 | 0.005 | +0.2 | ok |
| 01 | frame_gate_over | 0 | 0 | +0 | ok |
| 01 | pressed_lips_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | mouth_audio_mismatch_s | 0.0 | 0.0 | +0.5 | ok |
| 01 | joints_resid | 0.926 | 0.923 | +0.05 | ok |
| 01 | joints_lowfreq | 0.923 | 0.862 | +0.05 | ok |
| 01 | sharp_mouth | 0.946 | 1.007 | -0.03 | ok |
| 01 | chin_flicker | 0.904 | 0.901 | +0.1 | ok |
| 01 | jaw_jitter | 1.006 | 0.985 | +0.1 | ok |
| 01 | colour_worst_1s_dL | 1.11 | 1.31 | +1.0 | ok |
| 01 | spec_pass | 1 | 1 | -0 | ok |
| 01 | s_per_video_min | 94.5 | 108.2 | <= 600 | ok |
| 03 | sync_speech | 7.861 | 6.109 | -0.3 | **FAIL** |
| 03 | flash_strong | 44 | 34 | +3 | ok |
| 03 | frame_gate_worst_pct | 0.0 | 0.0 | +0.2 | ok |
| 03 | frame_gate_over | 0 | 0 | +0 | ok |
| 03 | pressed_lips_s | 3.2 | 3.7 | +0.5 | ok |
| 03 | mouth_audio_mismatch_s | 0.77 | 1.87 | +0.5 | **FAIL** |
| 03 | joints_resid | 0.909 | 0.893 | +0.05 | ok |
| 03 | joints_lowfreq | 0.837 | 0.818 | +0.05 | ok |
| 03 | sharp_mouth | 0.833 | 0.845 | -0.03 | ok |
| 03 | chin_flicker | 0.887 | 0.908 | +0.1 | ok |
| 03 | jaw_jitter | 0.958 | 0.946 | +0.1 | ok |
| 03 | colour_worst_1s_dL | 1.33 | 1.31 | +1.0 | ok |
| 03 | spec_pass | 1 | 1 | -0 | ok |
| 03 | s_per_video_min | 480.0 | 500.3 | <= 600 | ok |
| 05 | sync_speech | 3.472 | 3.648 | -0.3 | ok |
| 05 | flash_strong | 6 | 3 | +3 | ok |
| 05 | pressed_lips_s | 6.95 | 4.88 | +0.5 | ok |
| 05 | mouth_audio_mismatch_s | 1.84 | 1.54 | +0.5 | ok |
| 05 | joints_resid | 0.945 | 0.983 | +0.05 | ok |
| 05 | joints_lowfreq | 0.901 | 1.043 | +0.05 | **FAIL** |
| 05 | sharp_mouth | 0.959 | 1.009 | -0.03 | ok |
| 05 | chin_flicker | 0.939 | 0.95 | +0.1 | ok |
| 05 | jaw_jitter | 0.988 | 1.02 | +0.1 | ok |
| 05 | colour_worst_1s_dL | 1.65 | 1.49 | +1.0 | ok |
| 05 | spec_pass | 1 | 1 | -0 | ok |
| 05 | s_per_video_min | 414.4 | 438.8 | <= 600 | ok |

## Verdict: **FAIL**
- 01 sync_speech: 3.886 vs baseline 6.654
- 03 sync_speech: 6.109 vs baseline 7.861
- 03 mouth_audio_mismatch_s: 1.87 vs baseline 0.77
- 05 joints_lowfreq: 1.043 vs baseline 0.901
