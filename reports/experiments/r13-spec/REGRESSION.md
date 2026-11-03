# 12.3 release-candidate regression (r13, 03.11)

**Run:**
- Every video through the **backend interface** (`scripts/r13/dabai_job.py job.json`) with the defaults:
  - subtitles off;
  - k5 sync fix;
  - hallucination filter, pre-check;
  - `max_voices 2`.
- One job at a time on r13-t (RTX 5090). Script: `/workspace/dub/r13s/reg.sh`. Table: `scripts/r13/reg_table.py` → `data/regression.json`.
- SyncNet = the gate's `sync_speech` (`scripts/r11/sync_speech.py`), of the output vs its dub. The source is scored vs its own audio, for 0.8.

| video | min | status | flags | wall s/vmin | GPU-busy s/vmin (NVML busy / weighted) | VRAM peak GB | SyncNet out (r12 k5) | SyncNet source |
|---|---:|---|---|---:|---|---:|---|---:|
| m03 | 0.68 | SUCCESS | two_faces | 735 | 323 / 286 | 29.7 | 8.16 (8.21) | 5.50 |
| m15 | 0.56 | SUCCESS | – | 761 | 358 / 314 | 26.2 | 7.46 (–) | 6.37 |
| m18 | 0.33 | SUCCESS | – | 900 | 349 / 302 | 25.9 | 8.35 (8.33) | 3.61 |
| old01 | 1.69 | SUCCESS | variable_fps, low_res, long_video, face_partly_absent | 178 | 75 / 62 | 30.1 | 6.46 (–) | 4.49 |
| old02 | 1.85 | SUCCESS | low_res, long_video, no_face → voice only | 107 | 81 / 61 | 30.1 | – | – |
| old03 | 1.51 | SUCCESS | low_res, long_video | 542 | 320 / 290 | 29.5 | 7.58 (7.57) | 5.50 |
| old04 | 0.90 | SUCCESS | low_res | 562 | 316 / 275 | 29.7 | 6.05 (–) | 4.70 |
| old05 | 0.87 | SUCCESS | low_res, face_partly_absent | 545 | 339 / 308 | 29.5 | 4.31 (4.23) | 1.14 |

## Verdict: no regression vs r12
- **Status:** 8 / 8 SUCCESS through the interface. `status.json` is complete: flags, user messages, the output path, video minutes, GPU-busy. The output format equals the source on every video.
- **SyncNet = the r12 k5 baseline, within noise:** m18 8.35 vs 8.33, m03 8.16 vs 8.21, old03 7.58 vs 7.57, old05 4.31 vs 4.23. 01 / 04 / m15 had no r12 number; they are recorded here as the new baseline.
- **Audio:** unchanged by default. The r13 features are either off or only filter / flag; the hallucination filter dropped 0 segments on 01–05.
- **Timings:**
  - GPU-busy (NVML) is 316–358 s/vmin on the face videos, 264 s/vmin over the videos with a face, 223 over all 8.
  - Single face videos are still above the 300 norm. That is the face (generator + composite ≈ 280); the audio is ~44 weighted (AUDIO_SPEED.md).
  - The short clips (m18 0.33 min) carry fixed per-job costs.
- **Tempo (7.3):** every recorded atempo is within ×0.75–1.25; see TEMPO.md.

New per-job number for the backend: `gpu_busy_s` in `status.json` (old03: 486 s for 1.51 video-min).
