# Spec status (round 3, Agent D)

Columns: one per reference run. Cell = verdict + numbers per video (01 / 02 / 03 / 04 / 05). cand5: the canonical baseline reports (its mp4s are gone and cannot be re-rendered without the non-commercial models), so its video items are NOT MEASURED unless the report has them.

| spec | item | cand5 | clean0 | r2m25 | r2mfix30 |
|---|---|---|---|---|---|
| 9.4 / 5.6 | fps, resolution, frames, duration, H.264 vs source | **FAIL** (25 fps CFR master muxed; coordinator gap table 17.10) | NOT MEASURED | **FAIL** 04: 25fps 1349/1617f 464x848 Δ+0.06s h264 | NOT MEASURED |
| 8.5 | video jumps at LatentSync window starts (mouth d1 step vs inside the window; source = same frames) | x1.38 step, 41 % > 1.5x (03 only, CLEAN0.md) | NOT MEASURED | **FAIL** 04: 64 starts, >1.5x 0.30 (src 0.19), >2x 0.09 | NOT MEASURED |
| 8.5 | LS ↔ original switches: switch-zone frames, mouth d1 / p95 vs source (1 = like the source) | NOT MEASURED | NOT MEASURED | 04: 12 fr, d1 2.94 p95 2.12 | NOT MEASURED |
| 8.5 | audio clicks at joints (C's detector on the muxed audio), hot unit edges, bed steps | 01: clicks 4 / 02: clicks 1 / 03: clicks 1 / 04: clicks 0 / 05: clicks 0 | NOT MEASURED | PASS 04: clicks 0, hot edges 2, bed steps 12 | NOT MEASURED |
| 8.4 | mask feather 10-90 % edge on the output (px) | NOT MEASURED (same Gaussian rule as clean0) | NOT MEASURED | **FAIL** 04: 9.0 px | NOT MEASURED |
| 12.1 | SyncNet confidence / AV offset, final mix and dub-only | 01: mix 7.890, dub 7.860, AV 0 / 02: mix N/A, dub -, AV - / 03: mix 5.460, dub 5.690, AV 0 / 04: mix 3.690, dub 3.810, AV 0 / 05: mix 3.150, dub 3.660, AV 0 | NOT MEASURED | REPORT (threshold UNDECIDED) 04: mix 3.62/+0, dub 3.73/+0 | NOT MEASURED |
| 12.2 | Whisper large-v3 (GPU) back-transcription of the FINAL MIX vs the synthesised English text | 01: WER 0.019 CER(ctc) 0.051 / 02: WER 0.073 CER(ctc) 0.168 / 03: WER 0.009 CER(ctc) 0.043 / 04: WER 0.009 CER(ctc) 0.048 / 05: WER 0.000 CER(ctc) 0.169 (per-unit dub voice) | NOT MEASURED | 04: WER 0.024 | NOT MEASURED |
| 11.1-11.4 | replaced / invalid frames (held landmarks + composite fallbacks) vs the 2 % rule | NOT MEASURED | NOT MEASURED | PASS 04: 0 fr = 0.00 % | NOT MEASURED |
| 5.4 / 5.5 | head yaw / pitch per frame (MediaPipe matrix), share > 30°, would the 30 % pre-check fire | NOT MEASURED (no pose in cand5) | NOT MEASURED | pre-check fires on none 04: yaw p99 20.7° max 23.4°, pitch p99 11.4°, >30° 0.0 % | NOT MEASURED |
| 8.2 | second face present? only the active speaker touched? | NOT MEASURED | NOT MEASURED | single face everywhere (8.2 path not exercised)  04: tracks 1, 2-face share 0.00 | NOT MEASURED |
| norm | s / video-min (stage totals of the run) | 01: 548.8 s wall / 02: 255.9 s wall / 03: 880.9 s wall / 04: 425.9 s wall / 05: 499.6 s wall | NOT MEASURED | **FAIL** 04: 414 s/video-min (video stage only; audio + FIX30 see gpu_audit.md) | NOT MEASURED |

Per-video detail: `/workspace/dub/artifacts/r3d/score/<id>_<label>.score.json` (+ `.mjhr.json`); table of all metrics: `scorecard.md`.
