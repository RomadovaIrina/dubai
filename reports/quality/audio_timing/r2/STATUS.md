# Agent C round 2 (dub-C2, RTX 5090) — English dub audio: STATUS 2026-10-17

Branch `experiment/r2-audio` (local commits, not pushed). Changelog: `CHANGES.md`. Delivery: `/workspace/dub/artifacts/audio-r2/<id>/`
(`delivery_README.md`), listening clips `/workspace/dub/artifacts/audio-r2/sbs/` (54 clips, `sbs_README.md`; original frames, lips NOT re-synced).

## What changed in what is heard (runner defaults now = the delivery; 03 checked bit-exact)
1. **Timing** `--audio-policy r2` (default): cand5 + flow 0.2 + spread pauses + slow-down floor 0.875. 01+03+04+05: LatentSync frames inside
   "speaker talks, dub silent" gaps ≥ 0.35 s 449 → 281 (−37 %), gaps > 1 s 5 → 3, longest 2.0 → 1.3 s, weak-boundary pauses 4 → 2; 03 24.5 s gap
   0.5 → 0.1 s. Reproduces yesterday's offline numbers exactly on 03 / 04 (`s2_timing_e2e.md`).
2. **Layer 3** (GPU Qwen, guarded literal / natural refit): 24 target units → 13 pass guards + Qwen judges → 4 pass Agent C's meaning review →
   2 pass the timing gate (01 u23, 04 u05; 04 LS frames in gaps 67 → 57). Qwen-7B mostly cannot lengthen a phrase faithfully and naturally;
   the long residuals (03 u04 34.6 s / u06 66.8 s, 05 u01) stay for the video side. Table for human meaning review: `refit_table.md`.
3. **Bed** Spleeter 2stems (MIT) replaces HDEMUCS (NC) by default + residue duck: no Russian voice in dub pauses (02 0.45 → 0 s), ambience no
   longer dips when the speaker talks (02 −8.9 → −3.4 dB, 03 −4.1 → +0.2 dB) (`s4_bed.md`).
4. **Foreign sounds** declick (7 vocoder / atempo clicks → 0) + gentler duck −4 dB / 250 ms (pumping steps before phrases 03 9 → 1, 04 5 → 0)
   (`s5_detectors.md`). No new hot cuts.

## Delivery numbers (dub voice: CPU Whisper WER / CTC CER; residual = gaps ≥ 0.35 s)
| video | WER / CER / bad | residual n / s / LS frames | longest |
|---|---|---|---:|
| 01 | 0.029 / 0.047 / 0 (r2 E2E 0.027 / 0.052: u13 "9 mm" → "9 millimeter" normalisation, CER unchanged) | 4 / 2.6 / 13 | 1.1 |
| 02 | 0.102 / 0.159 / 3 (cand5 0.095: u03 "chaise longues", same take) | 12 / 7.0 / 0 (no face) | 0.9 |
| 03 | 0.004 / 0.038 / 0 | 8 / 5.6 / 143 | 1.3 |
| 04 | 0.009 / 0.051 / 0 | 4 / 2.2 / 57 | 0.6 |
| 05 | 0.000 / 0.163 / 0 (CER +0.019 vs cand5: slower speech at x0.875) | 5 / 3.5 / 58 | 1.1 |

## Runtime
Audio stage (runner `--stop-after audio`, Qwen + Chatterbox + QA on the GPU, Spleeter on the CPU, cold model loads): 36–61 s per video-minute
(01 46.9, 02 35.6, 03 51.2 → 40.3 with the final defaults, 04 48.1 → 51.6, 05 60.7). ≤ 20 % of the 300 s/min budget.

## Not done / open
* Human verdicts needed: floor 0.875 (CTC CER +0.019 on 05), the refit meaning table, the duck change (clips J/K), old vs new (clips B/C).
* Lips not re-rendered on the new audio (video agents). LS frames here are estimated (MediaPipe router × gate), not from a LatentSync run.
* Canonical 01 / 02 / 05 baselines did not arrive; today's own cand5 runs are the reference (05's cand5 translation differs from yesterday's).
* The pipeline's own TTS QA ladder can accept a take with a wrong number (refit: "0.0 %" for "0.5 %" at CER 0.078); the refit QA now rejects
  it, the main `run_tts_phrase` QA does not yet.
