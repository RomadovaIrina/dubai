# Agent C r2 — new English dub audio for the video agents (2026-10-17)

One self-contained dir per video, built by `scripts/quality/r2_deliver.py` (branch `experiment/r2-audio`). The same audio comes out of
`run_clean_pipeline_05.py` with its new defaults (checked bit-exact on 03); 01 / 04 additionally carry the accepted layer-3 refit units.

| file | use |
|---|---|
| `dubbed_16k.wav` | **LatentSync input + speech gate** (clean dub voice, 16 kHz) |
| `final_mix_aac.m4a` | **audio to mux** (dub voice + Spleeter 2stems bed, residue duck, duck -4 dB / 250 ms / 600 ms) |
| `final_mix_44k.wav`, `dubbed_24k.wav`, `dubbed_aac.m4a` | the same as wav / the voice only |
| `<id>_r2.manifest.json` | runner-schema manifest, all paths inside this dir: `units[].alignment.placements` (placed spans, atempo), `speech_gate` (source VAD ∪ dub VAD, margin 0.24 s, merge 0.6 s = what the optimized backend builds), `background`, `audio_policy`, `dubbed_track.declick`, `delivery` |
| `<id>_r2.residual.json` | remaining "speaker talks, dub silent" gaps: `t0/t1`, `frames` = master_25fps frame indices [f0, f1), `ls_frames`, `class` (natural < 0.35 s / residual), `position_in_burst`, `source_words` — for the silent-face handling |
| `<id>_r2.gaps.json` | the full dub_gaps.py report behind it |
| `audio16k.wav`, `aligned_u*.wav`, `tts_u*` | source audio, per-unit placed tracks, raw / cut TTS (for tracing) |

`candidate_video_stage.py --dubbed-dir /workspace/dub/artifacts/audio-r2/<id>` uses `dubbed_16k.wav` + `final_mix_aac.m4a`. Lips must be re-synced
(the old video was rendered on the cand5 audio).

| video | s | units | refit units | residual gaps n / s / LS frames | longest gap s | bed residue duck | clicks repaired |
|---|---:|---:|---|---|---:|---:|---:|
| 01 | 101.7 | 26 | u23 | 4 / 2.6 s / 13 | 1.1 | 0.0 s | 6 |
| 02 | 110.9 | 17 | – | 12 / 7.0 s / 0 | 0.9 | 0.6 s | 0 |
| 03 | 90.3 | 10 | – | 8 / 5.6 s / 143 | 1.3 | 0.15 s | 0 |
| 04 | 53.9 | 6 | u05 | 4 / 2.2 s / 57 | 0.6 | 0.0 s | 0 |
| 05 | 52.4 | 7 | – | 5 / 3.5 s / 58 | 1.1 | 0.36 s | 1 |

LS frames are estimated with the MediaPipe router × the speech gate (`face_ref.py`; matches the real cand5 LatentSync runs on 03 / 04).
Listening clips: `sbs/` (README inside; original frames, lips NOT re-synced).
