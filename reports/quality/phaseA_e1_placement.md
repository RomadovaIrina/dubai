# Phase A — E1 burst placement of the EXISTING TTS (04, 05), no translation / TTS / whisper changes (2026-09-29)

Runner: `scripts/quality/e1_burst_align.py --id <id> --out /tmp/dabai_quality/candidates/e1` (defaults: burst mode, extend on, atempo cap 1.3,
hard cap 1.6, no slow-down) on the baseline manifests + `tts_u*.wav` of this host; video stage: `candidate_video_stage.py --tag e1
--dubbed-dir ... --codeformer off` (LS-only, so SyncNet compares with the LS-only baseline); metrics: `quality_manifest.py --syncnet`,
new `dub_coverage.py` (Silero VAD of original vs dubbed track), new `e1_cut_words.py` (faster-whisper word timestamps of every
TTS clip vs E1's cut positions). Artifacts: `result_videos/week3_audio/phaseA_e1/` (baseline + E1 `dubbed_24k.wav`, E1 mp4, manifests,
`e1_alignment.json`, all metric JSON).

## Baseline vs E1

| metric | 04 baseline | 04 E1 | 05 baseline | 05 E1 |
|---|---:|---:|---:|---:|
| LS frames | 1099 | 982 | 899 | 878 |
| LS frames with silent dub (RMS < -50 dB) | 496 | 358 | 458 | 454 |
| …while original speaks, RMS definition (orig > -35 dB) — the pilot metric | **423** | **229** | **458** | **453** |
| …while original speaks, VAD definition (Silero VAD of the source) | 396 | 187 | 335 | 286 |
| VAD IoU (original speech vs dubbed speech) | 0.356 | 0.678 | 0.348 | 0.509 |
| original-speech seconds without dub | 15.4 | 7.8 | 15.3 | 11.3 |
| dubbed-speech seconds outside original speech | 10.1 | 2.5 | 8.3 | 4.7 |
| original bursts covered < 50 % / not at all | 9 / 3 of 15 | 1 / 0 | 10 / 6 of 17 | 6 / 0 |
| SyncNet (LS-only output; original 3.24 / 1.71) | 2.76 | **3.42** | 2.61 | **2.88** |
| AV offset (frames) | 0 | 0 | 0 | 0 |
| atempo: groups, max, > 1.15, > 1.25, > 1.3 | 5 units, 1.0, 0, 0, 0 | 19 groups, 1.30, 1, 1, 0 | 7 units, 1.0, 0, 0, 0 | 20 groups, 1.30, 1, 1, 0 |
| overflow / cut-at-slot-end groups | – | 1 / 0 | – | 1 / 1 |
| E1 valley cuts inside a word (of tight cuts) | – | **11 / 14** | – | 4 / 14 |
| original timeline | unchanged | unchanged | unchanged | unchanged |

Why the pilot metric does not move on 05: the 05 source track is above -35 dB on 896 of its 899 LS frames (outdoor noise), so
"original speaks" is always true there and the metric collapses to "dub is silent"; with Silero VAD as the speaking criterion the
same runs give 04: 396 -> 187 (-53 %), 05: 335 -> 286 (-15 %). The VAD definition is used as the primary target from here on
(`dub_coverage.py`); the RMS one is kept for continuity with the pilot numbers.

## Reading
1. Placement alone is worth a lot where the TTS has enough material: 04 loses half of the silent-while-speaking frames, every
   burst gets dub, dub outside original speech drops from 10.1 s to 2.5 s, and SyncNet rises by +0.66 (2.76 -> 3.42, above the
   original's 3.24) with AV offset 0 — lip-synced frames now mostly have speech in the driving audio.
2. The residual is the TTS budget, not the placement: the English TTS is shorter than the Russian speech (04: 24.2 s vs 29.5 s,
   05: 20.9 s vs 27.9 s), so every burst gets a fragment placed at its start and the burst tail stays silent (05: 6 bursts < 50 %
   covered, 11.3 s of original speech without dub). Filling bursts needs either a slow-down of short chunks (`--fill-slowdown`, not
   used here) or translations that are not shorter than the source.
3. E1 as it stands fails the "no cut words" criterion: with 1-2 VAD phrases per TTS clip it splits clips at energy valleys to get
   one piece per burst, and 11 of 14 such cuts on 04 (4 of 14 on 05) fall inside a word ('glasses', 'August', 'today', 'people's' …).
   The durations-only grouping also assigns text to bursts without any semantic correspondence.
4. Hence Phase B: chunks must be produced as whole phrases — one translated part per source burst (word groups from the whisper
   word timestamps), TTS per part — and placed with E1's window/extension/atempo logic (refactored into a shared function, not
   duplicated). WhisperX is not needed for this step; the cut-word check above already uses faster-whisper on the TTS.

Hypothesis confirmed (placement matters), E1 not shippable as is (cut words, duration-only grouping, unfilled burst tails).
