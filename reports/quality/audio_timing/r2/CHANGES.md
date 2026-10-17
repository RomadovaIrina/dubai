# Agent C round 2 (dub-C2, RTX 5090): English dub audio — changelog

Branch `experiment/r2-audio` (from `experiment/audio-timing` 07f4444). Local commits only (no push until the coordinator says so).
One entry per change: what / why / files / result, commit SHA (the SHA of an entry is the commit that adds it; `git log --oneline` for the rest).

## 1. Runner: `--audio-policy` (default `r2`) + clean bed `--bed-separator spleeter` (default)
* **What**: `run_clean_pipeline_05.py --audio-policy legacy|cand5|r2`, default `r2` = cand5 audio flags + `--sentence-flow-gap 0.2 --spread pauses
  --fill-slowdown 0.875` (yesterday's L1 + floor 0.875); explicit flags still win; the resolved values go to `manifest.audio_policy`.
  `--bed-separator spleeter|hdemucs` (default spleeter): `background_bed.separate_spleeter` runs Spleeter 2stems in its own TF venv on the CPU;
  HDEMUCS (weights without a licence) is refused unless `DUB_ALLOW_NC_BED=1`. `scripts/setup_spleeter.sh` builds the venv + sha256-checked weights.
* **Why**: the chosen timing policy was opt-in only; `--background on` (the default) still loaded the non-commercial HDEMUCS weights.
* **Files**: `scripts/pilot/run_clean_pipeline_05.py`, `scripts/pilot/background_bed.py`, `scripts/setup_spleeter.sh`.
* **Result**: Spleeter on 03 (90 s): 4.6 s wall incl. TF load (16 threads); the HDEMUCS guard raises. Reproduction of the offline numbers: entry 2.

## 2. Step 2 verified in the real pipeline: r2 vs cand5 E2E on 01–05 (`s2_timing_e2e.md`)
* **What**: audio-only E2E runs (`--stop-after audio`, Qwen on the GPU) with `--audio-policy cand5` and `r2`; `face_ref.py` (LS mask from the
  MediaPipe router × speech gate, no video run); `r2_metrics.sh`; `eval_cpu.py` tags `<root>_work_<id>` (cand5 / r2 no longer overwrite each other).
* **Why**: show that the integrated defaults reproduce yesterday's offline numbers (canonical baselines only partly uploaded: own baselines).
* **Result**: r2 and cand5 share all 66 TTS takes; 03 / 04 reproduce yesterday's offline table exactly, 01 within 0.2 s. 01+03+04+05: LS frames in
  gaps ≥ 0.35 s 449 → 281 (−37 %), gaps > 1 s 5 → 3, longest 2.0 → 1.3 s, weak pauses 4 → 2; WER unchanged except 02 +0.007 (one word);
  CTC CER 05 +0.019. Audio stage 36–61 s per video-minute.

## 3. Layer 3 refit on the GPU (`refit_table.md`, `refit/<id>/`)
* **What**: `refit_offline.py` v4: Qwen on the GPU, 5 alternatives per call under a JSON schema (unconstrained answers were malformed), natural-
  spoken-English prompt + `natural_check` judge, guards for sentence end / dangling word, mid-sentence capital, added vocative ("…, folks"),
  content bigram repeated across a cut; QA rejects a take whose ASR loses a number (01 u14 take 0 said "0.0 %" at CER 0.078); `build --ref-manifest`.
* **Why**: v3 (yesterday) accepted 6/17 with 2 wrong; with Qwen-7B judging itself, defects pass ("stunning swimming pool", "such a nice | what a nice").
* **Result**: 24 target units; guards + judges passed 13 candidates (v3 + v4), Agent C meaning review kept 4, the timing gate accepted 2: 01 u23
  (gap 0.6 → 0 s) and 04 u05 (0.4 → 0 s; 04 LS frames in gaps 67 → 57). The long residuals (03 u04 / u06, 05 u01) were not fixed by any
  faithful rewrite → video side. Qwen time: 1–24 s per unit.

## 4. Clean bed on 01–05 + residue duck (`s4_bed.md`)
* **What**: `background_bed.bed_speech_runs` + `mix_background(residue_duck_db)`, runner `--bg-residue-duck-db -15` (default): the bed is lowered
  only where the separated bed itself sounds like speech while the source speaks and the dub is silent. `bed_ab.py` uses the pipeline's Spleeter
  path (HDEMUCS reference separated on the fly behind `DUB_ALLOW_NC_BED`, measurement only).
* **Why**: Spleeter leaked a vocal sound into the bed on 02 (39.0–39.4 s, Silero p 0.99, dub silent) and a consonant onset on 03 (59.3 s).
* **Result**: voice residue in dub pauses 02 0.45 → 0 s, 03 0.13 → 0.1 s; ambience continuity (bed under source speech − pure ambience) Spleeter
  −2.7…+0.1 dB vs HDEMUCS −0.5…−8.9 dB; ducked ≤ 0.6 s per video.

## 5. Foreign sounds / clicks / hot cuts: declick stage, gentler duck (`s5_detectors.md`)
* **What**: `scripts/pilot/declick.py` + runner `--declick on` (default): isolated clicks of the assembled dub voice (audio_stages' detector)
  repaired in ±1.5 ms (4 kHz low-pass, 0.5 ms crossfades). Runner duck defaults −6 dB / 0.08 s / 0.35 s → **−4 dB / 0.25 s / 0.6 s**
  (`--bg-duck-attack-s`, `--bg-duck-release-s`). `audio_stages.py` measures the bed as heard (mix − dub at 44.1 kHz) when stems are not kept.
* **Why**: 7 dub-voice clicks (01: 4 in the raw Chatterbox takes + 2 after atempo, one of them new with r2 at 42.28 s; 05: 1 raw-take click
  at 41.99 s); 3–19 bed steps of 8–10 dB per video created by the fast duck attack right before each phrase ("pumping").
* **Result**: dub-voice clicks 7 → 0; mix-created bed steps 03 9 → 1, 04 5 → 0, 05 6 → 2, 02 19 → 11; bed "breathing" 7–8 → 5–6 dB;
  Whisper WER on the mix unchanged (01 0.019 / 0.022, 05 0.053 / 0.053). Hot cut edges r2 ≤ cand5 on every video.
