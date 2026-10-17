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
