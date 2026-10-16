# Agent C, P0: English dub timing — what exists, baseline residual gaps, test plan (2026-10-16)

Scope: minimise "source speaker talks, English dub silent" without hurting naturalness; hand the rest to the video side.
Box: server B, CPU only (`CUDA_VISIBLE_DEVICES=""`, 6 threads, `nice -n 10`), clone `/workspace/dub/dubai-audio`, branch
`experiment/audio-timing` from `origin/clean-base` (326e807). Data: copies of the cand5 baseline work dirs (raw TTS, manifests, source / dub
tracks; no Demucs stems) in `/workspace/dub/artifacts/audio-timing/baseline/<id>/`, manifests re-pointed there.

## 1. What exists (not re-measured here)

| layer | lever | state at clean-base | already measured |
|---|---|---|---|
| 1 placement | `--alignment burst` (TTS placed per source burst, spill into the pause before the next unit) | **opt-in** (runner default `slot`), but every production / baseline run passes it (`refresh_e2e.sh`) | week 3: silent-while-speaking −33…−68 % vs slot |
| 1 | `--burst-max-lead` (part may start up to N s before its burst into free silence instead of being sped up) | default 0.4 | never A/B-ed (only used against speed-up, not for fill) |
| 1 | `--burst-spill on`, `--burst-fallback-frac 1.01` (= never) | default | week 3 |
| 1 | `--tts-mode phrase` (one Chatterbox take per sentence, CTC word spans, cuts between words) + `--segmentation sentence` | default | 2026-10-02 coverage fix |
| 1 | `--phrase-cut acoustic` (+ `--cut-fade adaptive`) | opt-in (default `timing`); **in the baseline** | 10-12 / 10-13 |
| 1 | `--long-pause-func 1.2` (weak-boundary cost scaled by the source pause, + merge guard) | opt-in; **in the baseline** | 5 videos offline: weak pauses 26 → 6, source w/o dub 35.7 → 37.5 s |
| 1 | `--long-pause-grow`, `--pause-marks qwen` | opt-in, off | offline: weak-pause s 46.8 → 21.9, +1.1 s source w/o dub |
| 1 | `--sentence-flow-gap G` (inside a sentence, source pauses < G not reproduced) | opt-in, off | human A/B: 1.5 s "better than 0, not yet ideal"; 03 source w/o dub 10.6 → 19.4 s |
| 1 | `--ordered-translation` | opt-in, off (needs Qwen) | 04 u2 example only |
| 2 slow-down | `--fill-slowdown` (a span shorter than its burst may be slowed to this tempo) | **default 0.9** | week 3 (old parts path): 0.85 → 04 186 → 147, 05 305 → 275 frames, SyncNet −0.12 on 04; 0.9 adopted 10-02 |
| 2 speed-up | `--atempo-cap 1.15`, `--hard-cap 1.2` | default | first_res review |
| 3 fuller translation | `--fit-up on` (unit ratio < 0.8 → fuller Qwen rewrite, Qwen judge), `--fit-up-aim 1.0`, `--fit-up-rounds 3` | **default on** | 10-02: 03 source w/o dub 15.3 → 8.5 s |
| 3 | `--fit-up-guard` (reject appended clause / dangling auxiliary before the judge) | opt-in, off | iteration 3: WER 0.016 → 0.024, source w/o dub 35.2 → 38.0 s (5 videos) |
| tools | `recut_offline.py` (same raw TTS → cut → merge → placement → track), `pause_rhythm.py`, `dub_coverage.py`, `edge_cuts.py`, `e1_burst_align.py` (shared `place_groups`) | | |

Offline reproduction checked: `recut_offline.py --cut acoustic --long-pause-func 1.2 --cut-fade adaptive --background off --check` gives
IDENTICAL placements on 01 / 03 / 04 / 05 (35 / 33 / 19 / 21 parts); the dub track is sample-identical on 03 / 04 / 05 and on 01 except one
fade edge (56.69–56.85 s, start of u14). New: `scripts/quality/dub_gaps.py` (this report's numbers, residual list format for the video side).
`--background off` everywhere (HDEMUCS bed = verdict NO); the dub-track metrics do not depend on the bed.

## 2. Baseline (cand5) residual gaps — source VAD on, dub VAD off

| video | source speech s | source w/o dub s | IoU | gaps > 0.3 s n / s | > 0.5 s | > 1.0 s | gaps ≥ 0.35 s inside LS | tail / inner / head / whole s | LS frames silent-while-orig: RMS (table metric) / VAD |
|---|---:|---:|---:|---|---|---|---|---|---|
| 01 | 91.9 | 8.6 | 0.885 | 8 / 5.6 | 5 / 4.3 | 1 / 1.7 | 1 of 8 (01 is mostly NO_FACE) | 3.3 / 2.3 / 0 / 0 | 24 / 26 |
| 03 | 60.7 | 11.1 | 0.790 | 12 / 9.4 | 6 / 6.7 | 3 / 4.3 | 12 of 12 | 5.7 / 2.4 / 0.8 / 0.5 | **251** / 288 |
| 04 | 29.5 | 6.1 | 0.750 | 6 / 4.1 | 4 / 3.3 | 0 / 0 | 6 of 6 | 2.3 / 1.4 / 0.4 / 0 | 299 / 159 |
| 05 | 27.9 | 7.4 | 0.686 | 9 / 6.8 | 7 / 6.0 | 2 / 2.4 | 8 of 9 | 5.2 / 1.2 / 0.4 / 0 | 520 / 179 |

* The RMS metric reproduces the baseline table exactly (24 / 251 / 299 / 520). On 04 / 05 it counts source *noise* under dub pauses (05: source
  > −35 dB on almost every frame), so the VAD column is the primary target from here; RMS kept for continuity.
* The LatentSync mask is gated by source VAD ∪ dub VAD, so timing changes barely move it: re-estimating it from a new dub track matches the
  real run to ±1 frame. Recut A/B can therefore report LS frames without a video run.
* Position inside the source burst (tolerance 60 ms): **tails dominate** (dub part ends, speaker keeps talking: 16.5 of 25.9 s of gaps ≥ 0.35 s),
  then inner (between two parts inside one burst), few late starts.

### The decisive number: there is no slow-down headroom left
Every part that ends right before a gap ≥ 0.35 s already plays at **x0.90**, the current `--fill-slowdown` floor (28 of 28 such gaps on the
four videos; the other 7 gaps have no part ending there: late starts / between units). The tempo that would stretch that part over the gap:

| tempo needed | ≥ 0.9 | 0.85–0.9 | 0.75–0.85 | < 0.75 | no previous part |
|---|---:|---:|---:|---:|---:|
| gaps ≥ 0.35 s (01+03+04+05) | 0 | **0** | 5 | 23 | 7 |

So layer 2 (floor 0.85) can only shave ~6 % of a part's length off each gap (≈ 0.1 s on a 2 s part); it may push a few 0.35–0.5 s gaps under
the natural threshold but closes none. The shortfall is translation length (layer 3) and how the shortfall is distributed (layer 1).

### The two human-flagged windows on 03 (`sheet_24s`, `sheet_35s` in `coord/`)
* **24.5–25.0 s** (0.50 s, 13 LS frames), u3 "Today is Friday, August 28th, | the start | of our cooperation." The cut DP gave the burst
  23.9–25.0 only "the start" (2 words, x0.9); "of our cooperation." goes to the next burst; tempo needed 0.41. Layer 1 (cut placement /
  spreading) is the right lever: the burst is short and the phrase continues across a 0.6 s source pause.
* **34.3–35.5 s** (1.20 s, 31 LS frames) and **37.5–38.5 s** (1.00 s, 26 frames), u4 "…bringing us all the desired | result that we wished for, |
  so that…". The source says "нам всем" / "загадали" in the gaps; the English sentence is 7.8 s of TTS for 11.2 s of source speech (ratio 0.70).
  Fit-up triggered and was **rejected by the judge** ("added information: ensuring a seamless journey…"). Moving words between bursts only
  trades one 1.1 s gap for another: this window needs layer 3 or the video side.
* Fit-up is already on and triggers on most short units (03: 5 of 10, 04: 5 of 6, 05: 7 of 7); the worst residual gaps sit in the units where
  it was **rejected** (03 u04 / u06, 04 u02 / u03, 05 u01: "added information", "identical", "not closer").

Per-gap lists (t0 / t1 / frames / LS frames / cause / position / sentence / prev-next part / source words / fill diagnostics):
`reports/quality/audio_timing/baseline_gaps/0{1,3,4,5}_base.gaps.json`.

## 3. P1 plan (offline, `recut_offline.py` on the same raw TTS; 03 first, then 01 / 04 / 05)

Layer 1, placement (CPU, no Qwen / TTS):
1. **Spread inside the burst**: when a part is shorter than its burst after the 0.9 fill, split the remainder instead of leaving one tail gap:
   (a) centre the part (head + tail each < 0.35 s when the remainder < 0.7 s), (b) insert micro-pauses (≤ 0.2–0.25 s) only at the part's
   punctuation / clause boundaries from the CTC word spans. Measure: gaps > 0.3 / 0.5 s, natural-gap share, hot cuts.
2. **Balanced cut DP**: add a term that equalises fill ratio (part length / burst length) across the bursts of a sentence, so one burst is not
   full while its neighbour is half empty (03 24 s case); keep the weak-boundary cost.
3. `--burst-max-lead` 0.4 → 0.6 / 0.8 used for fill (late starts / short tails before the next part), `--sentence-flow-gap` 0.6 / 1.0 combined
   with (1), because flow alone moved the gap to the sentence end (10.6 → 19.4 s on 03).
Layer 2: fill floor 0.875 / 0.85 on top of the best layer-1 policy, listening clips at 03 24–26 s / 34–37 s + worst residuals
(`/workspace/dub/outbox/audio-<run>/`, original frames + new dub, lips NOT re-synced, dub voice only, no bed).
Metrics per policy: as section 2, plus `pause_rhythm.py` weak-boundary pauses, `edge_cuts.py` hot cuts, atempo distribution, and Whisper WER /
CTC CER on CPU for the finalists only.
Layer 3 (P2, design only, GPU on request): burst-targeted fit-up: ask Qwen for a *more literal* rendering of the exact source words that fall
in the gap ("нам всем" → "for all of us"), with a per-burst word budget, instead of a fuller rewrite of the whole sentence that the judge
rejects as added information. Needs Qwen (≈ 2 GB VRAM, GPU) + Chatterbox for the touched sentences only.
