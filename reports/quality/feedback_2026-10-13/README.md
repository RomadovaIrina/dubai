# Human feedback 2026-10-13 — twitching mouth / facial expression, foreign sounds, noticeable pauses

> «Вижу этот дергающийся рот постоянно и мимику и от этого меня тригерит. Так же обнаружил посторонние звуки, которых раньше не было.
> Обратите на это внимание тоже. По поводу пауз, тоже заметно конечно это всё.»

Status: **quality iteration done on all five videos; runtime NOT re-measured yet** (the GPU / CPU were shared with another process for
the whole session, see the caveat). Candidate for human review: `results_videos/cand5/0{1..5}_new.mp4` (not in git).

## Environment (clean box)

```text
repo/branch  RomadovaIrina/dubai @ tts-intelligibility-fix, start HEAD 73a30275 (remote HEAD checked = expected)
GPU          NVIDIA GeForce RTX 5090, 32607 MiB, driver 595.91.07 (CUDA 13.2 driver); torch 2.7.1+cu128 (pins unchanged)
CPU          cgroup quota 13.6 vCPU (previous box: 27.2) — DUB_THREADS stays 16 (production setting)
disk         50 GB overlay
setup        setup_dabai.sh 0-9 exit 0, download_models.sh exit 0, run_smoke.sh 11/11 PASS; pip drift = patch bumps of unpinned packages only
caveat       another process (/mem, 8.7-8.9 GB VRAM + CPU) ran on the same GPU during every run: wall / LatentSync / Qwen seconds here are
             1.5-2x the real ones and NOT valid for the 300 s/min budget. One run (05, first final) lost 4 LatentSync segments to a CUDA OOM
             (that process + a SyncNet job of mine) -> found by SyncNet, re-run exclusively; the runner now retries OOM segments and the
             validation fails on an OOM pass-through (it reported PASS before).
```

## What was reviewed (checked in code / manifests)

| flag | default at 73a3027 | reviewed candidate (cand4) | **new candidate** |
|---|---|---|---|
| `--alignment burst --video-backend optimized --codeformer optimized` | `refresh_e2e.sh` | same | same |
| `--background on` (Demucs bed) | default | on | on |
| `--phrase-cut` | timing (no fades) | acoustic (+5 ms fades) | acoustic **+ `--long-pause-func 1.2` + `--cut-fade adaptive`** |
| `--tts-gap-qa` | – | – | **on** |
| `--gate-merge-gap-s` | 0.6 | 1.2 | 1.2 (2.0 = option, see A) |
| `--mouth-only-pasteback` | off | on | on **+ `--codeformer-scope mouth` + `--mask-smooth 3` + `--ls-landmark-smooth 2`** |
| fit-down of short sentences | up to 4 rounds | same | **at most one rewrite** (now the behaviour, not a flag) |

Command: `DUB_EXTRA_ARGS="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on --gate-merge-gap-s 1.2
--mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2" bash scripts/pilot/refresh_e2e.sh 01 02 03 04 05`.
All new switches are opt-in flags (defaults unchanged) until the human review of these outputs.

## A. "Twitching mouth / facial expression" — CodeFormer re-rendered the eyes in every lip-synced frame

`scripts/quality/face_temporal.py` (LatentSync's own insightface 2d106 on master and output, face-normalised; ratios output / master, 1.0 =
as calm as the source; LS core = LS frames > 5 from a switch). Before = the reviewed video settings on the SAME new audio
(`candidate_video_stage.py --codeformer-scope face`), so only the video stage differs.

| upper face (eyes / brows) inside LS | 01 before -> new | 03 | 04 | 05 |
|---|---|---|---|---|
| sharpness vs source | **4.45 -> 1.01** | 0.86 -> 0.99 | 0.80 -> 1.01 | **2.09 -> 1.01** |
| pixel flicker vs source | **1.46 -> 1.01** | 1.15 -> 1.00 | 1.03 -> 1.00 | 1.11 -> 1.00 |
| sharpness step at a switch (source at the same frames) | – (1 switch) | 9.3 % -> 4.1 % (3.8) | 19.7 % -> 11.4 % (11.3) | **72.6 % -> 19.6 % (19.4)** |

| mouth inside LS | 01 | 03 (reviewed -> VA -> new) | 04 | 05 |
|---|---|---|---|---|
| lip landmark jitter vs source | 1.02 -> 1.02 | 0.95 -> 0.95 -> **0.89** | 1.00 -> 0.99 | 1.10 -> 1.04 |
| mouth pixel flicker vs source | 1.12 -> 1.10 | 1.01 -> 1.01 -> **0.90** | 1.00 -> 0.97 | 1.06 -> 1.00 |
| mouth-opening dynamics vs source | 1.01 -> 1.10 | 0.76 -> 0.74 -> 0.73 | 0.79 -> 0.80 | 0.88 -> 0.99 |

* Root cause: `MouthOnlyRunner` composited the LatentSync mouth into the source frame, **then** CodeFormer restored the whole face of that
  composite. CodeFormer re-synthesises eyes / lids / skin per frame (2-4x sharper than the source on 01 / 05, 46 % more flicker on 01) and
  everything jumps back at every LS -> original switch ("мимика"). `sheets/03_eyes_8.64s_reviewed_VA_master.jpg`: rows reviewed / new / source.
* `--codeformer-scope mouth`: CodeFormer on the raw LatentSync frame, mouth-only composite AFTER it -> outside the lower-face ellipse the output
  IS the source. Same work, no runtime cost.
* `--ls-landmark-smooth 2`: LatentSync builds its affine crop from 3 landmark points per frame (integer-rounded, almost unsmoothed) and the
  mouth lives in crop coordinates -> detector noise became mouth shake. Centred +-2-frame average of the replayed landmarks: 03 lip jitter
  0.95 -> 0.89, mouth flicker 1.01 -> 0.90, opening dynamics unchanged (no smoothing of the articulation itself), SyncNet 5.86 -> 5.87.
  `--mask-smooth 3` (mask bbox) had no measurable effect (kept, harmless).
* Switch frequency: 03 22 (defaults) -> 10 (merge 1.2) -> 8; 04 has 14 switches in 54 s (the speaker pauses 1.2-1.8 s between short phrases).
  `--gate-merge-gap-s 2.0` on 04: 14 -> 4 switches, mouth inside the merged pauses stays as calm (lip jitter 0.99, flicker 0.97), **+17 %
  LatentSync frames**. Not in the candidate: needs a human look (`results_videos/review_04/`) because it costs time.
* SyncNet, same frames with the dub voice only (comparable to the 2026-10-12 baseline): 01 7.86 (7.80), 03 5.79 (5.71), 04 3.92 (3.51),
  05 see table; AV offset 0 everywhere.
* Speed lever, quality-identical: the mouth composite works on the ellipse ROI (bit-identical, 10 -> 2.5 ms per frame).

## B. Foreign sounds — localised to the first stage where they appear

`scripts/quality/audio_stages.py`: detectors on the final timeline + trace raw TTS (`tts_u<id>_raw.wav`, now saved) -> cut part -> placed /
atempo -> dub voice -> final mix -> MP4, spectrogram sheets per event.

| source (first stage) | evidence | fix |
|---|---|---|
| **Chatterbox inserts a sound between words** (raw TTS) | 04 u03 "popular with ‹0.45 s held vowel› bloggers" (CTC: "with **the** bloggers"), 05 u01 "After a-a-all", 05 u05 "So ‹…› let", 02 u01/u05/u16. Whisper normalises these away -> the ASR QA scored them fine (0.09) | **`--tts-gap-qa`**: speech-level sound between the CTC-aligned words (>= 100 ms, within 15 dB of the speech level) -> new seed for that sentence (<= 2). Offline: 0 false positives on 03 (10 sentences), 1 on 04, 3 on 05; online: 05 3/3, 02 3/3, 04 1/1 fixed with the first retry, +4-8 s per video |
| a translation reduced to 2 words, unintelligible TTS (fit-down) | 02 u16 "The relaxation zone also includes lying down areas." -> "Relaxation zone: lying areas." -> "**Lying areas.**" (2nd rewrite of a 4-word sentence for a 0.31 s overflow); 3 QA-bad takes, Whisper hears "Thank you, Iris" | a short sentence is rewritten at most once (02 WER 0.138 -> 0.090) |
| cut inside voiced sound (part) | timing cut (defaults) on 03: 4 inner tails within 12 dB of the speech level, max -6.9 dB, 1 click | acoustic cut (reviewed) already fixed it: 0 / -18 dB, 0 clicks; with the new cut rule "Well, \| and" was voiced -> `--cut-fade adaptive` (25 ms fade-out / 8 ms fade-in only on still-voiced inner cuts): hot edges back to 1 |
| background bed (final mix) | Demucs voice residue in a dub pause (03 88.6 s, -19 dB rel. the dub voice), one separation transient (03 46.2 s); bed under the source speech is already 6 dB below the pause ambience | NOT changed: ducking it further (`--bg-src-duck-db`, implemented, off) would dig a -35 dB hole into a -21 dB ambience -> needs the reviewer's timestamps first |
| not found | clicks from atempo (the flagged ones are plosives present in the raw TTS), overlapping dub parts (0), repeated words, interjections (0 on all videos), mux artifacts (MP4 = mix) | – |

## C. Noticeable pauses — the source's hesitations copied onto weak English boundaries

`scripts/quality/pause_rhythm.py` (pause distribution, intra-sentence pauses: preserved / inserted / **weak boundary** = part ends without
punctuation on a function word / wh-word / number / inside a fixed phrase), `scripts/quality/recut_offline.py` (re-runs split -> cut -> span
merge -> placement -> track on the SAME raw TTS; reproduces a run's placements exactly) for policy A/B without Qwen / TTS noise.

* 03 reviewed: "Probably more in | 1.9 s | general", "I'll send it to our | 1.6 s | chat", "Well, and | 1.4 s | basically", "moving | 1.5 s |
  forward" — the acoustic cut DP had no syntax term; the dub reproduced every source pause but on the wrong English word.
* `--long-pause-func 1.2`: a cut on a weak boundary costs up to 1.2 more, scaled by the SOURCE pause there (0 at 0.3 s, full at 0.8 s), and
  the same cost is added to a merge over that pause (otherwise the DP dodges the weak cut by running the phrase through the pause and the
  source keeps talking after the dub has finished: 05 u03 +1.7 s). Short pauses keep the plain acoustic rule.

| offline, same TTS, five videos | reviewed rule | rule without merge guard | **new rule** |
|---|---:|---:|---:|
| weak-boundary pauses > 0.3 s | 26 (43.3 s) | 4 | **6** |
| source speech without dub (sum) | 35.7 s | 41.2 s | **37.5 s** |

  03 now: "Probably more in general \| what I want to say…", "…with you \| and in general…", "Well, \| and basically…", "…to our chat \| so
  let's…", "…we desire, \| moving forward." Residual: 02 "I will \| 9 s walk \| show the Russian sauna" (the only quiet boundary of that take).
* 01 (continuous speaker): all dub pauses > 0.5 s are sentence boundaries where the TTS is shorter even at the 0.9 slowdown floor; one 2.1 s
  gap (48.7 s, u12) = insufficient fill (fit-up), not cut placement -> residual; not fixed by slowing down further (would bring back drawn speech).

## Final five videos (new candidate) vs the 2026-10-12 baseline (defaults; SyncNet with the dub voice only)

| metric | 01 | 02 | 03 | 04 | 05 |
|---|---|---|---|---|---|
| valid (incl. new no-OOM check) | PASS | PASS | PASS | PASS | PASS |
| SyncNet dub-only / AV offset (baseline) | **7.86** / 0 (7.80) | no face | **5.79** / 0 (5.71) | **3.92** / 0 (3.51) | **3.70** / 0 (3.61) |
| SyncNet on the final mix (bed) | 7.76 | – | 5.76 | 3.77 | 3.33 |
| upper-face flicker / sharpness vs source (reviewed video) | 1.01 / 1.01 (1.46 / 4.45) | – | 1.00 / 0.99 (1.15 / 0.86) | 1.00 / 1.01 (1.03 / 0.80) | 1.00 / 1.01 (1.11 / 2.09) |
| LS <-> original switches / min | 0.6 | – | 5.3 | 15.6 (merge 2.0: 4.5) | 11.4 |
| Whisper WER (baseline) | 0.016 (0.019) | 0.090 (0.063) | 0.004 (0.004) | **0.009 (0.043)** | **0.000 (0.020)** |
| CTC CER (baseline) | 0.051 (0.055) | 0.182 (0.165) | 0.040 (0.042) | 0.050 (0.058) | 0.154 (0.155) |
| bad units / interjections / final word missing | 0 / 0 / 0 | 4 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| inserted sounds fixed by gap-QA | 0 (-10 dB run) | 3 / 3 | 0 | 1 / 1 | 3 / 3 |
| clicks in the dub voice | 0* | 0 | 0 | 1 | 0 |
| weak-boundary pauses > 0.3 s (reviewed rule, same TTS) | 1 (1) | 2 (7) | 2 (9) | 0 (5) | 0 (4) |
| source speech without dub s (baseline) | **7.2 (9.1)** | **10.0 (10.7)** | **9.3 (10.3)** | 5.0 (4.2) | 6.8 (5.2) |
| VAD IoU (baseline) | 0.894 (0.885) | 0.753 (0.740) | 0.824 (0.814) | 0.798 (0.832) | 0.699 (0.754) |
| dub pauses > 1 s: dub / original | 1 / 0 | 15 / 15 | 12 / 10 | 11 / 9 | 9 / 10 |
| atempo max / > 1.15 | 1.15 / 0 | 1.20 / 1 | 1.15 / 0 | 1.03 / 0 | 1.00 / 0 |

\* 01: the detector flags plosive releases that are in the raw TTS (atempo-mapped time shifts the match) — none is a cut / atempo click.
Regressions to watch: 02 WER 0.063 -> 0.090 (names "hamam / Finnish / Russian bath" as in the baseline + "Relaxation zone: lying areas." trimmed by
0.31 s at its slot end — the collision the old fit-down 2nd round "solved" by deleting the meaning; the other changes are TTS take variance:
Qwen segmented 02 into 19 sentences instead of 17, so the per-sentence seeds moved); 05 source-without-dub 5.2 -> 6.8 s (long-pause rule +0.7 s
offline, the rest different TTS takes); 04 +0.8 s.

## Runtime (NOT valid — shared GPU / CPU) and where the speed-up is

Shared GPU (another process with 8.9 GB + CPU load) during every run; 13.6 vCPU quota (previous box 27.2):

| video | src s | wall s | audio s (Qwen CPU) | video s (LatentSync / CodeFormer) | peak VRAM MiB | wall s/min | audio s/min | video s/min | pipelined s/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 101.7 | 987.3 | 785.0 (618.9) | 198.4 (111.968 / 11.949) | 31365 | 582.7 | 463.3 | 117.1 | 463.3 |
| 02 | 110.9 | 615.4 | 537.4 (383.5) | 63.1 (0.0 / 0.0) | 17422 | 333.1 | 290.8 | 34.1 | 290.8 |
| 03 | 90.3 | 1486.4 | 749.8 (612.2) | 747.9 (587.62 / 73.402) | 31578 | 987.3 | 498.0 | 496.7 | 498.0 |
| 04 | 53.9 | 748.1 | 362.7 (262.3) | 391.2 (295.251 / 37.315) | 31393 | 832.7 | 403.8 | 435.4 | 435.4 |
| 05 | 52.4 | 842.4 | 455.5 (352.1) | 395.9 (297.781 / 35.594) | 31890 | 965.2 | 521.8 | 453.6 | 521.8 |
| **median** | | | | | | **832.7** | 463.3 | 435.4 | **463.3** |
| **mean** | | | | | | **740.2** | 435.5 | 307.4 | **441.9** |

For reference, the 2026-10-12 baseline on an exclusive box with 27 vCPU: wall 519 / 277 / 841 / 456 / 470 s = 306 / 150 / 559 / 508 / 538 s/min (median 508).

* What the quality fixes cost: 0 for the video changes (CodeFormer scope / landmark smoothing / ROI composite saves time), gap-QA +4-8 s per
  video, long-pause rule / adaptive fades / fit-down rule ~0. Merge gap 2.0 (not enabled) would be +17 % LatentSync on 04.
* The bottleneck is the **CPU Qwen of the audio stage** (fit-up rewrites + judge 215-490 s per video here), not LatentSync: audio > video on
  4 of 5 videos, so even with the audio of the next job overlapping the GPU stage (the reason Qwen stays on the CPU) the CPU is the limit.
* Quality-neutral levers to measure next, on an exclusive box: llama.cpp threads vs the 13.6 vCPU quota (16 threads now = oversubscription),
  `--fit-up-guard on` (2026-10-12: fewer judge calls, 03 Qwen fit-up 215 -> 108 s), judge only when the rewrite changed content words; then
  the GPU side (LatentSync frames of the gate, CodeFormer on the mouth crop only).
