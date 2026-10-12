# DabAI quality pipeline — restore on a clean box, baseline reproduction, human-oriented review (2026-10-12)

## Environment

```text
repo/branch: RomadovaIrina/dubai @ tts-intelligibility-fix, HEAD de2559132a3b775f04026190fe5cd54a0c619185 (= expected)
GPU:         NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.159.04 (CUDA 13.0 driver), toolkit nvcc 12.8, torch 2.7.1+cu128
CPU:         AMD EPYC 9354 (Zen4, AVX-512), cgroup quota 27.2 vCPU (DUB_THREADS stays 16 = production setting)
disk:        50 GB overlay (≈19 GB free after venv + weights + runs)
setup:       scripts/setup_dabai.sh stages 0-9 exit 0 (~3 min, UV_HTTP_TIMEOUT=600), scripts/download_models.sh exit 0
             pip-freeze drift: patch bumps of unpinned packages only (aiohttp, pydantic, lmdb, ...); torch/CUDA/transformers/diffusers pins unchanged
inputs:      test_videos/01..05 from the README Drive link (05.zip), symlinked into the repo (git-excluded); HF_TOKEN in /workspace/dub/.env
smoke:       11/11 PASS (pyannote re-run after the token was provided) -> reports/smoke.md
```

## Baseline reproduction (runner defaults at de25591, `scripts/pilot/refresh_e2e.sh`, sequential, GPU idle before each run)

| metric | 01 | 02 | 03 | 04 | 05 |
|---|---:|---:|---:|---:|---:|
| valid | PASS | PASS | PASS | PASS | PASS |
| wall s (reference report) | 518.8 (714) | 277.1 (459) | 841.0 (956) | 455.6 (557) | 469.7 (594) |
| audio stages s | 406.9 | 230.1 | 346.3 | 189.2 | 225.8 |
| Qwen segmentation / translation s | 71.3 / 47.7 | 38.1 / 30.7 | 38.0 / 27.8 | 17.2 / 12.8 | 17.5 / 12.7 |
| TTS stage s (generation / ASR QA / CTC / rest = Qwen fit-up+judge+IO) | 251.2 (51.8 / 5.6 / 0.3 / 193.6) | 127.5 (34.3 / 3.6 / 0.2 / 89.4) | 246.7 (28.4 / 3.0 / 0.1 / 215.1) | 129.3 (15.0 / 1.6 / 0.1 / 112.6) | 164.9 (22.7 / 2.5 / 0.1 / 139.6) |
| LatentSync / CodeFormer s | 56.0 / 7.3 | 0 / 0 (no face) | 395.2 / 53.9 | 206.4 / 27.9 | 185.6 / 24.6 |
| SyncNet / AV offset (reference) | 7.80 / 0 (7.82) | N/A | 5.71 / 0 (5.85) | 3.51 / 0 (3.48) | 3.61 / 0 (3.30) |
| original SyncNet | 5.36 | – | 4.89 | 3.24 | 1.71 |
| original speech without dub s / VAD IoU | 9.1 / 0.885 | 10.7 / 0.740 | 10.3 / 0.814 | 4.2 / 0.832 | 5.2 / 0.754 |
| LS frames dub-silent while original speaks | 24 | 0 | 226 | 177 | 350 |
| Whisper WER / CTC CER (dub track, unit mean) | 0.019 / 0.055 | 0.063 / 0.165 | 0.004 / 0.042 | 0.043 / 0.058 | 0.020 / 0.155 |
| bad units / interjections / final word missing / lost numbers / cut groups | 0/0/0/0/0 | 4/0/0/0/0 | 0/0/0/0/0 | 0/0/0/0/0 | 1/0/0/0/0 |
| QA retried / ok / bad final · fit-up accepted / candidates | 0/0/0 · 9/12 | 0/0/0 · 5/7 | 1/1/0 · 4/5 | 0/0/0 · 3/4 | 0/0/0 · 7/7 |
| atempo max / mean · > 1.15 | 1.146 / 0.989 · 0 | 1.150 / 0.977 · 0 | 1.150 / 0.946 · 0 | 1.000 / 0.941 · 0 | 1.150 / 0.955 · 0 |

SyncNet aggregate (scoreable 4): **mean 5.157, median 4.660** (reference 5.11 / 4.67). AV offset 0 everywhere, 02 = no-face regression, valid.
The wall times are 100-190 s lower than the reference runs (faster CPU / quota for the Qwen calls; LatentSync / CodeFormer within noise).
Outputs: `result_videos/baseline/<id>_baseline.mp4` (not in git); full table `scripts/quality/e2e_report.py` -> `report_baseline.{md,json}`.

## Human-oriented review of the baseline outputs — remaining defects

New deterministic checks (no ASR): `scripts/quality/edge_cuts.py` (does a TTS span start / end while the voice is still loud?) and a cut-quality
count over the phrase splits (cut right after a function word, cut with < 60 ms pause).

| # | defect | evidence (video · time · text) | likely stage |
|---|---|---|---|
| 1 | **phrase broken after a function word, then a source pause** ("voice breaks off", words "swallowed") | 37 of 87 inner cuts (43 %) end on a/the/and/will/for/our…; 17 more with < 60 ms pause. 03 79.9-89.0 s: "Our interaction will be smooth and \| (1.7 s) will bring us the \| (0.5 s) result we desire. Forward \| (1.2 s) we go."; 03 5.1 s "I will \| record…"; 04 u01 4.98 s "we will test how our \| application" — Whisper on the dub hears "we will test our application" (the cut "how" is lost); 02 u16 106.8 s "Hamam \| and all this space" -> ASR "I'm in all this space" | `tts_burst.distribute_words` (timing-only cut choice, punctuation / pause bonus ≤ 0.25) |
| 2 | inner cuts still voiced at the cut point | edge_cuts: hot inner tails / heads 8/5 of 14 (01), 10/13 of 22 (02), 8/12 of 23 (03), 5/4 of 12 (04), 8/9 of 14 (05); the cut parts are saved with **no fade at all** (`cut_parts` computes faded copies, the caller re-slices `y`) | CTC word ends are early; `run_tts_phrase` part export |
| 3 | **dead air: the muxed audio is the dub voice only** — every pause is digital silence, the ambience/music of the video disappears ("creepy", "the voice breaks off and there is nothing") | frames < -60 dBFS: 16.2 s (01), 54.1 s (02), 38.9 s (03), 26.9 s (04), 27.2 s (05); the originals have 0.0-1.0 s | `build_dubbed_track` / final mux |
| 4 | fit-up rewrites that pad or change meaning | 01 u22 85.2 s "With proper use" -> "When used correctly, **it is.**"; 05 u00 1.0 s "+ **to see how it works**"; 04 u05 48.2 s "Классная погода, что, вперёд" -> "…isn't it? **That's great.**"; 01 u18 72.0 s "faucet" -> "mixer"; 01 u03 "…matte type of ceiling" (≈ 7 of 28 accepted fit-ups) | `fuller_rewrite` + Qwen judge (Qwen judging Qwen) |
| 5 | source-ASR word errors -> nonsense translation | 01 u10 34.4 s "высокую степень тепло-…" heard as "стебель" -> "a high stem of heat"; 01 u24 89.6 s "о делке" -> "the Prime deal"; 02 u01 7.4 s "парения" -> "the floating" | faster-whisper source ASR / translation without context |
| 6 | names / terms pronounced or recognised wrong | "hamam" (TTS) vs "hammam"; "Finnish sauna" -> ASR "finished sauna" (02 26.4 s); "Sochi" -> "Socky" (04 11.1 s); "sauna bath" -> "sonar bath" (02 103.0 s) | TTS pronunciation of rare words |
| 7 | short one-word TTS spans | "Let's \|", "Such \|", "The \|", "With \| moisture \|", "Hamam \|" (parts ≤ 2 words: 8 on 03) | same as #1 |
| 8 | original speaks, dub silent (lip sync draws a closed mouth) | LS frames 226 (03), 177 (04), 350 (05); 05 has a fast speaker | dub shorter than source; fit-up limited |
| – | not found | interjections (um/ah) 0, repeated words 0 (ASR "twenty twenty" for "20-25" only), final word missing 0, lost numbers 0, atempo > 1.15: 0, AV offset 0 | |

The 02 residual from the coverage-fix report (5-word unit cut by 0.4 s) did not reproduce in this run (cut groups 0 on 02); the fit-down rule for
4-5-word sentences is kept as a safety net (`--fit-down-short-words 4`, `--fit-down-short-overflow-s 0.3`).

## Fix 1 — background bed under the dubbed voice (`scripts/pilot/background_bed.py`, `--background on`)

Human feedback (2026-10-05): "the background goes silent — creepy", "the voice breaks off and then there is nothing", "can't we put the video's
own background noise there". Root cause: `build_dubbed_track` wrote digital silence everywhere outside the dub voice and only that track was muxed.

* Hybrid Demucs (`torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`, already in the env, weights 0.3 GB added to `download_models.sh`) separates the
  source audio; bed = drums + bass + other. 1.0-1.5 s GPU + 0.4-1.0 s load per video, 0.8 GB VRAM.
* Bed level = original voice/background balance (dub voice level − original vocal level), ducked −6 dB while the dub speaks (80 ms attack,
  350 ms release), never gated. Only the MUXED track changes: LatentSync, speech gate, TTS QA and every dub-only metric keep the clean voice.
* Room-tone fill: under the source speech HDemucs removes the room noise with the voice (02: bed −52 dB under speech vs the −45 dB pauses), and it
  files a steady noise floor under "vocals". Where the bed falls below the video's own noise floor it is topped up with the video's own room tone
  (steady non-speech frames, 30th-percentile floor, ≤ +3 dB, shuffled 200 ms frames with equal-power crossfades). Only for floors ≤ −42 dBFS
  (room tone); louder floors are music / street ambience that HDemucs keeps anyway (shuffling music frames would garble it).

A/B on the five baseline outputs (same frames, re-muxed; `scripts/quality/background_ab.py`, copies in `result_videos/background_ab/`):

| video | dead air (frames < −60 dBFS) OLD → NEW (original) | room fill | WER dub-only → mix | CTC CER dub-only → mix | bad units | source-voice leakage in the bed |
|---|---|---|---|---|---|---|
| 01 | 16.2 → 0.0 s (0.0) | skipped (floor −27.9 dBFS, music) | 0.019 → 0.019 | 0.055 → 0.065 | 0 → 0 | none (Whisper: hallucination "Продолжение следует…" only) |
| 02 | 54.1 → 5.9 s (1.0) | floor −55.0 dBFS | 0.063 → 0.066 | 0.165 → 0.172 | 4 → 4 | none |
| 03 | 38.9 → 0.1 s (0.0) | skipped (−40.5) | 0.004 → 0.020 | 0.042 → 0.049 | 0 → 0 | Silero 1.2 s of 60.7 s speech; 1 word "Хорошо." |
| 04 | 26.9 → 0.0 s (0.0) | skipped (−35.9) | 0.043 → 0.036 | 0.058 → 0.050 | 0 → 0 | none |
| 05 | 27.2 → 0.0 s (0.0) | skipped (−28.5) | 0.020 → 0.000 | 0.155 → 0.115 | 1 → 0 | none |

Mean WER 0.030 → 0.028, no new bad unit, no interjection. **SyncNet caveat:** the upstream SyncNet evaluator scores the muxed audio, and any
non-speech sound lowers its confidence (05: 3.53 with the dub voice only, 3.00 / 3.11 / 3.18 with the bed at −6 / −12 / −12−6 dB duck; the
original 05 with its own background scores 1.71). The frames are identical, so lip sync for a viewer is unchanged; the reports therefore give
SyncNet on the final mix AND on the same frames with the dub-only track ("SyncNet dub-only"), and the dub-only number is the one comparable to the baseline.

## Candidate — syntax-aware phrase cuts (NOT accepted yet; behind `--phrase-cut syntax --fit-up-guard on`)

`tts_burst.distribute_words_syntax`: the inner cut points of one synthesized sentence are chosen jointly (DP) by timing deviation from the
proportional burst point, punctuation, inter-word pause, a penalty after function words ("the | result") or inside number + unit, a < 60 ms
pause penalty, an acoustic penalty when the quietest 5 ms between the words is still within 15 dB of the speech level (CTC word ends are early),
and the option to leave a boundary uncut (phrase runs through a short source pause). Plus 5 / 20 ms edge fades on the exported parts (they had
none), and a deterministic fit-up guard (`appended_clause`: an appended clause / dangling "it is." is rejected before the Qwen judge).
TTS audio is byte-identical to the baseline (seeded); only cut points, the mixed track and Qwen fit-up outcomes can differ.

| | baseline | cand1 (syntax v1, no timing steepness) | cand2 (syntax + steep 4 + acoustic + fades + fit-up guard) |
|---|---|---|---|
| inner cuts after a function word / < 60 ms pause (offline count on the same clips) | 37 / 17 of 87 | 0 / 4 | 4 / 10 |
| hot inner tails / heads (edge_cuts, all videos) | 39 / 43 | 25 / 34 | 20 / 38 |
| SyncNet dub-only 01 / 03 / 04 / 05 · mean / median | 7.80 / 5.71 / 3.51 / 3.61 · 5.157 / 4.660 | 8.28 / 5.58 / 3.47 / 3.53 · 5.215 / 4.555 | 8.05 / 5.46 / 3.36 / 3.47 · 5.085 / 4.465 |
| SyncNet final mix (with the bed) | – | 8.28 / 5.35 / 3.43 / 3.06 | 8.09 / 5.22 / 3.27 / 3.18 |
| original speech without dub s 01..05 (sum) | 9.1 / 10.7 / 10.3 / 4.2 / 5.2 (39.5) | 9.6 / 11.1 / 12.0 / 4.9 / 7.0 (44.6) | 9.4 / 10.6 / 12.8 / 4.4 / 6.1 (43.3) |
| Whisper WER 01..05 | .019 / .063 / .004 / .043 / .020 | .019 / .088 / .004 / .024 / .020 | .019 / .078 / .009 / .040 / .000 |
| CTC CER 01..05 · bad units | .055 / .165 / .042 / .058 / .155 · 0/4/0/0/1 | .053 / .171 / .044 / .082 / .143 · 0/4/0/0/1 | .071 / .172 / .053 / .081 / .136 · 1/4/0/0/1 |
| wall s 01..05 (sum) | 519 / 277 / 841 / 456 / 470 (2563) | 505 / 272 / 885 / 448 / 466 (2576) | 494 / 273 / 754 / 448 / 445 (2414) |
| fit-up Qwen s on 03 (judge calls) | ≈215 (n/a) | 212.7 (16) | 108.3 (5) |

Verdict: **not accepted** — cut quality is clearly better (and the fit-up guard saves Qwen time), but speech coverage is worse (03 +1.7..2.5 s,
05 +0.9..1.8 s), SyncNet dub-only is lower on 03/04/05 and CTC CER is higher on 01/03/04. cand1 showed the mechanism: a punctuation bonus moved
cuts by up to 1 s (05 u02: a 2.1 s source burst got only "of course," = 0.3 s of dub); the timing steepness of cand2 removed most of that
(timing deviation max 0.55 s = baseline) but coverage did not return to baseline. Caveat for any per-unit reading: **Qwen on the CPU is
path-dependent** — the same prompt gave a different greedy answer when the preceding call sequence changed (llama.cpp prefix-KV reuse), e.g.
03 u08 fit-up round 1 differs between cand1 and cand2, which alone cost 03 ≈ 1.5 s of coverage. Defaults stay at the baseline behaviour.

## Accepted in this iteration

* `--background on` (default): see Fix 1. Runtime +3-6 s per video.
* `--fit-down-short-words 4 --fit-down-short-overflow-s 0.3` (default): the 02 residual (5-word unit cut by 0.4 s at the hard cap) — the rule
  never fired in these runs (no unit overflowed the hard cap), i.e. zero effect on the five videos; kept as a safety net, `6` restores the old rule.
* Instrumentation (no behaviour change): `tts.stats.fitup_qwen_seconds / fitdown_qwen_seconds / judge_calls / judge_seconds / fitup_gen_qa_seconds`;
  `scripts/quality/e2e_report.py` (one table per run set, SyncNet on mix + dub-only), `edge_cuts.py`, `background_ab.py`, `eval_run.evaluate()`.

## Runtime profile (baseline) and the Qwen lever

Audio stages are dominated by CPU Qwen, not by TTS: per video, Qwen segmentation 17-71 s + translation 13-48 s + fit-up rewrites/judge 89-215 s,
vs Chatterbox generation 15-52 s, ASR QA 1.6-5.6 s, CTC 0.1-0.3 s. Probe (same llama-cpp-python 0.3.35 built with CUDA for sm_120 into a
separate directory, production env untouched): 03 translation of 10 units **25.9 s CPU -> 1.4 s GPU** (7.5 -> 135 tok/s); 8/10 outputs identical
to the CPU, 2 differ ("Friends, hey everyone, greetings." vs "Friends, what's up, everyone hello."; one dropped subject). Expected saving
≈ 2.5-5 min per video; needs its own quality A/B because the texts change.

## Qwen on the GPU — A/B (`--qwen-gpu-layers -1`, CUDA build of the SAME llama-cpp-python 0.3.35 on PYTHONPATH; production env untouched)

| metric | baseline (CPU Qwen) 01 / 02 / 03 / 04 / 05 | GPU Qwen 01 / 02 / 03 / 04 / 05 |
|---|---|---|
| wall s (sum) | 519 / 277 / 841 / 456 / 470 (2563) | **218 / 135 / 569 / 323 / 305 (1550, −40 %)** |
| audio stages s | 407 / 230 / 346 / 189 / 226 | 105 / 86 / 79 / 57 / 64 |
| Qwen segmentation + translation + fit-up/judge s | 313 / 158 / 281 / 143 / 170 | 10 / 6 / 9 / 5 / 5 |
| peak VRAM MiB (nvidia-smi) | 25196 / 8244 / 25184 / 25204 / 24174 | 22460 / 15962 / 22440 / 22466 / 22949 |
| SyncNet dub-only · mean / median | 7.80 / – / 5.71 / 3.51 / 3.61 · 5.157 / 4.660 | 7.94 / – / 5.79 / 3.33 / 3.59 · 5.163 / 4.690 |
| original speech without dub s (sum) | 39.5 | 39.0 |
| Whisper WER | .019 / .063 / .004 / .043 / .020 | .026 / .091 / .004 / .067 / .020 |
| CTC CER · bad units | .055 / .165 / .042 / .058 / .155 · 0/4/0/0/1 | .063 / .174 / .050 / .075 / .164 · 1/3/0/0/1 |

Most WER differences are ASR/TTS noise on IDENTICAL texts (01 u09/u12/u16, 04 u02 "Sochi" -> "Hockey"); real text differences go both ways:
worse — 02 u05 "Russian **banya**" (ASR "banyan"), 02 u06 "…as it should be a hearth…", 02 u03 fit-up not accepted ("…zone for changing clothes
barrel chaise longues"); better — 04 u05 keeps the faithful "Nice weather, here comes." instead of the invented "That's great.". Same quality class,
−40 % wall time. **Not switched on**: it changes the environment (CUDA build of llama.cpp, setup stage 4) and the frozen "Qwen on CPU" decision —
needs an explicit decision.

## Human feedback on the first outputs (2026-10-12)

"01 — great, 04 — so-so, 03 — face glitches near the end, still swallowed endings."
* 03 end (82-90 s): LS / original pass-through alternate in 17-30-frame pieces (PASS 82.44-83.64, LS 83.64-87.56, PASS 87.56-88.24, LS 88.24-89.44,
  PASS 89.44-90.40); every switch changes stubble sharpness / lip shape within a 4-frame crossfade (88.5 -> 88.6 s: open LS mouth -> sharp closed
  original lips), and LS frames show a soft pinkish smear under the lower lip (87.2-87.4, 88.1-88.9 s).
* swallowed endings on 03: every hot tail is an INNER cut of a sentence, followed by a source pause: "record the first video |", "smooth and |",
  "will bring us the |", "taking with you |" — the word is cut before its natural decay.

Decision (user, 2026-10-12): **Qwen stays on the CPU** — in production the CPU Qwen of the next video overlaps the GPU video stage of the current
one; the GPU variant is kept only as the documented A/B flag `--qwen-gpu-layers` (default 0). CPU-side Qwen savings (fewer calls) remain the lever.

## Candidate 3 — acoustic phrase cuts (`--phrase-cut acoustic`)

Same timing split, but the DP prefers word boundaries whose quietest 5 ms is ≥ 15 dB below the speech level (no function-word rule, steep
timing cost beyond 0.4 s), 5 ms declick fades. TTS audio and fit-up outcomes identical to the baseline (same Qwen call sequence) -> clean A/B.

| metric | baseline 01 / 02 / 03 / 04 / 05 | cand3 01 / 02 / 03 / 04 / 05 |
|---|---|---|
| hot inner tails / heads (sum) | 8/5 · 10/13 · 8/12 · 5/4 · 8/9 (39 / 43) | 3/1 · 7/10 · 1/7 · 3/3 · 7/7 (21 / 28) |
| original speech without dub s (sum) | 9.1 / 10.7 / 10.3 / 4.2 / 5.2 (39.5) | 8.3 / 10.7 / 11.0 / 4.5 / 5.0 (39.5) |
| LS frames dub-silent while original speaks | 24 / 0 / 226 / 177 / 350 | 24 / 0 / 250 / 181 / 337 |
| Whisper WER | .019 / .063 / .004 / .043 / .020 | .019 / .063 / .004 / .031 / .000 |
| CTC CER · bad units | .055 / .165 / .042 / .058 / .155 · 5 | .051 / .160 / .045 / .068 / .149 · 4 |
| SyncNet dub-only · mean / median | 7.80 / – / 5.71 / 3.51 / 3.61 · 5.157 / 4.660 | 7.81 / – / 5.72 / 3.39 / 3.36 · 5.070 / 4.555 |
| wall s (sum) | 2563 | 2543 |

Regression found: 05 u06 "that" alone (0.14 s) on a 0.7 s burst (and 04/05 SyncNet −0.12 / −0.25) -> cand4 adds a fill penalty (part < 40 % of
its burst) and a one-word-part penalty (offline: parts < 40 % 9 -> 6, one-word parts 11 -> 4, voiced cuts 4, timing deviation 0.151 s).

## Face glitches at the end of 03 — video-stage A/B (`candidate_video_stage.py`, same dubbed track as the baseline)

| variant | LS <-> original switches | short original gaps < 1.2 s | LS frames | LatentSync s | SyncNet / offset | mouth sharp | upper-face sharp | flicker |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| baseline (gate merge 0.6 s) | 22 | 7 | 1870 | 395 | 5.71 / 0 | 0.749 | 0.818 | 0.974 |
| V1 gate merge 1.2 s | 10 | 1 | 1990 | 459 | 5.81 / 0 | 0.763 | 0.818 | 0.983 |
| V2 V1 + mouth-only paste-back (E4) | 10 | 1 | 1990 | 489 | **5.84 / 0** | 0.761 | **0.858** | 0.981 |

Visual check 87.2-90 s: V2 keeps 87.5-88.2 s lip-synced continuously (no jump between generated and original lips); the soft pinkish smear under
the lower lip in LS frames remains (a LatentSync lower-face generation artifact, not a switching problem) -> residual.
Videos: `result_videos/face_fix_03/` (before / V1 / V2, same audio with the background bed).

## Status at this commit

* Defaults: baseline behaviour + `--background on` + fit-down for 4-5-word sentences. Everything else is opt-in:
  `--phrase-cut acoustic|syntax`, `--fit-up-guard on`, `--mouth-only-pasteback on`, `--gate-merge-gap-s 1.2`, `--qwen-gpu-layers` (rejected: Qwen stays on CPU).
* Candidate 4 in progress (all five videos): `--phrase-cut acoustic` (with the fill / one-word penalties) `--gate-merge-gap-s 1.2 --mouth-only-pasteback on`;
  03 wall 929 s (+88 s vs baseline: more LatentSync frames + paste-back), 05 498 s. Accept / reject after the full table.
