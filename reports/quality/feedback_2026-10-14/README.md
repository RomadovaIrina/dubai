# 2026-10-14 — rollback to the 2026-10-12 pipeline, "thought" pause policy, closed mouth in dub silence (handoff)

> **Current candidate: R11 — see [R11.md](R11.md)** (command, commits, how it works, numbers). R10 below is superseded: its face
> (closed-mouth freeze) was rejected after a frame-by-frame analysis (`scripts/quality/face_defects.py`), its audio was kept and improved in R11.

**Start here on a new server.** Branch `quality-oct12` (from `73a3027`, the pipeline reviewed on 2026-10-12). Everything below is opt-in
flags; runner defaults are still the 2026-10-12 behaviour. Current candidate = **R11** ([R11.md](R11.md)); R10 in §3 is superseded. Time goal of the TZ: **<= 300 s of
processing per minute of video** (1:5); quality first, speed later (reviewer's decision 2026-10-14).

## 1. Restore on a clean box

```bash
git clone --branch quality-oct12 git@github.com:RomadovaIrina/dubai.git /workspace/dub/dubai   # (https for read-only)
cd /workspace/dub/dubai && export DUB_ROOT=$PWD
bash scripts/setup_dabai.sh            # stages 0-9, ~3 min
printf 'HF_TOKEN=hf_...\n' > /workspace/dub/.env   # env.sh reads ${DUB_ROOT}/../.env ; download_models.sh reads /workspace/.env
source scripts/env.sh && bash scripts/download_models.sh && bash scripts/run_smoke.sh   # 11/11 PASS on 2026-10-14
# test videos 01..05: Google Drive link at the end of the top-level README (zip) -> test_videos/ (git-excluded):
#   mkdir -p ../test_videos && unzip videos.zip -d ../test_videos && ln -s ../test_videos test_videos && echo test_videos >> .git/info/exclude
```

* Disk: the overlay is 50 GB, models 20 GB + venv 8 GB. On 2026-10-14 it filled up once (all tools fail with ENOSPC, even `rm` through the
  agent's shell): `eval_run` / `e2e_report` left temp wavs / mp4s in /tmp (fixed in `8c02e08`). Keep run work dirs pruned; watch `df -h /`.
* Pushing needs a key: generate `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_dubai`, give the user the `.pub` (repo deploy key with write
  access), `~/.ssh/config` Host github.com -> that IdentityFile, `git remote set-url origin git@github.com:RomadovaIrina/dubai.git`.
* Do not edit `scripts/pilot/*.py` while an E2E run is in progress: modules are imported lazily per stage, a run then mixes old and new code
  (happened once: `ThoughtOpts has no attribute underfill`).

## 2. Human feedback, in order

| date | on what | feedback | action |
|---|---|---|---|
| 10-12 23:30 | cand4 = `--phrase-cut acoustic --gate-merge-gap-s 1.2 --mouth-only-pasteback on`, CPU Qwen | mostly positive | baseline |
| 10-13 | cand5 (+ long-pause rule, gap QA, CodeFormer mouth scope, landmark smoothing) and then Qwen on the GPU + resident worker (`tts-intelligibility-fix` 559d700) | worse: 01 swallowed / clicked consonant endings; 02 sound gone; 03 face switches, "about to vomit" mouth; 05 street noise "breathing" with the dub; 01 Russian captions | rolled back to 73a3027 (`results_videos/12_oct_reproduce`, reproduced texts identical to the 10-12 report) |
| 10-14 | 12.10 reproduction 01/03/04 | mouth / facial twitching; **mouth opening when there is no sound**; illogical sentence structure, **"application" repeated with the pause in the wrong place** (04); rule: «не "день ……… сегодня хороший", а "день сегодня хороший ………"» — the dub may cover less of the source time, no speed-up / slow-down to fill it | thought pause policy + closed mouth (R1) |
| 10-14 | R1 / R2 | management: **pauses and speech of R1 and R2 good**, the face bad: jaw / cheek contour smeared when the source talks and the dub is silent («размазывание ебальников») | freeze (R6) |
| 10-14 | R6 | 04 8-12 s the face shakes; 01 face looks strained («как будто срать хочет») | R8 (ECC, blur match), R10 (slowdown 0.9, word QA, small-face fallback) |
| 10-14 | R10 01 / 04 | not reviewed yet | — |

01 "Срок службы более 10 лет": burned-in Russian graphics of the source video (also "ДО 10 ОКРАШИВАНИЙ", "20-25 ЛЕТ", "КЕРАМОГРАНИТ…"), not a
pipeline defect; translating on-screen text is out of the TZ scope.

## 3. Variants (all on `quality-oct12`; videos in `results_videos/` on the box, not in git)

Shared flags since R1 (`COMMON`): `--pause-policy thought --fit-up off --gate-merge-gap-s 1.2 --mouth-only-pasteback on --codeformer-scope mouth
--mask-smooth 3 --ls-landmark-smooth 2 --bg-duck-db 0 --bg-room-floor-db -30` + `--alignment burst --video-backend optimized --codeformer optimized`
(refresh_e2e.sh). Run: `PILOT_REFRESH_ROOT=/workspace/dub/runs/X DUB_EXTRA_ARGS="..." bash scripts/pilot/refresh_e2e.sh 04 01 03`.

| run | pause policy | dub silent while the source talks | verdict |
|---|---|---|---|
| 12.10 | one part per source burst (pauses inside sentences), slow-down 0.9 | LatentSync, reference = the frame itself (mouth follows the source) | reference |
| **R1** | thought, `--overflow tempo` (<= 1.1) | LatentSync with a closed-mouth reference (`--silent-mouth closed`) | pauses / speech liked; jaw contour smeared |
| **R2** | R1 + `--underfill clause` (a pause only after punctuation onto a source pause) | as R1 | pauses / speech liked |
| R3-R5 | R1 with `--overflow rewrite / split / shift` (01 only: 3-5 sentences overflow) | as R1 | same rhythm; tempo is the fastest (01: 194 vs 216-252 s/min); videos deleted, numbers in `overflow_variants_01.md` |
| R6 | R1 | `--silent-mouth freeze` v1 | shaking (04 8-12 s), strained mouth (01): deleted |
| R7 | R1 | closed + `--silent-mask big` | the contour step is inside LatentSync's own output, the mask does not help: deleted |
| R8 | R1 | freeze v2: ECC on the rigid upper head, blur match, illumination transfer | 04 / 03 not reviewed; 01 strained |
| R9 | — | — | stopped, deleted |
| **R11** | R10 audio + `--underfill semantic` (meaning chunks on the speaker's bursts) + `--thought-fill spoken` | `--silent-mouth self` (the source articulation, no seam) | **current candidate**, [R11.md](R11.md) |
| R10 | R1 + `--thought-slow-floor 0.9` + `--tts-word-qa on` | freeze v3 (keyframes from the speaker's pauses of the same shot, no pressed lips) on close-ups, closed-mouth LatentSync on small faces (< 140 px, 01) | audio liked, face rejected (04 22 s: ghost jaw / foreign background from far keyframes) -> R11 |

R10 command: `DUB_EXTRA_ARGS="$COMMON --silent-mouth freeze --tts-word-qa on --thought-slow-floor 0.9 --overflow tempo --underfill none"`.

## 4. Numbers (`compare_runs.md`, made by `scripts/quality/compare_runs.py`; WER / SyncNet only where e2e_report ran)

| | 12.10 | R1 | R2 | R8 | R10 |
|---|---|---|---|---|---|
| s per video-minute 04 / 01 / 03 | 525 / 410 / 638 | 412 / 194 / 438 | 409 / 202 / 443 | 356 / 219 / 375 | 361 / 226 / – |
| pauses inside a sentence > 0.3 s (04 / 01 / 03) | 12 / 1 / 20 | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / – |
| parts slowed (< 1.0) | 13 / 16 / 22 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 5 / 20 / – |
| source speech without dub s (04 / 01 / 03) | 3.9 / 8.3 / 11.0 | 19.4 / 22.0 / 32.3 | 15.6 / 22.0 / 32.3 | 19.4 / 22.0 / 32.3 | 16.4 / 15.5 / – |
| WER 04 / 01 / 03 | .031 / .019 / .004 | .009 / .047 / .004 | .009 / .047 / .004 | – | – |
| SyncNet dub-only 04 / 01 / 03 | 3.42 / 7.70 / 5.71 | 2.79 / 6.60 / 4.32 | 2.92 / 6.60 / 4.32 | – | – |

* Speed comes from `--fit-up off` (no Qwen fuller rewrites: 01 Qwen 535 -> 193-229 s) and, with the freeze, from LatentSync only where the dub
  speaks (03: 440 -> 257 s). 01 is under the 300 s/min goal; 03 / 04 are not (LatentSync dominates there).
* 01 WER .019 -> .047 in R1: Chatterbox swallowed the first sound of two sentences ("Bathroom" -> "Athrun", "Plaster" -> "Colossal"; already in
  the raw TTS, the QA score passed them). R10's `first_word_lost` rule (under `--tts-word-qa`) re-generated both — WER of R10 not measured.
* SyncNet drops with the thought policy. Hypothesis (not verified): the dub is silent for much longer (03: 32 s instead of 11 s) and SyncNet
  averages over all windows. `scripts/quality/syncnet_speech_only.py` scores only the dub-speech parts — **run it first** (12.10 vs R10).

## 5. What changed in code (`8c02e08`, `05ac4d3`, `b5ac853`, this commit)

* `thought_policy.py` + `tts_burst.run_tts_phrase(pause_policy="thought")` + `align_units_thought`: one part per sentence at natural speed from
  its first source burst; overflow tools; `cover_split` (underfill clause); energy-decay part ends (`tail_end`, `inner_cut`) instead of
  CTC end + 80 ms (the "swallowed / clicked consonant endings" of the feedback); `--thought-slow-floor`.
* `latentsync_accel._core(ref_index=...)`: per-frame reference image; `face_aware_latentsync_accel.closed_mouth_refs` picks the closed-mouth
  reference (no pressed lips); `--silent-mouth closed|freeze`; `mouth_freeze.py` (freeze v3) with the small-face fallback.
* `background_bed.py`: `--bg-room-floor-db` (room-tone fill of the HDemucs hole under the source speech up to -30 dBFS: 03 -5 dB, 04 -6 dB);
  runs use `--bg-duck-db 0` (05 "street breathing" was the -6 dB duck: no HDemucs hole on 05, bed -27.3 dB under speech vs -26.9 in pauses).
* ported from 2026-10-13: `--codeformer-scope mouth`, `--mask-smooth`, `--ls-landmark-smooth`, OOM retry + validation, clause-aware split
  of over-long sentences (`sentence_units.py`), `tts_qa.lost_words` (`--tts-word-qa`), temp-file cleanup. NOT ported: Qwen on the GPU,
  resident worker defaults, long-pause cut rule, gap QA, sentence flow / ordered translation.
* tools: `thought_metrics.py` (pauses inside / between sentences, weak-word pauses, tempo, coverage), `mouth_silence.py` (mouth opening /
  lip-shape motion where the dub is silent, `--scope all` for the freeze), `face_temporal.py`, `syncnet_speech_only.py`, `compare_runs.py`.

## 6. Open issues, in priority order

1. **Get human review of R10 01 / 04 and run R10 03** (`results_videos/14_oct_R10/`; R8 03 has the same face policy without slowdown / word QA).
2. **01 3.9-5.0 s still a closed mouth on a talking, expressive face** (looks strained): the English sentence is half as long as the Russian
   (1.9 vs 3.7 s), slow-down 0.9 cannot fill it. Needs a longer translation for such sentences WITHOUT the old fit-up padding ("That's great.",
   "it is."), or a different policy for small faces / continuous speakers — decide with the reviewer.
3. **Long silences between sentences with the speaker talking** (04 up to 7.8-9.6 s, 03 7.3 s): the reviewer accepted pauses after a thought,
   but a closed mouth for 8-9 s while the speaker gestures is a lot. R2 (`--underfill clause`) reduces it (04 19.4 -> 15.6 s) and was liked;
   R2 + freeze + word QA was never run.
4. SyncNet on the dub-speech parts only (§4); WER / CER of R10.
5. Freeze limits: close-ups only (>= 140 px face); keyframes need a pause of the speaker in the same shot (01 has none in its first shot);
   fast handheld motion is matched by ECC + blur but not perfectly.
6. 01 89.9-97.0 s: a full-body shot, face too small for LatentSync -> source mouth with the English audio (unchanged since 2026-09).
7. Background changes (duck 0, room fill -30) were not separately reviewed on 05 (05 / 02 were not re-run on this branch).
8. VRAM: R1 04 peaked at 31.9 GB of 32 (closed-mouth reference + CodeFormer mouth scope); freeze runs 22-27 GB.
9. Speed (after quality): LatentSync on 03 / 04, Qwen CPU segmentation + translation 50-230 s per video.
