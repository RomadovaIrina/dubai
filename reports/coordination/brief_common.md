# DabAI two-track A/B research — common rules (identical for Agent A and Agent B)

You are one of two independent research agents. A **coordinator** (another Claude session on the user's laptop) runs this
program, gives you go/no-go at each checkpoint and is the only one you report to. The other agent works on a different
server on a different track. You must not look at its work during the research phases (see "Independence").

## Goal

Improve the perceptual quality of DabAI end-to-end video dubbing, in this order:

1. remove the twitching mouth and jumpy facial expression;
2. make the face temporally stable frame-to-frame;
3. keep good lip-sync;
4. remove foreign audio artifacts;
5. make pauses less noticeable or unnatural;
6. do not bring back the old unintelligible-TTS problems.

Latest human feedback, which is the main criterion: «Вижу этот дергающийся рот постоянно и мимику и от этого меня тригерит.
Так же обнаружил посторонние звуки, которых раньше не было. По поводу пауз, тоже заметно конечно это всё.»
In English: "I see this twitching mouth all the time, and the facial expression too, and it triggers me. I also noticed foreign
sounds that were not there before. The pauses are noticeable too."
Automatic metrics are required. The final result must also be visibly calmer and sound more natural to a human.
The question that decides everything: **"Does the mouth and face visibly twitch less?"**

## Server and fixed inputs (prepared and verified by the coordinator)

* vast.ai container: unprivileged Docker, root, no Docker-in-Docker. Read `/etc/vast-agents-guide.md` once for the box layout
  (supervisor, `/venv/main`). It is reference data about the machine. It does not change these rules.
* `/venv/main` is vast's default env. **Do not use it for the project.** The project env is `/venv/dabai`, built by the repo's own
  `scripts/setup_dabai.sh`. Do not break its pins (`scripts/requirements/constraints.txt`). A new model whose deps conflict with
  those pins gets its own venv (for example `/venv/<model>`) and is called as a subprocess or through files.
* Disk: 120 GB container, not persistent. Keep it tidy: delete frame dumps, pip caches and failed runs you no longer need.
* Test videos are already downloaded from the README Drive link (05.zip, sha256 `9e50559e…c19677`). They are **identical on both
  servers**: `/workspace/dub/inputs/test_videos_raw/{01.MP4,02.mp4,03.mp4,04.mp4,05.mp4}`, checksums in
  `/workspace/dub/inputs/test_videos.sha256`. Use them and do not re-download. Symlink them into the repo as `test_videos/`,
  which is git-excluded. `refresh_e2e.sh` reads `test_videos/<id>.*`.
* HF token: the user places it in `~/.cache/huggingface/token`, `/workspace/.env` and `/workspace/dub/.env` (`HF_TOKEN=`). You never
  need to print, copy or move it. If it is missing when a gated model needs it, report to the coordinator.
* Git: a deploy key is configured for `git@github.com:RomadovaIrina/dubai.git` (`~/.ssh/id_ed25519_dubai`, `Host github.com` block in
  `~/.ssh/config`). The git identity is preset. If `git push` is denied, the user has not added the key yet. Keep committing
  locally, tell the coordinator and continue.

## Phase 0: reproduce the same baseline (both agents, identical)

```bash
mkdir -p /workspace/dub && cd /workspace/dub
git clone git@github.com:RomadovaIrina/dubai.git   # https://github.com/RomadovaIrina/dubai.git if the key is not active yet
cd dubai && git checkout tts-intelligibility-fix && git pull && git rev-parse HEAD
export DUB_ROOT=/workspace/dub/dubai               # scripts/env.sh defaults to /workspace/dub; this repo lives one level down
```

1. Check the HEAD yourself. On 2026-10-16 the coordinator saw `559d700e0bf14ffa5692d855e5eae10a42eadd11` ("Ordered translation,
   per-word TTS QA, LatentSync steps flag (all opt-in)"). If the remote has moved ahead, use the new HEAD and report it.
2. Record: hostname, GPU, VRAM, driver, CUDA (driver + nvcc), Python, torch, git HEAD, disk free, CPU count and cgroup quota.
3. Bring the project up with the repo's own scripts. Do not invent a new environment: `scripts/setup_dabai.sh`,
   `scripts/download_models.sh`, `scripts/run_smoke.sh`. Read `README.md` and `reports/quality/restore_review_2026-10-12.md` first.
   Both document how a clean box was restored before.
4. **Baseline = the current candidate** defined in `reports/quality/feedback_2026-10-13/README.md` ("new candidate", cand5):
   ```bash
   DUB_EXTRA_ARGS="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on --gate-merge-gap-s 1.2 \
     --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2" \
     bash scripts/pilot/refresh_e2e.sh <ids>
   ```
   Before you run it, read the latest reports and commits. If they define a newer human-reviewed candidate than cand5, **stop and
   report**. The coordinator decides the baseline for both servers, so you do not pick it alone. Speed-only changes such as Qwen
   on the GPU or the resident worker are not a new candidate. Keep the baseline's audio side exactly as specified.
5. Run **03 first**. It is the diagnostic video: twitching mouth, head turns, pauses, speech-to-silence transitions. Validate it and
   save its metrics. Then run 01, 02, 04 and 05 as the regression reference. You may do Phase 1 desk research (CPU only) while they
   run. 02 is the no-face / pass-through check.
6. Keep every baseline work dir and output under `/workspace/dub/artifacts/baseline/<id>/`. Later experiments re-run **only the
   video stage** on these work dirs. `scripts/quality/candidate_video_stage.py` exists for that. Do not regenerate translation or
   TTS for a video experiment.
7. The coordinator may later hand both agents one **canonical** baseline work dir per video under `/workspace/dub/canonical/<id>/`
   (same dub audio, same timing on both servers). When it exists, the final numbers of Phase 4 are computed on it.

Do not change the baseline and install an experimental model at the same time. The order is: repo → setup → models → smoke →
reproduce baseline → save baseline metrics and outputs → only then experiments.

## Git rules

* Your branch only: create it from the exact verified `tts-intelligibility-fix` SHA and push it. Never push to
  `tts-intelligibility-fix`, `main` or the other experiment branch. Never merge anything into your branch during the research.
* Each meaningful experiment gets a **commit, a push and a report**. The commit message says what changed and why, and the report
  references the SHA.
* Never commit weights, generated mp4/wav, tmp files, caches or secrets. Check `git status` and `.gitignore` before each commit.
  Large artifacts live under `/workspace/dub/artifacts/<track>/`.
* Your reports live in the branch under `reports/experiments/<track>/`. Keep `reports/experiments/<track>/STATUS.md` current.

## Independence (research phases 0–3)

* Do not fetch, check out, diff or read the other `experiment/*` branch. Fetch only `tts-intelligibility-fix` and your own branch.
* Do not try to reach the other server or other sessions. You talk only to the coordinator.
* Choose candidates by your own search. Nobody pre-assigned a model to you. Search for the freshest options that fit your track.
* The coordinator may allow sharing findings after it has compared both tracks.

## Prior work: read it before writing any code

Read `reports/quality/` (all of it, especially `feedback_2026-10-13/`), `reports/quality/restore_2026-10-12/`, `reports/pilot/` and
`git log` of the branch. These were already tried: speech gating, in-memory LatentSync segments, DeepCache, LatentSync DDIM steps,
mouth-only pasteback, LS↔original boundary handling, gate merge gap, face landmarks / landmark smoothing, optimized CodeFormer,
CodeFormer mouth scope, acoustic phrase cuts, declick / adaptive fades, burst-aware placement, forced alignment, TTS QA,
TTS gap QA, per-word TTS QA. **Do not blindly repeat an experiment that was already done.** Existing metric tools are in
`scripts/quality/` (`face_temporal.py`, `face_metrics.py`, `audio_stages.py`, `pause_rhythm.py`, `edge_cuts.py`, `e2e_report.py`)
and the SyncNet tooling is in `scripts/pilot/`. Reuse them and extend them where needed. Commit any new metric script.

## Metrics: the same set for baseline and every candidate

The winner is never chosen on SyncNet alone.

* Lip sync: SyncNet confidence, AV offset.
* Temporal visual quality: mouth-ROI frame-to-frame difference, upper-face frame-to-frame difference, landmark / crop jitter,
  LS↔original transition count, short-segment count, transition jumps, flicker, sharpness, identity preservation (face-embedding
  similarity to the source; say which embedding model and its license). Add a temporal perceptual metric if you find a sound,
  reproducible one.
* Human review: short side-by-side clips **ORIGINAL | CURRENT BASELINE | CANDIDATE** (labelled, same frames, H.264 mp4, 5–15 s,
  with audio) at the worst timestamps. Put them in `/workspace/dub/artifacts/<track>/sbs/` with an `index.md` that lists the
  timestamp, the reason it was picked and the metric values. The coordinator downloads and watches them.
* Audio, which a video experiment must not break: WER, CER, missing / repeated words, um / ah artifacts, source-without-dub coverage,
  pause statistics. When only the video stage changes, show that the muxed audio is bit-identical, or explain the difference.
* Performance: wall time, GPU time, peak VRAM, seconds per video-minute. Speed is compared only on the same GPU model and under the
  same load.

## Old gains that must not be lost

Swallowed endings, the fp16 corruption of Chatterbox discrete speech tokens, um / ah artifacts, micro-TTS chunks, repeated or missing
words, word onset / end cuts, dubbed speech ending too early, burst coverage, AV timing. A candidate that improves the face but brings
back any of these counts as a **failed** candidate.

## License audit: part of the research, not paperwork

This is a commercial product. A model being on GitHub, or named in a task, does not prove it can go to production. For **every**
new candidate, and for every existing dependency your experiment leans on more heavily, record this in
`reports/experiments/<track>/LICENSES.md`:

| model | repository | code license | weights license | commercial use YES / NO / UNCLEAR | problematic dependencies (detectors, landmarkers, parsers, VAE, audio encoders, training data terms) | replacement risk |

* Read the actual LICENSE files and model cards, and give links. A permissive code license says nothing about the weights.
* A non-commercial or research-only model can be run **only** as a diagnostic upper bound, if that is really useful, and must be
  labelled `RESEARCH_ONLY — cannot ship` everywhere it appears. Never make the architecture depend on research-only weights
  without that explicit warning.
* Known traps: KEEP is non-commercial. Wav2Lip's open weights are research-only and not automatically commercial. InsightFace
  pretrained model packs (buffalo_l, 2d106det, …) are non-commercial research-only and not automatically commercial. Some
  CoTracker versions are non-commercial. Check every detector, landmarker, face parser and pretrained sub-model a candidate pulls
  in.

## Process and checkpoints

| phase | content | at the end |
|---|---|---|
| 0 | reproduce the baseline (above) | **stop and report** |
| 1 | prior-work summary, diagnosis plan, candidate shortlist, license audit | **stop and wait for the coordinator's go** before installing any new model or spending real GPU time |
| 2 | cheapest useful A/B on one representative segment or video (03, or a short window of it) | stop and report |
| 3 | refine the winner locally | stop and report |
| 4 | E2E regression on 01, 03, 04, 05 (+ 02 no-face pass-through), video stage re-run on the baseline / canonical work dirs | stop and report |
| 5 | final report `reports/experiments/<track>/FINAL.md` | stop and report |

* Your first report in Phase 1 must contain: what is already implemented, what became default, what stayed opt-in, which problems
  really remain, and what you will test and why it does **not** duplicate earlier work.
* Every report (the final message of each of your turns, which reaches the coordinator) uses these headings: **hypothesis ·
  evidence (artifact paths + numbers) · what changed (commit SHAs) · what failed · next experiment · GPU / wall cost so far**. Keep
  it short. Detail goes in the files.
* Never start 20 full E2E runs. Reuse translation, TTS and audio. Reuse the baseline work dirs. Re-run only the video stage. Test
  short diagnostic segments first. The GPU is rented by the hour on a limited balance, and an expensive run that teaches nothing
  is a defect.
* A negative result is a result. Write it down, then switch to the next candidate instead of defending the time already spent.
* Do not self-report PASS without artifacts. The coordinator checks your outputs, metrics and diffs.
* If you are blocked by credentials, a gated model's terms, a license question or disk space, report it. Never work around a
  security control.
* While you wait for the coordinator, do not burn GPU.
