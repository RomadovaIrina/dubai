# You are AGENT A: lip-sync / temporal generation track (Server A)

Branch: `experiment/lipsync-temporal` (create it from the verified `tts-intelligibility-fix` SHA). Track id for paths: `lipsync-temporal`.
Server A: vast.ai, 1× RTX 5090 32 GB, AMD EPYC 7R32 (24 vCPU share), ~40 GB RAM allocation, 120 GB disk, Denmark.

## Your research question

Is LatentSync itself the main source of the visible jitter of the mouth and face? Or does the problem mostly come from what is around
it (crop / affine alignment, landmarks, mask, pasteback, speech gating, LS↔original boundaries)? Then find either **a more stable
lip-sync backend** or **a way to stabilise the current backend**.

Measure jitter **separately** for these situations. Find and list the frame ranges on 03 (and later 01, 04, 05) for each:

1. inside a long continuous speech segment;
2. at speech / no-speech boundaries (LS↔original switches);
3. during head turns;
4. with a closed or almost closed mouth.

## Your search space

* The lip-sync generator: current LatentSync (1.6 in this repo, with its own settings) versus modern video lip-sync models. You
  must at least evaluate **MuseTalk 1.5** and **VideoReTalking**, and search on your own for newer or better ones (2025–2026
  video-to-video lip-sync / visual dubbing models). Do not stop at the first one you find.
* Everything that directly feeds or wraps the generator: face detection / landmarks used for the crop, crop smoothing, affine,
  mask, pasteback of the generated mouth, gating, transitions. You may change integration, pasteback and gating **around your
  candidate**.
* Out of scope: the TTS / translation / audio pipeline (do not touch it), and swapping face-restoration models or adding generic
  post-hoc temporal filters on the composited output. Keep the baseline's CodeFormer settings fixed in your A/Bs so that the
  generator is the only variable.

## Before installing any candidate (Phase 1)

Check and record: code license, weights license, commercial-use restrictions, dependencies, and the licenses of every pretrained
detector / landmarker / parser / VAE / audio encoder it pulls in. Do not pick a model with a non-commercial or research-only license
as the production candidate. Such a model may run only as a diagnostic upper bound, labelled `RESEARCH_ONLY — cannot ship`.
Remember: Wav2Lip's open weights are not automatically commercial, and InsightFace pretrained weights are not automatically
commercial. Also check what LatentSync itself depends on for faces and landmarks, because your experiment leans on it.

Then pick the **1–2 most promising commercial-compatible** backends and A/B them against current LatentSync.

## A/B protocol

Use the same input video, the same dubbed audio, the same speech timing / gate, the same output resolution, FPS and frame count.
Compare current LatentSync against candidate 1 (and candidate 2 only if it is justified). Start with the cheapest useful test:
a short window of 03 that contains continuous speech, a head turn, a closed mouth and a speech↔silence switch. Only then run the
whole video. Report the four jitter situations separately, plus the full common metric set and side-by-side clips.

Then follow the common rules below.
