# DabAI round 3: whole-pipeline E2E quality (common part, all agents)

You continue the DabAI quality research. It is round 3. Rounds 1 and 2 (16–17.10) produced these branches on GitHub
(`git@github.com:RomadovaIrina/dubai.git`). Read the reports and CHANGES.md files in them before you start:
- `experiment/r2-lipsync-position`: agent A, round 2. LatentSync fp32 AlignRestore, flow crop anchors (flow4nr32), head-locked
  mouth mask (`--mask-track head`), and the 05 coverage fix (1025/1112 frames vs cand5 949, commit 92de423). This is the latest
  generator code.
- `experiment/r2-composite-position`: agent B, round 2. FIX30: compositing at the source frame rate, because a 25 fps master made
  with `ffmpeg -r 25` lags its own clock by ~60 ms. It works on 03 / 04; on 05 and 03 60–68 s it was still partial.
- `experiment/r2-audio`: agent C, round 2. Policy L1: flow 0.2 + spread + slow-down floor 0.875. Background bed from Spleeter
  2stems (MIT), declick, ducking −4 dB / 250 ms, layer-3 refit with Qwen on the GPU.
- `coordination/report-2026-10-17`: coordinator report for round 1 (decisions, licences, metrics).
- `clean-base` (326e807): the clean, commercially usable starting point (clean0).

**Important:** the round-2 servers are gone, and most round-2 video and audio outputs were lost. Only the code in git survived.
If you need an earlier output, regenerate it from the code. Round-2 audio from C does not exist any more.

## Goal (user, 19.10)
Push the E2E dubbing quality further:
- **voice:** intelligible TTS, no extra sounds, natural pauses and pace;
- **sync:** good lip-sync, including the parts where the dub is shorter than the source;
- **face:** no lag or slide, no jitter, no seams;
- **everything else** the spec asks for.

Quality first, speed second, but the norm of ≤ 300 s processing per minute of video binds the final candidate.

## Spec items (DabAI_описание_этапы_бэклог.docx) that bind every change
- 8.5: no jumps at window joints. This is a contractual liability.
- 9.4: output at the source frame rate and resolution.
- 8.4: mouth-mask feather 10–15 px.
- 8.2: only the active speaker is lip-synced.
- 5.4 / 5.5: head-angle pre-check.
- 11: robustness and the 2 % rule.

## Licence rule (user, final)
The licence of the FILES decides:
- **allowed:** MIT, Apache-2.0, BSD, ISC, OpenRAIL (respect its use restrictions);
- **not allowed:** non-commercial or research-only licences, and weights with no licence at all.

Training-data provenance is recorded for the lawyer but does not block anything.
- Still out: CodeFormer (S-Lab NC), InsightFace buffalo_l (NC), RetinaFace R50 (no licence), HDEMUCS weights (no licence).
- Keep `guard_nc()`. Never set `DUB_ALLOW_NC=1`.
- Before adding a new model, check its weight licence on the model card and write it to CHANGES.md.

## Server state (prepared by the coordinator)
- Repo: `/workspace/dub/dubai`. This server's deploy key is on GitHub, so you can push.
- `export DUB_ROOT=/workspace/dub/dubai`.
- Setup log: `/workspace/dub/setup.log`. It ends with `SETUP_DONE`. Wait for that line and do not start a second setup.
- Test videos: `/workspace/dub/inputs/test_videos_raw/{01.MP4,02.mp4,03.mp4,04.mp4,05.mp4}`, sha256 in
  `/workspace/dub/inputs/test_videos.sha256`.
- Canonical cand5 baseline work dirs (dub audio, TTS parts, placements, manifests): `/workspace/dub/canonical/artifacts/baseline/<id>/`.
- Common metrics: `/workspace/dub/common/`:
  - `mouth_jitter_hr.py` v4: resid, frozen_mouth, shear, per-second ranks;
  - `slide_metric.py` + `anchor_filters.py`: lower-face slide against the rigid upper face;
  - `shaky_windows.json`.
  - The same files are also in the round-2 branches.
- **Handoff between agents:** servers cannot reach each other. Put files for another agent into
  `/workspace/dub/handoff/out/<to>/<name>/`, where `<to>` is A, B, C or D, and write a `READY` file in that directory when it is
  complete. The coordinator relays it to `/workspace/dub/handoff/in/<from>/<name>/` on the target server. Tell the coordinator in your
  status what you handed off.
- **Videos for the user:** put every mp4 the user should watch into `/workspace/dub/outbox/` with a self-explaining name
  (`<video>_<what>_<variant>.mp4`). The laptop collects it automatically. Side-by-side layouts are welcome.

## Rules
- Work only on your own branch, `experiment/r3-<track>`, started from the base named in your track brief.
- Keep a short `CHANGES.md` on your branch: what changed, why, the numbers, and the licence of anything new.
- Commit and **push on every meaningful step**. Never commit weights, mp4 or wav files.
- **End a turn at least every ~45 min** with a 5-line status: hypothesis / evidence / what changed / what failed / next.
  Coordinator messages only arrive between your turns.
- Every quality claim needs numbers plus a clip in `outbox/`. Use the shared metrics so tracks stay comparable.
- Always report the processing time of your stage in s per video-minute.
- Disk is limited. Delete frame dumps and caches you do not need.
- Never work around a security control or a permission block. If you are blocked, report it.
