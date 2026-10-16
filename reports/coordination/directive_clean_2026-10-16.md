# COORDINATOR DIRECTIVE 2026-10-16 ~21:00 UTC — CLEAN-ONLY (user decision)

This supersedes earlier coordinator messages and the brief wherever they conflict. Older queued messages may arrive together with
this one; where they disagree, this directive wins.

1. **Clean only.** From now on only commercially clean components (verdict YES: code + weights + every pretrained sub-model
   permissive, no NC / research-only / unlicensed weights, no NC-trained restorers) may run in any product-path run on this server.
   cand5 stays only as the already-rendered visual reference: keep its videos and metrics and do not re-run it.
2. **Quarantine now**, at the first safe point (no job of yours is using them). Move, do not delete, into
   `/workspace/dub/quarantine_nc/`, keep relative paths, and add `README.md` with the sha256 of each file and the reason:
   - InsightFace buffalo_l: `third_party/latentsync/checkpoints/auxiliary/models/buffalo_l/` + `buffalo_l.zip` + any `~/.insightface` copy
   - CodeFormer: `third_party/CodeFormer/weights/CodeFormer/codeformer.pth`, `weights/facelib/parsing_parsenet.pth`,
     `weights/facelib/detection_Resnet50_Final.pth`
   - facexlib RetinaFace R50 `models/facexlib/detection_Resnet50_Final.pth` (UNCLEAR), any ParseNet / BiSeNet
   - GFPGAN v1.4, RestoreFormer(++), VQFR weights (FFHQ-trained, YES* is not clean enough for the user)
   Proof: a clean-pipeline run must succeed with these files absent. That is the test that nothing still loads them.
   **EVAL_ONLY** tools may stay. Label them in `LICENSES.md`, and the product path must never import them: SyncNet + S3FD (SyncNet
   scoring) and YuNet inside `mouth_jitter_hr.py`. Stop using any metric built on InsightFace landmarks (`face_temporal.py`, …).
3. **Clean face stack.** MediaPipe Face Landmarker / BlazeFace (Apache-2.0 code and model card) replaces buffalo_l + RetinaFace
   everywhere in the video stage. Use its own venv / subprocess if it conflicts with the dabai pins. No restorer at all in the
   product path.
4. **Ownership, so no work is duplicated:**
   - **Agent A** owns the clean face stack for the LatentSync input path: detector, landmarks, crop / affine, face router, and the
     landmarks the mouth-only composite replays. Deliver it as one opt-in flag (e.g. `--face-stack mediapipe`) plus
     `--codeformer off`, which together make the **clean0** pipeline. Commit, push, and end your turn right away with the SHA. The
     coordinator creates branch `clean-base` from it, and both agents then build on `clean-base`.
   - **Agent B**, until `clean-base` exists: continue only with clean code (no restorer, C2, source-texture transfer, mask /
     blending). Already-rendered LatentSync frames may be used as development data only. Port your paste-back / mask to take
     landmarks from the clean stack so that it plugs into `clean-base`. Final numbers come only from `clean-base`.
5. **After `clean-base`, both servers run the same clean base and tune different places:**
   - A, generator side: LatentSync window overlap / blending / shared noise, crop and landmark smoothing on the clean stack,
     DDIM / guidance, and clean alternative generators (TBDub etc., only if fully YES).
   - B, post side: window-boundary interpolation, texture transfer, mask / feather / blending, temporal filtering, LS↔original
     transitions.
6. **New reference: clean0** (cand5 audio + LatentSync 1.6 on the clean face stack, no restorer, mouth-only composite). Compare every
   experiment with clean0 and with the rendered cand5 on 03, including the human-marked windows 24–25 s and 35–36 s, then on
   01 / 04 / 05.
7. **Communication.** Coordinator messages are delivered only between your turns. End a turn at least every ~45 min of work, and
   at every milestone, with a 5-line status. Run long jobs in the background so that ending a turn does not kill them.
