# DabAI round 4: back to the v4 base, simplify, never regress (common part, both agents)

## Why this round exists
Round 3 ended with `r3final_fixed v4`. The user and a reviewer (Max) found v4 clearly better than everything before it, and the
lip "explosions" (single-frame flashes) are gone.

The next version (`e2e_v5`) then **regressed** in several places, because every fix was another layer of heuristics checked on one
or two videos. Max's verdict on v5 / 04:
- construction workers' noise came to the foreground;
- the pauses became strange;
- the mouth misbehaves again;
- in the first ~10 s half of the face "goes negative" (dark / inverted).

The user also reported:
- proper nouns pronounced wrongly ("Сочи Сити" → "Solki");
- a chin / jaw that moves like a wave (a second and third chin);
- lip motion that sometimes does not match the speech;
- in v5 the ORIGINAL Russian voice leaks through the background bed (v4 was better).

Details: `/workspace/dub/briefs/user_feedback_v5_2026-10-20.md`.

**The strategy (decided by the user):**
- The base is v4.
- **Fewer layers, not more.**
- Audio and video are developed and judged separately.
- Nothing is published without a regression check against the previous version on all five videos.

## Base
- Repo `git@github.com:RomadovaIrina/dubai.git`, base commit **`904c5d2`** (`experiment/r3-integration`, v4 finals). The one-command
  runner and the QA gates were added after it (D, `9220503`…`32a3605`, the v5 era). Cherry-pick the tools: `r3_e2e.sh`,
  `r3_qa_gates.py`, `r3_scorecard.py`, `r3_bgring.py`, `r3_flash.py`, `r3_joins.py`. Do NOT take v5's new defaults (sentence flow gap
  1.0 s, the bed remix of 53cfb22, `--sanitize-colour`) unless your A/B shows they help.
- The v4 reference finals are in `/workspace/dub/reference/v4/<id>_v4.mp4`.
  - Video agent: rebuild v4 exactly from 904c5d2 first. Same frame count and duration; visually identical; the metrics within noise.
  - Audio agent: rebuild v4's audio first, which shows that the base reproduces.
- Test videos: `/workspace/dub/inputs/test_videos_raw/`. Common metrics: `/workspace/dub/common/`. Feedback and spec notes:
  `/workspace/dub/briefs/`.

## The regression gate (mandatory before ANY clip goes to the user)
1. **One command, all five videos, from the raw input.** No per-video flags or thresholds.
2. **Automatic table, NEW vs PREVIOUS PUBLISHED (initially v4)**, per video:
   - frames / fps / duration;
   - SyncNet (dub-only and mix, AV offset);
   - slide / jitter v4 metric;
   - joint jumps;
   - background ring;
   - lip flash review count;
   - WER of the final mix;
   - audio hard cuts;
   - LUFS / true peak;
   - bed residue of the original voice under dub speech;
   - bed level in pauses vs the source;
   - pause-length distribution vs the source;
   - s/video-min, peak VRAM.
3. **Any metric clearly worse than the previous version on any video → no publish.** Fix it or revert.
4. **Human check.** For each video, a side by side PREVIOUS | NEW at normal speed. Also a review sheet, `outbox/<id>_review.md`, with
   one row per 10 s: voice / background / pauses / mouth / face / other. Fill in what changed and where to look. The user and Max
   then only confirm or refute, and do not have to hunt for timecodes.
5. Publish as `outbox/<id>_r4_<agent>_vN.mp4` plus `r4_<agent>_vN_regression.md`. Write the version and the commit in each file.

## Separate judging
- **Audio agent:** publish the NEW AUDIO OVER THE ORIGINAL VIDEO (no lip-sync), so that only the sound is judged.
- **Video agent:** publish with v4's audio, so that only the face is judged.
- **Merge:** the video agent merges at the end. Only gate-passing versions are merged, and the merge passes the gate too.

## Rules
- Your own branch, created from 904c5d2: `experiment/r4-audio` or `experiment/r4-video`.
- Commit and push every step. Keep a short CHANGES.md.
- Licence rule unchanged: only MIT / Apache / BSD / ISC / OpenRAIL weights. CodeFormer, InsightFace, RetinaFace R50 and HDEMUCS
  stay out. `assert_clean_tree()` must pass.
- One GPU per server: use it fully, and keep renders queued.
- End a turn at least every ~45 min with a 5-line status: hypothesis / evidence / what changed / what failed / next.
- Do not touch the other agent's server.
- Handoff goes through the coordinator: `/workspace/dub/handoff/out/<A|V>/<name>/` + READY, **≤ 150 MB**. Send flags, manifests and
  audio files, never frame dumps.
