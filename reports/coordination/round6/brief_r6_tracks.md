# DabAI round 6: track briefs (27.10.2026)

Read `/workspace/dub/briefs/NEXT_BRIEF.md` first; it is the master brief (rules, code map, open problems). Also read your own
round-5 branch: its CHANGES / STATUS.md and `reports/experiments/r5-*`.

The user's target: **ideal quality, both video and audio**. The user and the reviewer Max reject everything while the face
twitches.

Common rules:
- generic E2E defaults only;
- a regression gate against the previous published version on all 5 videos;
- a WATCH mp4 + sbs + `review.md` for everything the user should see;
- commit and push every step;
- a 5-line status every ~45 min;
- keep the GPU busy, cap CPU jobs (vast CPU quota);
- only licence-clean weights.

**If PyPI files are slow on your host, use `UV_INDEX_URL` / `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.**

The references (v4, r4_video_v1, m3) are in `/workspace/dub/reference/`. m3 exists only for 03 / 04 / 05; re-render 01 / 02 if
you need them.

---

## F: FACE + merged E2E (server F), branch `experiment/r6-face`, created from `origin/experiment/r5-face` (23965e1 + `backup/r5-face-wip-1024`)
Round-5 state:
- `scripts/r5/r5_twitch.py`, the whole-video twitch detector, works at 16 s per video. 03: m3 45.0 vs v4 52.8. 04 m3 45.5 and 05 m3
  36.4 had no v4 baseline then (the upload was truncated; it is fixed now).
- Fix flags `--seg-joint` and the eased / morph crossfades are committed but **untested**.

Tasks:
1. **Blocker #0, face twitch.**
   - Score v4 / r4_video_v1 / m3 on all 5 with the detector.
   - Find the top twitch moments, including the user's 23–24 s.
   - Test the crossfade / seg-joint fixes. Fix it generically.
   - Accept only if the detector is ≤ v4 on all 5 AND a human top-20 review (contact sheet + 2-s half-speed clips) shows nothing.
2. Single-frame flashes (05 33.8 s).
3. mr38 (chin).
4. Publish **m4 WATCH** on all 5.
5. Then merge the G winner (generator settings) and the A winner (audio) into m5, m6, …, each gated and published as WATCH.

## G: GENERATOR CEILING (server G), branch `experiment/r6-gen`, created from `origin/experiment/r4-merged-m4` (8ebb299)
Settle whether LatentSync 1.6 is the limit.
1. Best LatentSync settings:
   - guidance 2.0 / 2.5 + a deflash that keeps the +0.6–0.72 SyncNet (seen on 04 / 03) without the extra flashes;
   - inference steps;
   - bridge on / off (off is ~30 % faster);
   - `--max-ls-frames`;
   - a higher-res mouth crop / upscale.
2. 1–2 alternative lip-sync generators whose WEIGHTS have a clean licence. Verify on the model card and record it; drop NC /
   unlicensed / research-only. Respect 32 GB VRAM.
3. Compare on the 9 worst lip-sync windows + full 04 / 05:
   - SyncNet on dub speech;
   - mouth / dub mismatch;
   - flashes;
   - twitch (`r5_twitch.py` from `origin/experiment/r5-face`);
   - chin wave;
   - s/video-min;
   - VRAM.
4. Deliver `reports/experiments/r6-gen/README.md` with a recommendation, plus `outbox/gen_<id>_<t>s_ARMS.mp4` grids. Hand the winning
   flags / commits to F through the coordinator.

## A: AUDIO (server A), branch `experiment/r6-audio`, created from `origin/experiment/r5-audio` (4e688e7, with STATUS.md); also see `experiment/r5-audio-14b` (81e8da4) + `backup/r5-audio-14b-wip-1024`
Round-5 state (03, r5 vs pb1):
- source-talks / dub-silent 19.1 → 10.0 s;
- face-visible / dub-silent 12.4 → 3.1 s;
- big gaps 8 → 2.

Four gate axes failed:
- dub over silence +3.4 s;
- tempo +2.4 s;
- inside-phrase pauses 15 → 19;
- one more axis, see STATUS.md.

04 / 05 were still rendering.

Tasks:
1. Finish the duration-aware translation: piece-ordered candidates at source pauses, Qwen 14B or 7B, whichever wins on faithfulness
   and speed. Fix the 4 failing axes without losing the gains.
2. Run all 5 through the gate against pb1. Publish `<id>_r6_audio_vN_over-original.mp4` + the AB clip + `review.md` (the known issues
   from `pb1_listen.md` / `r4_audio_v1_known_issues.md`: fixed or not).
3. Hand passing audio to F as `audio-r6-vN` (≤ 150 MB).
