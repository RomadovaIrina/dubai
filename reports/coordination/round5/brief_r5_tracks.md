# DabAI round 5: track briefs (read `/workspace/dub/briefs/NEXT_BRIEF.md` first; it is the master brief)

All tracks start from `origin/experiment/r4-merged-m4` (`8ebb299`, which holds m3) on their own branch.

The user's target is **"ideal quality, both video and audio"**. The user and Max reject everything while the face twitches.

Common rules (see NEXT_BRIEF sections 2 and 6):
- generic E2E defaults only;
- a regression gate against the previous published version on all 5 videos;
- a WATCH mp4 + sbs + `review.md` for anything the user should see;
- commit and push every step;
- a 5-line status every ~45 min;
- keep the GPU busy, cap CPU jobs;
- only licence-clean weights.

The reference finals (v4, r4_video_v1, m2, m3) are in `/workspace/dub/reference/`.

---

## F: VIDEO agent (server G, the only video server), branch `experiment/r5-face`. You own the merged E2E. (User, 24.10: only 2 servers. The generator work below is yours too.)
1. **Blocker #0, face twitch** (NEXT_BRIEF section 5, #0).
   - Build the whole-video twitch detector.
   - Locate every twitch in m3 (the user saw one at 23–24 s, probably 04).
   - Fix it generically, then accept it as the brief says: the detector is at or below v4, AND the human top-20 shows nothing.
2. Single-frame mouth flashes (item 2): a sanitizer that catches mc_ratio ≈ 3.2–3.4 (05 33.8 s), plus the top-30 contact-sheet
   review.
3. Fold in mr38 (`--mouth-region 0.38`) for the chin wave.
4. Publish **m4 WATCH** on all 5. Later, merge the winners of G (generator) and A (audio) into m5, m6, …, each gated and published
   as WATCH.

## G: GENERATOR work (folded into F on server G, AFTER blocker #0 and m4). Point 2 (alternative generators) is DEFERRED.
Settle whether LatentSync 1.6 is the limit (NEXT_BRIEF section 5b, #2).
1. Best LatentSync settings:
   - guidance 2.0 / 2.5 + a deflash that keeps the +0.6–0.72 SyncNet without extra flashes;
   - inference steps;
   - `--max-ls-frames`;
   - bridge on / off (bridge off is ~30 % faster);
   - optionally a higher-res mouth crop / upscale.
2. One or two alternative lip-sync generators whose WEIGHTS have a clean licence. Verify the licence on the model card and record it.
   Drop anything NC / unlicensed / research-only. Respect the VRAM limit of 32 GB.
3. Compare on the 9 worst lip-sync windows plus full 04 / 05:
   - SyncNet on dub-speech seconds;
   - mouth/dub mismatch;
   - flashes;
   - twitch (use F's detector when it lands; until then slide / joints);
   - chin wave;
   - s/video-min;
   - peak VRAM.
4. Deliver `reports/experiments/r5-gen/README.md` with a recommendation, plus `outbox/gen_<id>_<t>s_ARMS.mp4` grids for the user.
   Hand the winning flags / code to F through the coordinator (≤ 150 MB, flags + commits).

## A: AUDIO agent (server A), branch `experiment/r5-audio`
Fix the root cause: the EN/RU length mismatch (NEXT_BRIEF section 5b, #3).
1. **Duration-aware translation.** For every source speech burst:
   - Qwen (GPU, Q8_0) produces N candidate translations within a word / syllable budget derived from the burst's duration and the
     speaker's measured rate;
   - a faithfulness check: Qwen judge + back-translation;
   - TTS for the top candidates;
   - pick the candidate whose duration best fits at tempo ≈ 1.0.
   Pauses then fall at the source pause points.
2. Keep:
   - pb1's boundary-only pauses;
   - the NE guard (proper nouns);
   - the source-speech bed duck −6 dB;
   - −16 LUFS / TP ≤ −1;
   - Chatterbox voice cloning.
3. Gates against pb1:
   - tempo outside 0.95–1.05 (s);
   - source-talks / dub-silent (s);
   - dub-talks / source-silent (s);
   - pauses inside a phrase;
   - WER;
   - faithfulness;
   - long merged silences (> source pause + 1 s).
4. Publish:
   - `<id>_r5_audio_vN_over-original.mp4`;
   - an AB clip prev-then-new;
   - `review.md`, which lists the moments from `r4_audio_v1_known_issues.md` / `pb1_listen.md` and whether each one is fixed.
   Hand passing audio to F as `audio-r5-vN` (≤ 150 MB).
