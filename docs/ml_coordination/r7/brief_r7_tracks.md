# DabAI round 7 (28.10.2026): track briefs

Read `/workspace/dub/briefs/NEXT_BRIEF.md`, especially sections 7–8 (the round-6 outcome and the user's screenshots review).

The user's verdict on round 6:
- **audio: good** (r6_audio_v1 is loved);
- **face: still bad, "nowhere near ideal"**.
Speed matters too. Quality comes first, but **1000 s/video-min is unacceptable**. Stay ≤ ~600 now; the target is 300.

A new independent test video (a different face, a scripted scenario, see `test_video_requirements.md`) will arrive later. Every fix
MUST be generic.

**Money rules (vast bills bandwidth):**
- no download over 5 GB without asking the coordinator;
- no new model downloads unless asked;
- report disk and bandwidth use in every status.

Common rules:
- one-command E2E;
- a regression gate against the previous published version;
- WATCH mp4 + sbs + review.md;
- push every commit;
- a 5-line status every ~45 min.

---

## F: FACE (server F), branch `experiment/r7-face`, created from `origin/experiment/r6-face` (2ec5a1c; also cherry-pick `backup/r6-face-wip-1027`)

Current best: **m5 = m4 face stack + audio r6_audio_v1**.

The user's blockers, in priority order:
1. **05 25–26 s: CATASTROPHIC paste.** The camera points at trees, and a rectangle with the face plus a background from another frame is
   pasted in the corner. Probable cause: face loss → held landmarks or crop → the stale crop is pasted.
   - Generic fix: if the CURRENT source frame has no confident face, paste NOTHING (pass-through). Never hold the position for more than
     ~2 frames.
   - New gate: whole-frame |out − src| OUTSIDE a small mouth ellipse. A large area means a defect; target 0 on all videos.
   - Scan ALL 5 videos for this.
2. **03 ~7 s: blurred / doubled lower face with a visible seam on the cheek and jaw** during head motion.
   - Mask-boundary seam metric: the sharpness and colour step across the feather.
   - Better registration on fast motion.
   - Under strong motion, lower the paste strength or region.
3. **The face still twitches overall.**
   - Contact sheets and top-N review on the **WHOLE FRAME**, not mouth crops: the crops hid defect 1.
   - Re-check the twitch detector so that it also counts paste-position jumps.
4. **m5 lips follow the dub less tightly than m4** (audio timing). The audio agent works on that; merge its handoff as m6.
5. **Speed:** report s/video-min per stage. Keep ≤ 600 now. Find the top-3 costs and cheap wins.

Publish **m6 WATCH**: face fixes + the audio agent's latest passing audio (or r6_audio_v1). All 5 videos, the whole-frame top-20 sheets,
sbs M5 | M6. Priority order of videos: 03, 01, 05, then 02, 04.

## A: AUDIO (server A), branch `experiment/r7-audio`, created from `origin/experiment/r6-audio` (a82b54b)

Keep r6_audio_v1's behaviour: the user loves it.

1. **Gaps / dub silences: no junk.**
   - Separation artefacts ("farts") and leaked original-voice fragments in the bed while the dub is silent.
   - A residual-voice detector on the bed stem (VAD / ASR) in every dub gap.
   - Replace detected spans with room tone from this video's non-speech background (crossfades).
   - Denoise the bed in the gaps with a clean-licence tool (noisereduce MIT / RNNoise BSD; no big downloads).
   - A metric + listen clips of every gap.
2. **Lip-friendly placement.** While the face is visible, put the dub speech INSIDE the source speech spans, so the mouth can follow it.
   m5 lost sync vs m4 because of placement:
   - 05 mismatch 1.74 → 2.64 s;
   - 04 0.77 → 2.60 s;
   - 05 SyncNet 5.08 → 4.06.
   Keep the tempo ≈ 1.0 and the pauses at the source pauses.
3. Gate it against r6_audio_v1:
   - no audible regression;
   - mouth / dub-placement metrics better.
4. Publish r7_audio WATCH (over-original + AB), then hand it to F (`handoff/out/F/audio-r7-vN/`, READY, ≤ 150 MB).
5. Speed: the audio stage s/video-min must stay reasonable (≤ ~150).
