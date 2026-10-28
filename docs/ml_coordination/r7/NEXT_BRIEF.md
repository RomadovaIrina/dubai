# DabAI round 5: handover brief for continuing on a fresh server (written 21.10.2026)

You continue the quality work on the DabAI Russian→English video dubbing pipeline. Read this whole file first. It is self-contained.

## 1. The product and the pipeline
Pipeline:
Silero VAD → faster-whisper large-v3 → pyannote 3.1 → Qwen2.5-7B GGUF (llama.cpp, **on the GPU**, Q8_0) → Chatterbox Multilingual
(TTS) → duration fit / placement → LatentSync 1.6 (25 fps) → composite at the source fps (FIX30) → mux.

Customer spec, which binds every change:
- 9.4: output at the source fps and resolution;
- 8.4: feather 10–15 px;
- 8.5: no jumps at joints, no clicks;
- 8.2: only the active speaker;
- 5.4 / 5.5: head-pose pre-check;
- 11: the 2 % rule;
- norm: **≤ 300 s of processing per video-minute on ONE RTX 5090**.

All measurements are single-GPU. Several servers only parallelise experiments.

## 2. Hard rules from the user
1. **Licence:** only MIT / Apache / BSD / ISC / OpenRAIL weights; LatentSync 1.6 (OpenRAIL++) is approved.
   - Banned: CodeFormer, InsightFace buffalo_l, RetinaFace R50, HDEMUCS.
   - `download_models.sh` has a tripwire, and `assert_clean_tree()` stops any run that can reach a banned file. Never set
     `DUB_ALLOW_NC=1`.
2. **Improve the E2E pipeline, not individual videos.** Every fix is a generic default of the one-command runner. No per-video
   flags or thresholds.
3. **Regression gate before every publish.** NEW vs the previous published version on all 5 videos (`scripts/r4/r4_gate.py`). Use the
   same frame set for per-frame metrics (00d647d). SyncNet is scored on dub-speech seconds.
4. **The user judges by eye and ear and can override the gate.** Publish WATCH candidates (`<id>_<ver>_WATCH.mp4` + `_sbs.mp4`
   PREV | NEW + a WATCH.md with the failing rows and timecodes) when the user asks to see them.
5. **While the face is visible, the mouth follows the DUB:**
   - dub speaks → lip-synced mouth;
   - dub silent → a closed / relaxed mouth, never the talking source mouth.
   Gate row: `mouth_audio_mismatch_s`.
6. **Pauses must be logical:** only at English punctuation or a clause start, never inside a phrase. The user chose audio **pb1** by ear
   over v1, even though it costs some alignment to the source.
7. Speed after quality, but report s/video-min and peak VRAM for every run.

## 3. Where the code is (GitHub `git@github.com:RomadovaIrina/dubai.git`, the repo is public, so HTTPS clone works)

| branch | head | what |
|---|---|---|
| `experiment/r4-merged-m4` | `8ebb299` | **start here**: m3 one-command E2E (`scripts/r4/r4_e2e.sh`) + m4 work in progress |
| `experiment/r4-merged` | `a81dc55` | m2 / m3 E2E, mouth-follows-dub rule (`1a7865a`, `--ls-split-dub-silence 1.0`), gate rows (`387ddf1`, `00d647d`) |
| `experiment/r4-audio` | `ea1f6c4` | audio: v1, pb1 (`--pause-boundary-only on`), NE guard (proper nouns), source-speech bed duck −6 dB, audio metrics `r4_audio_metrics.py`, generator sweep report |
| `experiment/r4-video` | `fd80085` (+ `backup/r4-video-wip-1020-2103`) | video: r4_video_v1 stack, gate, face metrics (colour, jaw, chin wave), `--mouth-region` (mr38), g25 arms |
| `904c5d2` | — | v4, the user's reference "best old version" |
| `coordination/report-2026-10-19` | — | coordinator reports, logs, briefs |

Setup on a fresh RTX 5090:
1. `scripts/setup_dabai.sh && scripts/download_models.sh && scripts/setup_mediapipe.sh`.
2. The CUDA llama-cpp build into `third_party/llama_cuda` (`JOBS=8 bash scripts/setup_llama_cuda.sh`; the recipe is in r3-integration CHANGES.md).
3. Spleeter weights: the r4 servers needed an extra install. See the r4-video CHANGES for the exact fix.
4. The HF token goes into `~/.cache/huggingface/token` and `/workspace/.env`.
5. Test videos: `05.zip` from the Google Drive link in README → `/workspace/dub/inputs/test_videos_raw/`.
6. References: v4 finals and r4 outputs are on the user's laptop. The coordinator uploads them to `/workspace/dub/reference/`.

## 4. State at handover (versions the user has seen)

| version | what | user verdict |
|---|---|---|
| v4 (`904c5d2`) | round-3 best | best old version |
| v5 | one-command E2E with new defaults | regressed (pauses, bed, half-face colour, mouth) and was dropped |
| r4_audio_v1 | v4 + NE guard + bed duck | better, but **pauses illogical** |
| pb1 | v1 + pauses only at boundaries | **chosen by ear** ("got better") |
| r4_video_v1 | v4 stack minus seam-LF (ablation: retime, colour match, face-clip, skin texture, lip sharpen and lip guard all needed) | ≈ v4 visually |
| m2 | v1 video + pb1 | watch only; joints / slide worse on 04 |
| **m3** | m2 + mouth-follows-dub rule + segment split in dub silences | watch; **current best**; the user still sees issues (below) |

### m3 gate numbers (vs v4-repro)

| video | verdict | notes |
|---|---|---|
| 05 | FAIL 1 row: strong flashes 29 / 22 | mouth/dub mismatch 3.04 → 2.04 s (the user's 11–13 s fixed); flashes in the newly lip-synced small-face span 8.8–12 s and the raw-generator flash at 33.7–36 s |
| 04 | FAIL 1 row: slide 0.175 / 0.154 | in the long pb1 dub silence 21.4–27.3 s (closed mouth over a talking jaw); mismatch 0.97 → 0.73 s; flashes 12 / 18 better |
| 03 | published (check `r4_merged_m3_WATCH.md`) | |
| 01 / 02 | rendering at handover | |

## 5. Open problems, in priority order (each needs a GENERIC fix)

**#0, THE BLOCKER (the user's final verdict on round 4):** "the face still twitches, and that moment simply kills it." The
user and the reviewer Max judge the whole result by face stability. Lip-sync gains count for nothing while the lower face / face
twitches.

Treat face twitching as the single top priority of round 5, above every other item below.

1. **Find every twitch automatically.** Build a whole-video "face twitch" detector:
   - the frame-to-frame motion of the pasted region relative to the rigid head, minus the source's own motion;
   - spikes at segment switches, window joints and retime frames;
   - rank them by visibility.
2. **Fix the twitches generically.** Candidates:
   - eased crossfades at every LatentSync segment start / end and at closed-mouth ↔ lip-sync switches;
   - no segment boundaries during fast head motion;
   - temporal smoothing of the pasted-region pose against the rigid head;
   - fewer and longer segments where the dub allows.
3. **Accept only when** the twitch detector is at or below v4 on all 5 videos, AND a human check of the top-20 twitch moments
   (a contact sheet plus 2-s clips at half speed) shows none visible.

1. **Face twitch at 23–24 s in m3** (the user, most likely 04; it falls inside the pb1 dub silence 21.4–27.3 s).
   - Suspect: the closed-mouth ↔ lip-sync segment switch, or a LatentSync window joint, over a moving jaw.
   - Diagnose with per-frame slide / jaw / joint values plus a contact sheet. Fix with eased crossfades at the segment switches, and
     boundaries away from fast head motion.
2. **Single-frame mouth flashes** ("explosions", the user's top visual complaint; e.g. 05 33.8 s, which is already in the raw
   generator output).
   - The sanitizer misses frames with mc_ratio ≈ 3.2–3.4.
   - The flash detector cannot separate real tears from articulation on whole videos. Every publish needs a top-30 contact sheet
     reviewed by eye.
3. **Chin / jaw "wave"** (the generator regenerates the chin soft tissue).
   - Winner: `--mouth-region 0.38` (mr38): 03 wave −30 %, 05 flashes 33 → 23, SyncNet within ±0.2.
   - Rejected: chin5 / chin3 (they kill the jaw drop: SyncNet −0.3 to −0.47).
   - **m4 = m3 + mr38 + the task-1 fix** was in progress on `experiment/r4-merged-m4`. Finish it, gate it, publish WATCH.
4. **Lip motion vs speech.**
   - guidance 2.5 gives +0.6 to +0.72 SyncNet (04, 03) and mouth/dub mismatch 1.0 → 0.3 s.
   - But strong flashes 14 → 32 (04), 49 → 58 (03), and the mouth residual rises. A stricter deflash / sanitizer was being tested.
   - Bridge off gives the same quality at ~30 % faster LatentSync, which is a free speed option.
5. **Audio leftovers** (from `r4_audio_v1_known_issues.md` and `pb1_listen.md`):
   - long merged silences in pb1 (04 0:33.4, a 7.8 s pause before "and") → allow one boundary pause or a gentle slowdown;
   - rushed speech (tempo ×1.15) in places;
   - "source talks, dub silent" up to 1.8 s.
   Root cause: EN/RU length mismatch. Faithful lengthening via Qwen only rarely passes.
6. **Speed:** v4 / v1 video stage is 390–475 s/video-min, so the norm FAILS.
   - 01 E2E reached 263 s/min, but the others are over.
   - Levers: bridge off (−30 % LatentSync), fewer CPU-contended jobs (`cpu.max` quotas of 23–31 CPUs on vast boxes), prep outside the
     GPU lock, LatentSync steps / DeepCache only with a human check.
7. **Half-face colour shift** (v5 04, first 10 s). Colour match is needed (without it 1.32 → 3.04). The v5colour variant was rejected.
   Keep the r4 colour gate row.

## 5b. Round-5 target and team (user, 21.10): "ideal quality, both video and audio"
Three agents on three RTX 5090 servers, all starting from `experiment/r4-merged-m4` (m3):

1. **FACE agent (high effort):** blocker #0, face twitch.
   - Then flashes (item 2) and the chin (item 3: mr38 in).
   - Acceptance: twitch detector ≤ v4 on all 5 + the human top-20 review.
2. **GENERATOR-CEILING agent (high effort).** Settle once and for all whether LatentSync 1.6 is the limit.
   - (a) The best LatentSync settings: guidance 2.0–2.5 + deflash, steps, `--max-ls-frames`, 512 crop / upscale.
   - (b) One or two alternative lip-sync generators with clean WEIGHT licences. Check the licence first and write it down; drop
     anything NC / unlicensed.
   - Compare on the 9 worst windows and on full 04 / 05: SyncNet on dub speech, flashes, twitch, chin wave, s/video-min.
   - Deliver a recommendation and a side by side for the user.
3. **AUDIO agent (high effort):** go from patches to the root cause.
   - Most audio defects (rushed ×1.15 speech, "source talks / dub silent", long merged silences, illogical pauses) come from the
     EN/RU LENGTH mismatch.
   - Do **duration-aware translation**: ask Qwen for a translation that fits the source burst's duration (a target syllable /
     word budget from the source timing), verify it with back-translation / judge for faithfulness, and pick the candidate whose TTS
     duration is closest. Then pauses come naturally at the source pause points and tempo stays at ≈1.0.
   - Keep the pb1 boundary rule, the NE guard, the bed duck, and −16 LUFS.
   - Gates:
     - tempo outside 0.95–1.05 (s);
     - source-talks / dub-silent (s);
     - dub-talks / source-silent (s);
     - pauses inside a phrase;
     - WER;
     - faithfulness score.

**Integration:** the face agent owns the merged E2E (m-series). Every merged candidate goes out as WATCH with a `review.md`, and the
user + Max confirm it before it counts.

**Human review protocol (with Max):**
- For each video, the 10 moments the detectors rank worst, each with a 3-s clip at normal and half speed.
- Max marks each one OK / NOT OK.
- Anything NOT OK becomes a regression test that all later versions must pass.

**Budget:** 3 × ~$0.8/h ≈ $2.4/h. The balance is $12.14 (~5 h), so the user should top up ~$30 for a full day.

## 6. How to work
- One command per experiment, all 5 videos, the gate against the previous published version, WATCH clips + `review.md` (one row per
  10 s: voice / background / pauses / mouth / face) so the user can confirm without hunting for timecodes.
- Keep the GPU busy; cap CPU jobs (`OMP_NUM_THREADS` 8, ≤ 2 heavy CPU jobs) because vast containers have a CPU quota.
- Commit and push every step. Short status every ~45 min: hypothesis / evidence / what changed / what failed / next.
- Handoffs between servers go through the coordinator, ≤ 150 MB. No frame dumps.
- First deliverable: **m4 WATCH on all 5** (m3 + mr38 + the 23–24 s fix), then the flash work, then g25 + deflash.

## 7. Round-6 outcome (27.10) and the next step
- Face twitch FIXED: the r3 sanitizer smeared 2–4 frames on small / moving faces → alignment guard (skin-ring NCC ≥ 0.8) + mr38 = **m4**
  (`experiment/r6-face` 2ec5a1c). Twitch ≤ v4 on 03 / 04 / 05.
- Audio **r6_audio_v1** (`experiment/r6-audio` a82b54b): the user LOVES it. Duration-aware translation, tempo ≈ 1.0, pauses at the source
  pauses.
- **m5 = m4 + r6 audio:** the face is fine, but the lips follow the dub less tightly (04 / 05 mismatch +1–2 s, 05 SyncNet −1), because of where
  the new audio places speech.
- **NEXT (#1):** audio placement must put dub speech INSIDE the source speech spans while the face is visible, keeping r6's tempo and pauses.
  Then build m6 = m4 + that audio, and gate it against m4 on the video axes AND against r6_audio_v1 by ear.
- Generator: LatentSync 1.6 at guidance 2.0 + bridge stays (`experiment/r6-gen` 5b04678). MuseTalk failed; InfiniteTalk is too heavy.
- Bandwidth on vast is billed. Avoid > 20 GB downloads.
- A blind test on unseen videos is still pending: 4 Wikimedia Commons clips were proposed (Kleiman FAL, Levkovich CC BY 3.0, Pokrov CC BY 3.0,
  BAIC CC BY 4.0).

## 8. User review of r6 m5 (27.10, screenshots) — round-7 blockers
1. **05 25–26 s: CATASTROPHIC paste.** The camera points at trees, and a rectangle with the face plus a background from another frame is
   pasted in the upper right. The suspect is face loss → held landmarks / crop (rule 11, "previous valid frame") → the old crop is pasted
   whole.
   - Generic fix: if the CURRENT source frame has no confident face, paste NOTHING (pass-through). Never hold the face position for more
     than ~2 frames.
   - New gate: whole-frame |out − src| OUTSIDE a small mouth ellipse. Any large area means a defect; target 0.
   - Contact sheets of the WHOLE FRAME, not mouth crops: the mouth-crop sheets hid this.
2. **03 ~7 s: blurred / doubled lower face with a visible seam on the cheek and jaw** during head motion. Fixes:
   - a mask-boundary seam metric (sharpness / colour step);
   - better registration on fast motion;
   - less repaint under strong motion.
3. **Audio in dub silences: "farts" (separation artefacts) and leaked original-voice fragments in the bed.** Generic fixes:
   - a residual-voice detector on the bed (VAD / ASR on the bed stem) in every dub gap;
   - replace detected spans with room tone taken from this video's non-speech background, with crossfades;
   - denoise the bed with a clean-licence tool (noisereduce MIT / RNNoise BSD) in the gaps;
   - a metric plus listen clips of all gaps.
   Keep r6_audio_v1 as it is otherwise (the user loves it).
4. The user says it is "better, but not by much". Face stability remains THE priority.
