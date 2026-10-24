# Coordinator log — DabAI two-track A/B

## Servers (vast.ai, rented 2026-10-16)

| role | instance | location | price | SSH alias (laptop) | address | GPU / driver | CPU / RAM | disk |
|---|---|---|---|---|---|---|---|---|
| A — lipsync-temporal | 56277909 | Denmark | $0.700/h | `dub-a` | root@85.218.235.6 -p 57630 | RTX 5090, 590.48.01 (CUDA 13.1) | EPYC 7R32, 24 vCPU / 40 GB | 120 GB |
| B — restoration-temporal | 56277943 | France | $0.834/h | `dub-b` | root@82.65.197.236 -p 40983 | RTX 5090, 580.178.04 (CUDA 13.0) | Ryzen 9 9950X3D, 16 vCPU / 46 GB | 120 GB |

fp16 matmul check: A 227 TFLOPS, B 237 TFLOPS (no foreign GPU load). Different hardware → final winners must swap servers.
Destroyed: 56266847 (old B, SSH reset), 56266811 (China, GitHub/HF blocked).

## Fixed inputs

* `tts-intelligibility-fix` HEAD seen by coordinator: `559d700e0bf14ffa5692d855e5eae10a42eadd11` (2026-10-13).
* Test videos: 05.zip sha256 `9e50559e44c2a0067fb164a19007ca6f52474687a22619d487bde3a9e3c19677`, identical on both servers,
  `/workspace/dub/inputs/test_videos_raw/`, per-file sha256 in `/workspace/dub/inputs/test_videos.sha256`.
* Baseline = cand5 (`reports/quality/feedback_2026-10-13/README.md`), see brief_common.md.

## Server prep done by coordinator

deploy key `~/.ssh/id_ed25519_dubai` + `Host github.com` block, github host keys, git identity `DabAI agent a|b`, test videos.

## Pending on the user

- [x] deploy keys A and B added with write access (verified: `Hi RomadovaIrina/dubai!` from both)
- [x] HF token placed on both servers (whoami = RomadovaIrina, fine-grained) — rotate after the program (posted in chat twice)
- [x] SSH connections `dub-A` / `dub-B` added to the app config (`ssh_configs.json`, backup `.bak-2026-10-16`)
- [ ] vast.ai balance topped up (was $8.43)

## Agent sessions

| agent | session | started | mode |
|---|---|---|---|
| A | [agent-a-lipsync] local_5d90db58-2f7e-419d-8ba3-e026cf1ab18b | 2026-10-16 18:32 UTC | auto, Opus 5.5 xhigh |
| B | [agent-b-restore] local_03de906a-e5c7-40db-a61b-68c8fa9d7261 | 2026-10-16 18:35 UTC | auto, Opus 5.5 xhigh |

## Decisions / checkpoints

- 18:35 both agents launched with brief_A/brief_B + brief_common. A cloned 559d700 at 18:35; B cloning.
- Baseline fixed to cand5 for both. User confirmed: LatentSync/CodeFormer = reference only, alternatives come in Phase 1.
- 18:58 video sync to the laptop: `coordination/sync_videos.sh` (background, every 2 min) → `C:\Users\irina\Videos\DabAI_AB\<A|B>__<run>__<file>.mp4`.
  Sources per server (`/root/coord_list_videos.sh`): outbox/<run>/*.mp4, artifacts/baseline/*/final, baseline face_temporal clips, */sbs/.
  Outbox contract sent to both agents (queued until their turn ends). Log: `coordination/sync_videos.log`.
- 19:50 B Phase 0 ACCEPTED (`reports/experiments/restoration-temporal/baseline_phase0.md`, branch 0fbe9e5): 5/5 PASS, SyncNet dub-only
  7.86/5.69/3.81/3.66 vs cand5 7.86/5.79/3.92/3.70, AV 0, switches 1/8/12/10, audio md5 recorded. Audio drifts slightly (CPU Qwen,
  path-dependent) → canonical work dirs needed for cross-track numbers; candidate: B's baseline work dirs.
- KEY: existing face_temporal rates cand5 mouth ≈ source (0.88–1.09) while the human sees constant twitching → metric blind.
  B tasked to build a high-res, face-stabilised jitter metric (standalone script) → coordinator distributes it to A as the common metric.
- User asked for max pace: both agents pre-approved to proceed without checkpoint waits; licence-clean (commercial YES incl. all
  sub-models) candidates may be installed + run on the 03 window without my go; UNCLEAR/NO/RESEARCH_ONLY and >30 min GPU runs need my go.
- 19:52 vast.ai credit $27.80 (user topped up) ≈ 18 h at $1.55/h.
- ~20:40 B Phase 1+2 ACCEPTED (eb1288e, 0f8ecbb; metric 25aedba). Findings: LatentSync 16-frame window-boundary jump (every 0.64 s);
  CodeFormer texture popping (mouth resid 1.43 vs source); no restorer + C2 boundary interpolation → resid 1.01, SyncNet 6.31 vs 6.12,
  but lower-face sharpness 0.60. Phase 3 = source-texture transfer.
- USER: human twitch ground truth on 03 = 24–25 s and 35–36 s (frames ~600–625, ~875–900). Sent to both.
- USER DECISION: shipping pipeline must be commercially clean (zero NO). Licence status (both audits): CodeFormer NO (S-Lab NC);
  InsightFace buffalo_l NO (LatentSync detector + paste-back landmarks; also shipped in infra/serverless/handler.py); LatentSync 1.6
  weights OpenRAIL++ (YES with use restrictions); RetinaFace R50 UNCLEAR; FFHQ-trained restorers YES* (legal sign-off).
  A: replace buffalo_l in LatentSync crop path (MediaPipe/YuNet), lead generator TBDub (Apache-2.0). B: drop CodeFormer, move
  paste-back landmarks to MediaPipe. Audio side (Whisper/Qwen/Chatterbox/pyannote/Demucs) not yet audited by anyone → TODO.
- Coordinator check of the user's windows: mouth_jitter_hr per-second ranks on baseline 03: 24 s resid rank 38/91, 35 s 77/91 → the
  metric does NOT single out the human-marked seconds. Asked B to look at those frames and extend the metric.
- ~20:58 USER DIRECTIVE CLEAN-ONLY (`directive_clean_2026-10-16.md`): quarantine NC weights on both servers, clean face stack
  (MediaPipe) owned by A → clean0 → branch clean-base; B continues post side with clean code only. Agents' turns were interrupted
  (stop_session) so the directive and the older queued messages could be delivered (A had not ended a turn since 18:32).
  New rule for agents: end a turn at least every ~45 min.
- 20:59–21:00 quarantine DONE on both servers (`/workspace/dub/quarantine_nc/` + README with sha256): buffalo_l (all 5 onnx + zip),
  codeformer.pth, parsing_parsenet.pth, RetinaFace R50 (+ GFPGAN / RestoreFormer on B). Find check: none left outside quarantine.
  Trap (B): insightface FaceAnalysis auto-re-downloads buffalo_l and download_models.sh re-fetches CodeFormer → re-check periodically.
- 20:59 backup: everything pushed — A `experiment/lipsync-temporal` d4d1c94; B `experiment/restoration-temporal` 0f8ecbb + WIP
  snapshot `backup/restoration-temporal-wip-1016-2059` (f9f812a, made with a temp index, B's working tree untouched).
  Metrics/reports mirror every 10 min → `C:\Users\irina\Documents\DabAI_AB_backup\{A,B}` (`coordination/backup_metrics.sh`).
- Common metric copied to A: `/workspace/dub/common/mouth_jitter_hr.py` (from B 25aedba, EVAL_ONLY, no insightface import).
- Audio-side licence audit delegated to a local research subagent → `coordination/LICENSES_audio.md` (stalled once, resumed).
- A clean0 ACCEPTED (326e807, `CLEAN0.md`): MediaPipe face stack (router + crop + verify + mask landmarks), no restorer, guard_nc()
  blocks NC loaders; 03 vs cand5: mouth resid 1.29 → 1.02, SyncNet dub-only 5.73 → 5.70, AV 0, crop accel −26 %, mask bbox −46 %,
  mouth sharpness 0.98 → 0.71, window-seam jumps 41 % → 60 % of starts (now the dominant artefact). Verified: 0 NC files on both.
- B human windows ACCEPTED (477c05c, `human_windows_03.md`, metric v4): 24–25 s and 35–36 s = dub silent while the source still
  talks → LatentSync holds a frozen closed mouth inside a talking face (+ onset snap at window boundary 604). A found the same.
  v4 adds frozen_mouth (34 s rank 8, 35 s rank 9, 24 s rank 22 / 90), shear, max_d1_jump. Root cause partly audio (dub shorter
  than the source speech burst) → raise to the user as a possible audio direction.
- 21:3x clean-base = 326e807 pushed. Metric v4 copied to A `/workspace/dub/common/`.
  A next: make clean the default (safety), 04 smoke, generator-side seam fix (overlap windows), then TBDub vs clean0.
  B next: merge clean-base, reproduce clean0 on B, frozen-mouth options (grown mask vs pass-through) with SBS for the user,
  C2 seam interpolation on clean0, sharpness via source texture. Same seam attacked from both sides on purpose.
- 00:38 keep-awake requested (laptop slept 00:00–00:35, sync paused; agents keep it awake only while mid-turn).
- 2026-10-17 ~00:45 USER shared the customer spec (`DabAI_описание_этапы_бэклог.docx`) → `quality_tasks_from_spec.md`.
  Gaps: outputs 25 fps vs 30 fps sources (spec 9.4, also judder); no head-yaw pre-check (5.4/5.5); CodeFormer's "no blur" intent
  (8.3) must be met cleanly; seams (8.5) = contractual liability; feather 10–15 px (8.4) unverified; 2 faces (8.2) untested;
  norm ≤ 300 s/video-min binds every change. Tasks sent to A and B (turns interrupted again so they apply now).
- USER VERDICT: clean0 at 24–25 s / 35–36 s "looks not that bad" → clean0 is a sound base. Forwarded to both.
- USER: wants the "English dub shorter than Russian" problem solved: logical pauses → silent face → partial sync; unsure how.
  Proposed a layered policy (pauses where the source is silent → mild slow-down → guarded fit-up → video fallback by gap length)
  and a 3rd agent for audio timing (CPU-only on server B first). USER: yes, on server B, CPU, no new rent.
- Audio licence audit DONE (`LICENSES_audio.md`): HDEMUCS bed separator NO → quarantine on both (A/B told; --background off for new
  E2E; bed regression = dead air returns → clean bed needed later); pyannote 3.1 stack YES*(NC) (MSDWild / VoxCeleb research-only)
  → user decision pending; Whisper / Qwen 7B / Chatterbox / Silero YES*(U); wav2vec2, librosa YES; ffmpeg = Ubuntu GPL build
  (--enable-gpl, libx264/x265): fine server-side, never ship the binary. LatentSync training data (VoxCeleb2/HDTF?) asked of A.
- Agent C started: [agent-c-timing] local_36c1f10e-a9c8-4af3-b86d-6f207826bc7d on dub-B, CPU-only, clone /workspace/dub/dubai-audio,
  branch experiment/audio-timing from clean-base. Layers: pause placement → slow-down corridor → guarded fit-up (GPU later) →
  residual-gap JSON for B's silent-face options.
- B: feather measured — clean0 / cand5 mask edge 7.7 px (10–90 %, face ~224 px) < spec 10–15 px; new composite `--edge-px 12` = 11.4 px.
- ~01:20 USER LICENCE RULE (final): the licence OF THE FILES decides. Allowed: MIT / Apache-2.0 / BSD / ISC / OpenRAIL (honour the
  use restrictions). Not allowed: non-commercial / research-only licences and weights with NO licence at all. Training-data
  provenance (YES*) is recorded for legal review but does not block. HF "grant access" on gated models (pyannote) = access terms only.
  → pyannote ALLOWED; GFPGAN / RestoreFormer / VQFR (Apache-2.0) allowed again as options; LatentSync allowed (OpenRAIL++).
  → still out: CodeFormer (S-Lab NC), InsightFace buffalo_l (NC), RetinaFace R50 (no weights licence), HDEMUCS weights (no licence).
  → C gets a clean background-bed separator task (Spleeter MIT candidate; verify weights licence).
- B was blocked by its auto-mode classifier (relay of the licence rule to un-quarantine GFPGAN; outbox hardlinks). Not laundered:
  told B to drop both (restorers stay quarantined; sbs/ dir is synced anyway).
- ~02:30 progress: A 05dae5c clean default, db440fd frame map + continuation overlap/24-frame windows (no seam numbers yet),
  65ff44c HDemucs quarantined + --background off, de61988 head pose 5.4/5.5 + robustness 11, 396835d two faces 8.2; 04 E2E PASS,
  LatentSync 255 s/min + MediaPipe 44 s/min (speed issue later). B: clean0 reproduced (SyncNet 5.74, resid 1.025), F0 source-fps
  composite: upper-face judder 1.36 → 1.03 (9.4 fixed), feather 11.4 px; sharpness sweep + 6 variants running. C: cbfec83 spread
  policy, 03 dub-silent gaps > 1 s 4.3 → 1.4 s; full grid running.
- PAUSE (user): agents pushed final A 1d2f78e, B 1a244cf, C 07f4444; WIP snapshots backup/*-wip-1016-2316/2317.
  Report → branch coordination/report-2026-10-17 (3ea3e0d, b424988).
- All videos pulled: A 127/127, B 220/220 → Videos\DabAI_AB\all (18 GB); non-video artifacts (wav, manifests, npz, sheets) →
  Documents\DabAI_AB_backup (A tar 722 MB, B 27 per-dir tars + loose, 2.3 GB). Git clean on all three clones.
- ~03:45 local: instances 56277909 (A) and 56277943 (B) DESTROYED per user. Sync / backup loops stopped.
  A third instance 56283185 (Latvia, $0.665/h, up 6 h, GPU 100 %, VRAM 29.5 GB) exists on the account — NOT ours, left untouched.
- 2026-10-17 ROUND 2 (lower-face slide, user saw nose/lower face sliding vs upper face on camera shake). Many vast hosts refused SSH
  after kex (host-side); working: A 56420413 (EE, dub-a / app dub-A2), B 56435140 (EE, dub-b / dub-B3, replaced 56420387),
  C 56456352 (FR 82.65.197.236:40508, dub-c / dub-C2, audio). Agents: A local_b820b686, B local_067c4d64, C local_070acb20.
  A: 3 causes (fp16 AlignRestore, lagging crop anchors, lagging mask) → flow anchors + fp32 + head-locked mask: slide −44…−85 %;
  USER verdict "no longer jitters badly, rather OK"; full 25 fps renders 01/03/04/05 delivered. 05 coverage issue: clean stack
  lip-syncs 379 frames vs cand5 ~1000 → A told to fix detector coverage now. B: 25 fps master lags its clock ~60 ms (ffmpeg -r 25)
  → FIX30 source-fps composite fixes slide on 03/04. C: L1+0.875 policy + Spleeter bed as defaults, HDEMUCS blocked; layer 3 running.
  Branches pushed: experiment/r2-lipsync-position, r2-composite-position, r2-audio (agents now push on every commit).
- USER: when ~1 h of balance remains → push all, pull all videos + non-video artifacts, destroy ALL servers (no confirmation needed).
  Script: coordination/final_shutdown.sh. Chrome got logged out of vast.ai → user must log in again for balance / destroy.
- 2026-10-19 ROUND 3. Round-2 final pull was INCOMPLETE (A dropped connections): laptop has only A 47 videos (212 MB) in
  Videos\DabAI_AB\all_r2; B/C videos and all r2 non-video artifacts lost with the destroyed servers. Code safe in git.
  Lesson: sync_r3.sh now pulls outbox continuously and keeps every handoff tar on the laptop (Documents\DabAI_AB_backup\r3_handoff).
- Rented (vast API from Chrome, credit $31.16): A 56682077 IT 5090 r3-a 95.253.220.115:50275 ($0.76/h, 256 CPU);
  B 56682102 FR 5090 r3-b 82.65.197.236:40604 ($0.78); D 56682079 JP 5090 r3-d 219.122.229.5:47341 ($0.73).
  3090 56682078 (for C) destroyed per user: C runs CPU-only on server A (root /workspace/dub/c, clone dubai-audio).
- Branches: A experiment/r3-lipsync (from r2-lipsync-position 92de423), B experiment/r3-face (r2-composite-position 6341e2e),
  C experiment/r3-audio (r2-audio a1fd764), D experiment/r3-integration (clean-base 326e807). Briefs in /workspace/dub/briefs.
- Agents: A local_c61b2829 (xhigh), B local_b8d69966 (xhigh), C local_9eaebee6 (high). D NOT started: user declined folder trust
  on dub-r3-D → server D idle.
- USER: C may use A's GPU too (shared, C ≤ ~10 GB VRAM, A ≤ ~20 GB, one GPU job at a time each). Told A and C; C's turn interrupted so it applies now.
- Agent D started on 2nd try: local_60d73a4c (medium). USER: use the GPUs to the full on all three servers → told A, B, D to keep the GPU busy (queue renders, parallel jobs when VRAM allows).
- ~10:35 GPU sample: A 10 %, B 80 %, D 9 %. USER: use every free GPU to re-measure open spec items. D → spec_status.md (9.4/5.6, 8.5, 8.4,
  12.1 SyncNet, 12.2 CER, 11 2 %-rule, 5.4/5.5 pose, 8.2, speed + VRAM) then the all-GPU audit (Qwen on CUDA llama.cpp, peak VRAM per
  stage, all-resident on 32 GB?). A → render 01–05 on C audio-r2 + 5.4/5.5, 8.2, 11, 8.5 measurements; A's turn interrupted to deliver.
- Relay C→B via laptop crawled (107 MB in 23 min from IT host) → killed; the identical tar already on the laptop is uploaded
  directly to B and D (chunked). Lesson: re-use the laptop copy for every target of the same handoff.
- USER asked: are we really on commercially clean models? Audit of all 3 servers: buffalo_l dirs hold only NOTICE_NC_QUARANTINED
  (setup quarantines them); no CodeFormer / ParseNet / RetinaFace R50 / GFPGAN. BUT download_models.sh re-fetched
  models/demucs/hdemucs_high_trained.pt on B and D → moved to /workspace/dub/quarantine_nc/demucs/ (sha256 + README), not in use.
  Told all agents: no download_models.sh re-runs, cand5 = reference numbers only (never re-render), every candidate uses C's clean
  audio-r2 (Spleeter), audio source in every table row. D to make download_models.sh skip banned files + fail loudly.
- USER (19.10): LatentSync stays — OpenRAIL++ is acceptable (weights openrail++ on HF, code Apache-2.0; S3FD/syncnet_v2 only used by the eval metric).
- ~13:00–15:30 agents hit the usage limit. USER: all agents → effort MEDIUM (save tokens). D GPU audit (e442b5e, gpu_audit.md): all
  resident FAILS 32 GB (03 OOM); per-stage unload peak 22–24.7 GiB, audio+LS resident 30.4 GiB; LatentSync 317–342 s/min alone,
  pipeline 460–510 s/min (norm 300 FAIL); peak grows with LS segment length → cap ~400 frames. r3F (audio-r2) vs clean0: slide
  0.05–0.13 px (vs 0.17–0.24), 9.4 + 8.4 PASS all, WER 1–4 %; SyncNet 05 −0.86 / 04 −0.34 (different audio → A/B needed);
  audio TP −0.2..−0.4 dBTP, LUFS −12.9..−20 → C. audio-r3 relayed to A, B, D (15:26). A/B/C resumed on medium.
- 19.10 evening: USER saw (1) a one-frame lip "flash" in 04 ~22–23 s and 03, (2) abrupt phrase cuts in the voice of 03/04.
  Coordinator diagnosis: (1) FIX30 25→30 flow-interpolated frames tear the lips (04 frames 670/675/680; ls_raw and the 25 fps composite
  are clean); (2) hard on/offsets at TTS part joins (04: 5.71/6.59/17.70/44.50/45.22/46.59 s; 03: 8.40/14.30/37.30/53.24/70.04 s).
  Servers B (56682102) and A (56682077) DESTROYED after all videos + C listen clips were pulled. D kept: USER → fix both on D (agent D,
  medium) and remake the finals.
- 19.10 late: r3final_fixed v4 (904c5d2) for 01–05. USER: 'much better overall'; still some twitching and some choppy moments in the voice.
- USER: improve the E2E PIPELINE, not individual videos. D told: r3final stack → generic defaults of the one-command E2E runner, auto thresholds, built-in QA gates (flash, hard-cut, bgring, fps) with auto-remediation; generic fixes for residual twitching and TTS part prosody jumps; validate one command on all 5 → <id>_e2e_v5.mp4.
- 20.10 ~00:15: USER verdict on v5: 'all crap, redo'. Everything pushed: experiment/r3-integration 32a3605 (v5 one-command E2E + QA gates). Agent D idle; its E2E batch (05, 03 re-run) still running on server D.
- 20.10: USER: stop and destroy D, redo from scratch. D run killed, instance 56682079 DESTROYED; no instances left on vast. Sync stopped. Code: experiment/r3-integration 32a3605 (clean).
- 20.10 USER feedback on v5 → user_feedback_v5_2026-10-20.md (proper nouns 'Сочи Сити'→'Solki', chin wave, residual lip inconsistency, v5 bed regression: Russian voice bleeds; v4 bed better; no more flashes = win).
- 20.10 ROUND 4 (base v4 904c5d2): rented A 56894197 TX r4-a 172.12.213.59:3833 (audio, $0.67/h), V 56894255 CA r4-v 192.234.50.233:1873 (video, $0.78/h); credit $5.90 (~4 h) — user will top up. Repo is public → HTTPS clone; push waits for deploy keys. Agents: V local_e73ca6f3 (high), A local_86fab407 (medium). Sync → Videos\DabAI_AB\r4.
- 20.10 ~17:35 USER topped up: credit $22.83 (~15 h at $1.45/h). Final-save timer cancelled; bundle backup every 20 min keeps running.
- 20.10 ~21:15: USER chose audio pb1 by ear and wants full merged videos. Server V (56894255) EXITED/stopped by host; restart queued (resources_unavailable). Code safe: experiment/r4-video e60218a on GitHub + laptop bundle. Merged m2 (r4_video_v1 stack + pb1) handed to the audio agent on server A.
- 20.10 ~21:30: old V 56894255 DESTROYED (per user). New V2 56934170 CA $0.89/h r4-v 192.234.50.233:6473 (same host IP), branch experiment/r4-video @ e60218a, setup running. New video agent local_c6ba5ee8 (high): tasks 4/5/6. Old V session unreachable. New deploy key pending.
- 20.10 late USER on 05 m2: 34–35 s mouth flash; ~11–12 s mouth talks with no dub; ~13 s mouth closed while dub talks. New generic rule (both agents): while face visible, mouth follows the DUB (dub speaks → lip-synced; dub silent → closed/relaxed, never source talking mouth) + gate metric mouth_audio_mismatch_s. Applied before m3 (audio agent turn interrupted).
- 21.10 USER: push everything; detailed brief for tomorrow's fresh-server continuation → brief_r5_next.md. Branches: r4-audio ea1f6c4, r4-merged a81dc55, r4-merged-m4 8ebb299, r4-video fd80085 (+ backup/r4-video-wip-1020-2103).
- 21.10 ~00:10 USER verdict: 'the face still twitches — that moment kills it'; Max's feedback not fixed. brief_r5_next.md: face twitch = #0 blocker. Final save done (bundles, 50 videos, reports/handoff tars). Destroying A 56894197 and V2 56934170.
- Both r4 servers DESTROYED; 0 instances; credit $12.14. NEXT_BRIEF bf609a8.
- 24.10 ROUND 5: rented F 57519702 (destroyed before setup), G 57519705 VN r5-g 118.68.92.236:43077, A 57519710 (SSH dead, destroyed) → A2 57520238 VN. USER: 2 servers are enough → G = video (face #0 + merged E2E + LatentSync settings), A2 = audio. Credit $9.35.
- 24.10: agents A local_ce9fc2c8 (audio, high, r5-a 115.73.216.179:49253), F local_c33101a9 (video, high, r5-g 118.68.92.236:43077). Sync → Videos\DabAI_AB\r5, bundles → r5_git.
- 24.10 17:23 local: credit $6.76, burn ~$1.73/h → zero ≈ 21:15 local. USER: no top-up yet, watch balance and save in advance. final_save_r5.sh ready; timer fires ≈ 20:15 local.
- 24.10 19:10: balance $1.48 → final save + push. Branches: experiment/r5-audio 4e688e7 (incl. 720061c STATUS), r5-audio-14b 81e8da4 + backup/r5-audio-14b-wip-1024, r5-face 23965e1 + backup/r5-face-wip-1024. Twitch 03: m3 45.0 vs v4 52.8. Audio r5 vs pb1 on 03: src-talks/dub-silent 19.1→10.0 s, face-visible dub-silent 12.4→3.1 s, big gaps 8→2; fails 4 gate axes. Destroying G 57519705 and A2 57520238.
