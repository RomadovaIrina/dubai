# r6 G: is LatentSync 1.6 the quality limit? (generator ceiling, 27.10.2026)

Branch `experiment/r6-gen` (from `experiment/r4-merged-m4` 8ebb299). One RTX 5090.

## Verdict

**Keep LatentSync 1.6 with the m3 / m4 generator settings unchanged:**
- guidance 2.0, 20 steps, bridge 8×4, seed 1247, `--max-ls-frames 400`;
- composite = m4 (`--mouth-region 0.38`).

**Is LatentSync 1.6 the limit?** With licence-clean weights on this hardware, yes.

- **No setting dominates.** Within LatentSync, every lever that raises lip-sync also raises frame-to-frame instability: flashes, twitch, chin wave. All the variants land on one trade-off curve, and with face twitch as the user's blocker #0, g2.0 + bridge is the right point on it.
- **No clean alternative generator beats it.** MuseTalk 1.5 loses badly. InfiniteTalk is the only other clean candidate, but it is a 14B model that re-renders the whole frame; it fills the disk and cannot meet ≤ 300 s/min.

**No new flags go to F.** The recommendation is "no generator change" (see `handoff/out/F/gen-reco/`).

## What was tested

### Generator arms
- One setting changed per arm.
- Generator only: `ls_window_ab.py`, with the same master frames, dub audio, speech gate and mouth-only paste as m3.
- Guidance variants were added to `ls_stable.py` via `DUB_LS_CFG`; candidate_video_stage routes them through StableRunner:
  - `rescale:φ` (CFG rescale);
  - `interval:a-b:g` (full guidance on part of the schedule);
  - `decay`;
  - `dsmooth:α` (guidance delta (cond − uncond) smoothed over frames, aimed at single-frame flashes);
  - `dclip:k` (per-frame guidance-delta norm clip).

### Windows
- The worst 8-s dub-only SyncNet windows of the m3 run, 3 per video: 03, 04, 05.
- 05 only had 2 windows with enough tracked dub speech, so the set is 8 windows, not 9.
- The core table below uses the 6 windows that every core arm has metrics for. 03 67 s and 03 4 s were still in metrics when the run was stopped.

### Metrics
| metric | source |
|---|---|
| SyncNet conf_at0 on dub-speech seconds | `r4_sweep_metrics.py` |
| lip_r, flash (r3_flash), jaw ratio | `r4_sweep_metrics.py` |
| twitch | `r5_twitch.py` from r5-face; master window = the reference |
| chin wave, rigid variant | `r4_chin_wave.py` |
| mouth / dub mismatch | `r4_mouth_audio.py` |
| s per LatentSync frame | measured inside the GPU lock, so clean |
| peak VRAM | from the run |

Full 04 / 05 used the m4 stack on m3's audio, scored by `r4_gate.py score` + `r5_twitch.py`.

## Window results

Six 8-s windows. Sums are over windows; the rest are means.

| arm | SyncNet dub speech | worst s | flash Σ | twitch Σ | twitch events | chin wave (rigid) | mouth/dub s Σ | s / LS frame | peak MiB | sync wins vs base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **G0 base (m3: g2.0, 20 steps, bridge 8×4)** | 4.25 | 2.47 | 27 | 37 | 22 | 2.01 | 1.44 | 0.222 | 11 209 | |
| G10 seed 7 (noise floor) | 4.12 | 2.40 | 28 | 38 | 24 | 1.93 | 0.96 | 0.227 | 11 209 | 1/6 |
| G1 guidance 2.5 | 4.70 | 2.79 | **79** | **63** | 28 | 2.25 | 1.52 | 0.227 | 11 209 | 6/6 |
| G4 g2.5 + CFG rescale 0.7 | 4.71 | 2.92 | 79 | 66 | 29 | 2.25 | 1.16 | 0.217 | 11 209 | 6/6 |
| G2 g2.5 + dsmooth 0.5 | 4.58 | 2.71 | 48 | 49 | 27 | 2.20 | 1.28 | 0.226 | 11 209 | 6/6 |
| G3 g2.5 + dsmooth 1.0 | 4.35 | 2.47 | 35 | 50 | 25 | 2.16 | 1.72 | 0.217 | 11 209 | 4/6 |
| G7 bridge off | 4.34 | 2.57 | 36 | 38 | 23 | 2.16 | 0.32 | 0.201 | 15 356 | 4/6 |
| G8 g2.5, bridge off | 4.79 | 2.87 | 76 | 56 | 30 | 2.40 | 1.24 | 0.201 | 15 356 | 6/6 |
| G9 g2.0, 12 steps, bridge off | 4.40 | 2.62 | 39 | 40 | 24 | 2.19 | 1.16 | **0.159** | 15 356 | 4/6 |
| MT MuseTalk 1.5 | **1.46** | 0.33 | 9 | **89** | 57 | **3.35** | 1.24 | **0.029** | 6 644 | 1/6 |

**Seed noise floor** (seed 7 vs seed 1247): SyncNet 0.13, flash 1, twitch 1.7, chin 0.08.

**Extra arms on the 5 windows of 04 + 05**, all with bridge:

| arm | SyncNet | flash | twitch |
|---|---:|---:|---:|
| base | 3.96 | 20 | 28 |
| guidance 2.25 | 4.09 | 39 | 32 |
| g2.5 + dclip 1.5 | 4.34 | 47 | 31 |
| g2.5 + interval 0–0.5 (2.0 after) | 4.06 | 31 | 35 |
| g3.0 + dsmooth 1.0 | 4.39 | 54 | 50 |
| g2.5, seed 7 | 4.22 | 56 | 45 |

Full table: `win_table.md`.

Reading:
- **Guidance is one dial between sync and stability.** Every variant (rescale, interval, smoothing, clipping, 2.25, 3.0 + smooth) lies on the same curve. Roughly, each +0.1 SyncNet costs about 7–10 flashes and about 4–6 twitch units over 6 windows. None moves off the curve: +0.45 SyncNet at the base's flash / twitch level does not exist.
- **Steps 12 and bridge off are within noise on sync**, and 10–28 % faster per frame. But they use more VRAM (15.3 GB, AccelRunner path) and have more flashes at the window level (seams).

## Full videos

04 / 05, m4 composite on m3 audio. LS s/video-min includes waits for the shared GPU lock (the sweep ran at the same time), so it is not a clean speed measure; use s / LS frame above.

| arm | id | SyncNet dub speech | strong flashes | mouth/dub s | chin rigid | twitch Σ | twitch ev ≥ 2 | peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **m4** (reference) | 04 | 5.86 | 9 | 0.77 | 1.93 | 43.9 | 8 | 14 480 |
| **m4** | 05 | 5.22 | 29 | 2.67 | 1.12 | 40.2 | 5 | 14 751 |
| g2.5 | 04 | **6.53** | 22 | 1.10 | 2.16 | 56.4 | 11 | 20 438 |
| g2.5 | 05 | **5.92** | 28 | 2.64 | 1.38 | **73.3** | 8 | 23 232 |
| g2.5 + composite deflash `--deflash-mc 1.5` | 04 | 5.80 | 12 | 0.73 | 2.12 | 50.9 | 7 | |
| g2.5 + composite deflash `--deflash-mc 1.3` | 04 | 5.67 | 10 | 0.70 | 2.12 | 50.3 | 6 | |
| g2.5 + `--retime nearest_rigid` | 04 | 6.63 | 30 | 0.77 | 2.19 | 69.8 | 12 | |
| g2.5, LatentSync at 30 fps (`LS_FPS=30`, no retime) | 04 | 6.18 | 11 | 0.73 | 2.00 | 57.0 | 8 | 16 551 |
| bridge off | 04 | 5.82 | 10 | 1.10 | 2.02 | 45.4 | 10 | 22 145 |
| bridge off | 05 | 5.17 | 27 | 2.64 | 1.18 | 50.3 | 5 | 22 393 |
| 12 steps, bridge off | 04 | 5.90 | 13 | 0.87 | 2.09 | 48.0 | 9 | 20 674 |

Not run (stopped at the coordinator's wrap-up): 12 steps on 05, g2.5 @ 30 fps on 05, and the g2.0 @ 30 fps control.

### Why g2.5 cannot be "deflashed" into a free win
- **The flashes are the sync gain.** On a contact sheet of g2.5's 22 strong flashes on 04 (scratch, not shipped), the defects sit at fast closures and "oo" puckers: a smeared pucker at 668, a pink lip blob with no lip line at 715, a torn lip line at 882. In the raw 25 fps layer, g2.5 jumps wide-open → closed → pucker within one frame (raw 555→557 and 595→596).
- **Blending makes it a blob; repeating makes it a jump.** The 25→30 fps retime blends two very different mouths into a blob. Nearest-frame retime avoids the blend but repeats frames, which reads as a jump (flashes 30, twitch 69.8).
- **Removing those frames removes the gain.** `--deflash-mc`, new in `fps_composite.py`, replaces layer frames whose motion-compensated lip residual is ≥ k× their neighbours'. It takes flashes 22 → 10–12 but SyncNet falls back to m4's level (5.67–5.80): the frames SyncNet rewards are the frames the flash detector flags.
- **LatentSync at 30 fps is the best compromise found:**
  - +0.32 SyncNet, flashes 11 vs 9, mismatch 0.73 vs 0.77;
  - but twitch Σ 57 vs 44 (strong events equal at 8);
  - 20 % more generated frames;
  - only 04 was measured, and the g2.0 @ 30 fps control was not run.

  It is a candidate for a user WATCH only if the user ever values lip-sync over stability. It is not recommended now, because twitch is the blocker.

## Other items from the brief

### `--max-ls-frames`
Not swept separately; the run was stopped before it. Window runs are < 400 frames, so the windows cannot measure it.

400 stays. The r4 reasoning still holds: the cap's cuts land at the quietest dub frame (`--cap-align on`). Bridge-on peak VRAM is 14.5–23 GB on full videos, so a larger cap fits 32 GB but would not change the generator.

### Higher-resolution mouth crop / upscale
This is moot for these sources:
- the inputs are 464×848 portrait;
- the lower face is about 220 px wide (MuseTalk crop mean 221–223 px);
- LatentSync 1.6's 512² crop therefore already upsamples the face about 1.5–2×, so the generator works above source resolution;
- the paste-back downsamples, so super-resolution of the generated mouth cannot add real detail.

Any perceived softness is the generator's own texture plus the composite's lip sharpen (`--sharpen-auto 0.95 --sharpen-lips-max 0.8`, already in m4).

### Speed
Clean, inside the GPU lock, per generated 25 fps frame:

| setting | s / frame | if every frame is lip-synced |
|---|---:|---:|
| g2.0 + bridge | 0.222 | ≈ 333 s / video-min |
| bridge off | 0.201 | |
| 12 steps, bridge off | 0.159 | ≈ 239 s |
| MuseTalk 1.5 | 0.029 | |

12 steps + bridge off is the only real LatentSync speed lever, but it costs stability (04: twitch 48.0 vs 43.9, flashes 13 vs 9) and VRAM (20.7 GB). Use it only if speed must win. The bigger speed levers are outside the generator: CPU prep outside the GPU lock, fewer contended CPU jobs (see NEXT_BRIEF §5.6).

## Alternative generators: licence check

The licence fields were read from the HF model cards (API `cardData.license`) and READMEs on 27.10.2026.

| model | weights licence | verdict |
|---|---|---|
| **MuseTalk 1.5** (TMElyralab/MuseTalk) | card `license: mit`; README: "The trained model are available for any purpose, even commercially" | **Tested.** Clean wrapper `scripts/r6/musetalk_window.py`: crop from our MediaPipe landmarks (no DWPose / S3FD), **no face-parse-bisent** (CelebAMask-HQ, non-commercial data), VAE = our sd-vae-ft-mse (MIT), whisper-tiny (MIT). **Rejected on quality**: SyncNet 1.46 vs 4.25, twitch Σ 89 vs 37, chin wave 3.35 vs 2.01 (256² crop, single-step). It is 8× faster and uses 6.6 GB. SyncNet best offsets of ±2 frames vary by window, so this is not a constant shift bug. |
| **InfiniteTalk** (MeiGen-AI/InfiniteTalk) | `apache-2.0`; Wan2.1-I2V-14B-480P `apache-2.0`; chinese-wav2vec2-base `mit` | **Clean but dropped** (agreed with the coordinator): 14B, re-renders the whole frame (head and background drift), 82 GB of weights filled the disk; far beyond ≤ 300 s/min. |
| KeySync | card apache-2.0, but initialised from SVD (SVD community licence) and trained on CelebV-HQ (NC) | risky, not tested |
| Wav2Lip / Wav2Lip-HD, Diff2Lip, TalkLip, JoyGen, VideoReTalking | non-commercial / CC BY-NC / no licence on the weights | dropped |
| LatentSync 1.6 (baseline) | `openrail++` | Upstream `face_detector.py` loads InsightFace (banned). Our pipeline replays MediaPipe detections and `guard_nc()` blocks that loader (`check_latentsync_clean.py`), so production stays clean. |

## Deliverables

### Outbox
`outbox/gen_<id>_<t0>s_ARMS[_half].mp4` for 04 7 s, 04 45 s, 05 29 s and 03 18 s. Each is a 3×2 grid with the dub audio:

| | | |
|---|---|---|
| source mouth | m3 base | g2.5 |
| g2.5 + guidance smooth | 12 steps, no bridge | MuseTalk 1.5 |

### Code (all on `experiment/r6-gen`)
- `scripts/quality/ls_stable.py`: `DUB_LS_CFG` guidance variants. Default off, so there is no behaviour change.
- `scripts/quality/candidate_video_stage.py`: routes `DUB_LS_CFG` through StableRunner.
- `scripts/quality/fps_composite.py`: `--deflash-mc` / `--deflash-floor`. Default off. Not recommended (see above).
- `scripts/r6/`:
  - `r6_gen_sweep.sh`, `r6_metrics_pass.sh`, `r6_win_extra.py`, `r6_report.py`: the window sweep;
  - `r6_full.sh`, `r6_recomp.sh`, `r6_full_table.py`: full-video arms;
  - `r6_grid.sh`: ARMS grids;
  - `musetalk_window.py`: MuseTalk 1.5 backend;
  - `infinitetalk_window.sh`: unused.
- `scripts/r5/r5_twitch.py`: copied from r5-face.

### Artifacts (server G only, not pushed)
- `/workspace/dub/artifacts/r6g/{win,full}`;
- the m3 / m4 references in `/workspace/dub/artifacts/r4m/{m3,m4}`.

### Setup note
`scripts/setup_spleeter.sh` must run before any audio stage. The PyPI mirror resolve hung; plain PyPI worked.
