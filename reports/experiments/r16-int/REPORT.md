# r16 T: KeySync in e2e + image (07.11.2026, server r16-qt, shared with Q and L)

Branch `experiment/r16-int` = `experiment/r15-keysync` (e61b056) + merge of `experiment/r15-tz` (c3ac1c5) + the r16 T commits. Not pushed.
Code: `scripts/keysync/r16/`, `scripts/r9/e2e.sh`, `scripts/job/`, `infra/docker/`. Outputs: `/workspace/dub/outbox/r16_t/`.

## Кратко (RU)
- **KeySync встроен в конвейер.** Одна команда: звук → KeySync fast2 → простая вклейка. TensorRT-движки собираются сами при первом запуске. Это теперь путь лица по умолчанию (`DUB_FACE=keysync`, в задаче `"face": "keysync"`).
- **Двое в кадре (#14).** Речь дубляжа сопоставляется с лицом по движению губ. KeySync идёт только на лицах, которым досталась речь, остальные не трогаются и вырезаются из маски вклейки. На m03 речь досталась маме, ребёнок как в исходнике.
- **Прошли целиком:**
  - m03;
  - вертикальное 4K (#23);
  - 5 минут 1080p60 (#24);
  - m03 в «чистой комнате» по рецепту образа (профиль clean, без интернета).
- **Docker-сборка (#26) не сделана.** На vast нельзя запускать Docker внутри контейнера. Нужна VM или машина с Docker; рецепт готов и проверен в чистой комнате.
- **Узкое место — вклейка на процессоре:** 3,5 с на секунду видео на m03, 6,3 с на 5-минутном (кроп 720p), 24 с на 4K-кропе.

## 1. The face stage (`scripts/keysync/r16/ks_face.sh`, called by e2e.sh when `DUB_FACE=keysync`, the default)
1. 25 fps master.
2. MediaPipe with 2 faces.
3. `ks_e2e_prep.py` plan:
   - one face → the r15 K recipe;
   - ≥ 2 co-occurring faces + dubbed units → `assign_speakers` (the r15 F `LS_FACE_UNITS` logic). KeySync runs per speaking face, each hearing its own units, the others muted;
   - faces with no speaker are untouched and subtracted from the paste mask (`--others`).
4. `ks_prep` crop (2 faces: `KS_BOX_K 1.25 / KS_BOX_M 0.4`).
5. ONE GPU-lock hold:
   - TRT engines (`ks_engines.sh`: cache keyed by GPU + driver + TRT, TRT_VERIFY ≥ 56 dB);
   - `ks_fast` fast2 (`--interp-steps 2 --vae-compile`);
   - the SyncNet offset on the first 60 s.
   In e2e the lock is inherited from the audio stage (`KS_HELD_FD`), so there is no second wait in the queue.
6. CPU: `ks_plain` paste (`--tex 2 --cavity 0.8`), one face after the other.
   - Output on the 25 fps master grid (flag `output_format_differs_from_source`);
   - 4K / large sources go back to full resolution through `hires_merge` on the 25 fps grid.

**Hooks for K1 / K2** (env only, nothing to merge):
- `KS_DUB_DELAY_MS` (40 = K1 step 0; only the generator input is delayed, the final mix never);
- `KS_PASTE_SHIFT` (`auto` | integer; K1: 0);
- `KS_PLAIN_ARGS`;
- `KS_INTERP_STEPS`;
- `KS_SYNC_MAX_S`.

m03 check vs K's r15 recipe:
- the crop is identical (223,246, side 476);
- raw SyncNet 8.03 / offset −1 vs K's 8.09 / −1;
- raw output 42.9 dB PSNR vs K's fast2 raw (sampling noise).

## 2. Runs (one RTX 5090, driver 575.57, TRT 10.16.1.11; times = GPU-held / CPU, NOT counting the shared-GPU queue)
| Run | Result | KeySync (ks_fast wall) | SyncNet | Paste (CPU) | RAM peak | Notes |
|---|---|---|---|---|---|---|
| m03 40.5 s, 2 faces | SUCCESS | 145 s (215 s/vmin; 163 without landmarks) | ~25 s | 143 s | ~25 GB | mother = SPEAKER_00 → face 0 (corr 0.39), child untouched |
| 4K vertical 20 s (#23) | SUCCESS, 2160×3840 | 61 s | 18 s | 478 s (crop side 1080) | 18 GB | face work at 1920, hires_merge 107 s |
| 5 min 1072×1904 60 fps (#24), `E2E_FACE_MAXSIDE=1280` | SUCCESS | 1036 s (207 s/vmin) | 438 s (→ ~90 s with the 60 s cap, db5c155) | 1888 s | **61 GB** (ks_fast) | audio 15 min, hires 7 min, disk ~3.5 GB |
| clean room m03 (image recipe, clean, HF offline, empty caches) | SUCCESS, CHECK_OK | 322 s cold → 145 s warm | 29 s | 147 s | – | cold start below |

**Cold start of a worker** (new GPU / driver, empty volume caches):
- TensorRT plans 639 s (kf 338 + student 301);
- first job +~350 s (VAE inductor autotune + keyframe warmup);
- model load 30 s.

That is ~17 min once. Later jobs: load 30 s + KeySync. The engines and the inductor cache live on the models volume (`models/keysync/trt_cache`, `inductor`).

**Warm face GPU per video minute:** ~210 s/vmin, KeySync incl. landmarks (K measured 137–148 without landmarks / encode). Audio stays ~95 GPU-s/vmin (r15).

## 3. Limits found
- **RAM: ks_fast keeps every frame, ~12 GB per video minute** (61 GB at 5 min). ~8 min fits in 128 GB. Fix later: chunk ks_fast per ~60 s (keyframes are independent per chunk).
- **Paste speed:** ks_plain is one process, ~3.5–6 s per video second at a 720p crop, 24 s/s at 4K's 1080 crop. Recommend `E2E_FACE_MAXSIDE=1280` as the KeySync default (KeySync draws 512 px; the paste cost grows with the crop side). Frame-chunked parallel paste = the next speed step (K1 owns ks_plain).
- **Static crop:** `ks_prep` takes one box per video (median face position). Cuts or big head travel in long videos are not handled; not seen on the test set.
- **Output frame rate:** 25 fps (the KeySync grid), not the source fps.

## 4. Image (#26): recipe done, `docker build` NOT run
- **Why:** vast is an unprivileged container: no Docker-in-Docker, and buildah fails on `unshare(CLONE_NEWUSER)` (tested).
- **What changed:**
  - Dockerfile builder stage: `ks_install.sh` → `/opt/keysync` (keysync 5e2503b + sm120 r15 patch, torch 2.7.1 cu128, TensorRT 10.16; **+~15 GB**: tensorrt_libs 6.2, nvidia 3.6, torch 1.9);
  - `fetch_models.sh` 8/8: toninio19/keysync checkpoints + KeySync's hidden runtime downloads, found by the clean room. These are facebook/hubert-base-ls960, the SVD VAE (stabilityai/stable-video-diffusion-img2vid, a **gated** repo, so the HF token must have accepted it) and face_alignment S3FD / 2DFAN4. Revisions / sha256 are pinned. Our 4 files come from `KS_KIT_DIR` (`lora_pick.pt`, `lora_meta.json`, `kf.pt`, `interp.pt`);
  - entrypoint: links the checkpoints;
  - `check.sh`: KeySync venv + weights + engines.
- **Clean room:** `scripts/keysync/r16/cr16.sh` = the same steps without Docker; m03 SUCCESS.
- **To finish #26:** on a Docker host, run
  ```
  DUB_PROFILE=clean bash infra/docker/build.sh
  fetch-models (HF_TOKEN, -v <kit>:/kit -e KS_KIT_DIR=/kit)
  check
  smoke m03
  ```
  then report the image size and cold start.

## 5. Server fixes made on r16-qt (affected Q too)
- `models/spleeter` was missing → audio failed. Installed with `scripts/setup_spleeter.sh`.
- `/venv/dabai` differed from `requirements.dabai.lock.txt`:
  - Chatterbox was ab55923 instead of the accepted 5de7a54;
  - noisereduce was missing;
  - ~10 minor versions had drifted.
  Synced to the lock; `/venv/main` too. `/venv/mediapipe` created from its lock.
- No htdemucs on this server: Spleeter bed only.

## 6. Interface v2
- New field `face`: `keysync` (default) | `latentsync`. `mode` applies to LatentSync only.
- New stage `face_keysync` in progress / ETA. New flags `face_keysync` and `face_keysync_multi_face` (technical).
- `scripts/job/example_job.json` added. INTERFACE.md / INTERFACE_RU.md updated.
- `test_interface.sh` 10/10. The HTTP part was skipped: no /venv/api on this server.
