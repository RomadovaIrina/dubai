# r15 T: the ML TZ tails + the pipeline packaged for the backend

Branch `experiment/r15-tz` (on top of `experiment/r14-distill`). Incremental bundle: `/workspace/dub/r15t/r15-tz.bundle`. Server r14-d, 05.11.2026.

## Кратко для Ирины (RU)

**Что сделано**
- **4K-видео (вертикальное 2160×3840) больше не падает.**
  - Причина: подготовка кадров для губ шла в полном 4K на процессоре. 20 секунд видео обрабатывались 15 минут, памяти уходило до 45–69 ГБ.
  - Теперь губы делаются на копии 1080p, а изменения переносятся обратно на 4K-кадры. Всё вне рта остаётся исходными пикселями.
  - Проверено: результат в 4K, синхрон в порядке, 22 минуты на 20 секунд, 32 ГБ памяти.
- **Длинное видео** (раньше в пайплайн никто не подавал ролики длиннее минуты).
  - На 5 и 10 минутах нашлись и исправлены 7 разных сбоев, все в озвучке и все из-за памяти видеокарты.
  - На коротких роликах исправления ничего не меняют.
  - 5-минутное видео 1080p теперь проходит озвучку целиком: без губ — до готового видео (оба варианта фона), с губами — озвучку и синхронизацию губ.
  - Полный прогон с губами упал в самом конце: на сервере кончилось место на диске (промежуточные файлы заняли 41 ГБ).
  - Что это значит для запуска:
    - 5 минут видео — больше 40 ГБ диска и около 45 ГБ оперативной памяти;
    - видео длиннее примерно 9 минут пока не поместится в память (генератор губ держит все кадры сразу), это нужно доработать;
    - видео 1080p стоит обрабатывать с ртом в 720p, иначе не уложится в лимит времени. Решение за вами.
- **Повёрнутые видео с телефона:** раньше их боковые кадры иногда читались как «лица нет». Исправлено.
- **Сравнения для глаза** (папка `outbox/r15_t` на сервере):
  - old03, первые 30 секунд, и m03, первые 20 секунд: «как сейчас» (CodeFormer + htdemucs) и «коммерчески чисто» (без CodeFormer, фон отделяет Spleeter);
  - на m03 ещё быстрый режим против качественного;
  - 4K против исходника.
- **«Пропадающие губы» на m03 найдены по кадрам:** 8,1–8,3 с и 14,3 с; около 15,5 с губы мелькают.
  - Губы бледнеют уже в генераторе губ. CodeFormer местами делает хуже, а встроенная починка ловила только часть кадров.
  - Решение агента F («без CodeFormer + умная починка губ») встроено в чистый профиль. Пока выключено (`DUB_CLEAN_LIPCON`): ждёт его проверок.
- **Готово для бэкенда:**
  - вход — JSON с параметрами задачи; выход — видео, субтитры, `metrics.json` и статус;
  - HTTP-сервис: поставить задачу, узнать статус, получить файлы, отменить;
  - коды ошибок и тексты для пользователя на RU и EN;
  - описание: `scripts/job/INTERFACE.md` (EN) и `INTERFACE_RU.md` (RU).
- **Рецепт Docker-образа** (`infra/docker/`):
  - все версии библиотек закреплены, веса лежат на отдельном томе, некоммерческих весов в образе нет;
  - ключей к реестру у нас нет, выкладку делаете вы — команды печатает `build.sh`;
  - «чистая комната» на сервере нашла и помогла исправить три ошибки рецепта. Главная: без токена Hugging Face не запускалось определение говорящих, в образе падала бы каждая задача.

**Что решить вам**
1. **Профиль образа: `clean` или `current`.** Значения по умолчанию нет, профиль задают при сборке.
   - `current` — как сейчас на серверах: CodeFormer и htdemucs, оба некоммерческие. Это отступление вы приняли 29.10.
   - `clean` — без CodeFormer, фон отделяет Spleeter. На m03 вы сказали: «current лучше». С починкой губ от F («примерно как прод») разница меньше.
   - Список компонентов и лицензий: `reports/experiments/r15-tz/LICENCES_PROD_PATH.md`.
2. **Видео 1080p: обрабатывать рот в 720p** (`E2E_FACE_MAXSIDE=1280`)?
   - При 1080p наложение рта идёт примерно в 5 раз дольше. 5-минутное видео не уложится в лимит времени.
   - В 720p получается то же качество, что в одобренных клипах: они все 720p. Рот масштабируется обратно в 1080p.
   - По умолчанию пока не меняю.
3. Если выберете `clean`: устраивают ли «серые» компоненты (pyannote, Spleeter, Real-ESRGAN: лицензия свободная, но в обучающих данных есть исследовательские наборы), или их тоже нужно проверить юристам.
4. Если образ будут отдавать третьим лицам (а не только запускать у себя): обязательства GPL у ffmpeg и pykakasi.

**Что НЕ проверено**
- Сам `docker build` не запускался: на vast нет Docker, а собирать образ на 20 ГБ на ноутбуке без вашего согласия я не стал. Рецепт проверен «чистой комнатой» на сервере: тот же код, что получит образ, без интернета и без токена.
- Полная загрузка весов с нуля (24 ГБ) не запускалась.
- 5 минут целиком с губами до готового видео (упало на нехватке диска) и 10 минут целиком: не пройдены.
- Починка губ для чистого профиля от агента F включена в код, но выключена до его проверок.
- Субтитры для китайского, арабского и хинди глазом не проверялись.

## 1. The 4K vertical `face_gen_failed` (TZ 5.4)

- **Root cause.** The repro ran with the old path (`E2E_FACE_MAXSIDE=0`).
  - At full 4K, the generator's frame preparation (`candidate_video_stage`) runs on the CPU.
  - A 20 s clip ran 15 min with the GPU at 0 % (13 GB VRAM parked) and 28–45 GB RSS. It was stopped by hand at the coordinator's request.
  - The first fixed run hit a separate bug (below) and timed out at full 4K with 69 GB RSS.
  - RAM and time grow with length, so a longer 4K clip passes the container limit (85 GiB) or the timeout, giving `face_gen_failed` / TIMEOUT.
- **Fix** (commits 87a5485, a5840b4).
  - Above `E2E_FACE_MAXSIDE` (1920) px on the long side, the face stages run on a lanczos ≤ 1920 copy.
  - `scripts/r15/hires_merge.py` then adds only the composite's change back onto the 4K frames: out = hi + up(mask · (lo_comp − lo_src)), YUV planes, adaptive threshold, feathered mask.
  - Everything outside the changed mouth keeps the source pixels; colour tags are kept.
  - The bug in the first fixed run: `ffprobe -of csv=p=0:s=' '` does not parse in ffprobe 6.1. The size came back empty, so the downscale silently did not happen. Fixed; it now warns if the size cannot be read.
- **Validated** (`t15_4k_fix`, res4k_vertical 2160×3840, 20 s):

| Measure | Value |
|---|---|
| Status | SUCCESS, flag `face_work_downscaled` |
| Output | 2160×3840, 30 fps, matches_source |
| Failed lip-sync frames | 0 |
| Wall | 1319 s (face_gen 342, composite 726, hires_merge 144) |
| GPU-busy | 145 s |
| VRAM peak | 26.5 GB |
| RAM peak | 31.6 GB |
| hires_merge | 569 / 601 frames changed, mask 6.7 % of the frame |

- The clip `outbox/r15_t/m18_4k_mouth_fix_vs_source_half_speed.mp4` (4K pixels, 0.5x).
- The lips are redder than the source (a* 22.5 vs 18.1). That is the generator on m18, not the 4K path: the eye-approved r14 q2_m18 output has a* 22.0 on the same frame.

## 2. Long videos (TZ 12.3)

No video longer than ~1 min had gone through e2e before (r13: "long_10min not run"). The r15 runs found seven separate long-video failures; all are fixed:

| # | Where | Cause (evidence) | Fix (commit) | Short clips |
|---|---|---|---|---|
| 1 | Audio: Spleeter bed | One float64 STFT of the whole track on the GPU next to the resident Qwen / TTS: `torch.OutOfMemoryError … 1.64 GiB, 1.17 GiB free, process 30.18 GiB` (long10) | On a CUDA OOM: free the cache + retry, then the CPU (13 s for 5 min) (439d793) | unchanged (recovery only) |
| 2 | Audio: named entities | One Qwen call with the whole script: "Requested tokens (4929) exceed context window of 4096" (long10): no glossary | Windows ≤ 3000 tokens, 2 sentences overlap, merged (ede78bb, `test_ne_windows.py`) | the identical single call |
| 3 | Audio: TTS stage in-process | The retake ran inside the pipeline process, which still holds Qwen + TTS + QA (27–31 GB). The QA Whisper OOMed, then CTranslate2 failed permanently ("invalid device ordinal") | Above 120 s the stage runs in its own process (r8_tts_stage.py, the pre-speed-L path) (ce83fcd) | unchanged (in-process) |
| 4 | Audio: retake batch | All selected units of a speaker in ONE ladder batch | Slices of ≤ 16 units (46d594c) | unchanged (≤ 16 units) |
| 5 | Audio: stage start | The exiting pipeline process still held 16.4 GB ("[python] <defunct>", CUDA teardown) when the stage started; plus another agent's GPU steps (4.1 / 0.8 GB) on the shared server | The stage waits until VRAM < 4 GB (≤ 120 s) (9ada8bf); agent F now yields to the marker before every step | unchanged |
| 6 | Audio: face-aware retake (lip-sync on) | With `--face` the retake keeps more and longer takes; 16-unit slices still OOMed (the lipsync-off check had passed) | Slices of 4 units for videos over 120 s (b727f53) | unchanged |
| 7 | Audio: the batched round itself | VRAM trace on a clean card: the stage climbs to 22.5 GB with its models, then ONE batched round of long takes (T3 decode + S3Gen CFM) adds +8 GB, to 31.9 GB → OOM | Over 120 s the stage's takes are synthesised one by one (`R8_STAGE_TTS_BATCH=0`: DUB_TTS_BATCH / DUB_S3_BATCH off for the stage only; the reference path the batched one was A/B-matched against) (b992c8d) | unchanged |

- **Observability:** `dabai_job` now keeps every stage log through the cleanup (8b75e5d); without that, the first 5-min failure had no traceback. A watcher logs foreign GPU processes during a run (`/workspace/dub/r15t/gpu_watch.sh`).
- **Result (5-min 1080p, `long5_1080p.mp4`, profile current):**

| Run | Stages reached | Result | Numbers |
|---|---|---|---|
| t15_long5_audio (lipsync off) | precheck → audio → sep → mux | **SUCCESS**; both beds 1080×1920, 300.0 s video + audio, matches_source | wall 1686 s, CPU-contended (another agent at load 93 / 64 cores: VAD 353 s vs 4.5 s); GPU-busy 473 s = 95 per video-min; VRAM 32.1 GB; RAM 15.1 GB; the stage in its own process after a 1 s VRAM wait |
| t15_long5, rerun 6 (lip-sync, `E2E_FACE_MAXSIDE=1280`) | face_res → audio (1054 s: main pass 350 s, stage 551 s) → sep → face_gen, all 300 s generated (16:48:47 → 17:29) | **failed at the end of face_gen: the disk was full (ENOSPC)**, then mux failed the same way, so ERROR | face_gen ≈ 41 min = 8.1 s per video-second at 720×1280; VRAM 13.1 GB; generator RAM 44.7 GB anonymous at 4.5 min (it keeps all frames), job total 55–60 GB; intermediates **41 GB** (two raw-frame memmaps of 8.9 GB) on a shared disk that had 44 GB free |

- **What this means for production:**
  - **Disk:** ≥ 10 GB per video minute at 720p face work (two uncompressed frame memmaps, plus the stage videos). A 15-min job needs ~150 GB.
  - **RAM:** the generator keeps all frames, ~9 GB per video minute at 720p face work. At 85 GiB that is ~8–9 min, so `DABAI_MAX_VIDEO_MIN=15` is **not safe** until the generator streams its frames (follow-up: write the face crops to the memmaps only and release them per segment).
  - **Time** (estimate for a full 5-min 1080p at face 1280): audio ~18 min + face_gen ~41 min + composite ~30 min + hires_merge ~10 min ≈ 1 h 40 min. That is within the TZ timeout (6000 s for 5 min) only barely, and only because the audio stage was contended.
- The disk-full window (~17:25–17:31) may also have affected the other agents' jobs; the coordinator was told at once.
- **Speed for long 1080p videos (OPEN DECISION).**
  - From the 4K job at a 1080×1920 face size: composite ~33 s and face_gen ~14 s per video-second. At 720×1280: ~6 and ~9.
  - So a 5-min 1080p video at the default face size 1920 needs ~3.75 h and hits the timeout (100 min).
  - With `E2E_FACE_MAXSIDE=1280` it would need ~1.3 h (estimate). The mouth is then rendered at 720p, like every eye-approved clip, and merged back onto the 1080p frames.
  - Not flipped: the coordinator is asking the user.
- **Not validated today:** the face stages (face_gen / composite / hires_merge) on a 5- or 10-minute input. Their RAM and time on 9000+ frames are unmeasured. A known risk: whole-video frame reads (`read_frames`, `r8_cf_detail`) grow RAM with length; the 4K repro showed the same pattern. Next step: one full 5-min run with `E2E_FACE_MAXSIDE=1280` on a dedicated GPU (~1.5 h).

## 3. A/B for the eye

Clips are in `/workspace/dub/outbox/r15_t/`:
- `<id>_mouth_current_vs_clean_half_speed.mp4`: the mouth at 0.5x;
- `<id>_full_current_vs_clean.mp4`: the full frame;
- `<id>_current.mp4` / `<id>_clean.mp4`: each arm with its own bed;
- `m03_20s_mouth_fast_vs_quality_half_speed.mp4`.

| Clip | Arm | Wall s | GPU-busy s | face_gen GPU s | Notes |
|---|---|---|---|---|---|
| old03 30 s (464×848) | current | 800 | 234 | 139 | VRAM 30.3 GB (audio), RAM 21 GB |
| | clean (clean room) | 520 | 203 | – | profile clean, check.sh OK |
| m03 20 s (720×1280, 2 faces) | current | 565 | 146 | 98 | VRAM 26.4 GB, RAM 25.6 GB |
| | clean (clean room) | 463 | 121 | 91 | |
| | fast (student 4 steps + c8, subs + watermark) | 514 | 133 | 74 (−25 %) | subs (karaoke) + watermark burned in |

- All runs SUCCESS, with 0 failed lip-sync frames.
- On 20–30 s clips the fixed load times dominate, so the per-minute GPU numbers are not the steady-state rate. The long run gives that (section 7).
- **The eye (via the coordinator):** "current is better (than clean), but current also has many defects; somewhere the lips vanish completely".
- **Lip-vanish analysis** (`scripts/r15/lip_vanish.py`, `lip_focus.py`; sheets `m03_lips_vanish_sheet.jpg`, `m03_lips_15s_stages.jpg`).
  - Method: per frame, lip-vs-skin a* and lightness contrast as a fraction of the source, at every stage.
  - Current: 16 washed frames (3.2 %); clean: 25 (5 %).
  - Real vanishing: 8.12–8.32 s and 14.32 s.
  - Around 15.5 s: redrawn / generator / pale frames alternate (15.36 s is washed in both arms and repaired; 15.60 s is mildly pale).
  - Cause:
    - it starts in the generator, every time;
    - the CodeFormer detail layer deepens it at 4.56, 5.24, 8.16 and 9.12 s;
    - the repair's wash detector missed 8.12–8.32 and 14.32.
  - Agent F built on this. "No CF + adaptive lipcon" (m03 dips 24 → 11 = prod CF) looks "about like prod CF" to the eye. It is wired into the clean profile behind `DUB_CLEAN_LIPCON` (commit 404e472), default 0 until F's gates pass (`/workspace/dub/r15f2/REPORT.md`).

## 4. The job interface (`scripts/job/`, contract `dabai-job/2`)
- `dabai_job.py` (CLI + library, stdlib only):
  - validates the job (unknown fields rejected, 22 target languages, modes `quality` / `fast`);
  - downloads a URL input without ever logging its query string;
  - runs `e2e.sh` in its own process group;
  - writes `status.json` every ~3 s (step, progress, ETA) and `events.jsonl`;
  - writes `<job_id>.mp4` / `.ass` / `metrics.json` and the status SUCCESS / PARTIAL_SUCCESS / ERROR / TIMEOUT;
  - error codes `E_*` and warnings `W_*`, with EN + RU texts;
  - cleans the per-job artifacts unless `keep_intermediate`. New in r15: the stage logs are kept (`e2e/logs/artifacts/`), so a failed job stays debuggable.
  - `versions.profile` reports what the job really loads.
- `job_api.py` (FastAPI):
  - `POST /jobs`, `GET /jobs/{id}`, `/events`, `/files/{video|subs|metrics|status|log}`, `DELETE`, `GET /health`, `/version`, `/schema/*`;
  - an optional bearer token;
  - one worker, FIFO;
  - restart-safe: a RUNNING job becomes `E_WORKER_RESTART` and its orphan pipeline is killed; QUEUED jobs resume in order.
- Tests:
  - `test_interface.sh`: 25 / 25;
  - `test_load.sh`: 9 / 9 (a burst of 10 jobs, kill -9 mid-job);
  - `scripts/r15/test_ne_windows.py`: pass.

## 5. The image recipe (`infra/docker/`)
- **Base:** multi-stage CUDA 12.8.1 + cuDNN on Ubuntu 24.04, CPython 3.11.17 (uv).
- **Locks:** the three ML venvs + the API venv, from the r14-d freeze, installed with `--no-deps`.
- **Native parts:** llama-cpp-python for sm_120; LatentSync a229c39; ffmpeg 6.1.1 (libass / libx264); Noto + DejaVu fonts.
- **Weights:** on a volume, filled once by `fetch-models`, pinned by `models.lock.json` (55 files, 23.2 GB, sha256). `HF_HUB_OFFLINE=1` is baked in.
- **`DUB_PROFILE=clean|current`:** REQUIRED at build and at run, no default. The NC parts of `current` come from a separate `/opt/dabai-nc` volume, never baked into the image.
- **Commands:** `entrypoint.sh` = api / job / fetch-models / check / smoke. `build.sh` builds and prints the push commands; it pushes nothing.
- **The clean room** (r14-d): the code as the image gets it (git archive minus the .dockerignore), no NC weights reachable, HF offline, no `HF_TOKEN`, profile clean.
  - `check.sh`: CHECK_OK.
  - Real jobs: old03 30 s and m03 20 s, both SUCCESS with profile clean.
  - It found three recipe bugs, all fixed (commit 6d31842):
    1. Diarization refused to run without `HF_TOKEN` even with the model cached offline. Every image job would have failed.
    2. `install_env.sh` read the generic `PYTHON_VERSION`, which the vast base exports as 3.12. The clean venvs came out 3.12.15; the variable is now `DABAI_PYTHON`, default 3.11.17.
    3. `check.sh` gave a false ffmpeg FAIL (`cmd | grep -q` under pipefail).
  - The clean-room jobs used the server's 3.11.17 venvs, whose freeze equals the locks. A 3.11.17 reinstall in the clean room was not repeated (~7 GB of downloads again).
- The research tree keeps CodeFormer / ParseNet on disk, so the licence guard (`assert_clean_tree`) rightly refuses `DUB_ALLOW_NC=0` there. That is why the clean arms ran in the clean room.

## 6. Licences of the current production path
`reports/experiments/r15-tz/LICENCES_PROD_PATH.md`:
- **NC:** CodeFormer (S-Lab NC), htdemucs weights (no licence, MUSDB NC data). ParseNet is NC but not loaded.
- **GREY:** pyannote, Spleeter, Real-ESRGAN.
- **LatentSync:** OpenRAIL++ with use restrictions.
- **GPL:** ffmpeg / x264 and pykakasi, if the image is distributed.
- **Replacements checked, none clean:**
  - GFPGAN, RestoreFormer and GPEN are FFHQ-trained; the clean data-chain alternative is a generic video SR on the mouth.
  - Every strong separator is MUSDB-trained.
  - So no third A/B arm was wired. F's no-CF + lipcon is the clean-profile answer instead.

## 7. Measured resources (one RTX 5090, quality mode unless noted)

| Run | Video | Wall s | GPU-busy s | VRAM peak | RAM peak | Notes |
|---|---|---|---|---|---|---|
| t15_ab_old03_cur | 30 s 464×848 | 800 | 234 | 30.3 GB (audio) | 21.0 GB | audio 245, face_gen 334, composite 206 s |
| t15_ab_old03_cln | 30 s | 520 | 203 | – | – | clean room |
| t15_ab_m03_cur | 20 s 720×1280, 2 faces | 565 | 146 | 26.4 GB | 25.6 GB | audio 140, face_gen 235, composite 177 s |
| t15_ab_m03_cln | 20 s | 463 | 121 | 26.4 GB | 24.5 GB | clean room |
| t15_m03_fast | 20 s | 514 | 133 | 27.2 GB | 25.1 GB | face_gen GPU 74 vs 98 s |
| t15_4k_fix | 20 s 2160×3840 | 1319 | 145 | 26.5 GB | 31.6 GB | face_res 9, face_gen 342, composite 726, hires_merge 144 s |
| t15_4k_repro (old path) | 20 s 4K | stopped at 900+ | ~0 | 13 GB parked | 28–45 GB | CPU-only frame prep |
| t15_4k_fix, 1st run (size-read bug) | 20 s 4K at full res | TIMEOUT 1803 | – | – | 68.8 GB | the downscale was skipped |
| t15_long5_audio | 5 min 1080p, lipsync off | 1686 (CPU-contended) | 473 (95 per video-min) | 32.1 GB | 15.1 GB | SUCCESS, both beds 1080×1920 300.0 s; main pass 952 s (VAD 353 s: another agent had 5 composites at load 93 / 64 cores; 341 s in the uncontended run), stage 314 s in its own process |
| long10 audio (failed, fix 1) | 10 min 464×848 | audio 683 until the OOM | – | 30.9 GB | – | – |

- **GPU-s per video minute:** the 20–30 s clips give 430–470, dominated by fixed load times (model loads, compile). The TZ norm (300) is about steady state. Short-clip slopes: face_gen ~3.5–4 GPU-s per video-second at 720p, i.e. ~210–240 per minute, plus audio.
- **Timeout:** default `max(1800, 1200 × video minutes)` s. It holds for 720p face work, not for 1080p at face size 1920 (section 2).
- **Disk:** weights 23.2 GB; a 20 s 1080p face job keeps ~4 GB of intermediates; plan 30–60 GB free for 5–15 min 1080p. `check.sh` warns under 60 GB.

## 8. Validated / not validated

| What | Result |
|---|---|
| Job interface v2: validation, errors, timeout, cancel, URL redaction, events, files | `test_interface.sh` 25/25 (CPU) |
| HTTP API, burst of 10 jobs, kill -9 restart | `test_load.sh` 9/9 |
| Real GPU jobs: quality (old03, m03), fast + subs + watermark (m03), clean profile in the clean room (old03, m03), 4K vertical | all SUCCESS, 0 failed lip-sync frames, output = source size / fps |
| The 4K root cause and its fix | repro + fixed run (section 1) |
| The image recipe | clean room: install from the locks, the code as the image gets it, HF offline, no token, check.sh OK, 2 real jobs. **`docker build` NOT run** (no Docker on vast; ~20 GB on the laptop not without asking) |
| `fetch-models` from zero (24 GB) | NOT run |
| llama.cpp CUDA build with nvcc 12.8 | NOT re-run (r14-d has nvcc 13.0; the clean room reused the r13 sm_120 build) |
| 5-min job | audio path SUCCESS end to end (lipsync off: both beds, mux); with lip-sync, audio + face_gen of all 300 s passed, then the job died at the end of face_gen on a full disk. Composite / hires_merge / mux on 5 min NOT validated |
| 10-min job | NOT run to the end (budget); its audio stage hit fix 1, now fixed |
| Face stages on 5–15 min input (RAM / time) | NOT validated |
| The `current` profile inside the image (the NC volume) | not run in an image; the profile itself is the servers' production path |
| F's clean lipcon (`DUB_CLEAN_LIPCON=1`) | wired (404e472), default off; F's gates pending (`/workspace/dub/r15f2/REPORT.md`) |
| CJK / Arabic / Hindi subtitle glyphs | not checked by eye |

## 9. GPU sharing on r14-d
- The queue (`/workspace/dub/r15t/gpu_queue6.sh`) starts a job after 60 s in a row with:
  - `gpu.lock` free;
  - no other agent's `/workspace/dub/r15*/E2E_RUNNING`;
  - < 2 GB VRAM used.
- Other agents' CPU processes do not block it (coordinator, 05.11).
- It takes the lock (fd inherited by the job), checks again inside, and writes `/workspace/dub/r15t/E2E_RUNNING`.
- Both are released when the job reaches its CPU-only tail (face_hires / subs / mux). The composite keeps the lock: it holds VRAM (CodeFormer + Real-ESRGAN).
- The job's nested `flock gpu.lock` calls pass through while `/workspace/dub/r15t/LOCK_HELD` exists (`bin/flock` shim).
