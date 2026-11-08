# r16 аудит багов + прогресс ТЗ, 08.11 (Q, только чтение)

Что проверяли:
- head интеграции T: `work-r16b` @ 2d53197 на r16-qt;
- бандлы K1/K2 в `r16in/day2/`. Слиты K1 a6ded57 и K2 3d66d9e / 9187ef1 / 688e28a. K1 64488de не слит и даже не скачан в репо.

Ничего не исправлялось и не коммитилось; на GPU ничего не запускалось. Каждая находка ниже указана с файлом и строкой; владельцы предложены.

## Коротко для Ирины (RU)

**Хорошее**
- Звук в финале = принятый микс бит в бит. Проверено на 12 выходах: золотые, full, long5, q03_alpha, q05. Совпадают MD5 пакетов и PCM и длительность. Нигде не перекодируется и не обрезается.
- Большинство пунктов прошлого аудита закрыты:
  - вклейка: ECC, тихое затухание в конце, параллельная = последовательная, `-c:a copy`, NaN в q05;
  - бороды идут в KeySync, Docker больше не обходит маршрут;
  - замок GPU;
  - дефолты KS_REF=0, гейт, сдвиг вклейки 0.
- Дефолты на самом деле в силе по всей цепочке: Dockerfile → dabai_job → e2e.sh → ks_face.

**Топ-10 багов, простыми словами**
1. **Плохой синхрон отчитывается как «успех».**
   - Если проба синхрона не сработала или рендер всё ещё сдвинут, статус всё равно SUCCESS: без предупреждения и без возврата минут.
   - Пример: q02, рендер сдвинут на 15 кадров → SUCCESS.
   - Это `dabai_job.py:89-92`. Владелец T.
2. **Проверка весов не видит SyncNet, детектор лиц S3FD и hubert.**
   - Без них проба тихо ставит задержку 0, маршрут лиц падает, а работа всё равно «успешна».
   - Падение самой проверки (код возврата 1) игнорируется.
   - Это `weights_check.py:12-14`, `e2e.sh:67`. Владелец T.
3. **Проба задержки m18 — не баг кода, а SyncNet против глаза.**
   - SyncNet честно хочет +35–44 мс на m18, а глаз хочет 0.
   - Предложение: правило «задержка 0, если запас уверенности над нулём < 2». На m18 / m03 / old03 оно даёт 0 / +40 / 0, как на глаз. Решение за Ириной, потом K2 / T.
4. **Шов ks_seam включается «всё или ничего» по порогу 2.0, а m03 сидит ровно на пороге.**
   - Один и тот же m03 дал 0 стыков в одном прогоне и 82 стыка (164 подменённых кадра) в другом.
   - Это `ks_seam.py:177`. Владелец K1.
5. **Вклейка смещена на 1 пиксель вправо и вниз при нечётных координатах кропа.**
   - Измерено на золотом m03; ECC это не исправляет.
   - Это `ks_prep.py:13-15`. Владелец K1.
6. **Ошибки «не то окружение голоса», «нет весов», «замок GPU» выглядят для клиента как «не удалось собрать видео, повторите».**
   - Повтор на том же сервере упадёт так же.
   - `e2e.sh` всегда выходит с кодом 0. Владелец T.
7. **Под замком GPU ~2 мин работы CPU на 5-мин видео:** кодирование, ориентиры лица, поиск монтажных склеек, кроп. Это тормозит очередь. Владелец T.
8. **Многоплановое видео: каждый план заново грузит KeySync (~15 с).** Если упал один план, всё видео отдаётся с исходной картинкой. Владельцы T / K2.
9. **Альфа-вклейка держит в памяти весь сегмент:** 5-минутный план с наплывом ≈ 46 ГБ в 1080p. Владелец K2.
10. **Паузы берутся из разметки дикторов исходника, а применяются к дубляжу.**
    - Где английская фраза длиннее русской, рот закрывается раньше звука, до 0,35 с (long5).
    - Владельцы T + Q.

Ещё:
- **Мои фиксы метрик не слиты в интеграцию:** мультилицевой маршрут, детекторы относительно исходника, slide без подбородка. Из-за этого золотые цифры T для q02 со 2 говорящими считаются по одному лицу.
- **Выход KeySync всегда 25 fps.** ТЗ 5.6 / 9.4 нужно решение.

**Хвосты ТЗ, которые закрываются за 1–2 дня**
1. Глаз Ирины на золотых / q03_alpha / q05 / long5. Параллельно разобрать q02 (PSNR 27 дБ).
2. Один прогон KeySync с субтитрами и вотермаркой (9.1–9.4): 2–3 ч.
3. Сценарий отказов на KeySync + правило «2 % кадров → частичный успех» (11.1–11.3): ~0,5 дня.
4. Проверка позы / поворота головы перед KeySync (5.5): ~0,5 дня.
5. 10-мин видео + набор граничных случаев на KeySync (12.3): ~3 ч GPU.
6. Сборка Docker и смоук на VM (5.1, холодный старт 0.10): 1 день, нужна VM.
7. Решения Ирины, каждое ≤ 15 мин:
   - fps (интерполяция или честные 25);
   - пауза a/b;
   - чёрная дыра (банк a);
   - «только clean» письменно;
   - порог SyncNet для договора;
   - A/B pace-fix;
   - ключ OpenAI.
8. Мелочи: таблица VRAM для KeySync; проверить флаг `many_speakers_one_voice` на long5; слить фиксы Q в интеграцию.

---

## Part A: bug audit (EN)

### A1. Orchestration (e2e.sh / ks_face.sh / dabai_job.py / face_route / locks)

The live entry point is `scripts/job/dabai_job.py`, run from `entrypoint.sh:42,45` and imported by `job_api.py:22`. `scripts/r13/dabai_job.py` is v1 and only `r14/eval_r14.sh` still uses it.

**Defaults in effect**

| Default | Where it is set | Verdict |
|---|---|---|
| face=auto → beards to KeySync | `Dockerfile:86` `DABAI_FACE_DEFAULT=auto` (825ace2) → `dabai_job.py:149` → `e2e.sh:157-160` pick_engine (LatentSync only with `DUB_BEARD_LATENTSYNC=1`, flag `beard_keysync`, 286b0ec) | OK |
| KS_REF=0 | `ks_face.sh:79` (2070d8b); the probe and the full run both use frame-0 ref; `ks_shots.sh:26` forces 0 per shot | OK |
| KS_GATE on | `ks_face.sh:94` `thr=0.12,cap=0.25,hi=8,lo=2`; one `--follow` ks_fast serves the probe and the full run (091be92) | OK |
| KS_PASTE_SHIFT=0 | `ks_face.sh:156` (825ace2); auto is opt-in | OK; the header comment at `ks_face.sh:8-9` is stale |
| Delay probe | `ks_face.sh:107-138`: clamp ±2 frames, conf < 3 → 0, no result → 0, each with a flag | Works, but see O1 and A4 |

**Findings**

| id | sev | file:line | bug | evidence | owner |
|---|---|---|---|---|---|
| O1 | MED | `dabai_job.py:89-92` (TECH set), `:411` | `ks_probe_failed`, `ks_probe_low_conf`, `ks_check_offset`, `face_route_failed` and `ks_dissolve_hard_switch` are TECH flags, so the job reports SUCCESS with no warning and no refund. | `r16t/golden/jobs/gold_q02`: `ks_probe_low_conf f0: conf 0.12 → delay 0` and `ks_check_offset … still 15 frames`, yet status SUCCESS, refund False. `q03_alpha` hard switch → SUCCESS. | T |
| O2 | MED | `weights_check.py:12-14`; `e2e.sh:67` | The precheck skips `syncnet_v2.model`, `sfd_face.pth` (sync_perframe.py:56, active speaker) and HF `hubert-base-ls960` (ks_fast.py:56). In Docker these are volume symlinks, so a missing file is a dangling link. A missing SyncNet means the route fails and the probe gives delay 0 → SUCCESS. A missing hubert means PARTIAL with the source picture. A weights_check crash (rc 1) is ignored; only rc 2 is acted on. | Read the code side by side | T |
| O3 | MED | `dabai_job.py:405-407` + MESSAGES; `e2e.sh:55` | `voice_env_mismatch`, `weights_missing` and `gpu_lock_failed` have no message, so they surface as `E_NO_OUTPUT` "could not be assembled", `retryable: True` (a retry fails the same way). `e2e.sh` always exits 0. | Trace `fail voice_env_mismatch` → `next(startswith("E_"))` finds nothing → `E_NO_OUTPUT` | T |
| O4 | MED | `e2e.sh:99-101` + `ks_face.sh:36-85,147,153,162` | CPU work under the GPU lock: route ffmpeg + beard, master encode, MediaPipe pass, shots detect, crop, k2_prep_track, ks_chunk split/join. | long5 KS_T `master 0 landmarks 18 shots 92`, s0 `crop 1 engines 26` ≈ 115 s; full_m18 crop 10 → engines 18 | T |
| O5 | MED | `ks_shots.sh:51,13,56` | Each shot segment starts its own ks_fast (load 12.7–16.6 s) and re-queues for the lock. One failed segment means the whole video gets the source picture. | long5: 7 segments; s1–s8 have `wait_gpu` | T / K2 |
| O6 | MED (small) | `face_route.py:97-103` + `ks_e2e_prep.py:129` | Silence ranges come from the SOURCE diarization but mute the DUB wav. Where the EN phrase runs past the RU turn, the mouth closes while the audio continues. | long5 tracks 0/2/3/4/6 lose 0.19/0.08/0.14/0.35/0.17 s of dub speech; m03 / q03 / q05: 0 | T + Q |
| O7 | LOW-MED | `e2e.sh:114` vs `:117`, trap `:43` | `sep_overlap` (htdemucs) starts before the voice-env check. On an early ERROR it is not killed and keeps running on the GPU, unlocked. Only with `DUB_ALLOW_NC_BED=1`. | Code | T |
| O8 | LOW-MED | `ks_engines.sh:15-16` | Old TRT key dirs (≈ 17 GB each) are never removed. | `trt/cache/…_fc1c2add` = 8.7 + 8.6 GB | T |
| O9 | LOW | `ks_face.sh:157-160` | A failed full-run SyncNet check (`O=x`) raises no flag; with shift=auto it silently becomes 0. | gold_q02_prev_1614 log `raw offset … x` | T |
| O10 | LOW | `dabai_job.py:149,480`; `face_route.py:12` | The docs still say "auto = LatentSync for beards". | grep | T |
| O11 | LOW | `r16q/e2e_audio.sh:69`; `r16q/q_face.sh:8` | Q helpers (off the live path) still use `exec 8>` + flock and a hard-coded engine dir for driver 575.57.08. | grep | Q |
| O12 | LOW (plausible) | `ks_face.sh:79`; `ks_shots.py:95-112` | With KS_REF=0, if frame 0 of a shot shows only the OTHER face, the ref is the wrong person; ks_shots only guarantees that some face is present. Not reproduced. | Code | K2 |

### A2. KeySync stage and paste

The kit is copied by `ks_face.sh:23` (`keysync/*.py`, then `r15`, then `r16`; no name collisions), so the live tsmooth is `scripts/keysync/ks_tsmooth.py` with `fillnan`. These unused copies lack the fix: `scripts/r12/`, `reports/…/r12-keysync/`, `/workspace/dub/keysync/`. `ks_cheeksmooth.py` and `k_post.sh` are not wired into the chain.

| id | sev | file:line | bug | evidence | owner |
|---|---|---|---|---|---|
| P1 | MED | `r16/ks_seam.py:177-178` | ks_seam is all-or-nothing at ratio ≥ 2.0, and m03 sits on the threshold. | Same m03 inputs: gold ratio 1.97 → 0 boundaries; warm ratio 2.02 → 82 boundaries (164 of 1067 frames replaced). The eye-accepted m03 had 0. | K1 |
| P2 | MED (confirmed) | `ks_prep.py:13,15` | Odd crop x/y: ffmpeg crop on yuv420p rounds down, but the paste uses the unrounded box, so it lands 1 px right/down. ECC registers to the shifted crop and does not fix it. | gold_m03 box (233, 257): the ffmpeg crop equals the numpy slice at (232, 256) exactly (MAD 0.0 vs 3.09). Also full_old03 y=303 and q05 s1 y=87. Fix: `//2*2` or `exact=1`. | K1 / T |
| P3 | MED (plausible) | `r16/ks_alpha.py:86-91,99-104` | The whole extended segment is kept at full resolution in a dict, plus the U frames. | 1080p ≈ 6.2 MB/frame, so a 5-min shot with a fade ≈ 46 GB (4K ≈ 185 GB). Only the 45 s q03 has been run. | K2 |
| P4 | LOW-MED | `r16/ks_shots.sh:67-70`, `ks_dissolve.py:206-207` | Any alpha zone means the whole video gets one more crf16 generation, including the non-alpha shots. | q03_alpha: all 1125 frames re-encoded for 50 alpha frames | K2 |
| P5 | LOW (plausible) | `r16/ks_plain_par.py:110,120` | A mix more than 13 frames shorter than the master makes par always exit 1; the serial retry doubles the time and raises a false `ks_paste_par_retry`. | Code; current mixes are ±1 frame | T |
| P6 | LOW (dormant) | `r16/k_ecc_par.py:25` + `r15/ks_plain.py:147-151` | With shift ≠ 0, ECC M[i] comes from the wrong source frame. | Default shift is 0 | K1 |
| P7 | LOW (plausible) | `r15/ks_plain.py:156-158,170-174` | par == serial holds only while the NaN runs of the align params are shorter than about margin − 6 (≈ 24). `--absent hold` restarts per chunk. | Longest NaN run measured over 31 golden/full/shot runs = 13 (q05 s1). The a6ded57 test is representative only for the current set. | K1 |
| P8 | LOW | `r16/ks_shots.sh:33-35` | The dissolve check runs with `2>/dev/null`, so a crash looks like an ordinary `ks_dissolve_hard_switch`. | Code | K2 |
| P9 | LOW | `ks_face.sh:182` | `--ks-lm` = raw_lm while `--ks` = tsmooth → snap → guard → seam output, so landmarks are stale on repaired frames. | old03: guard 52 frames + seam 177 boundaries, all with raw landmarks | K1 |
| P10 | LOW | `r15/ks_plain.py:303` vs `r16/ks_shots.py:162-168` | Different output frame-count rules: single-shot = min(master, floor(mix·25)); shots = master N. | m03 1012 / 1013 frames; long5 video 300.16 s vs audio 300.10 s. Cosmetic. | T |
| P11 | MED if merged | K1 64488de `ks_plain.py` fix_ks / FLE | The `--tex-flow-ema` state restarts at every par chunk boundary (about 20 frames to converge at α 0.95), so par ≠ serial. | Bundle code (`git merge-tree` is clean; flags are off by default; "eye check pending") | K1 |

### A3. Audio: bit-identical, full length

Every final mux point copies the audio:
- `e2e.sh` final mux: `-c copy`, no `-shortest`; the audio comes from `final_mix_aac.m4a`.
- `ks_shots concat` and `ks_dissolve assemble`: `-c:a copy`.
- `ks_plain` / `ks_plain_par`: `-c:a copy` with `-t`. Their output is intermediate only.

The 12 outputs checked against `artifacts/r4a/r9_<id>/<id>/final_mix_aac.m4a`: gold m03 / m18 / old03 / q02, full m03 / m18 / old03, long5, q03_alpha, q05_e2e, par_m03, shots_q02.
- The AAC packet md5, the decoded PCM md5 and byte count, the duration and the packet count are all identical. Examples: m03 40.500 s / 1746 packets; long5 300.103 s / 12926 packets.
- The outbox copies are identical too.
- The reused mixes have the same md5 as their sources in `HANDOFF_K` / `HANDOFF_Q`.

Only the picture length differs: the video is 1 frame shorter (m03 / m18 / old03), and long5's video is 57 ms longer than its audio.

### A4. Probe bias (m18: the probe picks +40, the eye wants 0)

The old `-ss` bias is fixed (825ace2):
- the window is cut on whole frames, video frame k0 ↔ audio sample k0·spf;
- the probe and the full run use the same ref and gate, through one process.

The probe agrees with SyncNet on the full render:

| run | o+1 | o0 | o−1 | o−2 | picked | parabola vertex |
|---|---|---|---|---|---|---|
| gold_m18 probe, 10 s | 2.55 | 6.37 | **8.18** | 5.28 | −1 (+40 ms) | −0.88 fr (≈ +35 ms) |
| gold_m18 full @+40 vs original dub | 4.99 | **6.93** | 5.27 | 2.27 | 0 | −0.04 |
| d0_m18 full @0 | 1.87 | 4.76 | **6.92** | 5.44 | −1 | −1.09 |
| gold_m03 probe | 1.71 | 5.04 | **8.43** | 6.24 | −1 | −1.1 |
| gold_old03 probe | 5.41 | **7.28** | 5.05 | 2.15 | 0 | +0.04 |

Every m18 probe across commits gives offset −1 with conf 7.4–8.4. So the probe is not biased: SyncNet really wants about +35–44 ms on m18, and this is SyncNet against perception. The post chain adds no lag: tsmooth is symmetric, and the paste runs at shift 0.

Lever: the margin over offset 0, conf − conf_at0. It is m18 1.81, m03 3.39, old03 0. The rule "delay 0 unless margin > 2" gives 0 / +40 / 0, which matches the eye on all three. It is tested on 3 clips only.

Other probe notes:
- Delay steps are 40 ms; a sub-frame `adelay` is possible.
- In multi-shot mode the first confident probe sets the delay for all segments, including the other person in a shot / reverse-shot (`ks_shots.sh:57-59`).
- A ks_fast crash during the probe kills the face stage (PARTIAL); it does not fall back to delay 0.

### A5. Eval / metrics harness (Q's own area)

| id | sev | file:line | bug | evidence | owner |
|---|---|---|---|---|---|
| M1 | MED | int `scripts/quality/sync_perframe.py:145`, `r16q/eval_final.sh`, `r16q_detect.py` | The Q fixes are not in int. Missing: 15fe38a (route v2 multi-lipsync faces: int scores only `lipsync_box_center_xy`, so q02 Vera + man are scored as one), 295d2bf (black / explode relative to the source, slide without the chin) and the r16q_spring / score_pair tools. `gold_all.sh` in int uses the old harness. | `git merge-base --is-ancestor 15fe38a HEAD` → no (same for 295d2bf, 47077c4, 802bb1a, 454e0cd, 3b567cf) | T merges experiment/r16-qa 3b567cf (bundle `/workspace/dub/r16q/r16-qa_latest.bundle`) |
| M2 | LOW | `r16/k_gate.sh:4-5` | Plain `flock` (not `-o`): a child can inherit the lock. Hard-coded `DUB_ALLOW_NC=1`, no n_sec check, no route (largest box). Off the live path (K helpers only). | Code | K1 |
| M3 | note | golden EVAL `noise_pm: null` | Not a bug: the golden runs pass `--noise 0`. The noise floors are in `reports/experiments/r16-qa/TRUST_v2.md`: m03 ±0.051, m18 ±0.009, old03 ±0.033, q02 ±0.075. | — | — |

### A6. Re-check of `briefs/R16_AUDIT_FINDINGS_EN.md`

**Paste stage**

| # | Status | Commit | Evidence |
|---|---|---|---|
| 1 ECC M not sliced | FIXED | 6de0c8e | `ks_plain_par.py:57-59` slices by the KeySync index; `ks_plain.py:148` |
| 2 silent end fade | FIXED | 6de0c8e | `ks_plain.py:309-318`: the fade uses global totals, with a WARNING and an NFADE count; par surfaces the warnings |
| 3 `--lm` sliced only if len == n | FIXED | 6de0c8e | `ks_plain_par.py:52-55` |
| 4 bridge across short cutaways | PARTLY | 8f7a5f2 / dbf8276 / 688e28a | Cuts split shots now. Still bridged: cutaways < 0.4 s, undetected cuts, and `KS_SHOTS=off` (`ks_e2e_prep.py:42-55`) |
| 5 cavity / sharp off in par | FIXED | 6de0c8e | `ks_plain.py:205-211`, `ks_plain_par.py:81-100` |
| 6 ks_seam not called / global phase | FIXED | a8bb5f4 | `ks_face.sh:175-177` `--chunks`. New issue: P1 |
| 7 `-c:a aac` / `-shortest` | FIXED | 6de0c8e | copy + `-t`; the final mux is verified bit-identical (A3) |
| 8 stale `--ks-lm`; NaN rotation centre | PARTLY | 6de0c8e | The rotation centre is fixed; the stale landmarks remain (P9) |
| 9 memory | NOT FIXED (mitigated) | — | `ks_plain.py:78` full layer at box size (m18 ≈ 11.7 GB / 5 min); tsmooth / snap / guard / seam ≈ 20 GB / 5 min single shot; plus P3 |
| 10 NaN crash in ks_tsmooth | FIXED | d67ceba | `ks_tsmooth.py:17-26,34,42,48`; the downstream steps handle NaN |
| 11 odd crop 1 px | NOT FIXED (confirmed) | — | P2 |
| 12 smoothing across cuts | FIXED for detected cuts | 8f7a5f2+ | per-shot ks_face (`ks_face.sh:41-43`); same caveat as #4 |
| 13 low items | Partly | — | cheeksmooth: not wired; `--others-grow 0`: default 1.3 is unaffected; all chunks n = 0: FIXED (`ks_plain_par.py:85-92`); SyncNet fail → shift 0: the probe is flagged, the post check is not (O9) |
| Merge notes | Mostly honoured | — | ks_plain_k1_tracked dropped; the int ks_plain has the tracked box etc. "int ⊇ everything" no longer holds: K1 64488de is unmerged (P11) |

**Orchestration**

| # | Status | Commit | Evidence |
|---|---|---|---|
| 1 lock hand-off / fd 8 / nesting | PARTLY | 825ace2 | `gpulock.sh:12-26` holder + `GPU_RDY` token; no `exec 8` on the live path. Open: CPU work under the lock (O4) |
| 2 beard routing / Dockerfile / codeformer | FIXED | 825ace2, 286b0ec, 66d6041 | `Dockerfile:86` auto; a missing CF → clean face + flag (`e2e.sh:211-214`) |
| 3 probe alignment / clamp / ref + gate | FIXED | 825ace2, 091be92 | `ks_face.sh:109-138`; A4 |
| 4 paste shift auto | FIXED | 825ace2 | `ks_face.sh:156` |
| 5 multi-shot identity / ref / blur | PARTLY | 8f7a5f2, 688e28a, 9187ef1 / d67ceba | Per-shot runs. Within a shot tracks are still linked by centroid; route faces on one track are merged (`ks_e2e_prep.py:122-125`) |
| 6 source silence on the dub timeline | NOT FIXED | — | O6 |
| 7 25 fps output | NOT FIXED | — | `e2e.sh:191`; flag `output_format_differs_from_source` only. Needs a TZ decision |
| 8 TRT key / build in the first job | PARTLY | e0ed80a | The key includes the LoRA / capture hash. Engines are still built in the first job; old dirs pile up (O8) |
| 9 split/join under the lock; two ks_fast | PARTLY | 091be92 | One `--follow` process. Split/join still under the lock; multi-shot loads once per segment (O5) |
| 10 refpick on absent frames | FIXED (and off) | 8f7a5f2, 2070d8b | `k2_refpick.py:12,42,55` |
| 11 job-id collisions | PARTLY | 18ecc18 | The API takes a per-id flock (`E_JOB_ID_BUSY`); direct `e2e.sh` runs still `rm -rf` the global per-ID dirs (`e2e.sh:108,113`) |
| 12 exit 3 → SUCCESS | FIXED | 825ace2 / c2ebf5a | `e2e.sh:193-194` → PARTIAL, refund |
| 13 low items | Partly | 67c8527, 8f7a5f2 | KS_REF_IDX leak: fixed. Hard-coded paths: partly (`q_face.sh:8`). lmk cache: not cleaned (13 MB, harmless). Disk check: only the API `/health`, no per-job check |
| Missing merge: k_post.sh (shift 0 + post-paste gate) | PARTLY | 825ace2 | Shift 0 is the default; there is no post-paste SyncNet gate |
| Missing merge: K2 refpick position / qseg / qsplice | PARTLY | 8f7a5f2 | The position term is merged; qseg / qsplice are replaced by ks_shots |

**Metrics and training**

| # | Status | Commit | Evidence |
|---|---|---|---|
| 1 multi-face per-frame D / global conf | PARTLY | 726427a (int); 15fe38a (Q only) | Speaker tracks are fixed in int; multi-lipsync routes only in Q's branch (M1) |
| 2 k_lag anchors = upper face | NOT FIXED in int | — | `k_lag_final.py:8` AN = 33/263/133/362/168 (6e0c6bc). Q's lower-face `r16q_klag.py` is in r16-qa, not validated |
| 3 stale det.npz | FIXED | 726427a | `r16q_detect.py:43` fingerprint (size + sha1 of the head / tail 4 MB) |
| 4 ks_lcm_jit cheekjaw band | N/A for e2e | — | Not in int (L's branch); not re-checked |
| 5–6 training leakage / λ comparability / meta | not re-checked | — | L's branch r16-train; nothing of it is in the e2e |
| 7 k2_lipmetrics abs() before the FFT | NOT FIXED in int | — | `k2_lipmetrics.py:77` `abs(...)` (2c404ce). Q's signed `r16q_k2lip.py` is in r16-qa, not validated |
| 8 k_gate drops seconds / no n_sec check | NOT FIXED | — | `k_gate.sh` (also M2); off the live path |
| 9 L's sync ≠ K's speech@0 | open (by design) | — | Compare only with `eval_final.sh` |
| 10 l_boundary phase argmax | not re-checked | — | L's file |
| 11 detectors (black / explode / slide / thresholds) | FIXED in Q, NOT in int | 295d2bf | The false-positive floor on golden is measured (TRUST_v2: old03 black 15, slide 111, texture 736, flicker 254). Use counts relative to golden |
| 12 beard threshold margin / low light | PARTLY, moot | 653ccf5 | `r16q_beard.py:69` refL < 12 → None (a low-light guard). The 2.0 threshold is unchanged, but beards now go to KeySync either way (only the flag differs) |
| 13 SyncNet rewards a dark mouth; short-window noise | still true | — | Use full-length speech@0 only; see A4 |

---

## Part B: TZ progress

**Source.** The original TZ is `DabAI_описание_этапы_бэклог.docx`, cited in `briefs/tz_status.md`. It was not found on r16-qt or the laptop, so its stand-ins are:
- `/workspace/dub/briefs/tz_status.md` (28.10): item IDs;
- `reports/experiments/r13-spec/TZ_REPORT_RU.md` (04.11): the last full status.

Newer evidence:
- `r15-tz/REPORT.md` (05.11) and `r16-int/REPORT.md` (07.11);
- `briefs/R16_CHECKLIST.md` (06.11) and `briefs/R16_АУДИТ_08-11.md`;
- the `work-r16b` git log and the 08.11 job statuses (`r16t/golden/jobs/*/status.json`, `outbox/r16_t/`).

Blocks 1–4 (backend, web, payments) are outside the ML zone.

| TZ item | Status | Evidence | What remains | Owner | Effort |
|---|---|---|---|---|---|
| **0 Pilot** | | | | | |
| 0.1 Blackwell sm_120 | done | CUDA 12.8 / torch 2.7.1; KeySync TRT on driver 575 / 580 | – | – | – |
| 0.2 LatentSync fork | done, superseded | LS frozen; KeySync is prod (06.11) | Written deviation LS → KeySync | Irina | 1 h |
| 0.3 FlashAttention | done | SDPA | – | – | – |
| 0.4 VRAM | done for LS, partial for KeySync | VRAM.md 04.11: audio 26–30 GB, face 13–14 GB. Nothing for the KeySync stage | Per-stage VRAM table for KeySync from the logs | T | 2 h |
| 0.5 ≤ 300 GPU-s / video-min on 1×5090 | done for videos ≥ 1 min; short clips over | whole job 08.11: long5 1263 / 5.0 = **253**; old03 443 / 1.51 = 294; m18 (20 s) 657 (fixed costs). Wall time long5 571 s/vmin | Agree "steady-state on ≥ 1 min" with the customer; re-measure after 091be92 | T + Irina | 0.5 d |
| 0.6 / 0.7 CodeFormer | dropped de facto | No CF weights anywhere; clean profile | "Clean only" in writing | Irina | 0.5 h |
| 0.8 SyncNet contract threshold | proposal only | SYNCNET_THRESHOLD.md: ≥ min(5.0, source − 0.5). Golden e2e: m03 8.29, m18 7.77, old03 7.84, q02 0.68 | Decision; check low-source-sync clips (q02-type) against the rule | Irina, Q | 1 h + decision |
| 0.9 Qwen 14B | done | Q5_K_M | Paperwork | Irina | – |
| 0.10 Cold start < 60 s | not started, worse | KeySync: TRT 639 s + first job +350 s ≈ 17 min once; then 30 s | Pre-built TRT + inductor cache on the volume per GPU / driver | T + infra | 1 d |
| 0.11 S value | done | – | – | – | – |
| **5 Preparation** | | | | | |
| 5.1 Working image | partial | recipe 0154dd6 / a4a88dc; voice pin 67c8527; **`docker build` never run** | Build on a Docker VM → fetch-models → smoke m03; size + cold start | T | 1 d (VM) |
| 5.2 VAD 45–90 s | done (option) | r13 | – | – | – |
| 5.3 Audio extraction | done | – | – | – | – |
| 5.4 Face detection | done, with deviation | MediaPipe instead of RetinaFace (licence) | Written deviation | Irina | – |
| 5.5 Pre-check (no face / head turned) | partial | Verified on LS (r13); **no pose gate on the KeySync path**; no face → voice-only dub + refund flag instead of an abort | Wire or prove the pose gate before ks_face; test q05 / q06; written deviation | T | 0.5 d |
| 5.6 Source aspect / resolution / fps | partial, regressed | KeySync output is always 25 fps (flag on every 08.11 job); resolution is kept | Decision: interpolate back or 25 fps + warning; then implement | Irina → T / K1 | 0.5–1 d |
| **6 ASR / translation** | | | | | |
| 6.1 Whisper large-v3 | done | – | – | – | – |
| 6.2 Hallucination filter | done | r13 | – | – | – |
| 6.3 Diarization / speakers ↔ faces | done | route.json 84031e1; `many_speakers_one_voice` fired on long5 | Verify the long5 flag | Q | 1 h |
| 6.4 Qwen translation | done | duration-aware, NE guard | – | – | – |
| 6.5 External-API languages | partial, blocked | 13/22 local; donor mock only | OpenAI key + decision; a real call | Irina → T | 0.5 d |
| **7 Voice** | | | | | |
| 7.1 Voice sample | done | – | – | – | – |
| 7.2 Chatterbox | done | 5de7a54 pinned; e2e refuses a wrong env (091be92) | – | – | – |
| 7.3 Duration alignment | done (accepted 30.10, frozen) | TEMPO.md; e2e audio == mix bit for bit (A3) | old03 pace-fix A/B: listen, or close as won't-fix | Irina | 15 min |
| 7.4 ChatterboxVC | partial (mock) | sim 0.55 → 0.87 (de) | Real donor calls | after the key | 0.5 d |
| 7.5 OpenAI donor + rate limiter | partial (mock) | donor_test PASS | Real key | after the key | 2 h |
| **8 Lip-sync and picture** | | | | | |
| 8.1 Engine (TZ: LatentSync 1.6) | partial (replaced by KeySync) | e2e default KeySync fast2 + TRT. Golden e2e vs accepted face-box PSNR: m03 35.6, m18 37.6, old03 33.2, **q02 27.3 dB** | Irina's eye on the golden worst clips; investigate q02 | T + Irina | 0.5 d |
| 8.2 Only the active speaker animated | done | route + others-mask (m03 child untouched) | – | – | – |
| 8.3 CodeFormer | dropped | clean | Written | Irina | – |
| 8.4 Compositing | partial | Fixed: par == serial, shift 0, seam, shots, alpha-paste, tracked crop. Open: edge / colour / softness eye check, hand over mouth (q02 34.8–38.6 s), q07 no face, nose tremor; bugs P1 / P2 | Eye check q03_alpha / q05; occlusion mask; q07 fallback | K1, K2, Irina | 2–3 d |
| 8.5 Smooth joins | partial | Audio done; m18 end fixed; seam; lag metric broken; pause a/b waits; black-hole bank (a) not in e2e | Irina: pause a/b, black-hole choice; then wire it in | Irina, K2, T | 1 d after the decisions |
| **9 Subtitles / final** | | | | | |
| 9.1–9.3 ASS, retiming, styles, karaoke | done on LS; **not run on KeySync** | r13 p90 lag 0.45 s; all 08.11 jobs have subs off | One KeySync e2e with subs + watermark; check the .ass timing at 25 fps; zh / ar / hi glyphs by eye | T + Irina | 2–3 h |
| 9.4 Watermark + source format | partial | watermark done; fps see 5.6 | – | T | – |
| **10 Infra** | | | | | |
| 10.1 RunPod serverless | partial | handler + job_api; no full RunPod run | after the image | infra | – |
| 10.2 Weights on a volume | done (recipe) | models.lock 55 files 23.2 GB + KeySync kit | Measure the size with KeySync | T | 1 h |
| 10.3 Cold start | not started | see 0.10 | | | |
| 10.4–10.5 Monitoring / prod | not started (not ML) | – | | infra | |
| **11 Fault tolerance** | | | | | |
| 11.1 try/except around the generator | done on LS; partial on KeySync | fault suite was LS only; KeySync face fail → visible PARTIAL; par → serial fallback | Re-run fault_suite with face=keysync; also O1 / O3 | T | 0.5 d |
| 11.2 Frame substitution | partial | ks_guard hold / in-between | Count the held frames in the status | T | 2 h |
| 11.3 2 % → PARTIAL | partial | rule verified on LS only | Feed the guard counts into collect.py | T | with 11.2 |
| 11.4 Refund | backend | `refund_suggested` flag | Note O1: degraded sync gives no refund | backend / T | – |
| **12 Acceptance** | | | | | |
| 12.1 SyncNet tool | done (caveats) | eval_final speech@0, noise floors in TRUST_v2 | Merge the Q fixes into int (M1) | Q → T | 1 h |
| 12.2 CER by language | done | 22 languages | – | – | – |
| 12.3 E2E real / long / 4K | partial | 08.11 SUCCESS: golden ×4, full ×3, **long5 5 min** (RSS 26 GB, 2854 s), q03_alpha, q05; 4K vertical on KeySync 07.11. No 10 min; the edge suite was LS only | 10-min run + the edge suite on KeySync; long 4K RAM (A2 #9, P3) | T | 1 d (~3 h GPU) |
| 12.4 Load: 10 parallel | partial | test_load 9/9 (LS); nothing on KeySync | Queue throughput on KeySync | T / infra | 0.5 d |
| 12.5 Legal | not ours | – | – | – | – |

**Tails closable in 1–2 days.** These are the same as in the RU summary:
1. Eye check of the golden clips + the q02 investigation.
2. Subs + watermark on KeySync.
3. Fault suite + the 2 % rule on KeySync (with O1 / O3).
4. Pose gate before ks_face.
5. 10-min run + the edge suite on KeySync.
6. Docker build + smoke.
7. Irina's decisions: fps, pause a/b, black hole, clean only, SyncNet threshold, pace-fix, OpenAI key.
8. Small items: KeySync VRAM table, long5 speaker flag, merging Q's fixes.

**Stale or contradictory statuses**
- **CodeFormer.**
  - `tz_status` says CF is back; r15-tz says "Irina decides"; the 08.11 audit says no weights exist.
  - The 08.11 statuses report `profile: "current"` while running without CF, so the label is misleading.
- **Speed.** `R16_CHECKLIST #21` says ✅ "~140 face", but that excludes landmarks and audio. The whole job measures 253–294 GPU-s/vmin on long videos and 657 on a 20 s clip.
- **fps.** The r13 ✅ "source fps" is LS-only; KeySync outputs 25 fps.
- **Long-video RAM.**
  - Memory note: 9 GB/vmin ≤ 8–9 min (LS).
  - r16-int: 12 GB/vmin.
  - long5 after ks_chunk: 26 GB peak.
  - 4K paste estimate: ~100 GB per 5 min.
- **4K vertical.** `R16_CHECKLIST #23` still says ❌; it passed on KeySync on 07.11.
- **R16_CHECKLIST (06.11).** Never updated. The day-2 work (shots, alpha-paste, ref0, beards → KeySync) is in neither the checklist nor the audit.
- **Beards.** Audit B6 / B7 (beards → LS + CF, silent fail) are stale: 66d6041, 825ace2, 286b0ec.
- **Black hole.** K2's bank (a) brings m03 from 16 % to 1 %, but none of it is in e2e (ks_face uses only `--tex 2`).
- **Docs.** README.md and CHANGES.md are r4/r5 era with no KeySync. TZ_REPORT_RU.md (04.11) is the last TZ-level summary, from the LS era.
- **No face (5.5).** The TZ says abort + refund; the code delivers a voice-only dub + `refund_suggested`. This is not yet a written deviation.

Scratch from the read-only checks is on r16-qt, in `/tmp/qaudit_orch/` and `/tmp/qaudit_paste/` (including a temp clone with the K1 64488de bundle). Nothing was written to the main repos.
