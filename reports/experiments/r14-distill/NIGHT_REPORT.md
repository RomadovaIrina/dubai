# r14-d night report: 4-step LoRA student, rounds 2 / 3 / 3b + agent F's smart reference

Server r14-d, branch experiment/r14-distill. Held out: old03 / m03 / m18 (never trained on).
Every E2E arm uses the q2 + F3 composite. Only the generator changes.
Clips are in /workspace/dub/outbox/r14_d/: `<video>_mouth_q2_vs_<arm>[_half_speed].mp4`, `<video>_full_q2_vs_<arm>.mp4`, and the decode check `val_mouths_*.jpg`.
Checkpoints: ckpt/r2_s4, ckpt/r3_s4, ckpt/r3b_s4 (every 500 steps, raw + EMA).

## Arms

| arm | generator | notes |
|---|---|---|
| q2 | LatentSync 1.6 teacher, production q2 + F3 | reference |
| s4 | r2 student, final step 3000 EMA | temp_w 1.0 over the whole crop |
| s4p1500 | r2 student @ step 1500 | early stop |
| s4p1500c8 | s4p1500 + LS_STRIDE 8 / continue overlap | |
| s4p1500c8rf | s4p1500c8 + `--ref-mode fixed` + LS_REF_SMART=1 | agent F's 0008 smart 16-frame source reference |
| r3 | r3 student @ step 500 | temp 0.25 off the lips, one-sided opening hinge open_w 2, sync_w 0.2 |
| r3c8 / r3c8rf | r3 + c8 / + smart reference | |
| r3b | r3b student @ step 1500 | symmetric \|open_s − open_t\|, open_w 0.5, 1500 steps |

## What happened tonight

1. **The r2 damping is real and grows with training.**
   - Opening vs q2 on old03: final ×0.80, step 1500 ×0.93.
   - Proxy val opening by step: 500 1.19, 1000 1.10, 1500 1.04, 2000 1.03, 2500 0.94, 3000 0.96.
   - Early stopping at 1500 alone takes old03 sync from 6.14 to 6.66.
2. **Checkpoint pick rule.** Val opening ratio ≥ open_min, then the lowest MSE.
   - Calibration: the dark-cavity proxy reads about 10 % high vs MediaPipe E2E. r2@1500 has proxy 1.04 and E2E ×0.93.
   - The coordinator therefore set open_min 1.03. At 0.95 the rule would pick r2@3000, the damped one.
   - On r3 the rule picked step 500: opening 1.43, MSE .0092.
3. **The r3 one-sided hinge was gamed.**
   - Val opening stayed 1.27–1.43 × the teacher's for the whole run. MSE climbed from .0092 to .0272 (untrained .0126).
   - The decode check (val_mouths jpg) shows r3@1500 and r3@3000 drawing a darker lip crease. That fools the dark-pixel proxy without opening the mouth more.
   - r3@500 looked sane, so it was evaluated.
   - E2E: the mouth sits more open (mean ×1.05–1.35, near-closed voiced share 7–35 % vs 16–58 % for q2) but moves less (std ×0.88, speed ×0.76 on old03).
   - Sync was mixed: old03 6.91 beats r2@1500, m03 6.20 is worse than every other arm, m18 7.15.
4. **r3b (symmetric term) behaves well on val but is still damped E2E.**
   - Val MSE .0060, jitter .0300 (lowest of all runs), sync sim .950.
   - Val opening drifts 0.96 → 0.94 → 0.89 over 500 → 1500 steps.
   - E2E old03 @1500: opening ×0.91, sync 6.59, 1 pop. That is no better than r2@1500.
   - The symmetric term at 0.5 does not stop the drift toward a closed mouth. It only slows it.
5. **c8 (stride 8 + continue overlap) helps sync a little, at about +50 GPU s / video-min.**
   - old03: 6.66 → 6.77 on r2@1500, 6.91 → 7.12 on r3.
   - m03 / m18 are mixed: r2@1500c8 m03 7.31 vs s4 7.47; m18 7.56 vs 7.46.
6. **The smart reference (agent F's 0008) is the best student arm, but its big win is on old03 only.**
   - old03: sync **7.739 vs 7.487 for q2 + F3**. Flicker 4 pops vs 9. Opening ×1.12 with speed ×1.04 (normal motion, not frozen). 276 GPU s / video-min vs 401.
   - m03: 7.686 (s4p1500c8 7.307, q2 8.162). Opening ×0.75 mean, but p99 ×0.94 and speed ×0.99, so it moves normally and rests more closed (silent mean 0.041 vs 0.075). Pops 12, same as q2.
   - m18: 7.409 (s4p1500c8 7.557, q2 8.388). Opening ×1.19, pops 8, same as q2.
   - Δ sync from rf on the r2@1500 c8 student: old03 +0.97, m03 +0.38, m18 −0.15. That matches F's finding on the teacher, where the smart reference helped mainly one video.
   - The same reference on r3 also lifts old03 sync (7.69), but r3's opening goes to ×1.37 with 10 pops, so r3c8rf is not a candidate.
   - Bottom line: the student still trails q2 + F3 by 0.5–1.0 sync on m03 / m18. Much of the old03 gap was the reference clip.

## Recommendation for the next session

- **Best student candidate: r2@1500 + c8 + smart reference (s4p1500c8rf).** Check it by eye against q2 + F3 on all 3 held-out videos (clips in the outbox), then on 2–3 fresh videos.
  - It costs about 280–340 vs 350–400 GPU s / video-min. Sync is above q2 on old03 but 0.5–1.0 below on m03 / m18, so it is not a drop-in default yet.
  - The smart reference is video-dependent. Make it a per-video choice (it never hurt much: m18 −0.15), or improve its frame picker.
- **The remaining student gap is in the generator.** At this point a 6-step student or more / cleaner distillation data is the lever, not loss terms on the opening.
- **Do not chase the opening with a loss term on the dark-pixel proxy.** The one-sided hinge is gameable. The symmetric one is too weak to stop the drift.
  - Better: early-stop on the val opening ratio (rule open_min 1.03, already in ckpt_val.py), or
  - a MediaPipe-landmark opening target computed offline on the teacher decodes. That cannot be gamed by darkening.
- **sync_w:** raising it to 0.2 (r3, r3b) gave no measurable val sync gain over r2 (stable-SyncNet sim .944–.951 in every run). Not worth raising further.
- **6-step student:** worth one run next. The 4-step student with the best reference still trails q2 by 0.5–1.0 on m03 / m18. That costs about +50 % generator time, which is still under the teacher's.
- **TREPA / temporal:** temp off the lips (r3, r3b) gave the lowest jitter (.0300). Keep it in any future round. TREPA is not needed for the current gap.
- If a new training round is wanted: r2 recipe + temp off the lips, at most 1500 steps, pick by open_min 1.03, plus an E2E with the smart reference.

## GPU time

Roughly 17 GPU-hours on 1 GPU this round:
- r2 training + teacher targets ≈ 9.5 h
- r3 3000 steps ≈ 2.3 h
- r3b 1500 steps ≈ 1.2 h
- night evals ≈ 4 h (13 arm-videos)

## Plain-Russian summary (для Ирины)

- Ученик (быстрая 4-шаговая модель) после долгого обучения «прикрывал» рот. Если остановить обучение раньше, на шаге 1500, это почти уходит.
- Попытка «заставить» рот открываться через штраф не сработала. Модель научилась рисовать тёмную складку на губах вместо настоящего открытия. Мягкий вариант штрафа (r3b) тоже ничего не дал.
- Лучший вариант ночи — ученик с шага 1500 + шаг 8 + «умный» опорный кадр от агента F. На old03 он синхронизирует губы даже лучше продакшена (7,74 против 7,49), меньше мигает и тратит примерно на 30 % меньше GPU.
- Но на m03 и m18 он всё ещё немного отстаёт от продакшена (7,69 против 8,16 и 7,41 против 8,39). Умный опорный кадр сильно помогает только на одном видео.
- Совет: посмотреть клипы глазами. Следующий шаг — ученик на 6 шагов, а не новые штрафы.

## Tables

### Training val (EMA, 24 held-out windows; teacher = 20 steps + CFG)

| run | step | latent MSE | mouth pix L1 | jitter | stable-SyncNet sim | opening / teacher |
|---|---|---|---|---|---|---|
| r2_s4 | 500 | 0.0067 | 0.0283 | 0.0314 | 0.947 | 1.190 |
| r2_s4 | 1000 | 0.0079 | 0.0281 | 0.0309 | 0.950 | 1.097 |
| r2_s4 | 1500 | 0.0061 | 0.0142 | 0.0316 | 0.944 | 1.038 |
| r2_s4 | 2000 | 0.0069 | 0.0195 | 0.0307 | 0.945 | 1.031 |
| r2_s4 | 2500 | 0.0062 | 0.0164 | 0.0309 | 0.946 | 0.945 |
| r2_s4 | 3000 | 0.0059 | 0.0158 | 0.0311 | 0.946 | 0.957 |
| r3_s4 | 500 | 0.0092 | 0.0420 | 0.0318 | 0.947 | 1.426 |
| r3_s4 | 1000 | 0.0141 | 0.0385 | 0.0317 | 0.950 | 1.345 |
| r3_s4 | 1500 | 0.0169 | 0.0399 | 0.0319 | 0.945 | 1.379 |
| r3_s4 | 2000 | 0.0214 | 0.0411 | 0.0325 | 0.945 | 1.386 |
| r3_s4 | 2500 | 0.0248 | 0.0434 | 0.0315 | 0.944 | 1.285 |
| r3_s4 | 3000 | 0.0272 | 0.0433 | 0.0314 | 0.945 | 1.273 |
| r3b_s4 | 500 | 0.0057 | – | 0.0307 | 0.947 | 0.959 |
| r3b_s4 | 1000 | 0.0062 | – | 0.0302 | 0.951 | 0.937 |
| r3b_s4 | 1500 | 0.0060 | – | 0.0300 | 0.950 | 0.888 |

### End-to-end on the held-out videos (dabai_job, q2 + F3 composite in every arm)

| video | arm | sync_speech | flicker mean | flicker pops (src p99) | mouth opening vs q2 | near-closed voiced | GPU s / video-min |
|---|---|---|---|---|---|---|---|
| old03 | q2 | 7.487 | 0.0062 | 9 | 1 | 0.18 | 401 |
| old03 | s4 | 6.144 | 0.0056 | 0 | 0.800 | 0.31 | 233 |
| old03 | s4p1500 | 6.657 | 0.0056 | 1 | 0.928 | 0.21 | 241 |
| old03 | s4p1500c8 | 6.767 | 0.0058 | 6 | 0.908 | 0.23 | 292 |
| old03 | s4p1500c8rf | 7.739 | 0.0062 | 4 | 1.117 | 0.13 | 276 |
| old03 | r3 | 6.912 | 0.0059 | 1 | 1.168 | 0.08 | 243 |
| old03 | r3c8 | 7.116 | 0.0061 | 4 | 1.165 | 0.07 | 293 |
| old03 | r3c8rf | 7.694 | 0.0068 | 10 | 1.373 | 0.03 | 280 |
| old03 | r3b | 6.588 | 0.0055 | 1 | 0.914 | 0.22 | 244 |
| m03 | q2 | 8.162 | 0.0098 | 12 | 1 | 0.16 | 352 |
| m03 | s4 | 7.473 | 0.0094 | 12 | 0.908 | 0.19 | 258 |
| m03 | s4p1500c8 | 7.307 | 0.0096 | 11 | 0.973 | 0.15 | 314 |
| m03 | s4p1500c8rf | 7.686 | 0.0096 | 12 | 0.748 | 0.28 | 299 |
| m03 | r3 | 6.204 | 0.0095 | 13 | 1.046 | 0.07 | 273 |
| m03 | r3c8 | 6.529 | 0.0097 | 15 | 1.072 | 0.07 | 317 |
| m18 | q2 | 8.388 | 0.0152 | 8 | 1 | 0.58 | 374 |
| m18 | s4 | 7.462 | 0.0150 | 8 | 0.815 | 0.65 | 270 |
| m18 | s4p1500c8 | 7.557 | 0.0153 | 8 | 0.975 | 0.56 | 338 |
| m18 | s4p1500c8rf | 7.409 | 0.0155 | 8 | 1.193 | 0.49 | 315 |
| m18 | r3 | 7.148 | 0.0153 | 9 | 1.349 | 0.35 | 261 |
| m18 | r3c8 | 7.157 | 0.0155 | 9 | 1.374 | 0.35 | 323 |

## Night notes (skips, calibration, decisions)

```
20:56:51 s4p1500 old03 done (== sync_speech s4p1500_old03: 6.657 over 59 s; global offset 0); its m03 / m18 skipped
21:05:57 r2 val opening: "step_00500_ema.pt" 1.189792103576604 "step_01000_ema.pt" 1.0973179469573726 "step_01500_ema.pt" 1.0379851438223424 "step_02000_ema.pt" 1.0306127125281659 "step_02500_ema.pt" 0.9447358202010758 "step_03000_ema.pt" 0.9569286109228318
21:08:09 SMOKE_OK
21:08:09 smart-ref patch applied
21:09:46 open_min set to 1.03 (coordinator): on r2 it picks step 1500 (0.95 would pick 3000)
23:35:36 night re-plan (r3 overshot); r3 eval checkpoint step_00500_ema.pt: opening ratio 1.426 >= 1.03, lowest MSE (0.0092) of the 6 that wide
23:35:58 decode check (outbox val_mouths_*.jpg, columns real | teacher | r2@1500 | r3@500 | r3@1500 | r3@3000): r3@500 sane; r3@1500/3000 darker lip crease = the hinge gaming the dark-cavity proxy
23:58:17 r3 old03 sync 6.912 (r2@1500 6.657), visual ok: yes -> r3 extended: 1
03:38:32 r3b trained: r3b_s4 step 1500/1500 (70 min): loss 0.0306 | last val @ 1500: mse 0.0060, sync 0.950 (teacher 0.927), jitter 0.0300 (teacher 0.0338), mouth opening / teacher 0.888 (untrained 0.869)
04:31:09 extra: s4p1500c8rf on m03 m18 (it beat q2+F3 on old03, checking it is not an old03 fluke)
```
