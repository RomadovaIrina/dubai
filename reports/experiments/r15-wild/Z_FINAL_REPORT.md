# DabAI r15, agent W: the wild track, the "worms" task and the generator-side audit (final hand-off, 05.11.2026)

## Коротко для Ирины
- **«Червяки» и «взрывы» рта рождаются в самом генераторе (LatentSync), а не при наложении.** Обрезка, наклейка, SR и починка слоя их не создают. Наложение даже вдвое их гасит.
- **Главный рычаг — «звуковой усилитель» (CFG, сейчас 2.0).** Чем он слабее, тем меньше мигания и взрывов, но тем хуже синхрон губ.
  - При 1.5 взрывов на m03 вдвое меньше (с замком цвета: сильных 18 → 4, мигание на уровне исходника), а SyncNet −0.5…−0.64.
  - При 1.25 сильных взрывов 22 → 4, а синхрон −1.1…−1.25.
- **Все хитрые способы ослабить усилитель только «в лишнем» не дали бесплатного выигрыша.** Пробовали обрезку по шагам, «rescale», сглаживание между кадрами — та же цена: меньше взрывов = хуже SyncNet.
- **Следующий шаг — глаз.** Видно ли на m03 панелях «1.5 + замок» ухудшение синхрона? Если нет — это готовый кандидат в прод, без доплаты по GPU. Если видно — нужен ученик с штрафом за мигание или раздельное усиление формы рта и текстуры (план ниже).
- **Аудит кода генератора.** Явные баги исправлены по умолчанию: кадровая карта, радужка в точках CodeFormer, провальные оценки движения головы. Остальное — флаги, ждут глаза. Из флагов CodeFormer-выравнивание даёт −12 % мигания без потери синхрона.

## Где что
- Code: branch `experiment/r15-wild-z`, head `__HEAD__` (r15-z: /workspace/dub/dubai). It contains:
  - `experiment/r15-wild` c1740ce, already pushed;
  - F's `experiment/r15-repair` 54b8509 (merge de9e8e7) plus F's explode.py from f74f9b4.
- Bundles in `bundles/`:
  - `r15-wild-z.full.bundle`: the whole branch, self-contained;
  - `r15-wild-z.incr.bundle`: `c1740ce..experiment/r15-wild-z`; it needs c1740ce + 54b8509 + f74f9b4, all on GitHub.
- Evidence:
  - `configs/`: per-arm env.txt + timing.json, eval jsonl (lip_flicker, flash, explode), SyncNet txt, zstage + audit-probe jsons, the calibration, setup_z.sh;
  - `tables/ZTABLE_FINAL.md`;
  - `videos/`: eye panels + key finals;
  - `ckpt/`: the student LoRA used for s4c8.
- The earlier W rounds are in the same branch: TBDub, best-of-N, keyframes. Their report is `reports/experiments/r15-wild/REPORT.md` and its tables are in the same dir. The r15-w eval dirs were cleaned to their finals.

## 1. What was tried
| step | result | verdict |
|---|---|---|
| W0 TBDub (Apache-2.0) through our chain + native | eye: "the whole face wobbles like jelly"; the tremble metric shows it is TBDub's own | dropped |
| W1 best-of-N per window (full / early / selective) on the student | SyncNet ±0.1 at +25..170 % cost | dropped |
| W2 keyframes / 2 + DIS interpolation in the aligned crop (tkf2c8) | SyncNet ≈ prod at −30 % GPU, E2E 215 / 275 GPU-s/vmin | **eye REJECTED**: 12.5 Hz shimmer (key vs interpolated texture, Nyquist share 6–12.5 % vs 0.4–1.3 %), m03 blurrier lips |
| Worms step 1: per-stage decomposition (m18 / old03 / m03) | lip colour / brightness flicker is born at the RAW LatentSync layer (old03 colour jumps: source 56 → raw 176 → final 103; m18 ×3–7). Not on the 16-frame seams. Geometry at the landmark-noise floor (old03 IMAGE-mode HF: source 1.67, raw 1.76 ‰ IOD). Input-crop affine stable (anchor HF 0.05 ‰ IOD, placement 0.1 px). LF slide ≤ 0.5 % IOD. In-generator CodeFormer adds +11 % geometric HF (1.76 → 1.95) | the generator is the source |
| Step 2: input-crop stabilisation LS_AFFINE_SMOOTH=9 (same transform for paste-back) | m18 blinks 40 vs 41, SyncNet lock 8.04 vs 8.13 | null (the crop is already stable) |
| Step 3: 2-noise latent ensemble (te2), + keyframes / 2 (te2k), stride 4 | m18 blinks 40–41 = prod; te2 / q2s4 cost ×2 | null |
| fixed reference (c8rf) / student s4c8 | blinks −17 % / −40 %; student SyncNet 7.33 | partial |
| **CFG dose-response** (trunc 3 / 5 / 7, guidance 1.25 / 1.5) | flicker AND snaps fall with CFG strength, and so does SyncNet. Even CFG-trunc 7/10 costs −1.6 SyncNet: the late-step CFG carries the sync. The DeepCache half-split code was checked and is correct | **CFG is the lever, with a sync price** |
| CFG rescale (Lin et al. 2023, per frame), φ 0.7 / 1.0 | sync kept, blinks unchanged (40.5) | the flicker is in the guidance DIRECTION, not its amplitude: null |
| CFG temporal smoothing of the amplification term (LS_CFG_TSMOOTH=0.25) | m03 flashes 69 → 57, strong 22 → 17; SyncNet −0.17 / lock −0.31; blinks −17 % | same snaps-per-SyncNet slope as lowering guidance |
| agent F's colour lock on top (`<arm>lk`) | halves the blinks and cuts the gate flashes 25–60 % in every arm, SyncNet −0.1..−0.25 | keep (F's) |

## 2. Results table (all arms; `lk` = + F's colour lock; flashes = the QA gate's r3_flash, strong = mc_ratio ≥ 2 and mc_vs_src ≥ 2 = F's m03 "snaps"; blinks = F's lip_flicker excess / 10 s)
Notes:
- m03 runs with face units. The CFG arms there use CF_EYE_IRIS=1, so CFG is the only change.
- m03 SyncNet sits at a global offset of 6 frames in every run, including r15-w q2u. That is the video, not the arm.
- m18 has no snaps (0 flashes / pop events in prod), so it judges blinks + sync only.

| video | arm | what | SyncNet | + lock SyncNet | blinks / 10 s | + lock blinks | flashes / strong | + lock flashes / strong | face GPU s/vmin |
|---|---|---|---|---|---|---|---|---|---|
| m03 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 8.774 | 8.681 | 17.78 | 4.69 | 69 / 22 | 50 / 18 | 357 |
| m03 | q2g15 | guidance 1.5 (prod 2.0) | 8.276 | 8.044 | 8.4 | 1.23 | 34 / 10 | 21 / 4 | 348 |
| m03 | q2g125 | guidance 1.25 | 7.647 | 7.425 | 3.95 | 0.74 | 16 / 4 | 13 / 2 | 350 |
| m03 | q2ts25 | LS_CFG_TSMOOTH=0.25 | 8.603 | 8.372 | 14.81 | 4.44 | 57 / 17 | 42 / 14 | 364 |
| m18 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 8.320 | 8.130 | 41.45 | 17.48 | – | – | 751 |
| m18 | q2v2 | prod + audit default fixes | 8.350 | 8.154 | 40.95 | 20.97 | – | – | 407 |
| m18 | q2as | LS_AFFINE_SMOOTH=9 | 8.260 | 8.039 | 39.95 | 17.48 | – | – | 389 |
| m18 | q2c8 | stride 8 continue | 8.588 | 8.423 | 40.95 | 18.97 | – | – | 467 |
| m18 | te2 | c8 + 2-noise latent ensemble | 8.534 | 8.310 | 40.45 | 17.48 | – | – | 739 |
| m18 | te2k | te2 + keyframes/2 | 8.542 | 8.236 | 40.95 | 18.97 | – | – | 425 |
| m18 | q2s4 | stride 4 continue | 8.443 | 8.291 | 40.95 | 20.97 | – | – | 750 |
| m18 | q2c8rf | c8 + fixed reference | 8.638 | 8.496 | 33.96 | 18.48 | – | – | 462 |
| m18 | s4c8 | student 4-step c8 | 7.583 | 7.327 | 24.97 | 12.98 | – | – | 362 |
| m18 | q2ct3 | CFG trunc 3/10 | 5.836 | 5.521 | 12.98 | 5.99 | – | – | 341 |
| m18 | q2ct5 | CFG trunc 5/10 | 6.053 | 5.708 | 12.98 | 5.99 | – | – | 347 |
| m18 | q2ct7 | CFG trunc 7/10 | 6.885 | 6.531 | 19.97 | 7.99 | – | – | 381 |
| m18 | q2g125 | guidance 1.25 | 6.624 | 6.310 | 22.47 | 6.49 | – | – | 409 |
| m18 | q2rs7 | CFG rescale 0.7 | 8.261 | 8.024 | 40.45 | 17.48 | – | – | 396 |
| m18 | q2rs10 | CFG rescale 1.0 | 8.378 | 8.167 | 40.45 | – | – | – | 360 |
| m18 | q2cfa | CF_ALIGN_FIT=lsq + CF_MOUTH=rigid | 8.359 | 8.090 | 35.95 | 17.48 | – | – | 409 |
| m18 | q2nb | AF_UPPER=nobrow | 8.268 | 8.014 | 40.45 | 18.97 | – | – | 380 |
| m18 | q2nr | LS_NOSE_RIGID=1 | 8.226 | 8.014 | 39.45 | 19.47 | – | – | 397 |
| old03 | q2d | production q2 + F3 (+ LS_DUMP_DIR) | 7.789 | 7.560 | 10.85 | -2.44 | – | – | 364 |
| old03 | q2as | LS_AFFINE_SMOOTH=9 | 7.823 | – | 9.85 | – | – | – | 357 |

### The CFG dose-response on m03 (the key numbers)
| m03 | SyncNet alone / + lock | gate flashes / strong, alone | + lock | blinks alone / + lock |
|---|---|---|---|---|
| production (guidance 2.0) | 8.77 / 8.68 | 69 / 22 | 50 / 18 | 17.8 / 4.7 |
| CFG tsmooth 0.25 | 8.60 / 8.37 | 57 / 17 | 42 / 14 | 14.8 / 4.4 |
| guidance 1.5 | 8.28 / 8.04 | 34 / 10 | 21 / 4 | 8.4 / 1.2 |
| guidance 1.25 | 7.65 / 7.43 | 16 / 4 | 13 / 2 | 4.0 / 0.7 |

The coordinator's target was snaps ↓ with SyncNet ≥ prod − 0.3.
- tsmooth 0.25 meets it alone (−0.17) but only with a modest snap cut (−23 %).
- guidance 1.5 cuts snaps by about 55 % (with lock: strong 18 → 4) at −0.5…−0.64 SyncNet.
- **Whether that SyncNet loss is VISIBLE is the open question.** SyncNet rewards exaggerated articulation, which is exactly what CFG adds.

## 3. Generator-side audit tracker (coordination/audit_r15_face.md)
| item | status | commit | evidence |
|---|---|---|---|
| frame_map.py greedy thumbnail master → source map | **FIXED by default** (FRAME_MAP=thumb restores) | 347a521 | Exact map = the master command replayed on an index video (as r3_master_timing). old03: the greedy map disagreed on 249 / 2258 master frames, by up to 3 source frames. The exact map agrees with r3 on 100 % and takes 6 s. Active only with FAST_MPREUSE seeding: the prod default when the audio stage left source_mediapipe.npz. Not active in our arms |
| CodeFormer eye centroids include iris points (468 ×2 + 470 / 473 ×2 + 475) | **FIXED by default** (CF_EYE_IRIS=1 restores) | 347a521 | Measured jitter effect nil (VIDEO-mode landmarks are smooth). m18 q2v2 = prod: blinks 40.95 vs 41.45, SyncNet lock 8.15 vs 8.13 |
| CodeFormer LMEDS 5-point fit + SOURCE mouth corners | flag CF_ALIGN_FIT=lsq, CF_MOUTH=rigid | 347a521 | Alignment HF −55 % on old03 (mouth 1.1 → 0.49 ‰ IOD), −5..15 % on m18, all sub-pixel. m18 arm q2cfa: blinks 36 vs 41, SyncNet 8.36 / 8.09 = prod. **Default candidate after eye** |
| rigid head motion: unchecked RANSAC / failed fit = identity | **FIXED by default**: no fit / < 12 samples / < 8 inliers → the frame pair's landmark similarity | 347a521 | Failed fits observed: 0 on m18 / old03. The audit's "< 0.3 inlier share" rule NOT adopted: at the 0.5 px threshold 23 % of m18 pairs are below 0.3 (flow noise on a big face). Flag AF_MIN_INLIER |
| UPPER includes brows / forehead | flag AF_UPPER=nobrow | 347a521 | LS crop anchors with vs without: m18 p99 5.1 px, max 6.6 px; old03 p99 0.55 px. m18 arm neutral (blinks 40.45, lock 18.97; SyncNet 8.27 / 8.01). Needs an eye check on brow-raise spots |
| nose anchor includes the nostril wings (MP 64 / 294) | flag LS_NOSE_RIGID=1 | 347a521 | Wing part of the anchor: p99 deviation 7.9 px m18, 1.4 px old03. The flag cuts the anchor wobble 39 % on old03. m18 arm: SyncNet lock 8.01; blinks in the table |
| steps 10 / 20 switch per window without hysteresis | flag LS_SIL_HYST=1; A/B `--steps-silent 0` | 347a521 | **NOT RUN** (stopped at wrap-up). Note: steps-silent only applies in the bridges path (stride None) |
| VAE posterior sampled, not its mean | no fix needed (flag LS_VAE_MEAN=1 exists) | 347a521 | Posterior std = 1e-4 of the latent. Decoded sample vs mean: 0.01 grey levels mean, p99 0.06 |
| composite-side items (retime pulse, colour offset, skin texture, boundary, seg-joint, layer_sr) | agent F | – | – |

## 4. Eye verdicts so far
- TBDub: REJECTED ("jelly").
- tkf2c8: REJECTED (old03 23.5–24.5 s stronger flicker; m03 "junk", blurrier lips).
- bon3: "not bad" but the student trembles less.
- **Pending (videos/ and r15-z outbox):**
  - `m03_mouth3_q2d_q2g15_q2g15lk_spots_half_speed.mp4`
  - `m03_mouth3_q2d_q2ts25_q2ts25lk_…`
  - `m03_mouth3_q2d_q2g125_q2g125lk_…`
  - spots 3.2–5.5, 11.3–14.3, 19.5–21, 26.7–28.7 s; 0.5x; panel = prod | arm | arm + colour lock;
  - m18 panels: CFG-trunc 3, student, fixed ref.

## 5. Open problems
1. The lip blinks and the m03 snaps are produced by CFG's per-frame audio-guidance DIRECTION. Every way of weakening CFG we tried trades sync for flicker at about the same rate: snaps −55 % ≈ SyncNet −0.5. Covered: truncation, lower scale, rescale, temporal smoothing of the amplification.
2. We don't know if the SyncNet drop at guidance 1.5 is visible: SyncNet likes exaggerated mouths. The eye has to decide.
3. The generator's CodeFormer adds geometric HF (+11 %). F's clean recipe (no CF + lipcon + lock) already removes it on the composite side.

## 6. Next steps (ranked)
1. **Eye test (0 GPU):** m03 panels guidance 1.5 + lock vs prod. If the sync loss is not visible → production `--guidance 1.5` + F's colour lock (cost-neutral; strong snaps 18 → 4, blinks at source level). Then confirm on old03 / m18 and the other videos (gate flashes + SyncNet).
2. If it is visible: **split the guidance by frequency / region** so the mouth-opening (jaw / shape, low-frequency) keeps g 2.0 and the per-frame texture gets g ≤ 1.5. Options:
   - CFG amplification on a spatially low-passed latent delta only (extend `_cfg` in ls_stable.py, next to LS_CFG_TSMOOTH);
   - or guidance 2.0 inside the inner-mouth / jaw mask and 1.5 on the lips' texture band.
   Cheap to try: one flag, ~1 GPU-min per video-min.
3. **Student with a flicker penalty:** s4c8 already gives −40 % blinks at 362 GPU-s/vmin. Train the 4-step student (scripts/r13/distill) against g 2.0 targets with a neighbour-frame x0-consistency loss on the lip region. That gets the g 2.0 shape without the per-frame texture noise.
4. **Audit flags → default after eye:** CF_ALIGN_FIT=lsq + CF_MOUTH=rigid (−12 % blinks, sync kept), AF_UPPER=nobrow, LS_NOSE_RIGID=1. Also run the `--steps-silent 0` A/B (not run).
5. Untested midpoint: guidance 1.75 or tsmooth 0.33, each + lock + cfa (stopped at wrap-up).
6. The audit arms ran on m18 only: repeat the useful ones on old03 / m03.

## 7. Reproduce
- Setup: `/workspace/dub/setup_z.sh` (configs/). Standard DabAI setup plus the CodeFormer / Real-ESRGAN / stable_syncnet downloads. Public weights only, no NC.
- Baseline dubs: `scripts/r15/base_q2.sh` (audio unchanged).
- One arm, e.g.:
  `V=q2g15 GEN_EXTRA="--guidance 1.5" bash scripts/r15/arm.sh m03`
  - Production flags (q2 + F3, k5 composite) are in arm.sh.
  - The exact env of every arm is in `configs/env/<arm>_<id>.env.txt`.
- Colour-lock variant:
  `SRC=<arm> V=<arm>lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh <id>`
- Metrics:
  - SyncNet: `scripts/r11/sync_speech.py`;
  - blinks: `zlf.sh` (F's lip_flicker);
  - gate flashes: `zflash.sh` (r3_flash strong);
  - pop events: `zexp.sh` (F's explode);
  - the table: `zreport_table.py`;
  - eye panels: `SPOTS="a:b ..." bash scripts/r15/panel3.sh <id> A B C "labels"`.
- New flags, all default off: LS_CFG_TSMOOTH, LS_CFG_RESCALE, CF_ALIGN_FIT, CF_MOUTH, AF_UPPER, AF_MIN_INLIER, LS_NOSE_RIGID, LS_SIL_HYST, LS_VAE_MEAN, LS_ENS, LS_DUMP_DIR, LS_AFFINE_SMOOTH, LS_KF, LS_BON*.
- Default fixes, with their restore flags: FRAME_MAP=thumb, CF_EYE_IRIS=1. The failed-fit fallback has no restore flag.
- Student checkpoint: `ckpt/r2_s4_step_01500_ema.pt` (agent D's r2 LoRA, 30 MB). Run it as:
  `R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=<ckpt> R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue`
