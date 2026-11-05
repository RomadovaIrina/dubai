# r15-F3: lip flicker ("worms") — the colour lock, the clean recipe, the audit tracker

Branch experiment/r15-repair (worktree /workspace/dub/dubai-f). Commits: 54b8509 (tools + lock), d5bfb26 + 038a077 (audit fixes).

## Сводка по-русски
- **Главная находка:** мигание губ лечит «замок цвета губ» плюс медиана по соседним кадрам, выровненным по оптическому потоку. Цвет и яркость губ сглаживаются по ±5 кадрам, форма рта и движение остаются свои.
- **Лишнее мигание** (средний скачок яркости губ сверх исходника) падает со 0.71 до 0.13 (old03), с 0.85 до 0.24 (m03) и с 1.32 до 0.42 (m18).
- **Резких скачков** в 2.5 раза меньше. На old03 почти уровень исходника, m18 ещё заметен.
- **Рецепт чистого профиля:** без CodeFormer + флаг контраста губ + замок цвета с медианой. Глаз уже одобрил «b + lipcon + замок» на m03 и old03.
- **Ученик с замком** даёт самый спокойный m03.
- **Не помогли:** флаги аудита для композита сами по себе, отказ от апскейла на губах. Дрожания движения из-за 25 → 30 нет.
- **Ещё открыто:** на old03 есть «пульс резкости»: каждый 6-й кадр на 14 % резче. Это следующий кандидат.
- **Все баги аудита из моего списка исправлены.** Простые включены по умолчанию, меняющие картинку стоят за флагами до гейтов.

## Metric (scripts/r15/lip_flicker.py, scripts/r15/lf.sh)
- Lip ring (outer minus inner polygon) of EACH video's own mouth:
  - the final uses the layer landmarks of the master frame it shows;
  - the source uses its own.
- W-style columns, comparable with W's table:
  - Wexc = mean |ΔL| of the final minus the source's;
  - jumps = frames with |ΔL| above the source's p99 (the source's own count in [ ]).
- My blink metric: L / a* jumps at a steady shape (opening change < 0.03 IOD), excess per 10 s over the source.
- Sharpness pulse: lip Laplacian variance on raw master frames / interpolated frames (1.0 = none).

| arm | old03 Wexc / jumps [28] | m03 [13] | m18 [6] |
|---|---|---|---|
| a prod (CodeFormer) | 0.712 / 120 | 0.846 / 98 | 1.323 / 68 |
| c8 (r3c8, eye-preferred r14) | (blink excess 5.5) | (4.7) | (19.5) |
| b + adaptive lipcon | – | – | 1.28 / 70 |
| a + colour lock | 0.247 / 68 | 0.300 / 44 | 0.520 / 34 |
| b + lipcon + colour lock | 0.232 / 60 | 0.297 / 31 | 0.454 / 31 |
| b + lipcon + colour lock r5 | 0.202 / 58 | 0.271 / 35 | 0.392 / 30 |
| b + lipcon + flow median | 0.234 / 55 | 0.341 / 42 | – |
| b + lipcon + lock + median (both) | 0.154 / 46 | 0.257 / 34 | – |
| **b + lipcon + lock + median r5 = RECIPE** | **0.131 / 47** | **0.238 / 36** | **0.421 / 30** |
| student s4p1500c8u (CF) | – | 0.454 / 60 | – |
| student + colour lock | – | 0.192 / 33 | – |
| student + lipcon + colour lock | – | **0.186 / 28** | – |
| audit composite flags alone (`aud`) | 0.710 / 131 | 0.865 / 107 | – |
| audit flags + b + lipcon + lock | 0.219 / 55 | 0.298 / 36 | – |
| no Real-ESRGAN on lips (a / b + lipcon) | 0.733 / 129, 0.698 / 120 | 0.893 / 103, 0.821 / 88 | – |

tone_flicker (jaw) is unchanged by the lock. mouth_pop excess drops:
- old03: 0.099 → 0.058 (lock) → −0.018 (median);
- m03: 0.131 → 0.084 → 0.029.

Eye (coordinator relay):
- m03: "b + lipcon + colour lock / flow median: better, almost no flicker".
- old03: "slightly worse than m03, but better than prod".
- b + lipcon (no lock) looked about like a.

## The recipe (DUB_PROFILE=clean face chain), all flags default OFF
Code: 5b5be66 + 6168df3 (lipcon) + 54b8509 (lip_temporal.py + r7_m7_comp LT_ARGS hook). Files needed:
- scripts/r15/lipcon.py
- scripts/r15/lip_temporal.py
- scripts/r8/r8_fallback.py
- scripts/r7/r7_m7_comp.sh

In e2e.sh face_comp, clean branch (DUB_ALLOW_NC=0, CFDV="", E2E_FX=1, E2E_SYNC=k5):
```bash
LCX="--thr-lipcon 0.45 --lipcon-mode adaptive --lipcon-rel 0.6 --lipcon-win 25 --lipcon-max-frac 0.03"
FBL=fb7sla; FBA="$FBA $LCX"                         # on top of the k5 FBA
export REP_LABEL=rep2sla REP2_FB_ARGS="$(cat reports/experiments/r8-face/fb7.args) $LCX"
export LT_ARGS="--mode both --radius 5 --mradius 2" LT_LABEL=bothr5   # NEW: lip colour lock (+-5) + flow-aligned median (+-2)
```
- What it costs: CPU only, about +3–5 min per minute of video (lip_temporal 2–4 min on a 40 s clip on a loaded box), plus lipcon.
- What it is: the colour lock replaces the low-frequency L / a* / b* inside the lip ring with its per-pixel median over ±5 flow-aligned neighbours. The median then replaces the mouth box with the median of ±2 aligned frames where the flow is consistent.
- Gates (sync / flashes / slide) are pending the GPU (after T's long5). Do not make it the clean default before they pass.

## Audit tracker (briefs/audit_r15_face.md)
| # | item | status | commit | evidence |
|---|---|---|---|---|
| 1 | 25→30 retime sharpness pulse | measured; `--retime-hf nearest` does NOT fix it; pending | – | pulse (raw / interp lip Laplacian var): m03 1.02–1.03, m18 1.06–1.07, **old03 1.14** (all arms, incl. retime-hf nearest 1.142); motion judder: none in prod (5 Hz share m03 0.091 vs src 0.092, m18 0.073 vs 0.134, old03 0.083 vs 0.091; the forced nearest retime shows it: m18 0.21) |
| 2 | CodeFormer aligned on source landmarks / iris points | n/a for clean (CF off); not mine (generator) | – | the CF sweep cf_layer.py uses the layer's eye-contour points (no iris) |
| 3 | cm-mouth-excl mean offset, unsmoothed | tested (`--cm-mouth-mode luma` in the aud arm): no lip-flicker gain | – | aud vs prod: m03 0.865 / 107 vs 0.846 / 98 |
| 4 | skin texture tex-tau 1000 | tested (`--skin-texture off` in aud): no gain | – | as 3 |
| 5 | rigid head incl. brows, no inlier check | not mine (anchor_filters / generator) | – | – |
| 6 | steps switch per window | not mine (generator) | – | – |
| 7 | boundary + bridges double seam | tested (`--boundary off` in aud): no lip-flicker gain | – | as 3 |
| 8 | seg-joint never set | tested (`--seg-joint 3` in aud): no gain | – | as 3 |
| 9a | layer_sr writes into its input list | **fixed by default** | 038a077 | neighbours' flow now reads raw frames; needs a GPU SR re-run to measure |
| 9b | layer_sr truncated crop origin | fixed behind `--stable-box on` | 038a077 | GPU SR A/B pending |
| P1 | r9_source_rigid decode_source removed | **fixed by default** | d5bfb26 | → fc.source_stream |
| P2 | fallback joint phase mixes segments | **fixed by default** (`--joint-seg on`) | d5bfb26 + 038a077 | m03 fb7soff identical with / without |
| P3 | cf_detail truncation (frozen mouth at end) | **fixed by default** | d5bfb26 | the base frames past the CF / landmark length are written unchanged |
| P4 | candidate stage links master_25fps unchecked | **fixed by default** | 038a077 | links only if the frame count matches the source within 2 frames and the fps is 25; m03 / old03 masters pass |
| S | repair / fallback / sanitize no hysteresis, paste guard hard switch, 3 re-encodes, mask on raw landmarks | pending | – | not tested yet |

Item 1 / old03 is the next lead: the 14 % sharpness pulse survives every arm.

## Clips (outbox/r15_f3, 0.5x)
- m03_lipflicker_05x.mp4, old03_lipflicker_05x.mp4: a | a + lock | b + lipcon + lock | b + lipcon + median
- old03_23-25s_lipflicker_05x.mp4: a | a + lock | b + lipcon + lock | source
- **old03_recipe_05x.mp4** (5.6–6.8, 15.3–16.4, 23–25 s), **m03_recipe_05x.mp4**, **m18_recipe_05x.mp4**: a prod | clean recipe | source

## Not done / pending
- Gates for everything (GPU, after T's DONE).
- The CF w sweep (GPU).
- The student on old03 / m18 (waits for D's r5).
- The old03 sharpness pulse.
