# r15-F2: CodeFormer replacement (source detail transfer) + the m03 "lips vanish" flag

Branch experiment/r15-repair (worktree /workspace/dub/dubai-f), bundle /workspace/dub/r15f/r15-detail.bundle.
Arms on the cached q2 layers (old03 = r9g_q2_old03, m18 = r9g_q2_m18, m03 = r9g_q2_m03fu, face-units fix):
- a = production (CodeFormer detail ia04);
- b = CodeFormer off (the pre-CF dump);
- c = CF off + source detail transfer (scripts/r15/src_detail.py).

The rest of the chain is unchanged: SR srt1, fb7s fallback, rep2s repair, q2 + F3 composite.

## Сводка по-русски (для Ирины)
- **Замена CodeFormer (вариант c) работает, но CodeFormer не догоняет.** Детали переносятся с самого исходного видео, без обучения и без чужих лицензий. Губы и зубы получаются чётче, чем без CF (b), но мягче, чем с CF (a). CF «дорисовывает» лишнюю резкость, а перенос ограничен резкостью исходника. Призраков и двойных контуров на стоп-кадрах нет.
- **Синхрон и мигание:** по old03 синхрон даже выше, чем в проде: 7.81 у c, 7.74 у b, 7.48 у a, гейт PASS. По m18 и m03 гейт ещё идёт: GPU занят прогоном агента T. Таблица допишется в этот файл сама. Мигание тона и «попы» рта у всех трёх вариантов на одном уровне.
- **«Пропадающие губы» на m03:** перенос деталей их НЕ лечит. Он добавляет резкость, но не цвет губ: провалов контраста у c 27 кадров, у прода 11.
- **Новый флаг контраста губ к коже на m03:** провалы 11 → 0 с CF, 24 → 10 без CF и 27 → 6 с переносом. Почти все такие кадры перерисовываются ремонтом из соседних кадров, а не подменяются исходником. Место 8.1–8.3 с ловится. 14.3 с не ловится: там губы бледнеют уже после апскейла, а флаг смотрит на слой до него.
- **На old03 флаг срабатывает слишком часто**, в 7–10 % кадров. Там губы бледнее исходника постоянно, а не вспышками, и метрика не отличает это от пропадания. Поэтому по умолчанию флаг ВЫКЛЮЧЕН: включать только для m03-подобных видео, пока не посмотрим глазом.
- **Что посмотреть:** m03_lips_lipcon_05x.mp4 (2–5.5, 8–9 и 14–16 с) и m03_mouth_a_b_c_05x.mp4 (8–9, 14–16 и 17–21 с), плюс old03 и m18 a|b|c.

## Layer sharpness (scripts/r15/layer_sharp.py; median |Laplacian| on the lip ring / teeth, relative to the source = 1.0)
| video | a CF (lips / teeth) | b off | c transfer |
|---|---|---|---|
| old03 | 0.97 / 0.65 | 0.80 / 0.60 | 0.88 / 0.66 |
| m18 | 1.73 / 1.79 | 1.00 / 1.13 | 1.17 / 1.16 |
| m03 | 1.25 / 0.96 | 0.87 / 0.71 | 0.97 / 0.82 |

src_detail per video:
| video | bank sharpness ratio p50 (vs the generated mouth) | bank switches | time |
|---|---|---|---|
| old03 | 2.08 | 300 / 2125 | 77 s |
| m18 | 1.71 | 70 | 34 s |
| m03 | 1.92 | 135 | 52 s |

All on CPU, with no training. The transfer is a no-op where the source is not sharper (weight = 0).

## tone_flicker / mouth_pop (finals)
| | old03 a / b / c | m18 a / b / c | m03 a / b / c |
|---|---|---|---|
| flicker low mean | 0.193 / 0.199 / 0.200 | 0.372 / 0.375 / 0.369 | 0.116 / 0.117 / 0.119 |
| pop excess mean | 0.099 / 0.108 / 0.110 | 0.123 / 0.121 / 0.140 | 0.131 / 0.125 / 0.143 |

The differences are within noise. c pops slightly more on m18 and m03 (+0.01..0.02).

## Gate (SyncNet etc., vs a)
old03:
- b: PASS, sync 7.476 → 7.742, flashes 35 → 35, pressed 2.93 → 1.93 s.
- c: PASS, sync 7.476 → 7.812, flashes 35 → 32, pressed 2.93 → 2.20 s.

m18 / m03, and the m03 lipcon arms: queued behind agent T's E2E (GPU rule). scripts/r15/gate_lc.sh appends them below and touches DONE.

## m03 "lips vanish" (agent T): lip-vs-skin contrast as a fraction of the source
Measure (scripts/r15/lipcon.py, T's lip_vanish metric): min(a* contrast ratio, lightness contrast ratio) on the lip ring vs a skin ring. A dip is a value below 0.45 where the source lips are visible.

| m03 final | median | dip frames |
|---|---|---|
| a CF (prod) | 0.91 | 11 |
| b off | 0.83 | 24 |
| c transfer | 0.80 | 27 |
| a + lipcon | 0.92 | **0** |
| b + lipcon | 0.83 | 10 |
| c + lipcon | 0.80 | 6 |

- (2) The **detail transfer does not fix the vanishing**. It moves luma high-frequency detail only, not lip colour or tone. In this measure CodeFormer actually helps lip contrast on m03; T's sheet attributes some dips to CF, but that was on T's own run.
- (1) **New flag `r8_fallback.py --thr-lipcon 0.45`** (default 0 = OFF; `comp_f2.sh LC=0.45`). It flags a transient lip-contrast drop of ≥ 0.1 below the ±6-frame median, at any opening; washl only covers mismatched openings. The repair re-draws the flagged frames from their neighbours.
  - m03: 27–32 lipcon flags; 39–40 frames repaired; 0–10 frames left to the source fallback.
  - Of T's misses, 8.16–8.20 is flagged and re-drawn. 14.2–14.3 is NOT flagged: on this run that dip appears only after the SR / repair stages (a_rep, c_rep 14.24), while the fallback scores the pre-SR layer. Its b-final 14.24 dip remains; the a + lipcon final has none.
- **Why the flag is not the default:** on old03 it flags 145–202 frames (7–10 %).
  - old03's layer lips are steadily paler than the source, and often the openings differ (dub closed over an open source).
  - Restricting it to matching openings (`--lipcon-same on`) or a layer-only transient rule did not separate the two cases. Either m03's real dips went or old03 still fired at 67–131 frames.
  - m18: 5–9 flags.
  - A shippable default needs an eye verdict on m03, plus a better discriminator, for example a relative-to-the-layer tone model.

## Clips (/workspace/dub/outbox/r15_f2/, 0.5x, source audio)
- old03_mouth_a_b_c_05x.mp4 (auto: 29.5–31.5, 50.4–52.4, 58.7–60.7 s)
- m18_mouth_a_b_c_05x.mp4 (3.2–5.2, 6.9–8.9, 10.4–12.4 s)
- m03_mouth_a_b_c_05x.mp4 (8–9, 14–16, 17–21 s)
- m03_lips_lipcon_05x.mp4: a CF | a CF + lipcon | b + lipcon | c + lipcon (2–5.5, 8–9, 14–16 s)

## Files
- scripts/r15/src_detail.py, layer_sharp.py, lipcon.py, run_f2.sh, comp_f2.sh (LC=), gate_lc.sh, clip3_f.sh
- scripts/r8/r8_fallback.py: --thr-lipcon / --lipcon-drop / --lipcon-same, all off by default; the production fallback is unchanged.
- scripts/r15/gate_f.sh: GT= override.

Note: during the session the m03 b/c repair caches (`_ls_cfd_{off,sdt1}_srt1_rep2s.mp4`) were truncated by a killed mis-launch and deleted. The b/c finals made before that are intact. A rerun of comp_f2.sh rebuilds them.

## Round 2: adaptive lipcon + post-SR pass (eye: "c + lipcon OK"; a > b > c without lipcon)

Code:
- `r8_fallback --lipcon-mode adaptive`: flag where the ratio is < 0.45 of the source AND < 0.6 x the video's own ±25-frame (2 s) median. Plus `--lipcon-max-frac 0.03`, a per-video cap that keeps the deepest dips.
- `--lipcon-only`, and `r7_m7_comp REP2_FB_ARGS`: lipcon is scored again on the REPAIRED post-SR layer, and a second repair pass runs on it.
- `comp_f2.sh LCA=0.45` uses all of this. Everything is off by default.

Adaptive-rule flag counts on the pre-SR layer, before the cap (same-frame ratio): old03 168–227, m18 8–9, m03 13–27.
- old03's dips are transient against its own 2 s median too, so the adaptive rule does NOT separate old03 from m03.
- Comparing against the source's typical contrast at the layer's opening did not separate them either (old03 169–222).
- The cap is what bounds old03: 63 frames = 3 %, in both passes.

Arms (m03 fubla / fucla, old03 bla / cla):

| | pass-1 lipcon flags | pass-2 flags | repaired frames | left to the source |
|---|---|---|---|---|
| m03 b + la | 27 | 2 | 40 + 2 | 0 |
| m03 c + la | 28 | 7 (incl. 14.24) | 36 + 7 | 0 |
| old03 b + la | 63 (cap) | 63 (cap) | 69 + 39 | 48 + 40 |
| old03 c + la | 63 (cap) | 63 (cap) | 81 + 49 | 25 + 26 |

Final-video dip frames (lip contrast < 0.45 of the source):

| | a (prod) | b | c | b + fixed lc | c + fixed lc | b + adaptive | c + adaptive |
|---|---|---|---|---|---|---|---|
| m03 | 11 | 24 | 27 | 10 | 6 | 11 | 11 |
| old03 | 24 | 64 | – | – | – | 27 | 33 |

Notes:
- **14.24 s on m03:** caught by pass 2 in c and gone from the c + adaptive final. In b the repaired layer has no dip there; it appears in the composite (colour match / sharpening), so b's final keeps a mild 0.41 frame.
- **8.6–8.7 s on m03:** a 3-frame pale run that the neighbour interpolation does not fully restore (0.18 in both adaptive finals).
- **The final-video measure is unreliable where the source fallback shows** (old03 5.7–5.9 s reads −0.07): there it measures on the layer's landmarks.
- **old03 still: b / c show a closed mouth at about 5.0 s where a is slightly open.** The redraw or fallback may cost sync there: check by eye.

Clips:
- outbox/r15_f2/m03_lipcon_adaptive_05x.mp4 (a | b + la | c + la; 2–5.5, 8–9, 14–16 s)
- outbox/r15_f2/old03_lipcon_adaptive_05x.mp4 (4–7, 17–18.5, 29.5–31.5 s)

Gates: scripts/r15/gate_la.sh, queued behind gate_lc, under the GPU rules; it appends below and touches DONE_LA.

RU:
- Адаптивный порог (провал ниже 0.6 от своей двухсекундной медианы) old03 НЕ успокаивает: там провалы тоже короткие. Поэтому сделан лимит: флаг трогает не больше 3 % кадров, самые глубокие.
- Второй проход после апскейла ловит 14.24 на m03 в варианте c.
- На m03 провалов 11 у b и c с новым флагом против 11 в проде.
- На old03 провалов 27–33 против 64 без флага и 24 в проде.
- Глазом проверить: old03 около 5 с, там рот в b/c закрыт, а в проде приоткрыт.

## How to enable: DUB_PROFILE=clean face = no CodeFormer + adaptive lipcon (eye: "b + lipcon about like a")

Code: commits **5b5be66 + 6168df3** on experiment/r15-repair.
- Files needed: scripts/r15/lipcon.py (new), scripts/r8/r8_fallback.py (lipcon flags), scripts/r7/r7_m7_comp.sh (the REP2_FB_ARGS second pass).
- The other files in those commits (src_detail, clips, gates) are not needed for clean.
- Everything defaults OFF, so the current profile is untouched.

Settings, in e2e.sh face_comp, clean branch (DUB_ALLOW_NC=0, CFDV="", E2E_FX=1, E2E_SYNC=k5):
```bash
LCX="--thr-lipcon 0.45 --lipcon-mode adaptive --lipcon-rel 0.6 --lipcon-win 25 --lipcon-max-frac 0.03"
FBL=fb7sla                                   # new fallback cache label (never reuse fb7s caches)
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2 $LCX"
export REP_LABEL=rep2sla REP_ARGS="--spikes off"
export REP2_FB_ARGS="$(cat reports/experiments/r8-face/fb7.args) $LCX"   # 2nd pass: lipcon alone, on the repaired post-SR layer
```
r7_m7_comp then runs:
1. pass 1: the fallback score with lipcon on the raw (no-CF) layer;
2. the repair → `_rep2sla.mp4`;
3. pass 2: lipcon on the repaired layer, a second repair → `_rep2sla_lc2.mp4`;
4. the composite fallback = max of both passes' left-overs.

What the settings mean:
- **Threshold:** lip-vs-skin contrast < 0.45 of the source at the same frame, AND < 0.6 × the video's own ±1 s median.
- **Cap:** at most 3 % of the LatentSync frames per pass, deepest dips first.

Cost: CPU only, about +45 s of repair and +60 s of pass-2 scoring per 40 s video (m03).

Verified so far:
- m03: dip frames 24 → 11, the same as prod CF (11).
- old03: 64 → 27, prod 24. old03 hits the 3 % cap.
- m18: gate pending.
- The gate (sync / flashes / slide) on all 3 is appended below when the GPU is free.
