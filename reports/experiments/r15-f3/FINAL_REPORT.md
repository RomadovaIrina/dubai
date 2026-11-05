# HANDOFF_F: agent F (r15-F / F2 / F3) final report, 05.11.2026

Branch **experiment/r15-repair**, head **78b28e5** (bundle `r15-repair.bundle` here, which requires d189fb5; origin had f74f9b4 pushed earlier).
Worktree on r14-d: /workspace/dub/dubai-f. Everything is on the cached q2 layers of old03 / m03 (face-units run r9g_q2_m03fu) / m18.

## Кратко по-русски
1. **«Второй рот» у мамы (m03, 19 с) исправлен:** LS_FACE_UNITS, принят глазом, по умолчанию ВКЛ (d494c67).
2. **Мигание губ (worms) лечит «замок цвета губ»:** лишнее мигание падает в 3–6 раз. Глаз: «справа лучше, почти не мигает».
3. **«Взрывы» рта рождаются в самом генераторе LatentSync:** резкие скачки формы за 1–2 кадра, а CodeFormer их усиливает. Их режет медиана по соседним кадрам, но сплошная медиана роняет синхрон (m03 8.60 → 8.05).
4. **Лучший чистый рецепт сейчас:** без CodeFormer + флаг контраста губ + лёгкий замок цвета + медиана только на кадрах-взрывах. На m03 синхрон 8.32 (порог 8.30), сильных вспышек 25 → 11. На old03 / m18 эта версия ещё не прогнана.
5. **Баги из аудита исправлены:** простые по умолчанию, меняющие картинку стоят за флагами. Что дальше, ниже в разделе NEXT STEPS.

## 1. What was done
- **r15-F, m03 double mouth:** a phantom diarization turn (the child's voice) mapped onto face 1. Fix: LS_FACE_UNITS (the face timeline follows the dubbed units), default ON. Accepted by eye: "второе выглядит нормас". Commit d494c67; diagnostics in 4511f30.
- **r15-F2, CodeFormer (NC) replacement:**
  - Source detail transfer (src_detail.py). Eye: CF > no-CF > transfer, so the transfer was dropped.
  - The lipcon flag (lip-vs-skin contrast fallback, adaptive, 3 % cap, a second pass after SR). Eye: "b (no CF + lipcon) about like a".
- **r15-F3, lip flicker:**
  - lip_temporal.py: colour lock / flow median / targeted median / outlier.
  - Metrics: lip_flicker (W-style ΔL excess + jumps), lip_jitter (retime judder), explode.py (pop events + stage of origin); r3_flash on the CPU; SyncNet on the CPU (SYNC_DEVICE=cpu).

## 2. Tables

### 2a. Lip flicker (W-style: Wexc = mean |ΔL| over the source; jumps = frames with |ΔL| > the source's p99, source count in [ ])
| arm | old03 [28] | m03 [13] | m18 [6] |
|---|---|---|---|
| a prod (CF) | 0.712 / 120 | 0.846 / 98 | 1.323 / 68 |
| b + lipcon + lock r3 | 0.232 / 60 | 0.297 / 31 | 0.454 / 31 |
| b + lipcon + lock r5 | 0.202 / 58 | 0.271 / 35 | 0.392 / 30 |
| b + lipcon + lock + median r5 ("recipe v1") | 0.131 / 47 | 0.238 / 36 | 0.421 / 30 |
| student s4p1500c8u + lipcon + lock | – | 0.186 / 28 | – |

### 2b. Sync vs explosions (m03, SyncNet speech conf on the CPU; CPU prod = 8.602 = GPU gate; gate needs ≥ 8.30; strong flashes = r3_flash, as the gate)
| arm | sync | strong flashes |
|---|---|---|
| a prod | 8.60 | 25 |
| b no CF | 8.68 (GPU gate PASS) | 26 |
| b + lipcon | 8.49 | 20 |
| b + lipcon + lock r3 | 8.33 | 12 |
| a + lock | 8.44 | 15 |
| b + lipcon + flow median | 8.08 ✗ | 6 |
| recipe v1 (lock r5 + median) | 8.05 ✗ (GPU gate FAIL: sync 8.60 → 8.04, pressed lips 0.57 → 1.10 s) | 7 |
| recipe v1 + repair spikes on | 7.91 ✗ | 5 |
| b + lipcon + targeted median only (--median-gate 1.5) | 8.35 | 11 |
| b + lipcon + lock r3 + targeted median 1.5 | 8.28 ✗ (borderline) | 10 |
| b + lipcon + lock r3 + targeted median 1.3 | 8.25 ✗ | 6 |
| **b + lipcon + light lock (L clamp 4) + targeted median 1.5 = recipe v2** | **8.32** | **11** |
| b + lipcon + light lock L4 (no median) | 8.38 | 13 |
| b + lipcon + light lock L2 (no median) | 8.44 | 16 |

The flicker price of the lighter lock (m03 Wexc / jumps, prod 0.846 / 98):
- lock r3 + targeted median: 0.280 / 31 (sync 8.28);
- **v2: 0.347 / 43 (sync 8.32)**;
- L4 alone: 0.416 / 53;
- targeted median alone: 0.576 / 58;
- L2 alone: 0.555 / 62.

So the lightness correction is what fights the flicker and also what costs sync. v2 is the balance point that still passes.

Strong flashes on old03 / m18:

| arm | old03 | m18 |
|---|---|---|
| prod | 35 | 5 |
| b + lipcon | 44 | 11 |
| + lock | 43 | 3 |
| recipe v1 | 27 | 2 |

So old03 needs the median; without it, old03 explodes more than prod.

### 2c. Where the explosions are born (m03, the gate's top-20 flash events; explode.py)
- Prod: raw generator 7, CodeFormer detail 6 (CF amplifies the generator's pop), composite / retime 7 (weak).
- Recipe: raw generator 12, composite 7 (weak), lip_temporal 1.
- Most are generator SHAPE SNAPS (open + teeth → pursed within 2 frames), not single-frame outliers. The outlier-only rule touched 0.01 % of the mouth pixels.
- Retime: no motion judder in prod (5 Hz share = the source's).
- old03 sharpness pulse: raw frames 14 % sharper than the interpolated ones, in every arm; `--retime-hf nearest` does not fix it.

## 3. Eye verdicts (relayed by the coordinator)
- F: the face-units fix is accepted.
- F2: CF > no-CF > transfer. no-CF + lipcon ≈ CF. c (transfer) + lipcon "OK".
- F3:
  - m03, b + lipcon + lock / + median: "better, almost no flicker".
  - old03: the same, slightly worse than m03, but better than prod.
  - old03 recipe v1: "prod AND the recipe explode everywhere; sticking a bit less with the recipe".
  - The user: the explosions are worst on m03 (mother), then old03.
- recipe v2 is NOT eye-checked yet: videos/m03_recipe_v2_05x.mp4.

## 4. The current best clean recipe (v2), all flags default OFF
Code: lipcon.py, lip_temporal.py, r8_fallback.py (lipcon flags), r7_m7_comp.sh (REP2_FB_ARGS second pass + LT_ARGS hook). The lipcon part is already wired by T in experiment/r15-tz 404e472 behind DUB_CLEAN_LIPCON=0. Add the LT_ARGS line. In e2e.sh face_comp, clean branch (DUB_ALLOW_NC=0, CFDV="", E2E_FX=1, E2E_SYNC=k5):
```bash
LCX="--thr-lipcon 0.45 --lipcon-mode adaptive --lipcon-rel 0.6 --lipcon-win 25 --lipcon-max-frac 0.03"
FBL=fb7sla; FBA="$FBA $LCX"; export REP_LABEL=rep2sla REP_ARGS="--spikes off" REP2_FB_ARGS="$(cat reports/experiments/r8-face/fb7.args) $LCX"
export LT_ARGS="--mode both --radius 3 --clamp 4 7 7 --median-gate 1.5 --mradius 2" LT_LABEL=bothLg15
```
The composite FC_EXTRA is unchanged (prod k5 + q2 + F3: `--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45`; EXTRA `--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5`).

CPU cost: lip_temporal is about 1.5–4 min per 40 s clip, plus lipcon passes about 1.5 min.

GPU gates not done for v2: m03 sync 8.32 is from the CPU SyncNet. The flashes, pressed lips and the rest are to be gated.

## 5. Audit tracker (briefs/audit_r15_face.md)
| item | status | commit / evidence |
|---|---|---|
| 1 retime sharpness pulse | open; old03 1.14 in every arm, `--retime-hf nearest` does not fix it; m03 / m18 1.02–1.07 | lip_flicker sharp_pulse |
| 2 CF aligned on source landmarks / iris | n/a for clean (CF off); generator side, not F's | – |
| 3 cm-mouth-excl mean offset | tested (`--cm-mouth-mode luma`): no flicker gain | aud arm |
| 4 tex-tau 1000 skin texture | tested (`--skin-texture off`): no gain | aud arm |
| 5 rigid head incl. brows | open, anchor_filters (not F's) | – |
| 6 steps switch per window | open, generator (not F's) | – |
| 7 boundary + bridges | tested (`--boundary off`): no gain | aud arm |
| 8 seg-joint never set | tested (`--seg-joint 3`): no gain | aud arm |
| 9a layer_sr in-place write | FIXED by default | 038a077 |
| 9b layer_sr truncated crop | fixed behind `--stable-box on` (needs a GPU SR A/B) | 038a077 |
| P1 r9_source_rigid decode_source | FIXED by default | d5bfb26 |
| P2 fallback joint phase | FIXED by default (`--joint-seg on`; m03 identical) | d5bfb26 + 038a077 |
| P3 cf_detail truncation | FIXED by default | d5bfb26 |
| P4 master_25fps symlink unchecked | FIXED by default (frame count + fps check) | 038a077 |
| S suspicious (repair / fallback / sanitize hysteresis, paste guard, re-encodes, raw-landmark masks) | open | – |

The aud arm = `--retime-hf nearest --cm-mouth-mode luma --skin-texture off --boundary off --seg-joint 3`: m03 0.865 / 107 vs prod 0.846 / 98; no gain alone or stacked with the lock.

## 6. Open problems
1. Explosions = LatentSync shape snaps. The composite median cuts them but costs sync. The targeted median is a compromise (m03 25 → 11 at −0.28 sync).
2. old03 needs the median the most (35 strong flashes in prod), and v2 is not run there yet; old03 also has the 14 % sharpness pulse.
3. m18 flicker is still 5× the source in jumps.
4. GPU gates for v2 on all 3 videos are not done, and neither are the F2 / F3 leftovers. The queue scripts are scripts/r15/gpu_queue*.sh, which lock only the GPU commands.
5. The CF w sweep (w 0.5 / 0.7) is half done: the cfd_w05 / w07 layers exist, but SR and the composite do not.

## 7. NEXT STEPS (ranked)
0. **LatentSync plain / registration-free paste A/B** (D's find, 06.11).
   - The m03 explosion the user saw at 8.64–8.71 s (source frames 259–261: torn, doubled lips with dark / teeth fragments) is born in the COMPOSITE. It is in every final (prod q2u, students, q2u + recipe) and in NO generator layer: q2pre / r6pre / r6raw / rep2s all show clean closed lips.
   - The source has an open mouth with teeth at 8.61–8.64 s, and the composite drags the clean closed mouth onto that open-mouth geometry.
   - Suspects: F3 `--reg-smooth 4 --mouth-region 0.45` and the k5 `--mouth-region 0.55` registration. D is testing F3="".
   - This matches the KeySync finding: per-frame registration / warps in the paste create floats and tears, and a plain inverse-crop paste removed them.
   - So: A/B the composite with registration off / the plain inverse-crop paste (comp_f2.sh F3="" and EXTRA without `--mouth-region`; fps_composite `--register` off).
   - Re-run explode.py with the stage list to confirm. My m03 attribution counted "composite / retime" events as weak, and this one may be among them: check the event list at 8.6–8.7 s.
1. Eye-check recipe v2 (videos/m03_recipe_v2_05x.mp4). Run v2 on old03 + m18 (CPU: comp_f2.sh with the LT_ARGS above), then the GPU gates on all 3 (gate_f.sh).
2. Fix the explosions AT THE SOURCE: the generator's shape snaps. Candidates (generator agents): steps hysteresis (audit 6), the student + lock (calmest m03: 0.19 / 28 jumps), a temporal consistency term / larger overlap at the windows.
3. If the composite has to carry it: a shape-aware easing. Only on snap frames whose opening jumps against BOTH neighbours AND where the dub audio does not have a plosive / onset there, so sync-relevant closures are kept.
4. old03 sharpness pulse: equalise the HF of interpolated vs raw retime frames (add HF on the interpolated frames, or soften the raw ones), then measure with lip_flicker sharp_pulse.
5. Flip the defaults in one batch once the gates and the eye pass: DUB_CLEAN_LIPCON=1 + LT_ARGS (clean); `--stable-box` after a GPU SR A/B.
6. The suspicious audit items, starting with the paste guard's hard switch and the repair / fallback hysteresis.

## 8. Contents of this folder
- r15-repair.bundle + HEAD.txt
- results/: every metric json / jsonl (lip flicker, flash, cpusync, jitter, explode events, gate face_gate / gate_vs_base, reports of F2 / F3)
- configs/: fb7.args, the exact env / flag lines
- videos/: the eye clips (0.5x mouth panels) of F2 / F3 and recipe v2
- MANIFEST.txt
