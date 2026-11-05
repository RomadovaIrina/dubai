# r15-D: 6-step student (r4_s6) + m03 with the face-units fix

Server r14-d, branch experiment/r14-distill. Held out: old03 / m03 / m18.
Every E2E arm uses the q2 + F3 composite. Clips are in /workspace/dub/outbox/r15_d/. Checkpoints are in ckpt/r4_s6/ (every 250 steps, raw + EMA, ckpt_val.json).

## Eye verdicts this morning (05.11, input to this round)

- s4p1500c8rf (smart reference) is **rejected**: m03 "face flew off" at ~19 s and trembles more; m18 trembles and flickers more.
- s4p1500c8 (no smart reference): on m03 "sync better and LESS trembling than prod". On m18 at ~4 s the lips are a darker, smeared band than prod's clean open mouth.
- m03 19 s cause (agent F): a 1 s SPEAKER_01 diarization turn (the child, 19.15–20.21 s) made two-face mode lip-sync the child's face.
  - Fix: d494c67, LS_FACE_UNITS, default ON. It acts only with ≥ 2 faces.
  - Cherry-picked here as 90fd069. Accepted by the user's eye.

## r4_s6: the 6-step student

### Recipe

- The r2 recipe from scratch: sync_w 0.15, pix 0.5, lpips 0.2, lr 1e-4 cosine, EMA 0.999.
- Temporal term temp_w 0.5 **off the lips** (the r3 lip box). No opening term.
- `--student-steps 6`, 2000 steps (103 min, 3.1 s/step, 17.9 GB). Val + checkpoint every 250 steps.
- The teacher targets of train_s4 were reused as they are. Training is single-step: a random student timestep, with the teacher x0 re-noised by the stored z. Only the student schedule depends on the step count, so nothing was recomputed.

### Val (EMA, 24 held-out windows)

| step | MSE | stable-SyncNet sim | jitter | opening / teacher |
|---|---|---|---|---|
| 0 (untrained, 6 steps) | 0.0043 | 0.922 | 0.0297 | 0.89 |
| 250 | 0.0055 | 0.942 | 0.0312 | 1.07 |
| 500 | **0.0047** | 0.936 | 0.0312 | 1.07 |
| 750 | 0.0089 | 0.937 | 0.0303 | 1.12 |
| 1000 | 0.0067 | 0.944 | 0.0296 | 0.87 |
| 1250 | 0.0067 | 0.946 | 0.0298 | 0.93 |
| 1500 | 0.0070 | 0.946 | 0.0299 | 0.95 |
| 1750 | 0.0067 | 0.946 | 0.0297 | 0.94 |
| 2000 | 0.0064 | 0.946 | 0.0297 | 0.945 |

The pick rule (opening ≥ 1.03, then lowest MSE) chose step 500. Its val sync is the weakest of the run, so the final step 2000 was evaluated on old03 too (s6fc8).

### End-to-end

| video | arm | sync_speech | flicker pops (src p99) | opening vs q2 (mean / speed) | GPU s / video-min |
|---|---|---|---|---|---|
| old03 | q2 + F3 | 7.487 | 9 | 1 | 401 |
| old03 | s4p1500c8 (r14 student) | 6.767 | 6 | 0.91 | 292 |
| old03 | **s6c8** (r4@500 + c8) | 5.820 | 1 | 0.89 / 0.82 | 373 |
| old03 | s6fc8 (r4@2000 + c8) | 6.373 | 1 | 0.91 / 0.86 | 350 |
| m18 | q2 + F3 | 8.388 | 8 | 1 | 374 |
| m18 | s4p1500c8 | 7.557 | 8 | 0.98 | 338 |
| m18 | **s6c8** | 6.104 | 8 | 0.92 / 0.82 | 389 |

m03 was not run with the 6-step student: the coordinator dropped it after old03 / m18.

**Result: the 6-step student is worse than the 4-step one on every video**: −0.4 to −1.5 sync, the mouth more closed and about 15–18 % slower. It also costs 350–389 vs 292–338 GPU s / video-min, over the 300 norm.
- The val proxies did not predict this. Step 500 had the best val MSE of any student so far (0.0047 vs r2@1500 0.0061).
- The pick rule's checkpoint was the worse of the two.
- Dropped (coordinator, 05.11).

## m03 with the face-units fix (LS_FACE_UNITS=1)

| arm | sync_speech (before the fix) | flicker pops (src 13) | opening vs q2u (mean / speed) | near-closed voiced | GPU s / video-min |
|---|---|---|---|---|---|
| q2u = q2 + F3 | **8.565** (8.162) | 12 | 1 | – | 341 |
| s4p1500c8u = r14 student r2@1500 + c8 | **7.621** (7.307) | 13 | 0.97 / 1.03 | 14 % | 311 |

Both arms gain +0.3 / +0.4 sync: the child-face bug was pulling every m03 number down.
- The student's mouth now opens and moves like prod's.
- The student still trails prod by 0.94 sync on m03 and 0.83 on m18, and by 0.72 on old03.

Clips:
- `m03_{mouth,full}_q2u_vs_s4p1500c8u[_half_speed].mp4`;
- the 17–21 s span at 0.5x: `m03_{mouth,full}_q2u_vs_s4p1500c8u_t17-21_half_speed.mp4`;
- old03 / m18 s6c8 vs q2 and vs s4p1500c8, incl. `m18_*_t2-6_half_speed.mp4`.

## Incidents

- **The first s6c8 E2E (09:29) failed on all 3 videos** (audio_failed, VRAM peak 32 GB). Another agent's 13 GB face stage ran at the same time.
  - dabai_job cannot hold gpu.lock for a whole run: e2e.sh takes the lock inside for sep / gaps, so a nested lock would deadlock.
  - The chain now waits for an idle GPU (< 2 GB used, no /workspace/dub/r15t/E2E_RUNNING, gpu.lock free) and writes /workspace/dub/r15d/E2E_RUNNING while a video runs.
  - The failed outputs are in r14d/failed_0933.
- **GPU yield for agent F** after training: 14 min, ended by F's GPU_DONE.

## Next student lever (no training today)

The best student, r2@1500 + c8 (4 steps), costs 292–338 GPU s / video-min. It trails q2 + F3 by 0.7–0.9 sync. The opening loss terms (r3 / r3b), the smart reference and more steps (r4_s6) all failed to close the gap. Two levers are left:

1. **Targets matched to production q2**
   - **Idea.** The current teacher is LatentSync 20 steps + CFG with the default reference. What the user accepts is the q2 stack: its step count, CFG, DeepCache, reference handling and stride. Regenerating the 1817 targets with exactly the q2 generator settings makes the student imitate the accepted output, not a different teacher.
   - **Cost (1 × 5090):**
     - targets ≈ 1.5–3 h (the 20-step teacher took 2.9 h; q2 uses fewer steps);
     - training 1500 steps ≈ 1.2 h;
     - ckpt_val 0.2 h;
     - E2E on 3 videos ≈ 1 h;
     - **total ≈ 4–5.5 GPU-h**, no download.
   - **Do first.** Cheapest, and it tests the hypothesis directly.
2. **More and cleaner data**
   - **Idea.** Today: 1817 windows from 11 own clips + NASA public-domain interviews. Add about 2× more frontal talking-head windows from US-government public-domain video (DVIDS, more NASA), filtered by the same face / sharpness / single-speaker gates.
   - **Cost:**
     - download ~20–40 GB — vast bills traffic, so check the instance's $/TB first;
     - CPU prep ~1–2 h;
     - teacher targets for ~2000 more windows ≈ 3–3.5 GPU-h (half with q2-matched targets);
     - training 3000 steps ≈ 2.4 h;
     - eval ≈ 1 h;
     - **total ≈ 6.5–8 GPU-h + traffic**.
   - **Approval.** DATA_SOURCES.md is approved for clean datasets (04.11). Confirm DVIDS by name with the user before downloading.
   - **When.** Only if lever 1 helps but does not close the gap.

Training rules for any next round:
- r2 recipe + temp off the lips;
- ≤ 1500 steps;
- pick by E2E on old03, not by the val proxies alone (the r4_s6 lesson).

## Простыми словами (для Ирины)

- Проверили ученика на 6 шагов вместо 4: он оказался хуже. Губы хуже попадают в звук, рот прикрыт сильнее, а GPU тратится больше. Это направление закрыли.
- Исправление агента F для m03 (в 19 с модель двигала губы ребёнку) подняло синхрон и у продакшена, и у ученика.
- Лучший ученик по-прежнему с шага 1500 на 4 шага + шаг 8. На m03 его рот теперь открывается и двигается как у продакшена. Синхрон чуть ниже продакшена (7,6 против 8,6), зато GPU тратится меньше. Клипы m03 17–21 с в замедлении лежат в outbox/r15_d.
- Следующая идея: переучить ученика на примерах, сделанных ровно с теми настройками продакшена, которые ты одобрила. Это около 5 часов GPU, без новых скачиваний. Новые данные (DVIDS) — потом и только с твоего согласия.
