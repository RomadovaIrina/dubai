# Agent D: LatentSync 4-step student. Final hand-off (r14 → r15, 04–06.11.2026)

**Branch:** `experiment/r14-distill` (repo RomadovaIrina/dubai). Its head is in `BRANCH_HEAD.txt`, and the bundle `r14-distill-final.bundle` sits next to this file. The bundle is incremental on top of `origin/experiment/r13-spec` and `origin/experiment/r13`.

## Итог простыми словами

- **Ученик r6 почти не уступает продакшену.** Это быстрая модель на 4 шага, обученная с CFG-дистилляцией. По синхрону губ он не хуже продакшена: old03 8,18 против 7,81, m03 8,78 против 8,78, m18 8,11 против 8,33. При этом GPU тратится примерно на 13 % меньше: около 263 с на минуту видео против примерно 300.
- **Глаз Ирины на r6@1250:**
  - клипы m18 2–6 с, old03 3–7 и 23–25 с, m03 17–21 с — «норм»;
  - единственный дефект — «взрыв во рту» на m03 около 8,5 с.
- **Этот «взрыв» сделал не ученик.** Слои генератора чистые, губы сомкнуты. Губы рвёт наложение в кадрах 8,64–8,71 с, когда в исходнике рот открыт, а в дубляже закрыт. Тот же разрыв есть и в продакшене q2u. Чинить его нужно в наложении, у агента F.
- **Что не сработало:**
  - ученик на 6 шагов;
  - штрафы за открытие рта;
  - учитель с настройками продакшена без CFG-дистилляции (r5);
  - «умный» опорный кадр для ученика.

## Arms and E2E

All arms ran on the same server, r15-w (1 × RTX 5090), with q2 + F3 composite, `LS_STRIDE 8 / continue` (c8) and `LS_FACE_UNITS=1`. Each cell: sync_speech / flicker pops / mouth opening vs q2u (mean, speed) / GPU s per video-minute.

| arm | old03 | m03 | m18 |
|---|---|---|---|
| q2u = production q2 + F3 | 7.805 / 15 / 1 / 305 | 8.779 / 11 / 1 / 286 | 8.331 / 8 / 1 / 326 |
| s4p1500c8u = r14 student (r2@1500) | 7.099 / 1 / 0.91, 0.98 / 257 | 7.586 / 12 / 0.98, 0.97 / 262 | – |
| r5fc8u = q2-matched teacher, no CFG term | 7.056 / 0 / 0.92, 0.90 / 260 | 7.322 / 11 / 0.97, 0.88 / 274 | – |
| r6c8u = r6@250 (the val pick) | 7.929 / 8 / 1.01, 0.98 / 262 | 8.467 / 12 / 1.03, 0.99 / 264 | 8.202 / 8 / 1.07, 0.93 / 314 |
| r6@500 | 8.075 / 11 / 0.99, 1.01 / 263 | 8.712 / 12 / 1.03, 0.95 / 264 | – |
| **r6@1250 = the candidate** | **8.175** / 16 / 1.01, 1.00 / 263 | **8.784** / 12 / 1.05, 0.91 / 262 | **8.105** / 7 / 1.06, 0.94 / 308 |

With F's lip recipe (CF off + adaptive lipcon + lip colour lock + median). Each cell: sync / lip-flicker W excess / W jumps [source's jumps]:

| arm | old03 | m03 |
|---|---|---|
| q2u + recipe | 6.950 / 0.007 / 42 [28] | 8.325 / 0.219 / 32 [13] |
| r14 student + recipe | 6.194 / −0.057 / 25 | 7.352 / 0.192 / 29 |
| r6@250 + recipe | 7.053 / −0.094 / 25 | 8.180 / 0.172 / 30 |

The recipe lowers sync_speech in every arm: q2u old03 7.81 → 6.95, and the old03 measure covers only 53 s instead of 59 s. F should check this.

Earlier rounds, all on r14-d (numbers are not comparable across servers; r15-w reads about +0.3):
- r2 (4-step, temp_w 1.0): it damped the mouth; the early stop at step 1500 fixed most of that.
- r3: the one-sided opening hinge was gamed.
- r3b: the symmetric opening term was too weak.
- r4_s6: 6 steps, worse than 4.
- The smart reference was rejected by eye: tremble, and the face "flew off".

Full tables: `metrics/R6_TABLE.md`, `metrics/R6_RECIPE_TABLE.md`, `metrics/R5_TABLE.md`. Earlier rounds: `reports/experiments/r14-distill/{NIGHT_REPORT,R15_REPORT}.md` in the repo.

## What made it work: CFG distillation (r6)

The q2-matched teacher targets (r5) alone did not help. The r5 student was as good as the r14 one, with slower mouth motion.

The CFG term does help. At every sampled student timestep, the student's ε is pulled toward the frozen teacher's guided ε at the same x_t:
- the teacher is the same UNet with the LoRA scaled to 0;
- guidance 2.0, unconditional branch = zero audio, as in production.

This restored the guided mouth dynamics that the x0 target alone loses. It costs nothing extra at inference (4 steps, no CFG) and ~0 % extra training time.

The val proxies (MSE / opening / stable-SyncNet) do **not** predict E2E. The rule-based pick (step 250) was the weakest of 250 / 500 / 1250. **Take the final step** and confirm by E2E on old03.

## Reproduce r6 (1 × 5090, ≈ 3.3 GPU-h incl. eval)

The runbook is `reports/experiments/r14-distill/R5_RUNBOOK.md`: setup, data, the file list. r6 needs the q2-matched teacher targets from r5 step 2.

```bash
# worktree of experiment/r14-distill with models / third_party / test_videos symlinked from the main checkout
cd /workspace/dub/dubai-d
# 1. q2-matched teacher targets (29 min): r5_chain.sh steps 0-2 (check, link 1817 windows from scripts/r15/r5_sel_list.txt, teacher)
R5_STEPS=1500 R5_IDS=m03 bash scripts/r15/r5_chain.sh     # or only its teacher step:
python scripts/r13/distill/distill2.py teacher --cap /workspace/dub/r15x/sel --out /workspace/dub/r15x/train_r5 --tbatch 4 \
  --teacher-steps 10 --teacher-cfg 2.0 --teacher-deepcache 5
# 2. r6 = train + pick + E2E old03 / m03 + F's lip recipe (scripts/r15/r6_chain.sh, train_r6/t -> train_r5/t)
R6_STEPS=1250 R6_CFG_W=5 bash scripts/r15/r6_chain.sh
#    the train command it runs:
python scripts/r13/distill/distill2.py train --out /workspace/dub/r15x/train_r6 --student-steps 4 --sync-w 0.15 --pix-w 0.5 --lpips-w 0.2 \
  --temp-w 0.5 --temp-nolips --lip-box 0.25,0.538,0.747,0.787 --ema 0.999 --lr 1e-4 --p16 0.7 --cfg-w 5 --cfg-g 2.0 \
  --steps 1250 --val-every 250 --val-max 24 --run r6_s4 --ckpt-root /workspace/dub/r15x/ckpt
# 3. production use: the student merged into the q2 generator (dabai_job debug_env / env)
R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=<ckpt>/r6_s4/step_01250_ema.pt R13_STUDENT_STEPS=4 \
  LS_STRIDE=8 LS_OVERLAP=continue LS_FACE_UNITS=1   # + the q2 + F3 composite env of scripts/r14/eval_r14.sh (COMMON)
# 4. eval one arm: Q2=q2u Q2ENV='"LS_FACE_UNITS": "1"' EVAL_DIR=... OB=... LORA=<ckpt> STEPS=4 TAG=<tag> \
#    XENV='"LS_STRIDE": "8", "LS_OVERLAP": "continue", "LS_FACE_UNITS": "1"' bash scripts/r14/eval_r14.sh old03 m03 m18
```

Checkpoints are listed in `CHECKPOINTS.txt` (size, sha256 prefix):
- r6_s4: all steps, raw + EMA + json, `ckpt_val.json`;
- r5_s4;
- the r14 control r2@1500.

They are already on the laptop: `C:\repos\chkpts\latentsync_student_r15_r6_s4` (r6) plus the earlier r2 / r3 / r3b / r4_s6 / r5. Each LoRA is ~30 MB (rank 32, fp16), and its step json holds the full training config.

Data:
- Training windows: 1817 = 746 own (`distill_cache.tar`, 1.6 GB) + 1071 NASA public domain (`cap_ext`, 2.2 GB). Both are on the laptop; the list is `scripts/r15/r5_sel_list.txt`.
- The r5 teacher targets (3.5 GB) are not copied, only indexed in `train/teacher_targets_r5_index.txt`. They regenerate in 29 min.
- Sources and licences are in `reports/experiments/r14-distill/DATA_SOURCES.md`.

## Open problems

1. **m03 "explosion" at 8.64–8.71 s.**
   - Seen in: every final, production included: q2u, r6, the r14 student, and with or without the recipe.
   - Not seen in: any generator layer; ls_pre, ls_raw and the SR / repair layers at master frames m217–218 are all clean.
   - The source mouth is open (teeth) at 8.61–8.64 s and closes by 8.68. The composite stretches the closed dubbed lips onto the open-mouth geometry and tears them.
   - None of the metrics catches it: F's explode.py (single-frame pop), multi-frame pop, texture burst, divergence from q2, darkness vs the layer.
   - The tiles in `analysis/` show it. The F3-off composite test was started and then dropped on the user's call.
2. Step choice: val ≠ E2E. A reliable val proxy is missing, so it is E2E-picked for now.
3. The F lip recipe lowers sync_speech, and its old03 measure covers 53 s instead of 59 s.
4. r6@1250: old03 has more flicker pops (16 vs prod 15 vs r6@250 8), and the m03 mouth speed is ×0.91. The user did not object.

## NEXT STEPS (ranked)

0. **(F) The composite tear on m03 8.64–8.71 s**, the "explosion": shared with F's list.
   - First test: the composite of the same layers with the F3 registration off (`F3=" "` for comp_f2.sh; `F3=""` falls back to the default!), then `--mouth-region` off.
   - Then a warp-tear detector: final vs layer at round(mu), on edges inside the lip hull.
   - Files: `analysis/boom_tiles2.jpg` and `boom_tiles3.jpg` (source | q2u | r6 | master | layers).
1. **Ship r6@1250 as the face-generator candidate** behind a flag (`R13_STUDENT_LORA` / `R13_STUDENT_STEPS=4` + c8 + face units). Run the release gates on old03 / m03 / m18 + 2–3 fresh videos, including two-face and long ones.
2. **Longer / extra r6 training.** Sync was still rising at 1250 on old03 / m03. Try 2000–2500 steps and cfg_w 5 vs 10, and pick by E2E on old03 + m03. ~2 GPU-h + eval.
3. **TensorRT / compile for the 4-step student.** Other agents measured KeySync fast4 + TRT at ~215; the same could bring r6 well under 250 s/vmin.
4. A CFG-distilled student from more data (DVIDS / more NASA; needs the user's OK by name; ~7 GPU-h + traffic). Only if fresh videos show a gap.
5. The val proxy: a 4-step decode + SyncNet on longer windows, or a mini-E2E on two 10 s clips per checkpoint, so the step choice stops needing full E2E.

## Contents of this folder

- `FINAL_REPORT.md`: this report.
- `BRANCH_HEAD.txt`, `r14-distill-final.bundle`: the code.
- `CHECKPOINTS.txt`: the r6 / r5 / r2@1500 checkpoints with sizes and hashes (the files themselves are on the laptop).
- `train/`: the r5 / r6 training logs and val, plus the index of the teacher targets.
- `logs/`: the chain logs and notes (R5_NOTES, R6_NOTES).
- `metrics/`: the tables, per-arm sync / flicker / mouth JSON, lip_flicker, explode, and the explosion-hunt JSON.
- `analysis/`: the mouth tiles of the m03 8.3–8.9 s explosion.
- `videos/clips/`: the eye clips (2-panels full / mouth / 0.5x, eye spots, recipe).
- `videos/finals/`: the final mp4s of q2u / r14 student / r5 / r6@250 / 500 / 1250 per video.
- `MANIFEST.txt`: every file with its size.
