# r13 SPEC agent: HANDOFF (04.11.2026, r13-t wrapped up)

The branch `experiment/r13-spec` holds everything. r13-t (vast.ai RTX 5090) is destroyed after this push, so **run data that is not in git is gone** except what the coordinator pulled to the laptop (see "Kept files" at the end).

## 1. Round 3 status
| item | state | where |
|---|---|---|
| Wall-clock cost (timeline, single vs queue, $/vmin) | ✅ done | WALLCLOCK.md, wallclock_timeline.png |
| Early release `R10_GEN_EARLY_RELEASE=1` | ✅ measured: wall 535 → **490 s/vmin**, $0.098 / vmin; the video differs from the plain queue only at run-to-run noise level (PSNR 54–58 dB; a plain single-job repeat is itself 54.5 dB) | WALLCLOCK.md; code 3a2080f (gen_worker.py + e2e.sh), default off |
| Edge robustness suite | 🟡 15 / 16 OK; res4k_vertical = face_gen_failed (LS agent); long_10min not run | ROBUSTNESS.md |
| OpenAI readiness | ✅ | OPENAI_READINESS.md |
| TZ report in Russian | ✅ (as of 04.11) | TZ_REPORT_RU.md |

## 2. Distillation pilot (03.11): done
- 4-step LoRA student of LatentSync 1.6, without CFG.
- SyncNet: m03 7.98 / m18 7.96 vs k5 8.16 / 8.35; old03 6.43 vs 7.58.
- Face GPU −35 %. The mouth is less articulate.
- Full report: DISTILL_PILOT.md. The user: "Да, доучить", "студент тоже неплохо справляется".

## 3. Student ROUND 2: the exact plan for tomorrow
Code is ready and committed but **not smoke-tested**: a 30-step smoke was started and killed at the wrap-up. Run the smoke first.
- `scripts/r13/distill/distill2.py`:
  - teacher: batched (`--tbatch 4`, same-length windows per UNet call) + `--flip` (each window again horizontally flipped; the teacher is run on the flipped conditioning);
  - train: cosine lr with 100 warmup steps, EMA 0.999 with warm-up, x0 MSE + 0.15 × stable-SyncNet + 0.5 × L1 + 0.2 × LPIPS(squeeze) on the decoded mouth half; `--student-steps 4|6`;
  - checkpoints every 500 steps: `/workspace/dub/r13s/distill/ckpt/<run>/step_XXXXX.pt` (live LoRA, fp16, ~30 MB), `step_XXXXX_ema.pt`, `step_XXXXX.json` (step, val MSE / SyncNet, the full config) and `best.pt` (the EMA with the lowest val MSE).
    - **Tell the coordinator each path as it appears**: the user wants every checkpoint on the laptop (`C:\repos\chkpts`).
- `scripts/r13/distill/student_gen.py`: `R13_STUDENT_STEPS` (default 4) sets the student's step count in the production generator.
- **Python:** the repo env (`. scripts/env.sh`), not `/venv/main` (that one has no omegaconf / lpips).

### Data (user-approved 04.11: "разрешаю использовать чистые датасеты на обучение")
| source | licence | what | size |
|---|---|---|---|
| our own inputs | own | the 11 non-test clips (746 windows already captured on r13-t: **lost with the server**, recapture ≈ 30–40 min) | – |
| GRID, https://zenodo.org/records/3625687 | CC BY 4.0 | 33 speakers, frontal studio; per speaker `sN.zip` 340–470 MB; `alignments.zip` 19 MB; `audio_25k.zip` 2.5 GB (not needed: the video zips carry audio) | ~13 GB of video zips |
| EC Audiovisual Service | CC BY 4.0 / © European Union, **only files whose page states it** | a few hours of press conferences / speeches | a few GB |
| NASA Image & Video Library | US public domain | interviews | a few GB |
- Keep `reports/experiments/r13-spec/DATA_SOURCES.md` with the URL, licence and attribution line per item. Not yet written: nothing was downloaded.
- Traffic is billed (~$2.7 / TB): fine, but **≤ 50 GB without asking**.
- **old03 / m03 / m18 never enter training.**

### Filtering (the LatentSync data-pipeline idea, with our commercially clean stack)
1. One face, MediaPipe (as production), yaw / pitch within the production pose precheck.
2. **Face size:** the face crop ≥ ~160 px. ⚠️ **GRID is 360×288 with ~100–130 px faces: most of it is expected to fail LatentSync's own resolution filter (≥ 256).** It still adds speaker / mouth-shape variety at low res. Test one speaker zip first (~400 MB), check the face size and the teacher output, then decide how much of it to use.
3. **Sync filter on the ORIGINAL clip:** `scripts/r11/sync_speech.py`, keep offset |≤ 1| frame and confidence ≥ 3 (LatentSync's threshold).
4. Visual quality: skip HyperIQA (licence not checked); drop clips with heavy blur / compression by our own rules.

### Capture with the original audio
- External clips have no dub: run the generator with the clip's **own** audio. That is LatentSync's own training setup: reconstruct the mouth for the real audio.
- `capture_gen.py` runs inside `r7_e2e.sh`, which expects an AUDIO dir (`$A/r9_<id>`). For external data, either build a minimal AUDIO dir holding the original 16 k wav as the "dubbed" track, or call `candidate_video_stage.py` through `capture_gen.py` directly. Check `r4_video.sh` for the exact arguments.
- **GRID:** concatenate one speaker's 3-second clips into ~60 s videos (one generator run per speaker; the ~40 s fixed overhead per run kills 3 s clips).

### Budget (≈ 5 h GPU)
| step | GPU h |
|---|---:|
| recapture of own clips | 0.6 |
| capture of external data | 0.5 |
| teacher, ~2200 windows (own 746 + flips + ~1000 external; ~4 s / window batched; measure in the smoke) | 2.4 |
| training, 3000 steps at ~1.5 s | 1.3 |
| eval: 3 E2E + SyncNet + flashes | 0.5 |
| **total** | **~5.3** |
- A 6-step student in addition costs +1.5 h (training + eval). Only if the 4-step result leaves time.
- If time is short, drop the flips of the external windows first.

### Eval (as in the pilot, plus the user's eye)
- Full E2E via `dabai_job` with `debug_env {R10_GEN_PY: student_gen.py, R13_STUDENT_LORA: …/best.pt, R13_STUDENT_STEPS: 4}` on old03 / m03 / m18.
- `sync_speech` SyncNet, `r3_flash` strong flashes, face GPU s/vmin.
- Full-length 2-panel k5 | student, plus the 0.5× mouth crop (`mouth_2panel.py`).
- Watch for lopsided / grotesque mouths (the user rejected the LS agent's uneven 3/4 mouth on m03).

## 4. Other open items (from rounds 1–3)
1. res4k_vertical face_gen_failed → the face agent.
2. long_10min edge case: not run.
3. The rotated clip's cosmetic `face_partly_absent` flag (the precheck reads unrotated frames).
4. The early-release hook: the LS agent merges it, then `R10_GEN_EARLY_RELEASE=1` goes on in the worker.
5. The OpenAI key (6.5 / 7.4 / 7.5), the SyncNet contract threshold (0.8), the old03 pace-fix A/B: the user's decisions.

## Kept files (pull to the laptop before the server dies)
- Pilot LoRA: `/workspace/dub/r13s/distill/lora.pt` (+ `train_log.jsonl`, `val.json`): already on the laptop per the coordinator.
- Pilot clips: `/workspace/dub/outbox/r13_spec/distill/*.mp4` (also in the laptop notes folder `r13-spec-clips/distill/`).
- Round-2 checkpoints: none yet.
