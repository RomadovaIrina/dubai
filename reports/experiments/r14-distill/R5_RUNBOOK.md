# r5 runbook: the 4-step student from teacher targets matched to production q2 (r15 lever 1)

**Why.** On 05.11 the user's eye preferred the r14 student (r2@1500 + c8, face units) over production on m03 ("правое лучше", "приятно выглядит"). Its remaining gap to q2 + F3 is 0.7–0.9 sync_speech.

**What changes.** The r14 teacher targets came from a teacher that is not what production runs: LatentSync 20 steps + CFG 2.0, no DeepCache. r5 regenerates the same 1817 windows with the exact production q2 generator settings:
- DDIM `--steps 10` on the speaking windows;
- CFG 2.0 (`scripts/r4/r4_video.sh` GEN default);
- DeepCache interval 5, branch 0 (`scripts/optim/latentsync_accel.py`).

It then retrains the 4-step student with the r14 recipe and evaluates it E2E.

**Approximation.** Production uses 20 steps on windows ≥ 25 % silent (`--steps-silent 20`). The captured windows carry no silence flag, so r5 uses 10 steps on all of them.

Server: a fresh 1 × RTX 5090 (r15-x). The whole chain is `scripts/r15/r5_chain.sh`: resumable, every step leaves `$X/done_<step>`, `X=/workspace/dub/r15x`.

## Files the new server needs (beyond the repo)

| what | size | from | put at |
|---|---|---|---|
| repo base: GitHub `RomadovaIrina/dubai`, branch `experiment/r13-spec` | – | GitHub clone (read access) | `/workspace/dub/dubai` |
| repo increment: `r15-distill.bundle` (`experiment/r14-distill` on top of origin r13-spec / r13) | ~0.1 MB | r14-d `/workspace/dub/r14d/r15-distill.bundle` (laptop watcher) | anywhere, then `git fetch` (below) |
| external training windows `cap_ext/` (1690 files, 1071 used) | 2.2 GB | laptop (staged by the coordinator) | `/workspace/dub/r15x/cap_ext/` |
| own training windows: `distill_cache.tar` (`cap/`, 1047 files, 746 used) | 1.6 GB | laptop | `tar xf distill_cache.tar -C /workspace/dub/r15x/own` → `/workspace/dub/r15x/own/cap/` |
| the r14 student checkpoint (control arm) `ckpt/r2_s4/step_01500_ema.pt` | 30 MB | r14-d `/workspace/dub/r14d/ckpt/r2_s4/step_01500_ema.pt` (copy it off before r14-d goes) | `/workspace/dub/r15x/ckpt/r2_s4/step_01500_ema.pt` |
| held-out test inputs old03 / m03 / m18 | 43 MB | laptop originals (r14-d `/workspace/dub/inputs/...`) | `/workspace/dub/inputs/test_videos_raw/03.mp4`, `/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV`, `/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4` |
| `/workspace/.env` (HF_TOKEN for the gated pyannote models, as on r14-d) | – | the coordinator | `/workspace/.env` (never printed) |

Everything else comes from the setup scripts. Measured on r14-d: ~25 min, ~23 GB models + ~15 GB venvs.
- That is about 40 GB of download: check the instance's $/TB traffic price first.
- Disk: ≥ 150 GB (the chain stops below 60 GB free).
- Not needed: `/workspace/dub/canonical`, `r8a` (created by runs), `data/` (raw external videos), and the old `train_s4/t` targets (r5 makes its own).

## Steps

```bash
# 1. repo: base from GitHub + the increment
git clone https://github.com/RomadovaIrina/dubai.git /workspace/dub/dubai && cd /workspace/dub/dubai
git fetch /workspace/dub/r15-distill.bundle experiment/r14-distill:experiment/r14-distill && git checkout experiment/r14-distill
# 2. stack (as r14-d; ~25 min, ~40 GB download)
bash scripts/r15/r5_setup.sh > /workspace/dub/setup.log 2>&1; grep -E "SETUP_FAIL|SETUP_CHAIN_DONE" /workspace/dub/setup.log
# 3. data (after the uploads above)
mkdir -p /workspace/dub/r15x/own && tar xf /workspace/dub/distill_cache.tar -C /workspace/dub/r15x/own
# 4. the chain (~5 GPU-h; log + notes in /workspace/dub/r15x)
mkdir -p /workspace/dub/r15x && setsid nohup bash scripts/r15/r5_chain.sh > /workspace/dub/r15x/r5_chain.log 2>&1 < /dev/null &
```

What the chain does, and its GPU cost:

| # | step | GPU | output |
|---|---|---|---|
| 0 | setup check: GPU, disk, venvs, LatentSync ckpt, `.env` present, inputs, control ckpt, `--teacher-deepcache` and `LS_FACE_UNITS` in the checkout | – | `R5_NOTES.txt` |
| 1 | link the 1817 r14 windows (`scripts/r15/r5_sel_list.txt`) into `$X/sel` | – | stops if any window is missing |
| 2 | teacher: `distill2.py teacher --teacher-steps 10 --teacher-cfg 2.0 --teacher-deepcache 5` | ≈ 0.5–1 h (the 20-step teacher took 2.9 h) | `train_r5/t/*.pt` (each records its `teacher` config) |
| 3 | smoke, 30 steps | 2 min | `smoke_r5.log` |
| 4 | train `r5_s4`: 2000 steps, val + ckpt every 250 | ≈ 1.6 h | `ckpt/r5_s4/` |
| 5 | pick: `ckpt_val.py --mse-tol 0.2`, i.e. opening ≥ 1.03, then of the checkpoints within 20 % of the lowest MSE the highest val sync | 0.2 h | `ckpt/r5_s4/ckpt_val.json` |
| 6 | E2E old03: q2u (q2 + F3, the baseline), s4p1500c8u (r14 student), r5c8u (pick), r5fc8u (final step 2000, if not the pick). The better r5 by old03 sync goes on. | ≈ 1.3 h | `eval/` |
| 7 | E2E m03 / m18: s4p1500c8u + the better r5 (baselines first) | ≈ 1 h | `eval/` |
| 8 | clips: r5 vs q2u and vs s4p1500c8u (full + 0.5x mouth; m03 t17-21, m18 t2-6) | – | `/workspace/dub/outbox/r15x/` |
| 9 | `R5_TABLE.md`: val per checkpoint + E2E (sync, flicker, opening, GPU s/vmin) | – | |
| 10 | commit (tables), `r5-distill.bundle` (incremental, verified), `R5_DONE` | – | |

**E2E settings.** Every E2E arm runs `LS_STRIDE 8 / continue` + `LS_FACE_UNITS=1`; the baseline q2u runs with `LS_FACE_UNITS=1` too. The control arm is re-run on the new server, so the comparison does not mix machines.

Total ≈ 4.5–5.5 GPU-h.

## Reading the result

- **r5 beats s4p1500c8u** on sync on ≥ 2 of 3 videos without a more closed mouth (opening mean ≥ 0.95 vs q2u, speed ≥ 0.95): the lever works. Look by eye next (m03 17–21 s, m18 2–6 s clips).
- **r5 is no better:** the teacher mismatch was not the gap. The next lever is data (DVIDS; needs the user's OK by name).
