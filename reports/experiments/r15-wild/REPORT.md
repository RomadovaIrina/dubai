# r15 agent W (wild track): TBDub bake-off, best-of-N per window, keyframes + interpolation

Server r15-w (RTX 5090, Taiwan), branch `experiment/r15-wild` (base `experiment/r14-distill` d189fb5). Held out: old03 / m03 / m18.

**Method**
- **Same dub everywhere.** Every arm runs on the dub of the q2 + F3 baseline (`artifacts/r4a/r9_q2_<id>`). The baseline itself is the one-command E2E via dabai_job (`scripts/r15/base_q2.sh`).
- **Same face path everywhere.** Every arm runs the face path of `e2e.sh` under dabai_job (`scripts/r15/arm.sh`):
  - E2E_FX=1, E2E_SYNC=1 (= k5), the R10_FAST_FACE stack and the q2 + F3 flags;
  - only the generator changes.
  - Identity check: arm `q2r` = q2 through `arm.sh` reproduces the E2E. SyncNet old03 7.80 vs 7.84; m18 8.308 vs 8.308.
- **GPU s/vmin** = NVML per-job GPU-busy seconds of the WHOLE face stage (face_gen + face_comp, model loads and cold start included) per video-minute. The E2E adds the audio stage, ~50 GPU-s/vmin on m03.
- **Metrics:**
  - SyncNet = `scripts/r11/sync_speech.py` (the gate's);
  - flicker / pops = `scripts/r14/flicker_metric.py`;
  - mouth opening = `scripts/r14/mouth_open.py`.
- **Clips** in `/workspace/dub/outbox/r15_w/`:
  - `<id>_mouth_<L>_vs_<R>_spots_half_speed.mp4` = the user's eye spots at 0.5x;
  - `<id>_mouth_<L>_vs_<R>.mp4` = the mouth 2-panel, full length;
  - `<id>_full_<L>_vs_<R>.mp4` = side by side, full length.

## TL;DR

1. **The wild winner is W2's teacher variant, tkf2.**
   - What it is: LatentSync q2 (the teacher) generating every 2nd frame (each keyframe with its own audio window). The in-between mouths are interpolated in the stabilised 512 crop space (DIS optical flow), lower face only, before paste-back.
   - SyncNet 7.45 / 8.02 / 8.12. q2 + F3 has 7.84 / 8.23 / 8.31 (−0.40 / −0.20 / −0.19).
   - Cost 145 / 154 / 192 face GPU-s/vmin. q2 costs 234 / 229 / 279 (−35 %).
   - So it beats the student s4p1500c8 on BOTH axes on all 3 videos: 7.07 / 7.19 / 7.49 at 194 / 218 / 231.
   - E2E ≈ face + ~50 audio ≈ 200–245, under the 300 norm, with the teacher.
   - **Needs the eye at 0.5x**, because LS_RATE was rejected for tremble. Unlike LS_RATE, head motion is never interpolated here (every frame keeps its own affine and upper face). Pops are 20 / 12 / 8 vs q2 14 / 11 / 8.
   - Clips: `<id>_mouth_q2_vs_tkf2_spots_half_speed.mp4`, `<id>_mouth_s4c8_vs_tkf2_spots_half_speed.mp4`.
2. **TBDub (W0) is not a drop-in replacement.**
   - Its own output ("native": its own paste-back, 25 fps) has SyncNet 7.62 / 7.80 / 6.34 (−0.22 / −0.43 / −1.97 vs prod).
   - Through our post-chain it drops to 6.1–6.3 / 5.8–6.2 / 5.5–5.8. Our fps_composite costs its layer −1.2 vs −0.5 for LatentSync; see W0.
   - Cheapest generator here: 140–190 face GPU-s/vmin, but 650 s/vmin wall per segment (CPU overhead).
   - The eye can judge the native clips `<id>_mouth_q2_vs_tbnat_spots_half_speed.mp4`. Even if they look good, using TBDub needs a composite re-tuned for a generator that redraws the whole lower face.
3. **Best-of-N (W1) is not worth it.**
   - SyncNet vs s4p1500c8: +0.12 / 0.00 / −0.02 (bon3), at 503–523 GPU-s/vmin.
   - The early-pick variants: bon3e 339–385, bon2e 289–339; SyncNet within ±0.1 of s4c8.
   - The lazy-mouth fix seen on old03 (opening ×1.00 vs ×0.91) does not hold on m03 / m18.
   - Picks ≠ seed 0 in 20–91 % of windows. A fixed better noise (`LS_NOISE_SEED`, the most-picked candidate) is not generic either: +0.13 / −0.09 / +0.02.
4. **Plain keyframes on the student (W2 option a) trade too much sync.**
   - kf2: −0.4…−0.5 SyncNet at −35 % GPU.
   - kf3: −1.7…−1.8 at −45 %.
   - My option (b), voiced-adaptive keyframes (kfv3: dense on dub-voiced frames), keeps or slightly raises SyncNet (+0.07 / +0.22 / +0.11) but saves nothing on these speech-dense clips (177 / 218 / 247).

## Results (`/workspace/dub/r15w/eval/TABLE.md`)

| video | arm | SyncNet | flicker mean | pops (src p99) | flicker @ eye spots | mouth opening vs q2 (mean / speed) | near-closed voiced | face GPU s / vmin |
|---|---|---|---|---|---|---|---|---|
| old03 | q2 | 7.841 | 0.0060 | 14 | 0.0057 | 1.0 / 1.0 | 0.141 | 233.8 |
| old03 | q2r | 7.804 | 0.0060 | 16 | 0.0057 | 0.999 / 1.014 | 0.141 | 233.8 |
| old03 | s4c8 | 7.065 | 0.0057 | 1 | 0.0054 | 0.914 / 0.993 | 0.188 | 193.5 |
| old03 | s4c8n2 | 7.191 | 0.0057 | 0 | 0.0055 | 1.073 / 0.947 | 0.133 | 196.7 |
| old03 | tb | 6.275 | 0.0060 | 41 | 0.0057 | 1.026 / 1.122 | 0.147 | 160.3 |
| old03 | tbw | 6.103 | 0.0056 | 6 | 0.0054 | 0.975 / 1.048 | 0.176 | 139.8 |
| old03 | tbnat | 7.624 | 0.0072 | 177 | 0.0064 | 0.77 / 1.195 | 0.393 | – |
| old03 | bon3 | 7.189 | 0.0057 | 1 | 0.0055 | 1.005 / 0.971 | 0.143 | 504.6 |
| old03 | bon3e | 7.172 | 0.0057 | 1 | 0.0054 | 0.965 / 0.98 | 0.161 | 338.9 |
| old03 | bon2e | 7.088 | 0.0057 | 1 | 0.0054 | 0.925 / 0.989 | 0.177 | 288.6 |
| old03 | kf2 | 6.551 | 0.0053 | 0 | 0.0050 | 0.87 / 0.943 | 0.228 | 122.9 |
| old03 | kf3 | 5.414 | 0.0050 | 0 | 0.0048 | 0.864 / 0.864 | 0.228 | 101.6 |
| old03 | kfv3 | 7.137 | 0.0056 | 3 | 0.0053 | 0.913 / 0.973 | 0.192 | 177.3 |
| old03 | tkf2 | 7.446 | 0.0057 | 20 | 0.0057 | 0.981 / 1.03 | 0.157 | 144.9 |
| old03 | tbnc | 6.266 | – | – | – | – / – | – | 227.7 |
| old03 | tbmr0 | 6.328 | – | – | – | – / – | – | – |
| m03 | q2 | 8.225 | 0.0098 | 11 | 0.0084 | 1.0 / 1.0 | 0.156 | 229.2 |
| m03 | s4c8 | 7.191 | 0.0095 | 11 | 0.0083 | 0.978 / 0.965 | 0.152 | 217.8 |
| m03 | s4c8n2 | 7.105 | 0.0095 | 10 | 0.0083 | 0.959 / 0.958 | 0.150 | 216.7 |
| m03 | tb | 6.236 | 0.0111 | 23 | 0.0095 | 1.079 / 1.039 | 0.114 | 168.9 |
| m03 | tbw | 5.837 | 0.0110 | 24 | 0.0097 | 1.065 / 1.051 | 0.116 | 154.5 |
| m03 | tbnat | 7.797 | 0.0171 | 156 | 0.0220 | 0.963 / 1.391 | 0.243 | – |
| m03 | bon3 | 7.192 | 0.0095 | 10 | 0.0087 | 0.951 / 0.951 | 0.162 | 502.7 |
| m03 | bon3e | 7.167 | 0.0095 | 10 | 0.0086 | 0.962 / 0.924 | 0.157 | 357.8 |
| m03 | bon2e | 7.230 | 0.0095 | 10 | 0.0083 | 0.967 / 0.94 | 0.148 | 300.7 |
| m03 | kf2 | 6.756 | 0.0093 | 11 | 0.0083 | 0.955 / 0.883 | 0.166 | 141.6 |
| m03 | kf3 | 5.355 | 0.0091 | 13 | 0.0082 | 0.957 / 0.795 | 0.148 | 119.6 |
| m03 | kfv3 | 7.408 | 0.0095 | 10 | 0.0083 | 0.984 / 0.953 | 0.148 | 218.4 |
| m03 | tkf2 | 8.022 | 0.0097 | 12 | 0.0084 | 1.006 / 0.956 | 0.160 | 153.9 |
| m18 | q2 | 8.308 | 0.0152 | 8 | 0.0153 | 1.0 / 1.0 | 0.570 | 278.9 |
| m18 | q2r | 8.308 | 0.0152 | 8 | 0.0153 | 1.0 / 1.0 | 0.570 | 279.2 |
| m18 | s4c8 | 7.487 | 0.0153 | 8 | 0.0154 | 0.975 / 0.902 | 0.568 | 231.0 |
| m18 | s4c8n2 | 7.510 | 0.0153 | 8 | 0.0154 | 0.832 / 0.871 | 0.643 | 236.4 |
| m18 | tb | 5.753 | 0.0181 | 16 | 0.0189 | 1.184 / 1.142 | 0.534 | 191.5 |
| m18 | tbw | 5.489 | 0.0179 | 15 | 0.0185 | 1.165 / 1.149 | 0.555 | 180.7 |
| m18 | tbnat | 6.340 | 0.0248 | 58 | 0.0259 | 1.219 / 1.42 | 0.548 | – |
| m18 | bon3 | 7.470 | 0.0153 | 8 | 0.0152 | 0.88 / 0.882 | 0.635 | 523.0 |
| m18 | bon3e | 7.387 | 0.0153 | 8 | 0.0153 | 0.86 / 0.867 | 0.647 | 384.8 |
| m18 | bon2e | 7.485 | 0.0153 | 8 | 0.0154 | 0.97 / 0.921 | 0.593 | 339.3 |
| m18 | kf2 | 7.212 | 0.0149 | 8 | 0.0148 | 0.924 / 0.813 | 0.618 | 156.8 |
| m18 | kf3 | 5.676 | 0.0148 | 9 | 0.0147 | 0.884 / 0.775 | 0.628 | 125.7 |
| m18 | kfv3 | 7.599 | 0.0153 | 9 | 0.0155 | 1.004 / 0.925 | 0.547 | 246.5 |
| m18 | tkf2 | 8.119 | 0.0149 | 8 | 0.0150 | 1.074 / 0.941 | 0.534 | 191.5 |

Notes on the table:
- "q2" GPU = its E2E face_gen + face_comp (old03 = the identical `q2r` arm; the old03 E2E NVML sampler did not write a file).
- tbnat (TBDub native) is a 25 fps video, so its flicker / pops are inflated by 1.2× larger inter-frame motion: not comparable.
- tbnc = TBDub with no CodeFormer anywhere; tbmr0 = tbnc with `--mouth-region 0`. Both are diagnostics on old03.

## W0: TBDub bake-off

### Licence (quoted exactly)
- **GitHub `TaoLiveAIGC/TBDub`:**
  - LICENSE = the full Apache License 2.0 text;
  - README: "The code and released model weights are available under the Apache License 2.0".
- **HF card `TaoLiveAIGC/TBDub`:**
  - license field `apache-2.0`;
  - "The released model weights and code are provided under the Apache License 2.0 unless otherwise specified. Third-party dependencies and checkpoints remain subject to their respective licenses."
  - Intended Use: "TBDub is intended for research and authorized applications involving visual dubbing, video localization, digital humans, and audio-driven facial animation." / "Users are responsible for obtaining appropriate consent and ensuring that generated content complies with applicable laws, licenses, and platform policies."
  - This is a use statement, not an NC clause: commercial use is allowed under Apache-2.0.
- **NOTICE.md:** derived from X-Dub and DiffSynth-Studio, both Apache-2.0.
- **Dependencies:**
  - Wan2.2 VAE from `KlingTeam/X-Dub`: HF license apache-2.0; GitHub KlingAIResearch/X-Dub Apache-2.0;
  - `facebook/hubert-large-ll60k`: apache-2.0;
  - MediaPipe face_landmarker: Apache-2.0;
  - pip deps (requirements.txt): no NC packages, no InsightFace.
- **Download:** 15.27 GB, the student only:
  - tbdub_student.safetensors 12.59 GB, sha256 4e86900f…ea11e6 verified;
  - Wan2.2_VAE 1.41 GB;
  - hubert 1.26 GB;
  - face_landmarker 4 MB (sha verified).
  - The teacher (+12.59 GB) was not fetched.
  - `scripts/r15/tbdub_setup.sh` reproduces this into `/workspace/dub/tbdub/{TBDub @19e18e4, ckpt, venv}`.

### Integration
- **Own venv.** TBDub runs in its own venv (diffusers 0.36 / transformers 5.1 clash with dabai). It reuses dabai's torch 2.7.1+cu128 through a `.pth` fallback.
- **Resident worker.** `scripts/r15/tbdub_worker.py` is a child of the generator stage, so it sits inside the stage's gpu.lock. It loads in 5–16 s and holds 14.6 GB resident (peak 18 GB per segment, 28.5 GB whole-video).
- **Drop-in.** `scripts/r15/tbdub_gen.py` (`R10_GEN_PY`) swaps the LatentSync runner in `candidate_video_stage.py`. Everything else is production: speech gate, our mouth-only paste-back, CodeFormer, raw-layer dumps, the whole face_comp.
- **Crop / paste-back.** TBDub's own MediaPipe crop is computed once over the whole master, and its own blend + paste-back is applied onto OUR frames.
- **Modes:**
  - default = per gate segment;
  - `TBDUB_WHOLE=1` = one generation over the whole master with the whole dub, its native clip / motion continuity, sliced per segment;
  - `TBDUB_FASTCC=1` = OpenCV instead of skimage for its per-frame Lab colour match, which was ~0.15 s/frame on CPU.
- **Fixes on the way:**
  - the MediaPipe child process wrote into the protocol stdout (fd-level redirect);
  - the whole-video run left 28 GB in the caching allocator, so CodeFormer OOM'd and every segment passed through (`empty_cache` after each request);
  - `arm.sh` now FAILs an arm when any generator segment passed through. Two runs had silently produced source-only videos: bon3 with stable_syncnet missing, and tbw with the OOM. Both were deleted and rerun.

### Where TBDub's sync goes (old03, SyncNet on each layer)

| layer | LatentSync q2 | TBDub |
|---|---|---|
| raw generator layer (25 fps) | 8.75 | 7.62 (= native) |
| + CodeFormer w 1.0 (generator stage) | 8.34 | 7.30 |
| no CodeFormer, + SR | – | 7.55 |
| + layer repair (rep2s) | – | 7.44 |
| final (fps_composite → source fps) | 7.84 | 6.27 (with / without CodeFormer: 6.28 / 6.27) |

- The loss is our fps_composite, about −1.2 for TBDub vs about −0.5 for LatentSync.
- `--mouth-region 0` (paste the layer's whole lower face, not lips + jaw drop) does not fix it: 6.33.
- TBDub trips the fallback more (3 % vs 1 %, mostly the dark-transient detector on its darker mouth cavity). Its paste-guard NCC to the source is lower (median 0.939 vs 0.985), because it redraws the whole lower face, skin included. Neither explains −1.2 alone.
- **Next step if the eye likes tbnat:** a stage-by-stage diagnosis of fps_composite on the TBDub layer (registration / retime / colour match).

### Speed
- Face GPU-s/vmin: 160 / 169 / 192 per segment, 140 / 155 / 181 whole-video.
- Wall is CPU-bound: 650 s/vmin per segment, mostly skimage colour match and PIL paste. Whole-video with FASTCC generates old03 in 276 s (184 s/vmin).

## W1: best-of-N per window (`LS_BON=N`, `LS_BON_EARLY=1`, `scripts/r15/bon.py`, default off)

**Design**
- **Candidates:** per continue-window (c8: 16 frames, stride 8), N initial noises denoised as one batch. Candidate 0 is the trajectory's own noise; candidate k is seed 1247 + 7919 k, LatentSync-style (one draw repeated over the frames).
- **Sequential pick:** the shared frames are pinned to the CHOSEN previous window, so the pick runs window by window.
- **Score:** z-free, with fixed generic scales:
  - stable SyncNet of the full window vs its mel (/0.03);
  - opening on dub-voiced minus dub-silent frames (/0.05 × 0.5);
  - lip-box log Laplacian variance (/0.15 × 0.5);
  - continuity jump vs the last pinned frame (/0.30 × −0.5).
- **Early mode:** the candidates run only the first step and are scored on the x0 preview; the winner runs the rest.
- **Scoring decode:** only the lower 288 rows, with the EAGER VAE decoder. The compiled `_Pad16` one re-autotuned for ~60 s per process.

**Pick statistics** (`eval/BON_STATS.txt`):
- Pick ≠ seed 0: bon3 64 / 91 / 78 %, bon3e 49 / 84 / 75 %, bon2e 38 / 59 / 20 %.
- The deciding terms are mostly continuity, sync and sharpness. The opening term almost never decides.
- Candidate 2 won most windows on all 3 videos. As a fixed seed (`LS_NOISE_SEED=17085`, arm s4c8n2) it is not a generic gain: +0.13 / −0.09 / +0.02, opening ×1.07 / 0.96 / 0.83.

**Verdict:** no consistent sync or opening gain; 1.5–2.6× the generator cost. Keep the flag for the record, not for production.

## W2: keyframes + interpolation (`LS_KF=k`, `KF_VOICED=1`, `KF_INTERP=dis|rife`, `scripts/r15/kf.py`, default off)

**Design**
- **Keyframes:** the generator (any mode, c8 included) runs on keyframes only: every k-th master frame plus the last. Each keyframe keeps its own 25 fps audio window (chunk selection, as LS_RATE). That is option (a): the temporal layers see a k× subsampled clip.
- **Crop space:** the restore is intercepted, so the keyframes' generated 512 crops stay in the aligned crop space.
- **In-betweens:** each in-between mouth is interpolated there between its two keyframes. It is composited over THAT frame's own aligned crop with LatentSync's lower-face mask, then restored with that frame's own affine. So only the lower face is interpolated, never the head motion.

**Interpolator**
- Default: bidirectional DIS optical flow (OpenCV, Apache-2.0; the repo's `ls_stable._flow_mid`).
- Practical-RIFE 4.25 is downloaded and verified: MIT code, and the README says the weight links are "under the same MIT license as this project". It is in `/workspace/dub/rife`, 23 MB, sha256 6615790e… for flownet.pkl.
- Running its downloaded model code was blocked by this session's auto-mode safety classifier. **RIFE (`KF_INTERP=rife`) is wired but untested; it needs the user's OK in the session.**

**Results**
- **Student kf2 / kf3:** smoothest (pops 0–13). SyncNet −0.43…−0.51 (kf2) and −1.65…−1.81 (kf3), with a lazier mouth (opening speed ×0.8–0.9).
- **Teacher tkf2:** see the TL;DR. It is the arm to show the user.
- **Option (b) kfv3** (dense on dub-voiced ±2 frames, keyframes / 3 only in dub silences): SyncNet ≥ s4c8 on all 3 videos. It saves nothing on these speech-dense clips (old03 −8 %, m03 0 %, m18 +7 %, the extra affine pass) and would only pay on pause-heavy videos.

## Files
- `scripts/r15/`:
  - `arm.sh` (face arm on the baseline dub, NVML timing, pass-through guard, `ARM_CFD`);
  - `base_q2.sh` (baselines, gpu.lock held over the unlocked audio stage);
  - `metrics.sh`, `pair.sh`, `table.py`, `bon_stats.py`, `pop_times.py`;
  - `tbdub_setup.sh`, `tbdub_worker.py`, `tbdub_gen.py`, `tbdub_smoke.py`, `native.sh`, `comp_variant.sh`;
  - `bon.py`, `kf.py`;
  - chains `chain_w0.sh`, `chain_tbw.sh`, `chain_w12.sh`, `chain_w1e.sh`, `chain_kfv.sh`, `final_metrics.sh`.
- `scripts/quality/ls_stable.py`: the flags `LS_BON`, `LS_BON_EARLY`, `LS_KF`, `LS_NOISE_SEED`, all default off. With them unset the code path is byte-identical to r14.
- Eval data: `/workspace/dub/r15w/eval/` (TABLE.md, BON_STATS.txt, *.sync.txt, flicker_*.json, mouth_*.json, <arm>_<id>/timing.json + logs).

### How to run the winner candidate (tkf2)
```
V=tkf2 LS_KF=2 bash scripts/r15/arm.sh <id>      # face stages only, on an existing q2 baseline dub
# full E2E: dabai_job debug_env {"LS_CMS_READ":"1","FC_CMS":"1","FC_EXTRA":"...q2+F3...","LS_KF":"2"}  (untested as a full E2E job)
```

## Gotchas (server sharing)
- **Unlocked audio stage.** The E2E audio stage (Qwen 14B + Chatterbox, ~27 GB) takes NO gpu.lock and OOMs next to any locked job. `base_q2.sh` holds the lock from the job start until the e2e's "sep" stage. Agent L used a whole-run lock (`locked.sh`), which also idled the GPU through its CPU-only composite stages.
- **Missing stable SyncNet.** `models/latentsync/stable_syncnet.pt` (1.6 GB, ByteDance/LatentSync-1.6, openrail++ like the LatentSync weights) was missing on this server and was downloaded for W1.

## Budget
About 2.4 GPU-h of 6: arms 1.95 h by NVML job attribution, plus baselines, syncs and the smoke test.

## Для Ирины (RU)
- **TBDub (Alibaba):**
  - в нашей цепочке синхрон сильно хуже прода (6.1–6.3 против 7.8–8.3);
  - «родной» TBDub ближе (7.6 / 7.8 / 6.3), но на m18 сильно проседает;
  - стоит дешевле всего (140–190 GPU-с/мин);
  - смотреть `*_mouth_q2_vs_tbnat_spots_half_speed`.
- **Лучший-из-N у студента:** синхрон почти не меняется, раскрытие рта улучшается только на old03, дороже нормы. Не берём.
- **Ключевые кадры у студента:** быстрее на 35–45 %, но синхрон падает.
- **Главная находка: учитель (прод) с генерацией через кадр + интерполяция рта в выровненном кропе:**
  - синхрон 7.45 / 8.02 / 8.12 (прод 7.84 / 8.23 / 8.31, студент 7.07 / 7.19 / 7.49);
  - цена лица 145–192 GPU-с/мин (прод 230–280, студент 194–231);
  - то есть лучше студента и по синхрону, и по цене, а весь E2E укладывается в норму 300.
- **Нужен глаз на 0.5x** (не дрожит ли, как старый LS_RATE): `*_mouth_q2_vs_tkf2_spots_half_speed.mp4` и `*_mouth_s4c8_vs_tkf2_spots_half_speed.mp4`.
