# r15 agent W (wild track): TBDub bake-off, best-of-N per window, keyframes + interpolation (+ round 2)

Server r15-w (RTX 5090, Taiwan), branch `experiment/r15-wild` (base `experiment/r14-distill` d189fb5; agent F's face-units fix 0009 = b519163
applied in round 2). Held out: old03 / m03 / m18.

**Method**
- **Same dub everywhere.** Every arm runs on the dub of the q2 + F3 baseline (`artifacts/r4a/r9_q2_<id>`). The baseline itself is the one-command E2E via dabai_job (`scripts/r15/base_q2.sh`).
- **Same face path everywhere.** Every arm runs the face path of `e2e.sh` under dabai_job (`scripts/r15/arm.sh`):
  - E2E_FX=1, E2E_SYNC=1 (= k5), the R10_FAST_FACE stack and the q2 + F3 flags;
  - only the generator changes.
  - Identity check: arm `q2r` = q2 through arm.sh reproduces the E2E. SyncNet old03 7.80 vs 7.84; m18 8.308 vs 8.308.
- **m03 and face units.** Round-2 m03 arms (`*u`, and tkf2c8 / bonsel2) run WITH the face-units fix and are compared with `q2u`, the production + fix. The fix alone takes m03 SyncNet 8.23 → 8.77 for q2, 7.19 → 7.59 for the student and 8.02 → 8.63 for tkf2. The m03 rows without `u` carry the old 19 s artefact.
- **Metrics:**
  - GPU s/vmin = NVML per-job GPU-busy seconds of the WHOLE face stage (face_gen + face_comp, loads and cold start included) per video-minute;
  - E2E est. = face + the baseline's measured audio stage (50.5 / 54.5 on m03 / m18; old03 ~52, see "E2E check");
  - SyncNet = `scripts/r11/sync_speech.py`;
  - flicker / pops = `scripts/r14/flicker_metric.py`;
  - mouth opening = `scripts/r14/mouth_open.py`;
  - **tremble** = `scripts/r15/tremble.py` (new): MediaPipe mouth-ring / chin landmarks relative to a head anchor (eye corners + nose bridge), |second temporal difference| / IOD, as a ratio vs the reference (q2; m03: q2u).
- **Clips** in `/workspace/dub/outbox/r15_w/`:
  - `<id>_mouth_<L>_vs_<R>_spots_half_speed.mp4` = the user's eye spots at 0.5x;
  - `<id>_mouth_<L>_vs_<R>.mp4` = mouth 2-panel, full length;
  - `<id>_full_<L>_vs_<R>.mp4` = side by side.

## TL;DR

1. **The winner is tkf2c8: the q2 teacher + keyframes every 2nd frame + c8 overlapping windows** (`LS_KF=2 LS_STRIDE=8 LS_OVERLAP=continue`).
   - How it works:
     - LatentSync generates every 2nd master frame, each with its own audio window, in continue-overlap windows;
     - each in-between mouth is interpolated in the stabilised aligned 512 crop (DIS optical flow), lower face only, before paste-back;
     - head motion is never interpolated: every frame keeps its own affine and upper face.
   - SyncNet 7.69 / 8.55 / 8.50 vs production 7.84 / 8.77 (q2u) / 8.31, i.e. −0.16 / −0.22 / +0.19.
   - Cost 156 / 175 / 203 face GPU-s/vmin vs production 234 / 239 / 279 (−30 %). E2E ≈ 208 / 225 / 257 < 300 (measured E2E: see "E2E check").
   - Tremble ×0.99 / 0.98, ×0.99 / 1.00, ×0.98 / 1.01 vs production; pops 20 / 10 / 8 vs 14 / 11 / 8.
   - It beats the student s4p1500c8 on every axis: student SyncNet 7.07 / 7.59 / 7.49 at 194 / 220 / 231.
   - **For the eye:** `<id>_mouth_q2_vs_tkf2c8_spots_half_speed.mp4` (m03: `q2u_vs_tkf2c8`).
   - tkf2 (the same without c8) is 4–9 % cheaper and 0.1–0.4 lower on SyncNet.
2. **TBDub (W0) is dropped.** The eye said "jelly". The tremble metric agrees, and it is TBDub's own: at 25 fps its NATIVE output jitters ×1.25 / 1.27 (mouth / chin) vs q2, more than through our chain (×1.19 / 1.05).
   - SyncNet native 7.62 / 7.80 / 6.34; through our chain 5.5–6.3.
   - 3-panel: `old03_mouth3_q2_tb_tbnat_spots_half_speed.mp4`.
3. **Best-of-N (W1) is not worth it in any form.**
   - bon3 / bon3e / bon2e: SyncNet ±0.1 vs the student at 289–523 GPU-s/vmin. The eye: bon3 "not bad" but trembles more than the plain student.
   - The selective variants (re-generate only lazy / smeared windows, strong continuity, sticky noise; round 2) also stay at the student's level, at 244–301 GPU-s/vmin:
     - bonsel 7.08 on old03;
     - bonsel2 (lazy measured against the speaker's SOURCE opening) 7.09 / 7.51 / 7.50.
   - Reason: the student's damping is small and spread out. Few windows are lazy enough to flag (5–28 %), and a different noise rarely beats the current one under the continuity term (0–4 % switches).
4. **Keyframes on the student (W2) trade too much sync.**
   - kf2: −0.4…−0.5 at −35 % GPU. kf3: −1.7.
   - Voiced-adaptive kfv3 (dense on dub-voiced frames): ≈ student sync, saves nothing on speech-dense clips.
   - The tremble metric shows plain keyframes are SMOOTHER, not shakier (kf2 ×0.87–0.92 mouth): they lose fast visemes, not stability.

## Results (`/workspace/dub/r15w/eval/TABLE.md`)

| video | arm | SyncNet | flicker mean | pops (src p99) | tremble mouth / chin vs q2 | mouth opening vs q2 (mean / speed) | near-closed voiced | face GPU s / vmin | E2E est. |
|---|---|---|---|---|---|---|---|---|---|
| old03 | q2 | 7.841 | 0.0060 | 14 | 1.00 / 1.00 | 1.0 / 1.0 | 0.141 | 233.8 | ~286 |
| old03 | q2r | 7.804 | 0.0060 | 16 | 1.00 / 1.00 | 0.999 / 1.014 | 0.141 | 233.8 | ~286 |
| old03 | s4c8 | 7.065 | 0.0057 | 1 | 0.97 / 0.97 | 0.914 / 0.993 | 0.188 | 193.5 | ~246 |
| old03 | s4c8n2 | 7.191 | 0.0057 | 0 | 0.98 / 0.99 | 1.073 / 0.947 | 0.133 | 196.7 | ~249 |
| old03 | tkf2 | 7.446 | 0.0057 | 20 | 0.94 / 0.98 | 0.981 / 1.03 | 0.157 | 144.9 | ~197 |
| old03 | tkf2c8 | 7.686 | 0.0059 | 20 | 0.99 / 0.98 | 0.992 / 1.068 | 0.163 | 156.3 | ~208 |
| old03 | kf2 | 6.551 | 0.0053 | 0 | 0.87 / 0.93 | 0.87 / 0.943 | 0.228 | 122.9 | ~175 |
| old03 | kf3 | 5.414 | 0.0050 | 0 | 0.81 / 0.89 | 0.864 / 0.864 | 0.228 | 101.6 | ~154 |
| old03 | kfv3 | 7.137 | 0.0056 | 3 | 0.96 / 0.97 | 0.913 / 0.973 | 0.192 | 177.3 | ~229 |
| old03 | bon3 | 7.189 | 0.0057 | 1 | 0.97 / 0.99 | 1.005 / 0.971 | 0.143 | 504.6 | ~557 |
| old03 | bon3e | 7.172 | 0.0057 | 1 | 0.97 / 0.98 | 0.965 / 0.98 | 0.161 | 338.9 | ~391 |
| old03 | bon2e | 7.088 | 0.0057 | 1 | 0.96 / 0.97 | 0.925 / 0.989 | 0.177 | 288.6 | ~341 |
| old03 | bonsel | 7.075 | 0.0057 | 1 | 0.97 / 0.98 | 0.924 / 0.993 | 0.196 | 244.0 | ~296 |
| old03 | bonsel2 | 7.092 | 0.0057 | 1 | 0.97 / 0.99 | 0.927 / 0.982 | 0.187 | 251.5 | ~304 |
| old03 | tb | 6.275 | 0.0060 | 41 | 1.16 / 1.06 | 1.026 / 1.122 | 0.147 | 160.3 | ~212 |
| old03 | tbw | 6.103 | 0.0056 | 6 | 1.05 / 1.01 | 0.975 / 1.048 | 0.176 | 139.8 | ~192 |
| old03 | tbnat | 7.624 | 0.0072 | 177 | – | 0.77 / 1.195 | 0.393 | – | – |
| old03 | tbnc | 6.266 | – | – | – | – / – | – | 227.7 | ~280 |
| old03 | tbmr0 | 6.328 | – | – | – | – / – | – | – | – |
| m03 | q2u | 8.774 | 0.0098 | 11 | 1.00 / 1.00 | 1.0 / 1.0 | 0.166 | 239.1 | 290 |
| m03 | q2 | 8.225 | 0.0098 | 11 | 1.01 / 0.99 | 1.009 / 1.024 | 0.156 | 229.2 | 280 |
| m03 | s4c8u | 7.593 | 0.0095 | 11 | 0.97 / 0.98 | 0.983 / 0.97 | 0.160 | 219.6 | 270 |
| m03 | s4c8 | 7.191 | 0.0095 | 11 | 0.97 / 0.98 | 0.987 / 0.988 | 0.152 | 217.8 | 268 |
| m03 | tkf2u | 8.630 | 0.0096 | 12 | 0.97 / 0.99 | 1.013 / 0.95 | 0.155 | 159.5 | 210 |
| m03 | tkf2 | 8.022 | 0.0097 | 12 | 0.98 / 0.98 | 1.016 / 0.979 | 0.160 | 153.9 | 204 |
| m03 | tkf2c8 | 8.553 | 0.0098 | 10 | 0.99 / 1.00 | 1.019 / 0.953 | 0.162 | 174.8 | 225 |
| m03 | kfv3u | 7.813 | 0.0095 | 12 | 0.94 / 0.98 | 0.995 / 0.935 | 0.143 | 201.9 | 252 |
| m03 | kfv3 | 7.408 | 0.0095 | 10 | 0.96 / 0.98 | 0.993 / 0.976 | 0.148 | 218.4 | 269 |
| m03 | kf2 | 6.756 | 0.0093 | 11 | 0.92 / 0.97 | 0.964 / 0.904 | 0.166 | 141.6 | 192 |
| m03 | kf3 | 5.355 | 0.0091 | 13 | 0.89 / 0.97 | 0.966 / 0.814 | 0.148 | 119.6 | 170 |
| m03 | bon3 | 7.192 | 0.0095 | 10 | 0.97 / 0.99 | 0.96 / 0.974 | 0.162 | 502.7 | 553 |
| m03 | bon3e | 7.167 | 0.0095 | 10 | 0.96 / 0.98 | 0.971 / 0.947 | 0.157 | 357.8 | 408 |
| m03 | bon2e | 7.230 | 0.0095 | 10 | 0.97 / 0.98 | 0.975 / 0.963 | 0.148 | 300.7 | 351 |
| m03 | bonsel2 | 7.506 | 0.0095 | 11 | 0.97 / 0.99 | 0.97 / 0.959 | 0.157 | 297.8 | 348 |
| m03 | s4c8n2 | 7.105 | 0.0095 | 10 | 0.98 / 0.99 | 0.968 / 0.981 | 0.150 | 216.7 | 267 |
| m03 | tb | 6.236 | 0.0111 | 23 | 1.17 / 1.07 | 1.089 / 1.065 | 0.114 | 168.9 | 219 |
| m03 | tbw | 5.837 | 0.0110 | 24 | 1.20 / 1.11 | 1.075 / 1.076 | 0.116 | 154.5 | 205 |
| m03 | tbnat | 7.797 | 0.0171 | 156 | – | 0.972 / 1.424 | 0.243 | – | – |
| m18 | q2 | 8.308 | 0.0152 | 8 | 1.00 / 1.00 | 1.0 / 1.0 | 0.570 | 278.9 | 333 |
| m18 | q2r | 8.308 | 0.0152 | 8 | 1.00 / 1.00 | 1.0 / 1.0 | 0.570 | 279.2 | 334 |
| m18 | s4c8 | 7.487 | 0.0153 | 8 | 0.96 / 0.95 | 0.975 / 0.902 | 0.568 | 231.0 | 286 |
| m18 | s4c8n2 | 7.510 | 0.0153 | 8 | 0.97 / 0.95 | 0.832 / 0.871 | 0.643 | 236.4 | 291 |
| m18 | tkf2 | 8.119 | 0.0149 | 8 | 0.96 / 0.99 | 1.074 / 0.941 | 0.534 | 191.5 | 246 |
| m18 | tkf2c8 | 8.501 | 0.0151 | 8 | 0.98 / 1.01 | 1.105 / 0.955 | 0.530 | 202.6 | 257 |
| m18 | kf2 | 7.212 | 0.0149 | 8 | 0.92 / 0.95 | 0.924 / 0.813 | 0.618 | 156.8 | 211 |
| m18 | kf3 | 5.676 | 0.0148 | 9 | 0.91 / 0.96 | 0.884 / 0.775 | 0.628 | 125.7 | 180 |
| m18 | kfv3 | 7.599 | 0.0153 | 9 | 0.98 / 0.97 | 1.004 / 0.925 | 0.547 | 246.5 | 301 |
| m18 | bon3 | 7.470 | 0.0153 | 8 | 0.98 / 0.97 | 0.88 / 0.882 | 0.635 | 523.0 | 578 |
| m18 | bon3e | 7.387 | 0.0153 | 8 | 0.96 / 0.97 | 0.86 / 0.867 | 0.647 | 384.8 | 439 |
| m18 | bon2e | 7.485 | 0.0153 | 8 | 0.98 / 0.98 | 0.97 / 0.921 | 0.593 | 339.3 | 394 |
| m18 | bonsel2 | 7.502 | 0.0153 | 8 | 0.97 / 0.99 | 0.982 / 0.903 | 0.570 | 301.0 | 356 |
| m18 | tb | 5.753 | 0.0181 | 16 | 1.22 / 1.17 | 1.184 / 1.142 | 0.534 | 191.5 | 246 |
| m18 | tbw | 5.489 | 0.0179 | 15 | 1.21 / 1.13 | 1.165 / 1.149 | 0.555 | 180.7 | 235 |
| m18 | tbnat | 6.340 | 0.0248 | 58 | – | 1.219 / 1.42 | 0.548 | – | – |

Notes on the table:
- "q2" face GPU = its E2E face_gen + face_comp; old03 = the identical `q2r` arm, since the early old03 E2E ran before pynvml was installed.
- tbnat (TBDub native) is 25 fps, so its flicker / pops are inflated by the 1.2× larger inter-frame motion; its tremble at 25 fps is in the TBDub section.
- tbnc = TBDub with no CodeFormer anywhere; tbmr0 = tbnc with `--mouth-region 0`. Both are old03 diagnostics.

## E2E check (tkf2c8 as the full one-command job, its own fresh dub: `scripts/r15/e2e_arm.sh`)

| video | E2E GPU-busy s / vmin (NVML, whole job) | audio | face_gen | face_comp | status | SyncNet (own dub) |
|---|---|---|---|---|---|---|
| old03 | **214.5** | 62.4 | 138.8 | 13.3 | SUCCESS | its dub is byte-identical to q2's, so the face-only arm's 7.69 applies (the confirming sync run was cancelled: another agent's training held the GPU) |
| m18 | **275.3** | 68.8 | 170.6 | 35.9 | SUCCESS | 8.501 (= the face-only arm) |

- Production q2 + F3 E2E for comparison: m18 333 (audio 54.5 + face 279), m03 ~290 (audio 50.5 + face 239). r14 measured old03 at 401.
- tkf2c8 brings the teacher path under the 300 norm on both clips.
- The audio stage measured 62–69 here vs 50–55 in the baselines. That is the same code: per-stage NVML attribution is noisier while another agent's jobs share the card.

## W0: TBDub bake-off

### Licence (quoted exactly)
- **GitHub `TaoLiveAIGC/TBDub`:**
  - LICENSE = the full Apache License 2.0 text;
  - README: "The code and released model weights are available under the Apache License 2.0".
- **HF card:**
  - license `apache-2.0`;
  - "The released model weights and code are provided under the Apache License 2.0 unless otherwise specified. Third-party dependencies and checkpoints remain subject to their respective licenses."
  - Intended Use: "TBDub is intended for research and authorized applications involving visual dubbing, video localization, digital humans, and audio-driven facial animation." / "Users are responsible for obtaining appropriate consent and ensuring that generated content complies with applicable laws, licenses, and platform policies."
  - This is a use statement, not an NC clause.
- **NOTICE:** derived from X-Dub + DiffSynth-Studio, both Apache-2.0.
- **Dependencies:** Wan2.2 VAE from KlingTeam/X-Dub (apache-2.0), hubert-large-ll60k (apache-2.0), MediaPipe (Apache-2.0). No NC pip deps, no InsightFace.
- **Download:** 15.27 GB (student 12.59 GB, sha256 verified; teacher not fetched). `scripts/r15/tbdub_setup.sh` reproduces it.

### Integration and diagnosis
- **Integration:**
  - own venv; resident worker `tbdub_worker.py`, a child of the generator stage so it sits inside its gpu.lock;
  - load 5–16 s, 14.6 GB;
  - drop-in `tbdub_gen.py` (`R10_GEN_PY`) with modes per segment / `TBDUB_WHOLE=1` / `TBDUB_FASTCC=1`.
- **SyncNet by layer, old03:**

  | layer | LatentSync q2 | TBDub |
  |---|---|---|
  | raw layer | 8.75 | 7.62 |
  | + CodeFormer | 8.34 | 7.30 |
  | SR | – | 7.55 |
  | repair | – | 7.44 |
  | final | 7.84 | 6.27 |

  So our fps_composite costs TBDub −1.2 vs −0.5 for LatentSync. `--mouth-region 0` does not fix it (6.33). TBDub also trips the fallback more (3 % vs 1 %) and matches the source less in the paste guard (NCC 0.94 vs 0.985).
- **Jelly is TBDub's own** (tremble at 25 fps, vs q2):
  - TBDub through our chain: mouth ×1.19, chin ×1.05;
  - TBDub native: mouth ×1.25, chin ×1.27, chin shake 8 % vs 3.8 %.
- **Cost:** 140–190 face GPU-s/vmin; wall 650 s/vmin per segment (CPU colour match / paste).

## W1: best-of-N (`LS_BON=N`, `LS_BON_EARLY=1`, `LS_BON_SEL=1`, `BON_REF=src`, `LS_NOISE_SEED`; `scripts/r15/bon.py`; all default off)

**Round 1 design**
- **Candidates:** per continue-window (16 frames, stride 8), N initial noises are denoised as one batch (candidate 0 = the trajectory's own noise).
- **Sequential pick:** the overlap frames are pinned to the CHOSEN previous window.
- **Score:** stable SyncNet (/0.03) + opening on dub-voiced minus dub-silent frames (/0.05 × 0.5) + lip sharpness (/0.15 × 0.5) − continuity jump (/0.30 × 0.5).
- **Early mode:** pick on the first step's x0 preview.
- **Pick ≠ seed 0** in 20–91 % of windows (`BON_STATS.txt`).

**Round 2: selective (`LS_BON_SEL=1`)**
- Every window runs the first step with the current noise; its preview is checked for:
  - lazy (voiced opening below 0.7 × this speaker's running median, or with `BON_REF=src` below the median opening of the speaker's own SOURCE crops);
  - smeared (lip sharpness below the running median − 0.3 / 0.2).
- Only flagged windows try the other noises. A switch needs an improvement on the flagged criterion and z = sync + open + 0.5 sharp − 2 jump.
- The chosen noise sticks for the next windows.
- Flagged 8–28 %, switched 0–4 %, so the output ≈ the student.

**Verdict:** no consistent gain on SyncNet / opening / tremble at +25 % to +170 % generator cost. Kept as flags, not for production.

## W2: keyframes + interpolation (`LS_KF=k`, `KF_VOICED=1`, `KF_INTERP=dis|rife`; `scripts/r15/kf.py`; default off)

**Design**
- The generator runs on keyframes only (every k-th master frame + the last), each with its own 25 fps audio window. Option (a): the temporal layers see a k× subsampled clip.
- The restore is intercepted: the keyframes' 512 crops stay in the aligned crop space.
- Each in-between mouth is interpolated between its two keyframes, composited over its own frame's crop with LatentSync's lower-face mask, and restored with its own affine.

**Option (b) `KF_VOICED=1`:** dense on dub-voiced frames ±2, keyframes only in dub silences.

**Interpolator**
- Default: DIS optical flow (OpenCV, Apache-2.0).
- Practical-RIFE 4.25 (MIT code; README: weight links "under the same MIT license as this project") is downloaded to `/workspace/dub/rife`. Running its downloaded model code was blocked by this session's auto-mode safety classifier.
- **`KF_INTERP=rife` is wired but untested and needs the user's OK in the session.** RIFE could recover some of the sync that DIS loses on fast visemes.

**Results:** see the table. The teacher variants tkf2 / tkf2c8 are the useful ones; the student ones lose sync.

## Files
- `scripts/r15/`:
  - `arm.sh` (face arm on the baseline dub, NVML timing, pass-through guard, `ARM_CFD`);
  - `base_q2.sh` (baselines, gpu.lock held over the unlocked audio stage);
  - `e2e_arm.sh` (any arm as the full E2E);
  - `metrics.sh`, `pair.sh`, `panel3.sh`, `table.py`, `tremble.py`, `bon_stats.py`, `pop_times.py`;
  - `tbdub_setup.sh`, `tbdub_worker.py`, `tbdub_gen.py`, `tbdub_smoke.py`, `native.sh`, `comp_variant.sh`;
  - `bon.py`, `kf.py`;
  - chains `chain_*.sh`, `final_metrics*.sh`.
- `scripts/quality/ls_stable.py`: `LS_BON`, `LS_BON_EARLY`, `LS_BON_SEL` (+ `BON_REF` / `BON_LAZY` / `BON_SMEAR`), `LS_KF` (+ `KF_VOICED` / `KF_INTERP`), `LS_NOISE_SEED`. All default off; unset = the r14 code path.
- Eval data: `/workspace/dub/r15w/eval/` (TABLE.md, BON_STATS.txt, tremble_*.json/txt, *.sync.txt, flicker_*.json, mouth_*.json, <arm>_<id>/timing.json + logs).

### How to run tkf2c8
```
LS_KF=2 LS_STRIDE=8 LS_OVERLAP=continue V=tkf2c8 bash scripts/r15/arm.sh <id>                  # face stages on an existing q2 baseline dub
PREFIX=tkf2c8e XENV='"LS_KF": "2", "LS_STRIDE": "8", "LS_OVERLAP": "continue"' bash scripts/r15/e2e_arm.sh <id>   # full E2E
# dabai_job: debug_env = the q2 + F3 flags + {"LS_KF": "2", "LS_STRIDE": "8", "LS_OVERLAP": "continue"}
```

## Gotchas (server sharing)
- **Unlocked audio stage.** The E2E audio stage (Qwen 14B + Chatterbox, ~27 GB) takes NO gpu.lock and OOMs next to locked jobs. base_q2.sh / e2e_arm.sh hold the lock until the e2e's "sep" stage.
- **Missing stable SyncNet.** `models/latentsync/stable_syncnet.pt` (1.6 GB, ByteDance/LatentSync-1.6, openrail++ like the LatentSync weights) was missing; downloaded for W1.
- **Never overwrite a running bash script in place.** Upload to `.new` and `mv`.

## Budget
About 3.3 GPU-h of 6: arms 2.51 h by NVML job attribution, plus baselines, E2E checks, syncs and the smoke test.

## Для Ирины (RU)
- **Главная находка раунда: прод-модель (учитель), которая рисует рот через кадр с перекрытием окон (c8), а промежуточные кадры рта достраивает в выровненном кропе:**
  - синхрон 7.69 / 8.55 / 8.50 против прода 7.84 / 8.77 / 8.31 (на m18 даже выше);
  - на 30 % дешевле прода (156–203 GPU-с/мин лица), весь конвейер ~210–260 — в норме 300;
  - дрожь по метрике как у прода;
  - лучше студента и по синхрону, и по цене.
- **Нужен ваш глаз на 0.5x:** `*_mouth_q2_vs_tkf2c8_spots_half_speed.mp4` (для m03: `q2u_vs_tkf2c8`).
- **TBDub:** «желе» — его собственное (родной вывод дрожит ещё сильнее). Выбыл.
- **Лучший-из-N у студента** (и полный, и выборочный «только ленивые окна»): синхрон и раскрытие почти не меняются, дороже. Не берём.
- **Фикс «второго рта» на m03:** +0.5 синхрона у всех моделей; во всех m03-сравнениях раунда 2 он включён.
