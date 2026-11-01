# r10 FACE QUALITY: hand-off (01.11.2026, servers destroyed tonight)

- **Branch:** `experiment/r10-face`.
- **Contents:** r9-e2e + `experiment/r10-base` (speed, 9e72755) merged + the r10 face work below.
- **Not merged yet:** the newer r10-base head f070124 (dynamic compile, LS_RATE, the gen worker). Merge it first on the new boxes; the generator levers build on it.
- **Default behaviour of the pipeline is unchanged:** every r10 lever is behind a flag, default off. The one exception is the streaming `fps_composite.py`, which is output-identical (see 3).

## 1. What was done
- **Metrics that match the eye**: `scripts/r10/face_metrics.py`, output vs source on the same frame grid.
  - **Lip flicker:** flow-aligned 2nd temporal difference on the lips. Rows: high-frequency detail; brightness; and brightness on the lip ring with the mouth cavity excluded, so articulation doesn't count.
  - **Chin:** source→output flow in the chin / jaw band (velocity, swim, wander) + the shading wander + MediaPipe chin points.
  - Synthetic check: 50 / 51 lip pops and 71 / 72 chin shifts found; source vs itself reads 0.
  - Tables: `metrics_table.md` covers all 6 videos, all layers, the ablations and the management's mgmt_r9 files.
- **Attribution (layer by layer, 6 videos):** both artefacts are born in the **raw LatentSync** layer (before CodeFormer / SR / composite):
  - lip-ring brightness pops are 2.1–2.6x the source raw, 1.8–2.3x final;
  - chin wander raw 0.006–0.012 IOD/frame, final 0.004–0.007;
  - CodeFormer adds a little; the composite's 0.38 IOD region cap halves the chin;
  - registration, retime, sharpening, skin texture and colour match are all minor (ablations).
- **F1 = composite `--reg-smooth 4` + `scripts/r10/r10_out_stab.py` (output-rate shading stabiliser).**
  - Gate PASS on old03 / m03 / m18; flashes 55→34, 20→10, 14→11; SyncNet −0.27 / −0.16 / +0.05.
  - old05 FAILs every arm, even composite-only: its landmark rows are unstable.
  - **REJECTED by the user's eye ("Не лучше")**: it damps brightness, but the eye follows the generator's frame-to-frame mouth SHAPE / definition jumps.
  - Kept behind flags, default off.
- **DIAGNOSIS.md** (+ clips in `/workspace/dub/outbox/r10_face/diag/`):
  - the eye sees lip definition jumping sharp → mushy → sharp and teeth on / off, already in the RAW layer;
  - the final draws a darker / redder outline that amplifies it;
  - the inner chin is generated and moves with the dub against the source jaw outline (the "slide").
- **Time-base fix (12552e8):** any re-encode of a final must use the SOURCE time base / pts (as `fps_composite` does).
  - With another time base, every downstream `-r 25` resample (SyncNet, flashes, `artefacts.py` stick / slide) picks neighbouring frames, which gives false gate regressions.
  - The g2 / g3 out_stab rows in `gate_tables.md` are affected; g4 / g5 are valid.
- **libx264 non-determinism:** via PyAV, identical frames give different CRF 13 bitstreams depending on the feeding timing (shown: fast vs paused feeding). A bitstream md5 is not an identity test; compare decoded frames at CRF 0.
- **r7_m7_comp.sh cache key** (this commit): the layer repair re-runs when the per-tag fallback npz is missing (it was skipped when the repaired layer was cached).

## 2. Flags added (all default off)
| where | flag | what |
|---|---|---|
| fps_composite.py | `--reg-smooth N` | temporal Gaussian average of the per-frame flow-RANSAC registration + the ring field (chin) |
| fps_composite.py | `--layer-lm-smooth N` | smoothed layer landmarks (no gain, keep off) |
| fps_composite.py | `--retime-hf nearest` | detail of the nearer frame on 25→30 retime blends (untested) |
| fps_composite.py | `--stream-chunk N` | streaming batch size (default 32) |
| r10_out_stab.py | `--shade on --region paste --keep-cavity on --radius-s 0.1 --shade-sigma 0.07` | the F1 shading stabiliser; streaming; source time base |
| r10_out_stab.py | `--shade-model poly0/poly1` | parametric shading (weak, keep field) |
| r10_out_stab.py | `--chin-reg on --chin-reg-vert X` | chin re-registration to the source. **vert > 0 removes articulation: bug, keep 0**; lateral-only gains are small |

## 3. Streaming fps_composite (production blocker for 5-min / 1080p60 uploads)
- **Done:** `scripts/quality/fps_composite.py` is now STREAMING; the in-memory version is kept as `fps_composite_inmem.py`.
  - Source + layer frames sit in bounded windows (SeqWin).
  - The layer stages (sanitize → seam → C2 boundaries → joints) are evaluated lazily in the original order.
  - Calibration samples are pinned in a pre-pass; the paste guard runs on a pending tail; the encode is in order.
  - Lazy stages use a per-thread Flow (a shared DIS object segfaults).
- **Proven:** decoded output identical at CRF 0, 0 differing frames on:
  - m18 (601 frames);
  - m03 (1214);
  - old03 (2710);
  - m03 with `reg_smooth=4` (the collect pass).
  - Peak RSS 7.1–10.9 GB → 3.0–4.7 GB. The streaming wall time is +5–40 % on short clips (extra decode passes for the pins).
- **Left:**
  - (a) **The m01 1080p60 E2E proof:** the job was stopped before it ran. The speed agent's repro OOMs in-memory at 61.5 GB.
  - (b) **Bounded `--reg-smooth`:** it still keeps ~0.1 MB/frame of ring fields for the whole video (bounded, not constant); window it if used.
  - (c) The pre-pass decodes the source 3–4 times: fine, but measure on 1080p60.

## 4. Generator-lever plan (the coordinator gave FACE temporary ownership of the generator QUALITY levers)
Build on `origin/experiment/r10-base` @ f070124. Each lever is a NEW env flag, default off, kept in the runner / `_core` (the gen worker reuses the pipe).
- Test with LS_RATE unset, in cold processes, and with `LS_UNET_COMPILE_DYNAMIC=1` or no compile.
- Keep the DeepCache bookkeeping consistent: the skip schedule counts from `start_timestep` per window group.

Order:
0. **Diagnostic: 20 steps everywhere, no DeepCache** (production is 10 in speech / 20 in silence, DeepCache interval 5) on m18, m03 and old03.
   - If the definition jumps largely vanish, the culprit is step reduction / DeepCache, and the speed budget must be argued separately.
   - **Status: started, stopped before m18 finished; no result.**
1. ~~Shared initial noise~~: already shared (`prepare_latents` repeats one noise frame; `set_seed(1247)` per `_core`). Dropped.
2. **x0 / latent temporal EMA**: ±1 frame, in the last denoising steps. ~free; gate on SyncNet.
3. **Overlapping windows + latent cross-fade** (16-frame windows, 4-frame overlap). About +25 % UNet.
4. **Chin-keeping generation mask**: the source jaw / chin kept, generator mouth only. Watch the mask edge; the model was trained on lower-half masks.

Measure each on the same segments first:
- face_metrics on the RAW layer: lip-ring pops, chin wander;
- SyncNet;
- `diag_arm.py` clips SOURCE | RAW r9 | RAW lever | final lever, 1x + 0.5x.

Only then the full gate and a full-length WATCH for the best 1–2. **The eye is the judge.**

Side A/B done, not judged yet: lip-outline softening (`r9_lipcolour.py`) on the r9 + fix finals. Clips are in `outbox/r10_face/diag/levers/*_lipc_*`. Metric: ring pops unchanged (m03 1.80, m18 2.16); it changes only the outline look.

## 5. Resume commands (new box, after the setup + `scripts/r9/r9_models.sh` + the inputs)
```
git fetch origin; git checkout experiment/r10-face; git merge origin/experiment/r10-base      # f070124 or newer
bash scripts/r10/base.sh old03 m03 m18 old05 m15 m19               # r9 + fix baselines (E2E_FX=1) + fx+stab; ~2 h GPU
bash scripts/r10/layers.sh m03 [ablations...]                      # face_metrics on every layer + finals (+ composite ablations via recomp.py)
# step 0 (generator diagnostic): arm s20 = 20 steps, no DeepCache, same seed / inputs; then the r9 + fix composite on it
bash scripts/r10/run_gen_arm.sh m18 s20 "--codeformer optimized --codeformer-w 1.0 --steps 20 --no-deepcache --codeformer-parse off --codeformer-batch 16"
python scripts/r10/face_metrics.py --video <arm ls_pre> --src <work>/video/master_25fps.mp4 --lm-src <work>/video/master_mediapipe.npz --out x.json
python scripts/r10/diag_arm.py m18 s20 3.8,15.0                    # SOURCE | RAW r9 | RAW s20 | final s20 -> outbox/r10_face/diag/levers/
bash scripts/r10/gate.sh m18 s20X <final video>                    # gate rows; then: python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/m18base --cand /workspace/dub/r10/gate/m18s20X
# streaming composite: identity at CRF 0 vs the in-memory version, then the m01 E2E proof with peak RSS
EQ_V="fps_composite_inmem fps_composite" bash scripts/r10/run_eq_test.sh m03 crf=0
bash scripts/r10/run_m01_e2e.sh                                     # E2E_FX=1 m01 + a VmHWM sampler on fps_composite
```
- The run scripts expect `/workspace/dub/r10/{runs,metrics,abl,gate}`.
- `gate.sh` needs the baseline `gate/<id>base` first: `bash scripts/r10/gate.sh <id> ""`.

## 6. r11 (01.11): KeySync in our composite
Context:
- The user prefers KeySync (sync + a nicer face), but its raw square paste draws artefacts around the face.
- The comparison target became r11-sync's **k5** (LatentSync + the composite sync fixes, branch `experiment/r11-composite-sync`): m18 8.33, m03 8.21, old03 7.57 (old03 r9+fix 7.19).
- The fine-tune branch is CLOSED: see `R11_FINETUNE.md`.

Code (all on this branch, every flag default off):
- `scripts/r11/sync_speech.py`: the gate SyncNet number for any video.
- `scripts/r11/upper_lip.py`: per-frame upper-lip redness vs the moustache, lip-line edge, open interior L, chin texture / chin a* vs the source, plus a contact sheet of fixed or worst frames.
- `scripts/r11/ks_fix.py`: KeySync layer repair.
  - `--guard 3`: explosion guard;
  - `--skin-cm 12`: skin-only colour match;
  - `--lip-chroma 1`;
  - `--chin tex|src`.
- `scripts/r11/ks_softpaste.py`: a soft lower-face paste OUTSIDE fps_composite.
  - Mask: the face oval below the cheeks / nostrils, the union of the source and KeySync landmarks.
  - `--interp mci|nearest|hybrid`: hybrid takes the nearest generated frame inside the lips.
- Runners in `scripts/r11/ks/`:
  - `ksv.sh` composite variants; `KSV_ROOT=/workspace/dub/r11_sync` runs r11-sync's checkout;
  - `ksp.sh` soft-paste variants;
  - `attr*.sh` attribution;
  - `gate_ksp.sh`.

Findings:
- **Our fps_composite costs KeySync ~1 SyncNet.**
  - m18: layer 7.78 → composite 6.78.
  - old03: layer 7.65 → 6.2-6.6; the speed agent's r7_m7_comp KeySync composite is 5.69.
  - Every 'off' switch only gets m18 back to 7.43 (bare + retime nearest + registration off + face-clip off + paste-guard 0).
  - **So drop fps_composite for KeySync and use `ks_softpaste.py`.**
- **Soft paste** (repaired layer, guard + skin cm + lip chroma + chin tex):

  | clip | k5 | mci | nearest | hybrid |
  |---|---|---|---|---|
  | old03 | 7.57 | 7.39 | 7.55 | 7.51 |
  | m18 | 8.33 | 7.30 | 7.49 | 7.41 |

  - old03 = parity with k5.
  - m18 can NOT reach k5: the KeySync layer itself is 7.78.
  - Lip-ring flicker on old03: 0.86 (soft paste) vs 1.41 (our composite).
  - m18 trade-off: mci gives ring 1.23 / sync 7.30; nearest gives ring 2.0 / sync 7.49.
  - Chin swim on m18 is 0.019-0.021 vs r9+fix 0.008: KeySync's chin swims more.
- **Explosion guard works:**
  - old03 KeySync frames 2195-2196 (87.8 s, a melted face) → the source, ±2 frames + fade;
  - 0 false flags on m18 / old03.
- **Chin:**
  - chin src (the source pixels under the KeySync mouth) KILLS sync: m18 6.95 → 5.93, old03 6.31 → 5.66. KeySync opens the mouth with the jaw.
  - chin tex (KeySync shape + the source detail / chroma) restores the stubble texture (0.68 → 1.0 of the source) at no sync cost. Use chin tex.
- **old03 'upper lip vanishes'** is a SHAPE issue: KeySync draws a thin upper vermilion with stubble on it.
  - Upper-lip redness vs the moustache: 7.7 vs source 9-11.7, LatentSync 12.
  - The lip chroma restore only takes it 7.9 → 8.4.
  - Not the 512 resolution (old03's box is 464 px), not the moustache regenerated as skin, not the mouth-only paste (no help).
  - A lip-line warp was skipped (coordinator).
- Sheets: `/workspace/dub/r11/ulip_qo3/{worst_upper_lip,step1}.jpg`.

- **Gate of the old03 soft paste (hybrid) — FAIL vs k5** (`gate_ksp.sh`, rows in `/workspace/dub/r11/gate_ks.md`):

  | | k5 | KeySync soft paste |
  |---|---|---|
  | sync | 7.57 | 7.51 (PASS) |
  | strong flashes | 38 | 53 |
  | stick | 0 | 5 |
  | slide | 0 | 15 |
  | teeth | 0 | 25 |
  | amp | 0 | 9 |
  | pressed lips | 2.93 s | 2.13 s |
  | occluder ring | 1 | 0 |

  - The base (old03 r9+fix) is 7.20 / 36 flashes.
  - So the soft paste keeps KeySync's sync but exposes its own mouth artefacts (teeth / slide / stick).
  - Next lever: per-frame KeySync → source fallback on those events (r8_fallback-style detection on the KeySync layer), or the k5 fallback flags on the KeySync layer.
- Not done:
  - the r11-sync k5 FALLBACK flags on KeySync (fb7 + `--open-gate 0.04 --dark-transient on --thr-wash-lips 0.2`; repair `--spikes off`). The soft paste has no fallback. Only the guard is used;
  - the full-length labelled + blind clips ("сейчас (LatentSync)" | "KeySync + наша обработка"): made only if a candidate passes the gate vs k5. Today that is old03 only.

Resume (r11-q layout; KeySync layers from the speed agent's runs):
```
# data: artifacts/r4m/ksg_m18 (= q18 KeySync, same dub audio as m18), ksga_qo3 (old03 = qo3, same audio as old03);
# box_*.json in /workspace/dub/ksx/keysync/data; the old03 manifest needs /workspace/dub/r11f/runs/qo3/work/in/qo3.mp4 (symlink to r10/runs/qo3)
P=/workspace/dub/artifacts/r4m/ksga_qo3/work_qo3_ksga_qo3/qo3_ksga_qo3_25fps
python scripts/r11/ks_fix.py --layer ${P}_ls_srt1.mp4 --master $(dirname $P)/video/master_25fps.mp4 --box /workspace/dub/ksx/keysync/data/box_qo3.json \
  --layer-lm ${P}_ls_srt1.mediapipe.npz --src-lm /workspace/dub/artifacts/r4m/ksga_qo3/timing/qo3_source_lm_F.npz --out ${P}_ls_kfglt.mp4 \
  --guard 3 --skin-cm 12 --lip-chroma 1 --chin tex
bash scripts/r11/ks/ksp.sh qo3 ksga_qo3 box_qo3.json ls_kfglt sp_fixh --interp hybrid     # -> /workspace/dub/r11/ksp/qo3_sp_fixh.mp4 + SyncNet / ring / chin
bash scripts/r11/ks/gate_ksp.sh                                                              # gate vs old03base and old03Zk5 -> /workspace/dub/r11/gate_ks.md
python scripts/r11/upper_lip.py --src <master_25fps> --src-lm <source_lm_F> --ls <LS layer> --ls-lm <its mediapipe> --ks-lm ${P}_ls_srt1.mediapipe.npz \
  --rend ks=${P}_ls_srt1.mp4 --rend fix=${P}_ls_kfglt.mp4 --frames 224,440,790,856,894,1634,2022,2112,2194,2196,2208 --out DIR --tag step2
```

r11-sync notes for the KeySync resume (its branch `experiment/r11-composite-sync` @ f5f6856, `reports/experiments/r11-sync/REPORT.md`):
- `--colour-mouth-flat D` ≈ its `--cm-mouth-excl D --cm-mouth-mode mean`.
  - On LatentSync: +0.08 / +0.14 SyncNet, but strong flashes rise on the old videos (old03 35 → 44).
  - Check flashes before using it on KeySync.
- Fallback = the biggest lever: fb7 + `--open-gate 0.04 --dark-transient on --thr-wash-lips 0.2`, r9_layer_repair `--spikes off` (m18 7.74 → 8.01 PASS).
  - Try it as the per-event KeySync → source fallback that the soft paste needs: the teeth / slide / stick events.
- Rejected on LatentSync:
  - `--retime-mouth` (+10-13 flashes on the 30 fps in-between frames);
  - `--mouth-region 0.55` (occluder ring 0 → 4).
- Its old03 numbers use its own fresh baseline (r9g_old03 / r9f_old03, gate base `r10/gate/old03base`). The KeySync gate above uses the same base.
- **Exact k5 recipe** (r11-sync, `origin/experiment/r11-composite-sync` @ 430e3a1, checkout `/workspace/dub/r11_sync`):
  - r8_fallback: `$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2`
  - r9_layer_repair: `--spikes off`
  - fps_composite: V1 + `--cm-mouth-excl 0.5 --mouth-region 0.55 --tex-tau 1000 --tex-sigma 2.5`
  - NOT `--retime-mouth` or `--cm-mouth-mode zero`.
  - Driver: `scripts/r11s/fb_arm.sh`. It assumes the r9g_<id> layer names; copy it and change the layer paths for KeySync.
