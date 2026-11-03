# r13 LS (LatentSync speed + quality) HANDOFF (04.11.2026, r13-l; destroyed after this push; LS work paused by the user)

## Status in one screen
- **q2 = the LatentSync quality stack: gated, to keep.**
  - Flags (all default off): `LS_CMS_READ=1 FC_CMS=1` + `FC_EXTRA="--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"`.
  - Use 0.2: the 0.5 lip cap gave m18 lips 1.68 x the source.
  - Full gate PASS on m03 / old03. m18 trips only the occluder ring 0 -> 1, which the plain k5 control does too.
  - User on the per-channel-sharpen version "q": "цвета немного поплыли, но само по себе стало лучше". q2 is the luma-only + capped answer to that; not yet judged by eye.
- **E2E (one command, audio + face, single video, no gen worker), q2**, NVML GPU-busy (util-weighted) / wall s/vmin:
  - m03: 341 (304) / 1023;
  - old03: 346 (309) / 712;
  - m18: 359 (316) / 1259.
  - Over the 300 norm by 15-20 %. q2 costs no GPU vs k5.
- **LS_RATE (frame skipping) is NOT accepted:**
  - plain 2/3: "лицо подергивается и дрожит";
  - adaptive: "да, лицо дрожит";
  - even 2/3: "подрагивает, но сильно лучше";
  - **even 3/4 + q2 (e34q): "ой блять это пиздец"** on m03 at 0:02 of the 0.5x crop.
- **The grotesque mouth: analysed** (`m03_e34q_bad_mouth_566-572.jpg`; rows = source / k5 / q2 / even 3/4 / even 3/4 + q2, source frames 556-574):
  - At 566-572 both even-cadence arms show the wrong mouth: a flat teeth band, a torn dark left corner at 570, then pursed lips.
  - k5 and q2-only show the correct round "o".
  - **Cause: the even-cadence resampling (`LS_RATE_EVEN`), not q2.** Either the flow-interpolated half-step input frame (which LatentSync then lip-syncs) or the flow fill between generated frames tears at the corner. The audio window at the half-step time may also differ from the frame's true time.
  - The dark corner gap may be the same kind of thing as the management's "кончики губ" defect; that is still unconfirmed (no timestamps).
  - Not done: a scan of all e34q videos for mouth-shape outliers (per-frame mouth landmarks vs k5).
- **E2E with q2 + even 3/4:** started and stopped at the wrap-up (old03 was in face_gen). No numbers.

- **Branch:** `experiment/r13`. All levers are behind flags, default off.
- **Gate:** the r9-speed-face gate vs the k5 base of each video, `r10/gate/<id>base`.
- **Face GPU:** NVML GPU-busy s/vmin of the generator + the k5 composite, warm process.
- **SyncNet:** the gate's `sync_speech`.

## k5 controls (r13-l)
| video | k5 base SyncNet / strong flashes | warm control face GPU | control SyncNet |
|---|---|---|---|
| m03 | 8.168 / 18 | 280 | 8.160 (gate PASS) |
| old03 | 7.534 / 42 | 286 | 7.512 (PASS) |
| m18 | 8.291 / 7 | 281 | 8.073 (occluder ring 0→1, noise at identical settings) |

E2E k5 baselines (cold-ish box, NVML GPU-busy E2E): m03 510 (first run, VAE autotune), old03 346, m18 371.

## Speed: LS_RATE (frame skipping)
| arm | flags | m03 face GPU / SyncNet | old03 | user's eye |
|---|---|---|---|---|
| 2/3 | `LS_RATE=2/3` | 197 / 8.131 (gate PASS, pressed at limit) | 216 / 7.471 (flashes +3) | **"лицо подергивается и дрожит"**: OUT |
| 2/3 + ss10 | `+ --steps-silent 10` | 176 / 8.165 (PASS) | 177 / 7.462 (pressed lips 4.23 s FAIL) | out |
| fill hf / near | `LS_RATE_FILL=hf/near` | 204 / 8.02; 209 / 7.81 | | the metric does not move: out |
| adaptive | `LS_RATE_ADAPT=0.5` (skip in calm blocks only, 16 %) | 233 / 8.189 | | **"да, лицо дрожит"**: OUT |
| even 2/3 | `LS_RATE_EVEN=1` (generate on an even 16.7 fps grid) | 213 / 8.259 | | **"подрагивает, но сильно лучше"** |
| even 3/4 | `LS_RATE=3/4 LS_RATE_EVEN=1` + full colour stack | 227 / 8.142 | | pending |
| even 4/5 | `LS_RATE=4/5 LS_RATE_EVEN=1` + full colour stack | 243 / 8.031 | | pending |
| **q2 + even 3/4** | e34q | **223 / 8.028**, flashes 13 | **231 / 7.296**, flashes 34 | pending |

Jitter metrics (`scripts/r12/`):
- `jerk.py` (MediaPipe landmarks) and `wobble.py` (lower-face flow vs the source) are BLIND: they don't separate the trembling versions.
- `mouth_pop.py` (the mouth's flow-compensated temporal residual): its share at the skip cadence is 0.073 base / 0.153 plain 2/3 / 0.106 adaptive / 0.134 even 2/3.
  - The even 3/4 and 4/5 arms equal the control at their own cadence.
  - A fill frame lowers the residual by construction, so this metric is a hint, not the eye.

## Quality
- **Colour.**
  - The sources are BT.709-tagged. OpenCV read the master with another matrix (m03 +1.7 L / +2.7 b*), the finals were untagged, and every yuv420p layer step lost ~0.7 L / +0.4 b*.
  - k5 finals vs the source (same reader): m03 background +0.99 L / +3.17 b*, box +1.75 / +4.70, lips +1.90 / +5.62; old03 background −1.39 L, box b* +2.55.
- **Flags:**
  - `LS_CMS_READ=1`: the generator reads the master with its own matrix.
  - `FC_CMS=1`: `fps_composite` decodes / encodes with explicit BT.709 + a BT.709-tagged output.
  - `LS_LAYER_RGB=1`: lossless RGB intermediate layers. Best colour, but +4–6 strong flashes on m03 (the layers keep the generator's frame-to-frame detail): NOT in the stack.
- **Detail:** `--sharpen-luma on` (the unsharp mask on luma only, a*/b* untouched) + `--sharpen-auto 1.05 --sharpen-lips-max 0.5`.
  - The colour + per-channel sharpen version "q" was judged **"цвета немного поплыли, но само по себе стало лучше"**. q2 is luma-only and capped.
- **`ks_tsmooth` on the LS layer:** no gain. The layer's mouth jerk-spike rate = the source's; it costs −0.10 / −0.05 SyncNet.

**q2 = `LS_CMS_READ=1 FC_CMS=1` + `FC_EXTRA="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.5"`, full gate PASS on m03 and old03:**

| video | SyncNet (k5) | strong flashes (k5) | box ΔL / Δb* vs source (k5) | lip sharpness vs source (k5) |
|---|---|---|---|---|
| m03 | 8.202 (8.168) | 19 (18) | −0.65 / +0.58 (+1.75 / +4.70) | 1.10 (0.98) |
| old03 | 7.557 (7.534) | 30 (42) | −0.99 / +0.63 (−0.19 / +2.55) | 0.79 (0.67) |

m18:
- q2 (cmsr): face GPU 301, SyncNet 8.041 (the k5 control rerun 8.073), flashes 5, occluder 0 -> 1.
- q2 + even 3/4: 249, SyncNet 8.005, flashes 3, gate PASS on every row.

## NEXT (when LS resumes)
1. **Eye check of q2 alone** (lip cap 0.2) on m03 / old03 / m18, then make q2 the LS default: `E2E_Q2=1` in `scripts/r9/e2e.sh` setting the three flags + FC_EXTRA.
2. **Speed without frame skipping:**
   - the resident gen worker (`scripts/r10/worker.py --gen-worker`) on the q2 E2E;
   - measure the queue's GPU-busy vs 300.
   - r10 had 308 with the worker; q2 single-video is 341-359.
3. **LS_RATE_EVEN (if retried):**
   - scan for mouth-shape outliers (layer landmarks vs the k5 layer: per-frame lip-contour distance / IOD, top spikes);
   - check the audio chunk times at the half-step frames;
   - try generating on the TRUE source frames nearest the even grid, instead of flow-interpolated inputs.
   - Until then it is rejected.
4. **Lip corners:** management timestamps -> `scripts/r12/corner_sheet.py` (source | layer | final) -> generator vs composite.
5. **m03 flash count with the lossless layers (LS_LAYER_RGB):** +4-6. If the lossless layers are ever wanted for colour, add temporal HF consistency on the layer first.

## Resume
```
git checkout experiment/r13; bash scripts/r9/r9_models.sh; uv pip install --python /venv/main/bin/python nvidia-ml-py
NOFX=1 bash scripts/r12/ls_base.sh m03 old03 m18                      # k5 baselines + gate bases
export LS_CMS_READ=1 FC_CMS=1 FC_EXTRA_ADD="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
QUICK=1 bash scripts/r12/ls_arm.sh m03 q2 ""                           # q2 arm (drop QUICK for the full gate)
bash scripts/r12/clip_ls.sh m03 q2 "k5 + q2"                            # 2-panel + 0.5x mouth clip
# E2E: see reports/experiments/r13-ls/queues/e2eq.sh
```

## Not in git
On r13-l only, gone with the box:
- the clips in `/workspace/dub/outbox/r13_ls/`;
- the artefact caches.

## Open
- The eye on q2 (cap 0.2).
- The gen-worker E2E.
- The lip-corner timestamps.
- The e34q outlier scan.

## Clips
`/workspace/dub/outbox/r13_ls/` on r13-l. Each `<id>_k5_vs_<arm>.mp4` is full length; `_mouth05x` is 0.5x, the window where the two differ most.
