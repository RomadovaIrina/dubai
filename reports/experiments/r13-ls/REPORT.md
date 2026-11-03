# r13 LS (LatentSync speed + quality) report (03.11.2026, r13-l)

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

## Open
- **The eye:** even 3/4 (m03_k5_vs_e34*, e34q*) and q2 (m03 / old03 _k5_vs_cmsr*).
- **m18:** q2 / e34q (running at this commit).
- **E2E with the gen worker** on the final stack.
- **The lip-corner defect:** a corner close-up sheet (`corner_sheet.py`) shows no obvious composite artefact on m03; timestamps from management are needed.

## Clips
`/workspace/dub/outbox/r13_ls/` on r13-l. Each `<id>_k5_vs_<arm>.mp4` is full length; `_mouth05x` is 0.5x, the window where the two differ most.
