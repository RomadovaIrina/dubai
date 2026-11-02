# r11 KeySync bake-off vs LatentSync r9+fix (r11-s, 01.11)

## Licence
- Code: antonibigata/keysync 5e2503b, Apache-2.0.
- Weights: SVD-derived; Stability Community License. Commercial use needs Stability registration and revenue < $1M. User decision 01.11.
- Other inference dependencies:
  - hubert-base-ls960: Apache-2.0
  - WavLM-Base+: MIT
  - face_alignment: BSD-3
- Not downloaded: SAM2, the SVD base, the VSR model.

## Setup
- /workspace/dub/keysync: venv with torch 2.7.1+cu128, 13 GB of weights.
- Patches for sm_120 / torch 2.7 are tagged "keysync bake-off" in code/:
  - numpy weights_only allowlist;
  - SDPA instead of xformers (xformers 0.0.31 FA3 fails on sm_120);
  - skip-attention mask shape;
  - optional face-ID imports;
  - WavLM path.
- Input: a static square crop, 25 fps, 512², plus our dub speech (dubbed_16k.wav). The authors' default settings (12 steps keyframe + interpolation).
- Output: raw paste back (ks_paste.py): mci interpolation to the source fps, 24 px feather, our final mix.
- old03 = test_videos_raw/03.mp4: the face agent's "q03" (there are no m03 artefacts on r11-s).

## Speed (generator only, NVML GPU-busy, 100 ms)
| video | dur s | wall s | GPU-busy s | GPU-busy s/vmin |
|---|---|---|---|---|
| m18 | 20.0 | 347 | 327 | **979** |
| old03 | 90.3 | 1528 | 1487 | **988** |
- LatentSync face_gen is ~210-230 s/vmin; the whole-pipeline norm is 300.
- Peak VRAM ~22 GB.

## Quality (r4_gate compare, KeySync / r9+fix; metrics/<id>/gate_ks_vs_r9f.md)
| metric | m18 | old03 |
|---|---|---|
| SyncNet dub speech (gate: >= base - 0.3) | 5.20 / 7.74 FAIL | 5.26 / 7.18 FAIL |
| strong lip flashes | 3 / 8 better | 21 / 35 better |
| colour worst 1 s dL | 6.08 / 1.37 FAIL | 3.78 / 1.38 FAIL |
| chin wave (2 s std) | 10.9 / 3.8 FAIL | 4.8 / 1.2 FAIL |
| below-jaw band | 9.8 / 2.3 FAIL | 26.2 / 2.5 FAIL |
| mouth sharpness (src=1) | 1.08 / 1.16 FAIL | 0.79 / 0.83 FAIL |
| face_metrics slide events ev / S | 106 / 29 vs 18 / 5 | 351 / 68 vs 3 / 0 |
| whole-frame gate | FAIL | FAIL |

- The colour / below-jaw / bg-ring fails come partly from the raw square paste and the 512 -> face-box upscale; our composite would cut those.
- The SyncNet -2 and the slide / chin wave are the generator's own: it re-renders the whole crop every frame, so the face swims.
- They also hold without interpolation: a nearest-frame paste on m18 gave swim 0.0207 vs 0.0199, and flashes 39.

## Verdict
- KeySync is not a better generator for us: lip sync is clearly worse (SyncNet -2.5 / -1.9), the lower face swims, and it costs 3.3x the norm in the generator alone.
- Its one plus: fewer strong lip flashes.

## Clips
/workspace/dub/outbox/r11_keysync/{m18,old03}_2panel_{1x,0.5x}.mp4: 7 s at the most mouth movement, face crop, left r9+fix, right KeySync, about 9.2-9.4 MB each.

# Round 2 (01.11, the user reopened KeySync): KeySync as the layer in our composite

## What changed
- **Sync offset:** KeySync's output has a constant AV offset per video: m18 −1 frame, old03 +1 frame (58 of 58 speech seconds at +1).
  - `ks_mkgen.sh` (`KS_SHIFT=auto`) measures it with SyncNet on KeySync's own output and shifts the layer.
- **Layer:** `ks_layer.py` puts the crop into the master_25fps frames, Lab mean/std colour match to the source box (±12 frames).
- **Composite:** `r7_m7_comp.sh` runs unchanged with the r9+fix recipe (SR=srt1, FB=fb7, tex-tau 1000), **without REP**.
  - r9_layer_repair flags KeySync's lips as landmark_slide / lip_colour.
  - With REP=1, m18 fell back to the source mouth on 143 of 475 frames (29.6 %).
  - There is no CodeFormer-detail stage, because KeySync has no pre-CF dump.

## Gate (KeySync in composite / r9+fix): FAIL on both videos
| metric | m18 (ksfn_q18) | old03 (ksfa_qo3) |
|---|---|---|
| SyncNet dub speech | 6.70 / 7.74 | 5.69 / 7.18 |
| strong flashes | 6 / 8 | 33 / 35 |
| colour worst 1 s | 2.76 / 1.37 | 1.23 / 1.38 |
| chin wave 2 s | 9.25 / 3.79 | 2.33 / 1.24 |
| twitch mouth | 1.20 / 1.18 | 1.43 / 1.01 |
| jaw jitter | 1.13 / 1.05 | 1.08 / 0.93 |
| mouth / dub mismatch s | 0.0 / 2.6 | 1.47 / 1.67 |

- SyncNet on m18 by stage: KeySync layer 7.83 → final 6.70.
- LatentSync for comparison: layer 8.72 → final 7.74.

## Steps vs cost (m18; GPU-busy for the KeySync stage only; the composite is extra)
| config | sampler iterations | sampling s | GPU-busy s | s/vmin | SyncNet (composite) | chin wave |
|---|---|---|---|---|---|---|
| default 10 | 12 | 266 | 327 | 979 | 6.70 | 9.25 |
| 8 | 10 | 220 | 282 | 845 | 6.79 | 9.19 |
| 6 | 8 | 192 | 262 | 783 | 6.76 | 9.16 |

- Quality is flat from 6 to 10 steps.
- The fixed part (model load, landmarks, VAE, HuBERT/WavLM) is ~60-70 s per 20 s video, about 200 s/vmin.
- So the generator stage alone stays above 700 s/vmin. LatentSync's generator stage is ~210-230 s/vmin.

## Clips
`m18_comp_2panel_{1x,0.5x}.mp4` and `old03_comp_2panel_{1x,0.5x}.mp4`: r9+fix | KeySync in our composite (10 steps), the same crop on both panels.

# Round 3 (01.11): KeySync speed on one 5090. The user prefers KeySync

## Method
- Resident worker `ks_fast.py`: models loaded once; the authors' `sample()` runs unchanged with the infer_raw.sh arguments; per-stage cuda-synced timing.
- s/vmin = worker wall per video-minute. The GPU is ~94 % busy across the run (NVML), so this is ~GPU-seconds.
- Covers KeySync only, not the rest of the pipeline.
- Gate vs the **KeySync baseline**, per `ks_eval.sh`:
  - SyncNet AV offset of the raw output → paste with that shift → SyncNet on dub-speech seconds at offset 0, on a 25 fps rendition;
  - strong lip flashes (r3_flash).
  - Rule: SyncNet ≥ base − 0.3; flashes no worse (noise across identical-quality arms is ±1-2).

## Lever table (m18 unless noted)
| arm | s/vmin | SyncNet @0 | flashes | verdict |
|---|---|---|---|---|
| authors' settings, my first port (3-D SDPA) | 1002 (old03 955) | 7.68 (old03 7.54) | 5 (22) | baseline |
| SDPA 4-D fix (flash kernel; 3-D input fell to MATH) | 971 (old03 921) | 7.68 (7.53) | 6 (22) | = (new baseline) |
| resident worker (model load) | load is 7.7 s per process | - | - | negligible |
| VAE decode 7 frames / call (fp32; 14 = OOM) | 1011 | 7.72 | 6 | no gain |
| interpolation chunk 6 segments / call (authors 2) | 998 | 7.68 | 6 | no gain (GPU saturated) |
| 6 steps (8 sampler iterations) | 714 | 7.81 | 5 | pass |
| DeepCache 2:0 (KS_DC) | 659 | 7.70 | 5 | pass |
| DeepCache 3:0 | 549 | 7.57 | 8 | borderline flashes, not taken |
| 6 steps + DeepCache 2:0 | 520 | 7.26 | 11 | **FAIL** |
| DeepCache 2:0 + fp16 VAE decode | 629 | 7.68 | 6 | pass (decode 38 -> 29 s) |
| + attention fast path (no index copy / scatter / sync) | 611 | 7.70 | 4 | pass |
| 8 steps + DeepCache 2:0 + fp16 VAE | 554 | 7.75 | 5 | pass |
| DeepCache 2:0 + fp16 VAE + torch.compile UNet (warm) | 531 | 7.71 | 5 | pass |
| **best: 8 steps + DC 2:0 + fp16 VAE + fast path + compile UNet (warm)** | **450 (old03 444)** | **7.71 (old03 7.72)** | **5 (22)** | **pass** |
| torch.compile of the VAE decoder | - | - | - | fails (temporal decoder attribute) |
| landmark batch 40 | = | - | - | no gain (5.5 s warm) |
| skip silent + closed-mouth spans | m18 0 %, old03 7.7 % of frames skippable | - | - | small, not built |
| keyframe density | - | - | - | not a lever: interpolation generates every frame in 14-frame windows |

- Warm compile: the first video in a fresh process pays the compile; m18 then took 565 s/vmin.
- Inductor caches kernels across runs (/tmp/torchinductor_root).

## Where the best arm's time goes (old03, 669 s for 90.3 s)
| stage | s |
|---|---|
| interpolation sampling | 422 |
| interpolation VAE decode | 133 |
| keyframes | 45 |
| landmarks | 26 |
| VAE encode | ~25 |
| save | ~11 |

## Bottom line
- KeySync alone is **~445 GPU-s/vmin** at baseline-equal quality, down from 921-1002. On m18 the arm runs about 2.2x faster.
- Add the rest of the pipeline: audio + sep ~55, the composite GPU part ~25-35. End to end that is ~530 s/vmin vs the 300 norm.
- Untried levers that could still matter:
  - CFG-pass caching: the uncond pass every other step, ~-15 %.
  - A compiled / fp16 temporal VAE decoder: decode is ~20 % now.
  - TensorRT for the UNet.
  - Step distillation, which needs training.
