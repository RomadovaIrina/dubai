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
