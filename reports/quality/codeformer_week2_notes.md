## Reading

### Where the legacy time went (04, 1349 frames, this host)
The upstream CLI takes 411 s (457 s/video-min; the pilot 0.6 host measured 197 s = 222 s/video-min for the same command — this
box is slower on the CPU-bound parts, so all comparisons below are same-host). The instrumented replay (321 s) shows the
neural network is a minority of the time: paste-back over the whole (upscaled) frame 41 %, RetinaFace detection 15 %,
PNG writes 14 %, CodeFormer net 11 %, ParseNet 5 %, mp4 re-assembly 5 %, per-sample D2H 3 %, per-face
`torch.cuda.empty_cache()` 2 %. `read_image` also upscales every 464x848 frame to 512x936 before detection, which is why
the pilot 0.7 outputs changed geometry.

### What the optimized stage removes, and what each change is worth (micro-benchmarks, 200 real LATENT_SYNC frames)
| step | legacy execution | optimized | evidence |
|---|---|---|---|
| second face detector | RetinaFace per frame inside CodeFormer (36 ms/frame on the upscaled frame) | none: the 5 points come from LatentSync's insightface verify pass (already computed for every LATENT_SYNC frame, 8.7 ms/frame paid anyway); mapping measured, not assumed | landmark check: derived points within RetinaFace's own 2-3 px self-disagreement; `b8_retina` (fresh RetinaFace per frame) is 1.6x slower and not better (SyncNet 2.44 vs 2.46) |
| PNG / mp4 round trips | 4047 PNGs, 1.9 GB, re-read for assembly | frames stay in RAM, no intermediate files | profile: 14 % + 5 % of the legacy wall |
| model load | per CLI call | once per process | `load_s` ~1.4 s |
| geometry | 464x848 -> 512x936 | preserved | every candidate: output == input geometry, frame count, duration |
| paste-back | whole frame, CPU, float64 masks | same arithmetic on the ROI the warped square can reach (+ blur margin); identical result | `--check-reference`: PSNR 76-88 dB, max diff <= 5 vs the upstream helper functions with the same landmarks |
| ParseNet mask blur | 2 x 101x101 Gaussian on CPU per face (10 ms) | separable reflect-101 conv on the GPU for the batch (0.1 ms/face) | max diff 2.4e-4 on the [0,1] mask |
| `empty_cache()` per face | yes | no | b8 vs b8_empty: 28.4 -> 26.2 faces/s (-8 %); b1: 25.4 -> 24.1; peak VRAM unchanged (the caching allocator reuses LatentSync's freed blocks: E2E nvsmi peak 23.4 GB with or without CodeFormer) |
| batch | 1 | 8 | 25.4 -> 28.4 faces/s (b16 29.7 for 9.5 GB); outputs differ from batch 1 by cuDNN algorithm choice only (PSNR 69 dB mean / 46 dB min, ~260 px per frame > 2 levels) |
| D2H | `tensor2img` per sample (8 ms/face) | batched GPU uint8 conversion, one copy (2.4 ms/face), bit-identical | micro-profile |
| CPU / GPU overlap | sequential | GPU batch k+1 on its own thread while batch k is pasted; 4 CPU workers | b8_nooverlap 22.2 -> b8 28.4 faces/s; workers 1 -> 4: 21.1 -> 28.4 |
| fp16 | fp32 | autocast for CodeFormer net + ParseNet (opt-in `--codeformer-precision`, now default) | net 20.6 -> 16 ms/face, ParseNet 13 -> 9 ms/face, 0 label flips; E2E 04: 39.2 -> 29.7 s, SyncNet 2.45 vs 2.46, same sharpness/flicker; `.half()` weights break upstream code (dtype mismatch in the quantizer) and were not pursued |

Not adopted: `use_parse=False` (25 faces/s but changes the mask semantics), reuse of affine transforms across neighbouring
frames (the face moves; every frame gets its own landmarks), TensorRT / torch.compile (out of scope this week).

### Where the floor is now
At batch 8 the GPU is the bottleneck: CodeFormer net 16-21 ms/face + ParseNet 9-13 ms/face; everything else overlaps.
04 is the worst case of the dataset (1099 of 1349 frames are LATENT_SYNC): 29.7 s fp16 = 33 s/video-min, 39 s fp32 = 44 s/video-min.
The business target (<= 30 s/video-min) is therefore met on the videos with fewer lip-synced frames and missed by ~10 % on 04;
the remaining lever without touching the model/torch stack is the number of eligible frames, and with it TensorRT or
torch.compile of the two networks (explicitly out of scope in week 2).

### Quality (04, SyncNet on the full output; LS-only 2.76, original 3.24)
w 0.5 / 0.7 / 0.9 / 1.0 -> 2.46 / 2.55 / 2.61 / 2.67, AV offset 0 for every weight; mouth sharpness 1.06-1.07x, upper-face
sharpness 0.85-0.89 of the master (LS-only: 0.62-0.76), eyes 1.26-1.32x sharper than LS-only, flicker unchanged (0.98-0.99),
seam gradient ratio 0.99 (no edge added at the mask boundary), no pixel changed outside the warped face square before
encoding (feather zone only), pass-through frames untouched (their differences to the LS-only file are libx264 re-encode noise,
PSNR ~44 dB). Fidelity weight chosen: **w = 1.0** — smallest SyncNet drop (-0.09) with the face still visibly restored; the pilot
0.7 finding that lower w costs lip sync is reproduced (monotonic). Pilot 0.7's strict "no drop" policy is still not met
by any weight; that decision stays a project one.

### Landmark source decision
insightface-derived points (reused) vs a fresh RetinaFace pass per frame: same SyncNet within 0.02, same seams, the
RetinaFace variant is 1.6x slower (fallback detection on all 1099 frames). The alignment is a real 5-point similarity
transform in both cases — never a bbox resize.

### Dataset (chosen config: w 1.0, batch 8, fp16 autocast, insightface landmarks)
| video | LATENT_SYNC faces / master frames | CodeFormer s | s/video-min | SyncNet LS-only -> +CodeFormer (orig) | AV offset | valid |
|---|---|---:|---:|---|---:|---|
| 01 | 260 / 2542 | 7.1 | 4.2 | 6.84 -> 6.57 (5.36) | 0 | PASS |
| 02 | 0 / 2773 (no face anywhere) | 0.0 | 0.0 | N/A (CodeFormer never invoked: 0 eligible frames, 0 faces, 0.0 s) | – | PASS |
| 03 | 2031 / 2260 | 56.0 | 37.2 | 4.65 -> 4.37 (4.89) | 0 | PASS |
| 04 | 1099 / 1349 | 29.7 | 33.0 | 2.76 -> 2.67 (3.24) | 0 | PASS |
| 05 | 899 / 1311 | 24.0 | 27.4 | 2.61 -> 2.50 (1.71) | 0 | PASS |
Dataset: 116.8 s of CodeFormer for 6.82 video-min = **17.1 s/video-min** (legacy CLI on 04 alone: 457 s/video-min on this host,
222 on the pilot host). Per-video the stage is <= 30 s/video-min on 01/02/05 and 33-37 on 03/04, whose frames are 81-90 %
lip-synced: the stage cost is proportional to the number of LATENT_SYNC frames, not to the video length. Every output keeps
width/height/fps/frame count/duration, no pixel changes outside the warped face square before encoding, seam ratio 1.00-1.06,
AV offset 0 on all four scoreable videos, SyncNet drop 0.09-0.28 (w 1.0).

### Full E2E proof (audio stages included), `run_clean_pipeline_05.py ... --codeformer optimized` on 05
validation PASS, 464x848 / 1311 frames / 52.44 s preserved, 899 faces restored in 27.7 s (CodeFormer load 0.6 s), LatentSync 196.8 s,
total 303.5 s; nvidia-smi peak 22.5 GB — no VRAM increase over the LatentSync-only run (the caching allocator reuses freed
LatentSync blocks; CodeFormer alone needs ~5 GB at batch 8). `--no-codeformer` / `--codeformer off` still runs the frozen
baseline path (the 03/05/02 baselines above were produced with the integrated code and validate identically).

### Verdict against the acceptance criteria
Correctness: 5/5 valid, no crash, exact frame count, resolution and duration preserved, changes confined to the face square + feather. PASS.
Lip sync: AV offset never worsens (0 -> 0); SyncNet confidence drops by 0.09-0.28 (w 1.0) — reported, not hidden; the strict
"no drop" policy of pilot 0.7 is still not met by any weight. Visual: faces visibly sharper than LatentSync-only (eyes 1.3-4x,
mouth 1.06-1.9x on 01/04/05), no rectangular seams, flicker unchanged (0.97-1.13). Performance: dataset 17.1 s/video-min
(PASS vs <= 30), worst-case videos 03/04 33-37 s/video-min (FAIL vs 30 by 10-25 %); the floor is the CodeFormer net + ParseNet
GPU time (~26 ms/face in fp16 at batch 8), everything else is overlapped or removed.
