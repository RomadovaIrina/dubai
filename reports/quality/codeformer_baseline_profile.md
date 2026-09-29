# CodeFormer legacy profile — 04_baseline.mp4 (week 2, step 1)

Input: `/tmp/dabai_quality/baseline/04_baseline.mp4` 464x848 25.0 fps 1349 frames 53.96 s. GPU NVIDIA GeForce RTX 5090 (sm_120). Script `scripts/quality/codeformer_profile.py`; raw `/tmp/dabai_quality/codeformer/codeformer_profile_04_baseline.json`.

## A. Legacy CLI (`inference_codeformer.py -i video.mp4 -w 0.5 -s 1 --detection_model retinaface_resnet50`, own process)

| wall s | s/video-min | peak VRAM MiB (device / process) | PNG files | disk MB | output geometry |
|---:|---:|---|---:|---:|---|
| 411.406 | 457.46 | 3202 / 2188 | 4047 | 1945 | 512x936 (input 464x848) CHANGED |

## B0. Same loop in-process with timers (1349 frames, 1349 faces, wall 321.377 s = 357.35 s/video-min, peak VRAM 2306 MiB device / 2188 MiB process)

| step | seconds | % of wall | ms / frame |
|---|---:|---:|---:|
| paste_back | 131.728 | 41.0 | 97.65 |
| detection | 48.798 | 15.2 | 36.17 |
| png_write | 44.878 | 14.0 | 33.27 |
| net | 35.397 | 11.0 | 26.24 |
| parsenet | 17.202 | 5.4 | 12.75 |
| video_assembly | 14.804 | 4.6 | 10.97 |
| tensor2img_d2h | 10.569 | 3.3 | 7.83 |
| empty_cache | 6.387 | 2.0 | 4.73 |
| read_image | 4.272 | 1.3 | 3.17 |
| img2tensor_h2d | 3.053 | 0.9 | 2.26 |
| decode | 1.444 | 0.4 | 1.07 |
| model_load | 1.387 | 0.4 | 1.03 |
| align_warp | 1.355 | 0.4 | 1.0 |
| other (untimed) | 0.104 | 0.0 | |

read_image output shape [936, 512, 3] vs input [848, 464, 3] (facelib upscales the short side to 512 before detection; the CLI output keeps that size).

