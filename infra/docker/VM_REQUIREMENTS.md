# What a Docker VM needs for the DabAI image (r16, TZ 5.1 / 0.10 / 10.2)

Vast containers cannot run Docker, so `docker build` has never run. This is what one VM needs to build the image, fill the models volume, and measure image size, cold start and warm job time in one go (`infra/docker/build_smoke.sh`).

## The machine
| Item | Need | Why |
|---|---|---|
| GPU | 1× RTX 5090 32 GB (sm_120); driver ≥ 595 (the TensorRT plans are built for GPU + driver + TRT) | the pipeline peaks at ~30–32 GB in the audio stage; KeySync TRT |
| Docker | Docker Engine ≥ 24 with BuildKit / `docker buildx`; **NVIDIA Container Toolkit** (`--gpus all` works) | build + GPU containers |
| CPU / RAM | ≥ 16 cores, ≥ 64 GB RAM (128 GB comfortable) | the composite / paste are CPU-bound; long videos |
| Disk | **≥ 200 GB free** under /var/lib/docker: build cache ~60 GB, image ~20 GB, models volume ~37 GB, TensorRT plans + inductor ~17–35 GB, jobs | |
| Network | ~45 GB down once (~10 GB base layers + wheels, ~37 GB weights); HF ≥ 40 MB/s recommended (~20 min) | fetch-models |
| OS | Ubuntu 22.04 / 24.04, root or docker group | |

Typical offers: a bare-metal / VM 5090 host (RunPod "GPU Pod" with Docker is a container, not enough; needs a VM such as RunPod / Lambda / Hetzner GEX / OVH bare metal / Vast "VM" instances where available).

## Inputs to bring
- The repo at the release commit (`git clone` + `git checkout experiment/r16-int` or the bundle).
- `HF_TOKEN` with the terms accepted for `pyannote/speaker-diarization-3.1` and the SVD VAE repo (KeySync).
- `KS_KIT_DIR` with 4 files of ours: `lora_pick.pt`, `lora_meta.json` (laptop `C:\repos\chkpts\keysync_student_r15_lcm_r1\`), `kf.pt`, `interp.pt` (`C:\repos\chkpts\keysync_trt_r15\keysync\trt\cap\`).
- Optional: `STUDENT_LORA_FILE` (the LatentSync student, mode fast on the LS engine).
- One smoke clip (e.g. m03 20 s).

## One command
```bash
export DUB_PROFILE=clean HF_TOKEN=... KS_KIT_DIR=/path/kit
bash infra/docker/build_smoke.sh /path/m03_20s.mp4 build_smoke_out
```
Writes `build_smoke_out/report.json`: build_s, image_bytes, fetch_s, volume_gb, check, smoke_cold_s (first job = TensorRT build), smoke_warm_s, trt_cache_gb, api_healthy_s. Pushes nothing.

## What the numbers answer
- **5.1** the working image exists and passes `check` + a real job.
- **0.10 / 10.3 cold start**: smoke_cold_s − smoke_warm_s = the one-time TensorRT + inductor build per GPU / driver (r16 estimate ≈ 10–17 min); api_healthy_s = container start to `/health` 200. A < 60 s cold start needs the plans pre-built on the volume for the exact GPU / driver (build once per fleet type, keep the volume).
- **10.2** the models volume size with KeySync.
