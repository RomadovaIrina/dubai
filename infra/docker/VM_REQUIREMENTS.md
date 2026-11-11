# What a Docker VM needs for the DabAI image (r16, TZ 5.1 / 0.10 / 10.2)

Vast containers cannot run Docker, so `docker build` has never run. This is what one VM needs to build the image, fill the models volume, and measure image size, cold start and warm job time in one go (`infra/docker/build_smoke.sh`).

## The machine
| Item | Need | Why |
|---|---|---|
| GPU | 1× RTX 5090 32 GB (sm_120); driver ≥ 595 (the TensorRT plans are built for GPU + driver + TRT) | the pipeline peaks at ~30–32 GB in the audio stage; KeySync TRT |
| Docker | Docker Engine ≥ 24 with BuildKit / `docker buildx`; **NVIDIA Container Toolkit** (`--gpus all` works) | build + GPU containers |
| CPU / RAM | ≥ 16 cores, ≥ 64 GB RAM (128 GB comfortable) | the composite / paste are CPU-bound; long videos |
| Disk | **≥ 200 GB free** under /var/lib/docker: build cache ~60 GB, image ~20 GB, models volume ~37 GB, TensorRT plans + inductor ~17–35 GB, jobs | |
| Network | ~48 GB down once (~10 GB base layers + wheels, ~38 GB weights: models.lock.json 37.6 GB); one TCP connection may be throttled (r16-q: PyPI 0.06–0.2 MB/s, HF 4 MB/s; many connections 60–200 MB/s) | fetch-models |
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

## After the smoke: publish (Irina 11.11)
- Target: the PUBLIC `ghcr.io/romadovairina/dubai` (free), linked to the repo by the image label `org.opencontainers.image.source`.
- From the VM (datacentre bandwidth; ~12–15 GB compressed up), never from the laptop:
  `DRY=1 bash infra/docker/push_ghcr.sh <local tag>` (checks only), then `GHCR_TOKEN=... bash infra/docker/push_ghcr.sh <local tag> <tag> latest`.
- The checks (refuse to push on any hit): profile clean; every layer < 9 GB (GHCR: 10 GB / layer, 10 min / upload); `docker history` + config free of
  secret-looking strings; a file scan with no weights / caches / `.env` / credentials / token patterns outside the libraries' site-packages.
- Fixed-dub smoke (the picture compared on the SAME audio): `SMOKE_DUB_DIR=<dir with dubbed_16k.wav + final_mix_aac.m4a>`.

## Vast VM: what to rent (11.11 search, nothing rented)
Vast "VM" instances (`vms_enabled`) can run Docker; ordinary Vast containers cannot. Search: 1× RTX 5090, VM-capable, ≥ 250 GB disk, rental ≥ 7 days.
Estimate = 4 h × ($/h incl. 250 GB disk) + 48 GB down + 15 GB up.
| Offer | Where | $/h incl. 250 GB | $/TB down / up | Ends in | CPU / RAM | Net | Driver | 4 h total |
|---|---|---|---|---|---|---|---|---|
| 47905086 (pick) | Texas | 0.686 | 3.89 / 3.89 | 19.6 d | 32 / 60 GB | 4.6 Gbps | 610.57 | ≈ $2.99 |
| 58914553 (cheaper, slow net) | West Virginia | 0.647 | 2.60 / 3.91 | 8.0 d | 24 / 63 GB | 0.2 Gbps | 595.91 | ≈ $2.77 (+~1 h of downloads) |
RAM ~60 GB is under the 64 GB wanted for the 10-min video tail: run the smoke + load first, the 10-min run last (watch RAM).
