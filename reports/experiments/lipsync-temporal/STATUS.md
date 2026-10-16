# Track A — lip-sync / temporal generation (`experiment/lipsync-temporal`) — STATUS

Research question: is LatentSync itself the main source of the visible mouth / face jitter, or is it what is around it (crop / affine,
landmarks, mask, paste-back, speech gate, LS<->original switches)? Then: a more stable lip-sync backend, or a way to stabilise the current one.

| phase | state | report |
|---|---|---|
| 0 reproduce baseline (cand5) | **done**: 5/5 PASS, metrics saved | `PHASE0.md`, `baseline/` |
| 1 prior work, diagnosis plan, shortlist, licenses | **done** (coordinator pre-approved continuing) | `PHASE1_PLAN.md`, `LICENSES.md`, `research/` |
| 2 cheapest A/B on a 03 window (40-58 s) | **running**: L0/L1/L3 done, L2/S1/S2/S3 + TBDub (installed in /venv/tbdub, all sub-models Apache-2.0) next | `p2/` (soon) |
| 3 refine winner | – | – |
| 4 E2E regression 01/03/04/05 (+02) | – | – |
| 5 FINAL.md | – | – |

## Server A (2026-10-16)

```text
host         vast.ai container 01d9e6967bbe (Denmark), unprivileged Docker, /workspace NOT a volume (nothing survives recycle/destroy)
GPU          NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0, driver 590.48.01 (CUDA 13.1 driver)
CUDA         toolkit nvcc 13.0 (V13.0.88, image); project torch 2.7.1+cu128 (wheel CUDA 12.8), onnxruntime-gpu 1.26.0 (CUDA EP + TensorRT EP)
CPU          AMD EPYC 7R32 48-Core (host 96 threads visible), cgroup quota 2304000/100000 = 23.0 vCPU; DUB_THREADS stays 16 (production)
RAM          cgroup limit 75.4 GiB (host 157 GiB)
disk         120 GB overlay, 88 GB free after env + weights (33 GB used)
Python       /venv/dabai = Python 3.11.17 (conda), torch 2.7.1+cu128, diffusers 0.32.2, transformers 4.48.0, insightface 0.7.3
repo         RomadovaIrina/dubai @ tts-intelligibility-fix HEAD 559d700e0bf14ffa5692d855e5eae10a42eadd11 (= the coordinator's SHA)
branch       experiment/lipsync-temporal created from 559d700, pushed; fetch refspecs limited to tts-intelligibility-fix + this branch
third_party  latentsync a229c3948406bc2cf6eaf4873e662e70c6a04746, CodeFormer b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b (= reports/third_party-commits.txt)
setup        scripts/setup_dabai.sh stages 0-9 exit 0 (2 min 28 s); pip drift vs reports/pip-freeze.lock.txt = unpinned packages only
             (aiohttp, lmdb 2->3, multidict 6->7, pydantic, ...; torch / CUDA / transformers / diffusers / numpy pins unchanged);
             freeze + diff kept in /workspace/dub/artifacts/lipsync-temporal/env/, the committed lock files are left unchanged
weights      scripts/download_models.sh exit 0 (3 min 56 s, 20 GB); latentsync_unet.pt sha256 0a478e89...316d3 = HF LFS sha256
             (one read timeout was retried by huggingface_hub). LatentSync's insightface buffalo_l pack is auto-downloaded on first use
             into third_party/latentsync/checkpoints/auxiliary/models/buffalo_l (det_10g, 2d106det, ...) -> license note in LICENSES.md
smoke        scripts/run_smoke.sh 11/11 PASS (10 min 41 s; latentsync 280 s, codeformer 164 s)
inputs       /workspace/dub/inputs/test_videos_raw/{01.MP4,02..05.mp4}, sha256 -c OK on all five (05.zip sha256 9e50559e44c2a006...),
             symlinked per file into test_videos/ (git-excluded via .git/info/exclude)
formats      01 480x848 avg 25.0 fps (r 30), 02 464x848 29.98, 03 464x848 30.0 CFR, 04 464x848 30.0 CFR, 05 464x848 29.9;
             the pipeline works on a 25 fps master -> on 30 fps sources every 6th frame is dropped in the whole output
```

## Cost so far

GPU: baseline 5 videos 3725 s wall (LatentSync 1023 s, CodeFormer 129 s) + smoke 10.7 min + metrics (SyncNet, face_temporal,
lipsync_jitter) ≈ 30 min. No candidate model installed, no experiment GPU time yet.

## Phase 2 notes (live)

* L0 (no generation: crop -> inverse warp -> CodeFormer -> mouth composite) on 03 40-58 s, metric v2 (native resolution): mouth
  warping error vs source = 0.96 BEFORE CodeFormer, **1.29 after CodeFormer** (upper face 0.96 -> 1.31 before the mouth composite).
  CodeFormer's per-frame re-synthesis alone adds ~33 % unexplained mouth texture change with no lip-sync at all -> relevant for track B;
  CodeFormer stays fixed in this track's A/Bs, every arm is measured at three stages (raw LatentSync / after CodeFormer / composite).
