# r10 SPEED: REPORT (31.10–01.11.2026, r10-s, one RTX 5090, 24 vCPU, 63.8 GB cgroup RAM)

Branch: `experiment/r10-base` (pushed). The spec norm: **≤ 300 GPU-seconds per video-minute** for the full E2E on one RTX 5090.

**Clocks:**
- **wall** = elapsed s per video-minute;
- **GPU-busy** = s per video-minute with NVML device utilisation > 10 % (100 ms samples);
- **util-weighted** = Σ util/100 · dt. It is the lower bound of "GPU-seconds" and is reported next to GPU-busy.

Production stages only: the metrics / gate passes are not counted.

## Bottom line

- **Single video, default stack:** GPU-busy 302–365 s/vmin (m19 365, m03 318, old03 302; m01 1080p60 378). That is 0–20 % over the norm.
  - Short clips are worst: the fixed per-video costs, about 20–30 GPU-s, weigh more on them.
- **Production form (queue worker, 2 jobs in flight, resident face generator), 4-video mix of 3.04 vmin:**
  - wall 581 → **467** s/vmin;
  - GPU-busy 318 → **308**;
  - + UNet leaf compile (static shapes, first pass): 559 / 310. It recompiles for every new video's shapes.
  - The dynamic-shape variant on an unseen-video queue is **not yet measured** (NEXT).
  - **The norm is not met yet:** best 308.
- **The face generator (LatentSync) is ~75 % of the GPU time:** 145 ms per master frame, steady. Audio is ~60 GPU-s/vmin; the composite is ~15–30 (it is ~90 % CPU).
- **Rate cuts (half-rate, 2/3 rate) are rejected:** they fail the strong-flash gate on old03, and on m03 for half-rate.
  - The extra flashes come from LatentSync generating at a lower frame rate (weaker temporal coherence of its own output). They do not come from the flow fill or the audio windows.
  - They are not fixed by the out_stab output stabiliser either. Those out_stab numbers predate the face agent's time-base fix: not final.

## 1. Integration (task 1)

- **Merges:** `experiment/r10-base` = a00b0f0 + merge r9-speed-face (2cdffa3) + merge r9-speed-audio (111aee4), with 0 conflicts.
- **e2e.sh defaults:**
  - the approved r9fast-audio (round 2: O / C / K);
  - **R10_FAST_FACE=1** = the face stack S2 + VAE compile (GEN_COMP=0, FAST_SRCPREP, FAST_FB_BG, FAST_ENC=ultrafast, FAST_MPREUSE, CF / FC threads, LS_VAE_COMPILE). The mouth-only decode (LS_DEC_ROW0) is **not** on;
  - E2E_FX=1 (r9 + fix);
  - half-rate behind `E2E_HALF_RATE=1`.
- **Smoke:** m19 E2E SUCCESS on r10-s.
- **New flags** (all opt-in except the sampler):
  - `E2E_LS_RATE`;
  - `R10_HEAVY_LOCK` / `R10_COMP_GATE` (set by the worker);
  - `R10_GEN_SPOOL`;
  - `E2E_FRAME_GATE=off`;
  - `LS_UNET_COMPILE`;
  - `LS_PROFILE`.

## 2. Honest GPU-time accounting (task 2)

- **The sampler:** `scripts/r10/gpu_sampler.py`, NVML at 100 ms (nvidia-ml-py in /venv/main). It runs inside every e2e and writes `<out>/gpu_time.json` per stage; collect.py also writes `gpu_busy_s_per_video_min`.
- **Limitation:** per-process SM samples (nvmlDeviceGetProcessUtilization) come back almost empty on this driver, so the per-job split under overlap is not reliable. The device total is.

Warm E2E, default stack, one video at a time (wall / GPU-busy, s per video-minute):

| video | min | audio | face_gen | face_comp | **wall** | **GPU-busy** | util-weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| m19 (720p30, 18 s) | 0.30 | 158 / 84 | 418 / 254 | 188 / 27 | 772 | **365** | 319 |
| m03 (720p30, 41 s) | 0.68 | 103 / 62 | 377 / 237 | 149 / 19 | 635 | **318** | 274 |
| old03 (90 s) | 1.51 | 77 / 54 | 297 / 234 | 110 / 13 | 488 | **302** | 271 |
| m01 (1080p60, 86 s) | 1.44 | 84 / 56 | 506 / 260 | 480 / 63 * | 1071 | **378** | 319 |

\* m01: the composite was OOM-killed (section 6), so these are lower bounds.

A cold first run on a fresh machine adds ~120 s of VAE autotune, once per machine (inductor disk cache).

**Where the face_gen GPU time is** (m03, kernel profile `runs/ls_profile_m03.txt`, ~139 ms/frame CUDA):

| part | share |
|---|---:|
| convolutions (cuDNN) | 22 % |
| linear / matmul | 20 % |
| flash attention (SDPA already) | 14 % |
| memory-bound copies / elementwise / norms / gelu | ~29 % |

GPU-busy ≈ LatentSync inference + CodeFormer. The rest of face_gen (MediaPipe prep, CodeFormer glue, dumps, loads: ~40 % of its wall) is CPU while the GPU idles.

## 3. Generator levers (task 4)

Face-only runs on the acct audio, gated with `gate_cmp.py` vs the default stack of the same video (`gates.txt`):

| lever | flag | face GPU-busy s/vmin (m19 / m03 / old03) | gate |
|---|---|---|---|
| default stack | – | 291 / 249 / 249 | reference |
| half-rate (12.5 fps + flow fill) | `LS_HALF_RATE=1` | 184 / 156 / 155 | m19 PASS; **m03 FAIL** flashes 14→22; **old03 FAIL** flashes 37→52 + sync −0.31 |
| half-rate, exact 25 fps audio windows | `LS_RATE=1/2` | – / 154 / – | m03 FAIL flashes 14→22: the same 22, so the audio window is not the cause |
| 2 of 3 frames (16.7 fps), exact windows | `LS_RATE=2/3` | 221 / 185 / 184 | m19 PASS, m03 PASS; **old03 FAIL** flashes 37→54 |
| half-rate + out_stab (both sides) | face agent's r10_out_stab os_sh3fk | – / – / – | m03 FAIL flashes 8→17 (non-final, see above) |
| 2/3 + out_stab (both sides) | | – | old03 FAIL flashes 27→34 (non-final) |
| **UNet leaf compile** (warm) | `LS_UNET_COMPILE=default` | – / **224** (face_gen 231→206) / – | m03 **PASS on every row incl. the frame gate** (sync 7.59, flashes 13 vs 14, stick +1) |
| resident generator worker | `R10_GEN_SPOOL` | – / 243 (face_gen 225) / – | m03 **PASS** on every row vs a cold process (not bit-exact: mean 0.014 grey, kernel choice depends on the memory layout from window group 2 on; job 2 ≠ job 1 either) |

- **Rate levers:** `LS_RATE=k/m` (ls_stable.py) generalises half-rate. Each kept frame gets its own 25 fps audio window through `chunk_hook`, and dropped frames are flow-filled at their true time. `LS_HALF_RATE=1` is unchanged.
  - 2/3 even improves sync (m03 +0.14). The flash count is what fails, on the most flash-prone video.
- **UNet compile:** whole-UNet compile was dropped in r9 (one ~290 s compile per shape).
  - Here each ResNet / Transformer3D / motion-module leaf (25 modules) is compiled alone, and DeepCache's Python skip logic stays eager.
  - The dynamo recompile limit has to be raised: one cache per *code object* is shared by all 9 ResNets × widths × batch shapes. At the default 32, half the leaves silently fell back to eager.
  - The warm per-process tracing costs ~+12 s wall per video (CPU) and pays once in a resident worker.
- **TensorRT / ONNX:** not built.
  - With DeepCache (interval 5), only the full UNet calls (~half the UNet time, the UNet ~half of LatentSync) fit a static engine.
  - At 1.3–1.5× on those, the estimate is −10…−20 GPU-s/vmin: less than the leaf compile gives, for a far larger engineering cost (r9: 16 GB workspace OOM + an unsupported motion-module node).
  - Not worth it before the compile path is exhausted.

## 4. Pipelining worker (task 3)

`scripts/r10/worker.py --queue q.txt --out DIR [--inflight 2] [--gen-worker]`, one NVML sampler for the queue.

**Schedule (VRAM-safe on 32 GB):** audio peaks at ~30 GB, face_gen at ~14 GB, the composite at ~12 GB.
- `R10_HEAVY_LOCK` holds audio → sep → face_gen exclusively.
- Job N+1 is admitted when job N enters face_gen.
- Job N's composite waits for a go (`R10_COMP_GATE`) until no other job is before / in its audio stage. So the composite overlaps job N+1's face_gen, never an audio stage.
- Order: A(N) G(N) A(N+1) [G(N+1) ‖ C(N)] A(N+2) …

`--gen-worker` adds the resident face generator (`scripts/r10/gen_worker.py` + `gen_client.py`, hook `R10_GEN_PY` in r4_video.sh):
- LatentSync and the compiled graphs are loaded once;
- the pipe is parked on the CPU between jobs (0.2 s up / 0.6 s down), so the audio stage keeps its 30 GB;
- each job runs with its own argv / env / cwd;
- DeepCache is re-wrapped per runner.

Queue m19 + m03 + old03 + m15 = 3.04 vmin, default face stack, production only, all jobs SUCCESS:

| config | wall s/vmin | GPU-busy s/vmin | util-weighted | GPU busy % of wall |
|---|---:|---:|---:|---:|
| serial (inflight 1) | 581 | 318 | 281 | 55 % |
| worker, 2 in flight | 485 | 318 | 279 | 66 % |
| worker + resident generator | 467 | **308** | 275 | 66 % |
| worker + resident generator + UNet leaf compile (static shapes), first pass | 559 | **310** | 267 | 56 % |

**UNet compile in the queue, first pass (static shapes, inductor cache warm from m03 only):**
- The util-weighted GPU time drops, but GPU-busy does not, and wall gets worse.
- Every video brings window / tail shapes the cache has not seen, so the worker compiles them mid-queue: the old03 generator job took 590 s vs ~443 s.
- This is NOT the steady state, but a repeat pass of the same videos is not a valid production number either (every upload is a new video).
- The honest measurement: `LS_UNET_COMPILE_DYNAMIC=1` (symbolic batch / frame dims; added, NOT yet measured), the cache warmed on *different* videos, then a 4-video queue unseen by the graphs.
- It was queued when the servers were shut down: see NEXT.

**Status vs the norm:** **not under 300 yet.** The best measured queue is 308 GPU-busy s/vmin (worker + resident generator); single videos are 302–365.

Floor of this schedule = Σ(audio + face_gen) ≈ 439 s/vmin wall. To get closer: the face_gen CPU prep (MediaPipe master pass, frame map, rigid motion, dumps; ~40 % of face_gen wall, more at 1080p) runs under the heavy lock. It can move in front of the lock: generator internals, next step.

## 5. Gate and config summary (task deliverable)

| config | wall s/vmin | GPU-busy s/vmin | gate |
|---|---:|---:|---|
| E2E default, single video (m19 / m03 / old03) | 772 / 635 / 488 | 365 / 318 / 302 | reference (= r9 speed stack S2 + VC, r9 gate PASS) |
| E2E default, m01 1080p60 | 1071 | 378 | **PARTIAL_SUCCESS**: composite OOM (section 6) |
| queue serial (4 videos) | 581 | 318 | all SUCCESS |
| queue worker, 2 in flight | 485 | 318 | all SUCCESS, same outputs |
| + resident generator | 467 | 308 | m03 PASS vs cold |
| + UNet leaf compile (static), first pass on the queue | 559 | 310 | m03 single video: **PASS on every row incl. the frame gate** (face_gen GPU 231 → 206) |
| + UNet leaf compile DYNAMIC, unseen-video queue | not measured | not measured | NEXT |
| + LS_RATE=2/3 (not taken) | – | face −25 % | FAIL old03 flashes |
| + half-rate (not taken) | – | face −38 % | FAIL m03 / old03 flashes, old03 sync |

## 6. 1080p60 (m01)

- **face_gen 506 s/vmin wall / 260 GPU.** LatentSync 394 s; the CPU prep is heavy (MediaPipe 113 s, frame map 44, rigid motion 42, dumps 69).
- **The composite (`fps_composite.py`) was OOM-killed** at the 63.8 GB cgroup limit (rc 137, `memory.events` oom_kill 2): 5180 output frames at 1072×1904. The result was the source picture with the dub (PARTIAL_SUCCESS).
- A re-run of the composite alone **without** `--pre-threads` OOMs too: peak RSS 61.5 GB, killed at 502 s, oom_kill 3. So it is not the speed flag.
- `fps_composite.py` holds the full-resolution frame sets in RAM and needs streaming / chunking. That is the face agent's file; handed off via the coordinator.
- Until it is fixed, a 1080p60 clip of ~80 s or more ends PARTIAL_SUCCESS.
- Repro: `runs/scripts/after5.sh`. The per-tag `*_fb_fb7_<TAG>_rep2.npz` must exist when the repaired layer is cached (a cache-key quirk of r7_m7_comp.sh).

## 7. Files

- **This folder:**
  - `acct_table.md`;
  - `gates.txt` (every gate_cmp output);
  - `runs/`: e2e gpu_time / timing, face_only timings, queue.json ×N, the LatentSync kernel profile.
- **Code:**
  - `scripts/r10/`: gpu_sampler, worker, gen_worker / gen_client / gen_worker.sh, acct_table;
  - `scripts/quality/ls_stable.py`: LS_RATE, LS_UNET_COMPILE, LS_PROFILE;
  - `scripts/r9/e2e.sh`, `scripts/r9/collect.py`, `scripts/r4/r4_video.sh` (R10_GEN_PY hook).
- **Runs on r10-s:**
  - `/workspace/dub/r10/{acct,q4_*}`;
  - `/workspace/dub/r9/face/<id>_<variant>`;
  - artifacts `r4m/r9{g,f}_<id>_<variant>`.

## 8. NEXT (exact commands to resume on a new box)

**Ownership:** the LatentSync generator files (ls_stable.py, candidate_video_stage.py, …) are temporarily owned by the FACE agent for QUALITY flags (coordinator, 01.11). The steps below only *run* committed code.

**Setup:**
- setup + `bash scripts/r9/r9_models.sh`;
- `uv pip install --python /venv/main/bin/python nvidia-ml-py` (for the NVML sampler);
- inputs in `/workspace/dub/inputs/{mamakris,test_videos_raw}`.
- The queue files are in `runs/scripts/q3warm.txt` and `q4u.txt` (ids → paths; adjust the paths if they move).

```bash
cd /workspace/dub/dubai; mkdir -p /workspace/dub/r10; cp reports/experiments/r10-speed/runs/scripts/q*.txt /workspace/dub/r10/
# 0. (once per box) warm the VAE autotune + the face path: any single E2E
ID=m19 E2E_METRICS=off bash scripts/r9/e2e.sh /workspace/dub/inputs/mamakris/story-1774250832175-746471562.mp4 /workspace/dub/r10/warm0
# 1. dynamic UNet compile: warm-up on DIFFERENT videos (old01, old02, m19)
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q3warm.txt --out /workspace/dub/r10/q3warm_dyn --inflight 2 --gen-worker \
  --env E2E_METRICS=off LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1
# 2. THE RESULT: the unseen 4-video queue (m03, old03, m15, m18) -> /workspace/dub/r10/q4u_dyn/queue.json (gpu_busy_s_per_vmin)
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4u.txt --out /workspace/dub/r10/q4u_dyn --inflight 2 --gen-worker \
  --env E2E_METRICS=off LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1
# 3. the reference: same queue, gen worker, no compile
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4u.txt --out /workspace/dub/r10/q4u_ref --inflight 2 --gen-worker --env E2E_METRICS=off
# 4. the m03 gate of the dynamic compile (needs the stack baseline m03_b: face_only on the m03 audio of a finished e2e)
for id in m03; do mkdir -p /workspace/dub/r9/runs/$id/work; ln -sfn /workspace/dub/r10/q4u_ref/$id/work/in /workspace/dub/r9/runs/$id/work/in; done
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
V=_b bash scripts/r9/face_only.sh m03
LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1 V=_ucd bash scripts/r9/face_only.sh m03
# (after the background frame gates finish, ~15 min)
python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_b --cand /workspace/dub/r9/face/m03_ucd
```

**Decision rule:**
- If step 2 is ≤ 300 GPU-busy s/vmin and step 4 PASSES, make `--gen-worker` + `LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1` the worker default.
- Otherwise report the miss plainly.

**After that:**
1. Move the face_gen CPU prep out from under the heavy lock (wall; generator files, once the ownership returns).
2. Fixed per-video audio costs: the r9 resident audio worker exists (exact). Under the 32 GB budget it needs a slot plan: audio worker ~14 GB resident + generator 13 GB + composite 12 GB > 32 GB.
3. The m01 composite memory (face agent).
