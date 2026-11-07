# Wall-clock cost of the pipeline (r13 round 3, 03–04.11)

**What we pay for on a rented GPU is WALL time, not GPU-busy time.**
- Price: $0.72 / h (RTX 5090).
- Jobs: the k5 E2E (production defaults) on old03 (90 s), m03 (40 s), m18 (20 s), m15 (34 s) = 3.08 video-min.
- GPU time: the NVML sampler at 10 Hz plus the e2e stage stamps.
- Figure: `wallclock_timeline.png` (GPU util over time; stage bands per job).

## Where a single job idles (old03; m03 / m18 similar)
| stage | wall s | GPU-busy s | idle | why |
|---|---:|---:|---:|---|
| audio | 120 | 82 | 39 | process start, model loads, the CPU chain (lip placement, hush, remix) |
| face_gen | 495 | 382 | 113 | ~40 s CPU before LatentSync (frame decode, MediaPipe master, crops) + ~50 s CPU after it (assembly, encode, muxing) |
| face_comp | 188 | 18 | **169** | the composite is CPU: fallback, layer repair, flow, x264 |
| total | 816 | 482 | 334 | the GPU is busy **59 %** of the wall (m03 44 %, m18 39 %) |

## Single job vs a 4-job queue (same 4 videos)

| mode | wall s | wall s/vmin | $/video-min | GPU busy % of wall |
|---|---:|---:|---:|---:|
| 4 single jobs, one after another (the regression) | 2040 | **664** | **$0.133** | 39–59 % |
| r10 worker: 2 jobs in flight + a resident generator (`scripts/r10/worker.py --inflight 2 --gen-worker`) | 1647 | **535** | **$0.107** | 62 % |
| the same + `R10_GEN_EARLY_RELEASE=1` (`r13s/queue_we`) | 1510 | **490** | **$0.098** | 70 % |
| steady-state bound of the current lock design (audio + sep + face_gen serialised, composites overlapped) | – | ≈ 489 | ≈ $0.098 | |
| GPU-busy floor (if the GPU never idled) | – | 332 | $0.066 | 100 % |

- The worker overlaps job N's CPU composite with job N+1's audio + face_gen.
- **−19 % wall.** GPU-busy per video-minute is unchanged (332 vs ≈ 332).

## Is the queue output identical?
- **Audio:** the voice track is **bit-identical** to the single run (the pipeline is deterministic), and the Spleeter mix matches to ~120 dB (float rounding). The delivered AAC differs at 35–39 dB SNR, but that is the AAC encoder's own quantisation choices on microscopically different input: both are equally valid encodings at the codec's noise level, nothing audible.
- **Video:** PSNR 48–50 dB vs the single run (visually identical, not bit-exact).
  - **Baseline: a plain repeat of the same single job is not bit-exact either.** m18 single vs single (`reg` vs `reg2`): different md5, PSNR 54.5 dB. The generator itself is not bit-deterministic (GPU kernels).
  - **Early release vs the plain queue** (`queue_we` vs `queue_w`): PSNR old03 58.0 / m03 56.4 / m18 54.5 / m15 57.5 dB. That is the same as the run-to-run noise (54.5 dB on m18), so the hook adds no change of its own.
  - So "bit-identical" cannot be demanded of the video for any lever. The fair bar is "within the run-to-run noise", and early release meets it.

## Early release result (04.11)
- **Wall 535 → 490 s/vmin (−8 %), $0.107 → $0.098 / video-min.** The GPU is busy 70 % of the wall (was 62 %).
- This reaches the steady-state bound of the current lock design (≈ 489). Going further needs the composite CPU work to be cheaper or parallel (face-agent code), or a second job's face_gen to overlap the first's composite (already the case).
- Recommendation: switch `R10_GEN_EARLY_RELEASE=1` on in the worker once the LS agent has merged the hook.

## What still costs wall time, and the levers
1. **The generator's CPU head and tail** (~90 s per job, inside the heavy lock).
   - `R10_GEN_EARLY_RELEASE=1` (new, default off): the resident generator parks the model and signals "GPU done" right before its CPU assembly, and e2e releases the heavy lock, so the next job's audio starts during the assembly.
   - Hook point: `scripts/r10/gen_worker.py` wraps `face_aware_latentsync_accel.assemble_frames` at runtime; no generator file is edited (diff below).
2. **The composite CPU** (~170 s per job). Already overlapped in a queue; on a single job it is the biggest idle block. Parallelising it inside the composite is face-agent code (not touched).
3. **Fixed per-job costs** (process start, model loads ~15–20 s): the resident audio worker (`R9_WORKER_SPOOL`) would remove the audio half.

### The early-release hook (for the LS agent to merge)
- `scripts/r10/gen_worker.py`: `early_release(pipe, off)` wraps `faa.assemble_frames`.
  - Before the original call it moves the pipe to the CPU (when the worker offloads) and calls `torch.cuda.synchronize(); empty_cache()`.
  - Then it touches `$R13_GPU_DONE_FILE`.
  - It is installed in `main()` when `R10_GEN_EARLY_RELEASE=1`.
- `scripts/r9/e2e.sh` (face_gen stage):
  - exports `R13_GPU_DONE_FILE=$W/gen_gpu_done`;
  - runs `r7_e2e.sh` in the background;
  - with `R10_GEN_EARLY_RELEASE=1` and a worker lock, it polls the marker and calls `flock -u 8` when it appears.
- The computation is unchanged: `assemble_frames` gets the same arguments; only the model parking moves earlier.
