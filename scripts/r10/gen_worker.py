#!/usr/bin/env python3
"""r10 speed (R10_GEN_SPOOL=<dir>): ONE resident face-generator worker per GPU. LatentSync (+ the LS_VAE_COMPILE / LS_UNET_COMPILE graphs) is
loaded and compiled ONCE for every video it serves; a video's generator job is the same code as without the worker
(scripts/quality/candidate_video_stage.py main()) with the job's own argv, environment and cwd.

What stays resident: the LatentSync pipeline (UNet / VAE / whisper-tiny audio encoder / scheduler) and the dynamo / inductor graphs compiled on
it. What is reset per job so a job equals a cold process: the DeepCache wrapping (each runner enables its own helper on the shared UNet: the
previous one is unwrapped first), the environment (replaced by the job's), the seeds (every LatentSync call seeds itself, set_seed(1247)).
    python scripts/r10/gen_worker.py --spool <dir>          (scripts/r10/gen_worker.sh start|stop <dir>)
Spool: jobs/<n>.json {argv, env, cwd} -> out/<n>.log (stdout + stderr) + out/<n>.rc; <dir>/alive = heartbeat (1 s); <dir>/stop ends it."""
import argparse, contextlib, json, os, pathlib, sys, threading, time, traceback

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (ROOT / "scripts/pilot", ROOT / "scripts/quality", ROOT / "scripts/optim", ROOT / "scripts/r7", ROOT / "scripts/r8", ROOT / "scripts/r9"): sys.path.insert(0, str(p))


def resident():
    import resident_vram_probe as rvp
    from DeepCache import DeepCacheSDHelper
    load = rvp.load_latentsync; box = {}
    def load_latentsync():
        if "pipe" not in box:
            t0 = time.perf_counter(); box["pipe"] = load(); print(f"[gen_worker] LatentSync loaded once ({time.perf_counter() - t0:.1f} s)", flush=True)
        return box["pipe"]
    rvp.load_latentsync = load_latentsync
    en = DeepCacheSDHelper.enable
    def enable(self, pipe=None):   # one DeepCache wrapping at a time on the shared UNet: unwrap the previous runner's first
        prev = getattr(self.pipe, "_r10_dc", None)
        if prev is not None and prev is not self:
            with contextlib.suppress(Exception): prev.disable()
        en(self, pipe); self.pipe._r10_dc = self
    DeepCacheSDHelper.enable = enable
    return box


class Tee:
    def __init__(self, f): self.f = f
    def write(self, s): self.f.write(s); return len(s)
    def flush(self): self.f.flush()
    def isatty(self): return False
    def fileno(self): return self.f.fileno()


def run_job(job, out_log):
    import candidate_video_stage as cvs
    env0, argv0, cwd0 = dict(os.environ), list(sys.argv), os.getcwd()
    keep = {k: v for k, v in env0.items() if k in ("LS_VAE_COMPILE", "LS_UNET_COMPILE")}   # the resident graphs were built with these
    rc = 1
    with open(out_log, "w", buffering=1) as f:
        fd1, fd2 = os.dup(1), os.dup(2); os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)   # subprocesses (ffmpeg) and C-level prints too
        so, se = sys.stdout, sys.stderr; sys.stdout = sys.stderr = Tee(f)
        try:
            os.environ.clear(); os.environ.update(job["env"]); os.environ.update(keep); os.chdir(job["cwd"])
            sys.argv = ["candidate_video_stage.py"] + job["argv"]
            rc = cvs.main() or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except BaseException:
            traceback.print_exc(); rc = 1
        finally:
            sys.stdout, sys.stderr = so, se; os.dup2(fd1, 1); os.dup2(fd2, 2); os.close(fd1); os.close(fd2)
            os.environ.clear(); os.environ.update(env0); sys.argv = argv0; os.chdir(cwd0)
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
    return rc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spool", required=True); a = ap.parse_args()
    d = pathlib.Path(a.spool); (d / "jobs").mkdir(parents=True, exist_ok=True); (d / "out").mkdir(exist_ok=True)
    stop = threading.Event()
    def beat():
        while not stop.is_set(): (d / "alive").touch(); time.sleep(1)
    threading.Thread(target=beat, daemon=True).start()
    box = resident(); import resident_vram_probe as rvp, ls_stable
    pipe = rvp.load_latentsync(); ls_stable.vae_compile(pipe); ls_stable.unet_compile(pipe)
    import torch
    off = os.environ.get("R10_GEN_OFFLOAD", "1") == "1"   # the pipe parked on the CPU between jobs (in place: the compiled graphs keep their parameters)
    if off: pipe.to("cpu"); torch.cuda.empty_cache()
    print(f"[gen_worker] up: spool {d}, LS_VAE_COMPILE={os.environ.get('LS_VAE_COMPILE', '')} LS_UNET_COMPILE={os.environ.get('LS_UNET_COMPILE', '')} offload {off}", flush=True)
    n = 0
    while not (d / "stop").exists():
        jobs = sorted((d / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not jobs: time.sleep(0.2); continue
        j = jobs[0]; job = json.loads(j.read_text()); name = j.stem; t0 = time.perf_counter()
        if off: pipe.to("cuda")
        t1 = time.perf_counter(); rc = run_job(job, d / "out" / f"{name}.log"); n += 1; t2 = time.perf_counter()
        if off: pipe.to("cpu"); torch.cuda.empty_cache()   # between jobs the GPU belongs to the other stages (audio peaks at ~30 GB of 32)
        print(f"[gen_worker] to-GPU {t1 - t0:.1f} s, job {t2 - t1:.1f} s, to-CPU {time.perf_counter() - t2:.1f} s", flush=True)
        (d / "out" / f"{name}.rc").write_text(str(rc)); j.unlink()
        print(f"[gen_worker] job {n} {name} rc {rc} {time.perf_counter() - t0:.1f} s", flush=True)
    stop.set(); (d / "alive").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
