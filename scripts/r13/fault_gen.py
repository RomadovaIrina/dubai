#!/usr/bin/env python3
"""r13 spec 11.1-11.3 fault injection: the face generator stage (scripts/quality/candidate_video_stage.py) with injected failures,
WITHOUT touching the generator files: r4_video.sh runs ${R10_GEN_PY:-candidate_video_stage.py}, so
    R10_GEN_PY=scripts/r13/fault_gen.py DUB_FAULT=<mode> bash scripts/r9/e2e.sh ...
modes (DUB_FAULT, or per job DUB_FAULT_JOBS="<id>=<mode>;<id>=<mode>" keyed by R10_JOB, which e2e.sh exports):
  crash        the generator process dies (RuntimeError after CUDA init) -> e2e: face_gen_failed, source picture + dub, PARTIAL_SUCCESS
  calls:1,4    LatentSync calls 1 and 4 (0-based, in process order) raise RuntimeError inside the model call (11.1: the segment's
               try/except -> PASS_THROUGH_LS_FAILED, the source frames stand in = 11.2; 11.3 counts them against the 2 % rule)
  oom:2        call 2 raises torch.cuda.OutOfMemoryError once (the retry path)
  p:0.1        each call fails with probability 0.1 (seed DUB_FAULT_SEED, default 0)
Every injected fault is logged as "[fault] ..." in the stage log."""
import os, pathlib, random, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality"))


def mode():
    job = os.environ.get("R10_JOB", "")
    for kv in os.environ.get("DUB_FAULT_JOBS", "").split(";"):
        if "=" in kv and kv.split("=", 1)[0].strip() == job: return kv.split("=", 1)[1].strip()
    return os.environ.get("DUB_FAULT", "").strip()


M = mode(); RNG = random.Random(int(os.environ.get("DUB_FAULT_SEED", "0")))
print(f"[fault] job {os.environ.get('R10_JOB', '?')} mode {M or 'none'}", flush=True)
if M == "crash":
    import torch
    torch.zeros(1, device="cuda")
    raise RuntimeError("[fault] injected: generator process crashed (DUB_FAULT=crash)")
if M:
    import face_aware_latentsync_accel as faa
    kind, _, arg = M.partition(":")
    calls = {int(x) for x in arg.split(",") if x.strip().isdigit()} if kind in ("calls", "oom") else set()

    class FaultyRunner(faa.AccelRunner):
        n = 0

        def __call__(self, *a, **k):
            i = FaultyRunner.n; FaultyRunner.n += 1
            if kind == "calls" and i in calls or kind == "p" and RNG.random() < float(arg):
                print(f"[fault] injected RuntimeError in LatentSync call {i}", flush=True)
                raise RuntimeError(f"[fault] injected: LatentSync call {i} failed inside the model")
            if kind == "oom" and i in calls:
                calls.discard(i); import torch
                print(f"[fault] injected CUDA OOM in LatentSync call {i}", flush=True)
                raise torch.cuda.OutOfMemoryError(f"[fault] injected OOM in call {i}")
            return super().__call__(*a, **k)

    faa.AccelRunner = FaultyRunner
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
