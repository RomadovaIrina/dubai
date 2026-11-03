#!/usr/bin/env python3
"""r13 spec 11.1-11.3 fault injection: the face generator stage (scripts/quality/candidate_video_stage.py) with injected failures,
WITHOUT touching the generator files: r4_video.sh runs ${R10_GEN_PY:-candidate_video_stage.py}, so
    R10_GEN_PY=scripts/r13/fault_gen.py DUB_FAULT=<mode> bash scripts/r9/e2e.sh ...
modes (DUB_FAULT, or per job DUB_FAULT_JOBS="<id>=<mode>;<id>=<mode>" keyed by R10_JOB, which e2e.sh exports):
  crash        the generator process dies (RuntimeError after CUDA init) -> e2e: face_gen_failed, source picture + dub, PARTIAL_SUCCESS
  calls:1,4    LatentSync calls 1 and 4 (1-based) raise RuntimeError inside the model call = the generator's own round-3 injector
               (--inject-ls-error): 11.1 the segment's try/except -> PASS_THROUGH_LS_FAILED, the source frames stand in (11.2),
               11.3 counts them against the 2 % rule
  oom:2        call 2 raises torch.cuda.OutOfMemoryError once (the retry path); patched on BOTH runner classes (faa.AccelRunner and
               ls_stable.StableRunner: the E2E default --bridge-len 8 uses the latter)
Every injected fault is logged as "[fault] ..." / "injected fault" in the generator log."""
import os, pathlib, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality"))


def mode():
    job = os.environ.get("R10_JOB", "")
    for kv in os.environ.get("DUB_FAULT_JOBS", "").split(";"):
        if "=" in kv and kv.split("=", 1)[0].strip() == job: return kv.split("=", 1)[1].strip()
    return os.environ.get("DUB_FAULT", "").strip()


M = mode(); kind, _, arg = M.partition(":")
print(f"[fault] job {os.environ.get('R10_JOB', '?')} mode {M or 'none'}", flush=True)
if kind == "crash":
    import torch
    torch.zeros(1, device="cuda")
    raise RuntimeError("[fault] injected: generator process crashed (DUB_FAULT=crash)")
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"
sys.argv[0] = str(target)
if kind == "calls":
    sys.argv += ["--inject-ls-error", arg]
elif kind == "oom":
    calls = {int(x) for x in arg.split(",") if x.strip().isdigit()}; n = [0]
    import face_aware_latentsync_accel as faa, ls_stable

    def wrap(cls):
        orig = cls.run_frames

        def run_frames(self, *a, **k):
            n[0] += 1
            if n[0] in calls:
                calls.discard(n[0]); import torch
                print(f"[fault] injected CUDA OOM in LatentSync call {n[0]}", flush=True)
                raise torch.cuda.OutOfMemoryError(f"[fault] injected OOM in call {n[0]}")
            return orig(self, *a, **k)
        cls.run_frames = run_frames
    for c in {faa.AccelRunner, ls_stable.StableRunner}:
        if hasattr(c, "run_frames"): wrap(c)
runpy.run_path(str(target), run_name="__main__")
