#!/usr/bin/env python3
"""Resident worker (2026-10-13): several videos in ONE process with DUB_RESIDENT=1 (model_cache.py), the way one worker per GPU node serves
jobs in production (TZ 3.4 / 0.4). Every job is the unchanged run_clean_pipeline_05.main(); the first job pays the model loads (cold), the
following ones reuse the loaded models (warm). Writes <run-root>/final/<id>_final.mp4 + manifest (same layout as refresh_e2e.sh, so the
report tools work), <run-root>/worker.json (per-job wall, cold / warm, device-wide peak VRAM per job).
    PYTHONPATH=third_party/llama_cuda:$PYTHONPATH python scripts/pilot/run_worker.py --run-root /workspace/dub/runs/worker --ids 04 05 03 01 02 -- <runner flags>
"""
from __future__ import annotations
import argparse, glob, json, os, pathlib, subprocess, sys, threading, time
os.environ["DUB_RESIDENT"] = "1"
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class VramSampler:
    def __init__(self):
        self.peak = 0; self.stop = False; self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        while not self.stop:
            try:
                v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(1.0)

    def take(self) -> int:
        p = self.peak; self.peak = 0; return p


def main() -> int:
    argv = sys.argv[1:]; extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--target-lang", default="en")
    a = ap.parse_args(argv)
    import run_clean_pipeline_05 as rp, model_cache
    root = pathlib.Path(a.run_root); (root / "final").mkdir(parents=True, exist_ok=True); (root / "work").mkdir(parents=True, exist_ok=True)
    src_dir = HERE.parents[1] / "test_videos"; vs = VramSampler(); jobs = []
    for k, vid in enumerate(a.ids):
        inp = sorted(glob.glob(str(src_dir / f"{vid}.*")))[0]; out = root / "final" / f"{vid}_final.mp4"
        sys.argv = ["run_clean_pipeline_05.py", "--input", inp, "--output", str(out), "--target-lang", a.target_lang, "--alignment", "burst",
                    "--video-backend", "optimized", "--codeformer", "optimized", "--work-dir", str(root / "work")] + extra
        print(f"===== worker job {k + 1}/{len(a.ids)} {vid} start {time.strftime('%H:%M:%S')} (loaded: {model_cache.loaded()})", flush=True)
        vs.take(); t0 = time.perf_counter(); rc = rp.main(); wall = round(time.perf_counter() - t0, 2); peak = vs.take()
        jobs.append({"id": vid, "rc": rc, "wall_s": wall, "warm": k > 0, "peak_vram_mib": peak, "models_cached_after": model_cache.loaded()})
        print(f"===== worker job {vid} exit {rc} wall {wall}s peak {peak} MiB", flush=True)
        (root / "worker.json").write_text(json.dumps({"jobs": jobs, "extra_args": extra}, indent=1))
    vs.stop = True
    return 0 if all(j["rc"] == 0 for j in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
