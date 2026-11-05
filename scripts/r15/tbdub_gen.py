#!/usr/bin/env python3
"""r15 W0 TBDub bake-off: the PRODUCTION face generator stage (scripts/quality/candidate_video_stage.py, every flag as in e2e.sh) with the
LatentSync runner swapped for TBDub (TaoLiveAIGC/TBDub student, 2 steps, 512 px, Apache-2.0); nothing else edited: the speech gate,
the mouth-only paste-back (our MediaPipe boxes), CodeFormer, the raw-layer dump and the whole face_comp chain (q2 + F3) run as usual.
TBDub runs in its own venv (scripts/r15/tbdub_worker.py), a child of this process, so it sits inside this stage's gpu.lock.
    R10_GEN_PY=scripts/r15/tbdub_gen.py bash scripts/r15/arm.sh ...      env: TBDUB_STEPS, TBDUB_SEED, TBDUB_OFFLOAD=1,
    TBDUB_WHOLE=1 (one generation over the whole master with the whole dub, sliced per segment), TBDUB_FASTCC=1 (OpenCV Lab colour match)"""
import atexit, json, os, pathlib, runpy, subprocess, sys, time, types
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality"))
import numpy as np


class TBDubRunner:
    """Same interface as face_aware_latentsync_accel.AccelRunner / ls_stable.StableRunner (run_frames / set_segment / calls / config)."""
    silence_mix = None

    def __init__(self, *a, **k):
        t0 = time.perf_counter(); env = {k_: v for k_, v in os.environ.items() if k_ not in ("PYTHONPATH", "VIRTUAL_ENV")}
        logf = open(os.environ.get("TBDUB_LOG", "/workspace/dub/r15w/tbdub_worker.log"), "a")
        self.p = subprocess.Popen([os.environ.get("TBDUB_PY", "/workspace/dub/tbdub/venv/bin/python"), str(HERE / "tbdub_worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=logf, env=env, text=True, bufsize=1, cwd="/workspace/dub/tbdub")
        r = json.loads(self.p.stdout.readline() or '{"ok": false, "error": "worker died at load (see tbdub_worker.log)"}')
        if not r.get("ready"): raise RuntimeError(r.get("error"))
        atexit.register(self.close)
        self.load_s = round(time.perf_counter() - t0, 2); self.steps = int(os.environ.get("TBDUB_STEPS", "2")); self.guidance = 1.0; self.seed = int(os.environ.get("TBDUB_SEED", "42"))
        self.config = {"generator": "TBDub student (TaoLiveAIGC/TBDub, Apache-2.0)", "steps": self.steps, "seed": self.seed, "resolution": 512, "offload": os.environ.get("TBDUB_OFFLOAD") == "1"}
        self.calls = []; self.acc = types.SimpleNamespace(); self._seg = None
        print(f"  [tbdub] worker ready in {self.load_s} s", flush=True)

    def set_segment(self, start, n): self._seg = (int(start), int(n))

    def run_frames(self, frames_rgb, audio_wav, detections=None):
        t0 = time.perf_counter(); s, n = self._seg; work = pathlib.Path(audio_wav).parent; tag = f"/dev/shm/tbdub_{os.getpid()}_{s}"
        fr = np.ascontiguousarray(np.asarray(frames_rgb)); assert len(fr) == n, (len(fr), n); np.save(tag + "_in.npy", fr)
        q = {"master": str(work / "master_25fps.mp4"), "start": s, "n": n, "wav": str(audio_wav), "in": tag + "_in.npy", "out": tag + "_out.npy"}
        if os.environ.get("TBDUB_WHOLE") == "1":   # one TBDub generation over the whole master + the whole dub (sliced per segment by the worker)
            dd = sys.argv[sys.argv.index("--dubbed-dir") + 1]; q["full_wav"] = str(pathlib.Path(dd) / "dubbed_16k.wav")
        self.p.stdin.write(json.dumps(q) + "\n"); self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline() or '{"ok": false, "error": "worker died (see tbdub_worker.log)"}')
        os.unlink(tag + "_in.npy")
        if not r.get("ok"): raise RuntimeError(f"TBDub: {r.get('error')}")
        out = np.load(tag + "_out.npy"); os.unlink(tag + "_out.npy")
        dt = round(time.perf_counter() - t0, 2); self.calls.append({"start": s, "frames": n, "seconds": dt, **{k: r[k] for k in ("gen_s", "crop_mode", "peak_mib", "got") if k in r}})
        print(f"  [tbdub] seg @{s} {n} frames in {dt} s (gen {r.get('gen_s')} s, peak {r.get('peak_mib')} MiB, crop {r.get('crop_mode')})", flush=True)
        return out, dt

    def close(self):
        if getattr(self, "p", None) and self.p.poll() is None:
            try: self.p.stdin.close(); self.p.wait(timeout=60)
            except Exception: self.p.kill()


import ls_stable, face_aware_latentsync_accel as faa
ls_stable.StableRunner = TBDubRunner; faa.AccelRunner = TBDubRunner
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"; sys.argv[0] = str(target)
sys.argv += ["--steps", "2", "--steps-silent", "2", "--guidance", "1.0", "--no-deepcache", "--bridge-steps", "1"]   # the TBDub runner ignores them (manifest only)
runpy.run_path(str(target), run_name="__main__")
