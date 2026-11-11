"""r17 resident worker, the KeySync half: ks_fast.py --serve <spool> keeps the KeySync-fast models (both UNets as TensorRT engines + the LCM
student, HuBERT, WavLM, the SVD VAE with its compiled decode, face_alignment) loaded ACROSS videos, and PARKS them off the GPU between requests,
so the audio stage of the next job gets the whole card (TZ 0.4: audio peak ~30 GB, KeySync ~28 GB of 32 GB: never both on the GPU).

Spool <D>:
  pid, alive (heartbeat, every s), ready (JSON: load_s, park state, VRAM) once loaded + parked, stop (ends it)
  req/<id>.json  {"jobs": <follow file>, "out": <dir>, "log": <log>, "sig": <settings signature>, "env": {..}, "owner": <pid>}
A request = exactly what one cold `ks_fast.py --follow --jobs <jobs> --out <out> > <log>` did (ks_face.sh / ks_shots.sh): its job lines are read
from <jobs> until END, a KS_JOB line per job and KS_FAST_DONE at the end go to <log>. The daemon only serves requests whose settings
signature (the ks_fast args + the KS_* env that patch the models at start: ks_serve.sh ks_sig) equals its own; otherwise KS_SERVE_REJECT and
the client starts a cold ks_fast. A dead owner (no END) or a failing job -> KS_SERVE_ERROR, the models are parked, the daemon goes on."""
import gc, json, os, pathlib, sys, threading, time, traceback

import torch


def _modules(objs, depth=3):
    """every torch module reachable from these objects (face_alignment / keysync wrappers keep their nets as attributes), each once"""
    seen, out = set(), []
    def walk(o, d):
        if id(o) in seen: return
        seen.add(id(o))
        if isinstance(o, (torch.nn.Module, torch.jit.ScriptModule)): out.append(o); return
        if d <= 0 or not hasattr(o, "__dict__"): return
        for v in vars(o).values(): walk(v, d - 1)
    for o in objs: walk(o, depth)
    return out


def _trt(mods):
    return [t for m in mods for t in m.modules() if getattr(t, "_ks_trt", False) and hasattr(t, "park")]


def _vram():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2), "torch_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            "gpu_used_gb": round((total - free) / 2**30, 2)}


_HOME = {}   # id(module) -> the device it lived on before the first park (unpark puts it back there, never on a device it was not on)


def _dev(m):
    for t in list(m.parameters()) + list(m.buffers()): return t.device
    return None


def park(objs):
    t0 = time.perf_counter(); mods = _modules(objs)
    for t in _trt(mods): t.park()
    for m in mods:
        if id(m) not in _HOME: _HOME[id(m)] = _dev(m)
        m.to("cpu")
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    return round(time.perf_counter() - t0, 2)


def unpark(objs):
    t0 = time.perf_counter(); mods = _modules(objs)
    for m in mods:
        dv = _HOME.get(id(m))
        if dv is not None and dv.type == "cuda": m.to(dv)
    for t in _trt(mods): t.unpark()
    torch.cuda.synchronize()
    return round(time.perf_counter() - t0, 2)


def _alive(pid):
    try: os.kill(int(pid), 0); return True
    except (OSError, TypeError, ValueError): return False


def serve(spool, load_s, objs, job_lines, run_line):
    D = pathlib.Path(spool); (D / "req").mkdir(parents=True, exist_ok=True)
    (D / "pid").write_text(str(os.getpid()))
    def beat():
        while True:
            (D / ".alive.tmp").write_text(str(time.time())); os.replace(D / ".alive.tmp", D / "alive"); time.sleep(1)
    threading.Thread(target=beat, daemon=True).start()
    sig = os.environ.get("KS_SERVE_SIG", "")
    ps = park(objs)
    (D / "ready").write_text(json.dumps({"load_s": load_s, "park_s": ps, "parked": True, "sig": sig, **_vram()}))
    print("KS_SERVE_READY", (D / "ready").read_text(), flush=True)
    out0, err0, n = sys.stdout, sys.stderr, 0
    while not (D / "stop").exists():
        rs = sorted((D / "req").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not rs: time.sleep(0.1); continue
        rf = rs[0]
        try: rq = json.loads(rf.read_text())
        except Exception: time.sleep(0.1); continue   # (half written: the client writes atomically, so only a foreign file)
        rf.unlink(missing_ok=True)
        log = open(rq["log"], "a", buffering=1); sys.stdout = sys.stderr = log; env0 = dict(os.environ)
        try:
            if rq.get("sig") != sig:
                print(f"KS_SERVE_REJECT settings differ: request {rq.get('sig')!r} vs resident {sig!r}", flush=True); continue
            os.environ.update({k: str(v) for k, v in (rq.get("env") or {}).items()})
            us = unpark(objs); print("KS_SERVE_ACCEPT " + json.dumps({"request": n, "unpark_s": us, **_vram()}), flush=True)
            os.makedirs(rq["out"], exist_ok=True); owner = rq.get("owner")
            try:
                for line in job_lines(rq["jobs"], True, alive=lambda: owner is None or _alive(owner)):
                    run_line(line, rq["out"])
                print("KS_FAST_DONE", flush=True)
            except Exception:
                traceback.print_exc(); print("KS_SERVE_ERROR", flush=True)
        finally:
            try: print("KS_SERVE_PARKED " + json.dumps({"park_s": park(objs), **_vram()}), flush=True)
            except Exception: traceback.print_exc()
            os.environ.clear(); os.environ.update(env0); sys.stdout, sys.stderr = out0, err0; log.close(); n += 1
    print("KS_SERVE_STOP", flush=True)
