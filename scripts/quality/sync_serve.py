#!/usr/bin/env python3
"""r17 speed (output-identical by construction, KS_SYNC_SERVE=1): ONE process for every sync_perframe.py call of a video (ks_face.sh's probe +
check per face / shot). Each request runs sync_perframe.main() with its own argv, in its own cwd, stdout / stderr to its own log, exactly as the
cold `python sync_perframe.py ...` would; the detector and SyncNet are still built per call (no state carried between calls). Saved per call:
the interpreter start, the torch / cv2 / LatentSync imports and the CUDA context (~5-10 s each; 2 calls per shot).
    python sync_serve.py --spool <dir>      jobs/<id>.json {argv, cwd, log} -> out/<id>.rc ; <dir>/alive (heartbeat), <dir>/pid, <dir>/stop ends it
Client: ks_face.sh sync_run (falls back to the cold call when the server is not alive)."""
import argparse, json, os, pathlib, sys, threading, time, traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spool", required=True); a = ap.parse_args()
    D = pathlib.Path(a.spool); (D / "jobs").mkdir(parents=True, exist_ok=True); (D / "out").mkdir(parents=True, exist_ok=True)
    (D / "pid").write_text(str(os.getpid()))
    def beat():
        while True:
            (D / ".alive.tmp").write_text(str(time.time())); os.replace(D / ".alive.tmp", D / "alive"); time.sleep(1)
    threading.Thread(target=beat, daemon=True).start()
    import torch, cv2, numpy  # noqa: F401  (the imports the per-call process paid for)
    import sync_perframe
    if torch.cuda.is_available(): torch.zeros(1, device="cuda")   # the CUDA context, once
    out0, err0, cwd0, argv0, env0 = sys.stdout, sys.stderr, os.getcwd(), list(sys.argv), dict(os.environ)
    ppid0 = os.getppid()
    while not (D / "stop").exists():
        if os.getppid() != ppid0: break   # the script that started it is gone (an error exit before sync_serve_stop): never linger on the GPU
        js = sorted((D / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not js: time.sleep(0.05); continue
        jf = js[0]
        try: job = json.loads(jf.read_text())
        except Exception: time.sleep(0.05); continue
        jf.unlink(missing_ok=True); jid = jf.stem; rc = 1
        log = open(job["log"], "w", buffering=1)
        try:
            os.chdir(job["cwd"]); sys.argv = ["sync_perframe.py"] + list(job["argv"]); sys.stdout = sys.stderr = log
            try: sync_perframe.main(); rc = 0
            except SystemExit as e: rc = 0 if e.code in (None, 0) else (e.code if isinstance(e.code, int) else 1)
        except BaseException:
            traceback.print_exc(); rc = 1
        finally:
            sys.stdout, sys.stderr, sys.argv = out0, err0, argv0; os.chdir(cwd0); os.environ.clear(); os.environ.update(env0); log.close()
            (D / "out" / f"{jid}.tmp").write_text(str(rc)); os.replace(D / "out" / f"{jid}.tmp", D / "out" / f"{jid}.rc")


if __name__ == "__main__":
    main()
