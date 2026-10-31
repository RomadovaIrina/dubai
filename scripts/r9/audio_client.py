#!/usr/bin/env python3
"""r9 speed R: hands one audio job (this argv, env, cwd) to the resident worker (scripts/r9/audio_worker.py, spool $R9_WORKER_SPOOL), streams its
log to stdout and exits with its exit code. No live worker (heartbeat older than 10 s) -> the job runs cold in this process (audio_proc.py).
    python scripts/r9/audio_client.py <run_clean_pipeline_05 argv>       (r4_audio.sh's R9_PIPE_PY)"""
import json, os, pathlib, sys, time, uuid

HERE = pathlib.Path(__file__).resolve().parent


def beat_age(alive):
    """seconds since the worker's last heartbeat (inf: none)"""
    for _ in range(5):
        try: return time.time() - float(alive.read_text())
        except Exception: time.sleep(0.05)
    return float("inf")


def main():
    D = pathlib.Path(os.environ["R9_WORKER_SPOOL"]); alive = D / "alive"
    if beat_age(alive) > 10:
        print("[client] no live audio worker: running cold", file=sys.stderr, flush=True)
        sys.argv = [str(HERE / "audio_proc.py")] + sys.argv[1:]
        for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7", HERE.parent / "r8", HERE): sys.path.insert(0, str(p))
        import audio_proc
        audio_proc.main(); return
    jid = f"{time.time():.6f}_{uuid.uuid4().hex[:8]}"; tmp = D / "jobs" / f".{jid}.tmp"
    tmp.write_text(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()})); tmp.rename(D / "jobs" / f"{jid}.json")
    log, rcf, pos = D / "out" / f"{jid}.log", D / "out" / f"{jid}.rc", 0
    while True:
        done = rcf.exists()
        if log.exists():
            with open(log, "rb") as f:
                f.seek(pos); b = f.read(); pos += len(b)
            if b: sys.stdout.buffer.write(b); sys.stdout.flush()
        if done: break
        if beat_age(alive) > 30:
            print("[client] the audio worker died", file=sys.stderr); sys.exit(1)
        time.sleep(0.2)
    sys.exit(int(rcf.read_text().strip() or 1))


if __name__ == "__main__":
    main()
