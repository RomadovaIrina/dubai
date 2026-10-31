#!/usr/bin/env python3
"""r10 speed: stands in for `python scripts/quality/candidate_video_stage.py <args>` when R10_GEN_SPOOL is set (scripts/r4/r4_video.sh via
R10_GEN_PY): the job (argv + this process's environment + cwd) goes to the resident generator worker (scripts/r10/gen_worker.py); its log is
streamed to stdout and its exit code returned. A dead worker (no heartbeat for 10 s) = exit 3."""
import json, os, pathlib, sys, time, uuid

d = pathlib.Path(os.environ["R10_GEN_SPOOL"]); name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
job = {"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()}
tmp = d / "jobs" / f".{name}.tmp"; tmp.write_text(json.dumps(job)); tmp.rename(d / "jobs" / f"{name}.json")
log, rc = d / "out" / f"{name}.log", d / "out" / f"{name}.rc"; pos = 0
while True:
    if log.exists():
        with open(log, errors="replace") as f: f.seek(pos); s = f.read(); pos = f.tell()
        if s: sys.stdout.write(s); sys.stdout.flush()
    if rc.exists():
        time.sleep(0.2)
        with open(log, errors="replace") as f: f.seek(pos); sys.stdout.write(f.read())
        sys.exit(int(rc.read_text().strip() or 1))
    alive = d / "alive"
    if not alive.exists() or time.time() - alive.stat().st_mtime > 10: print("[gen_client] generator worker not alive", flush=True); sys.exit(3)
    time.sleep(0.5)
