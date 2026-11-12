#!/usr/bin/env python3
"""DabAI HTTP wrapper (r15 T): a minimal FastAPI service around dabai_job.run_job. One GPU = one service = jobs run ONE AT A TIME (FIFO queue);
two jobs on one 32 GB card would OOM (TZ 3.5), so the queue is the GPU lock of this node.

    DABAI_JOBS_DIR=/data/jobs uvicorn job_api:app --app-dir scripts/job --host 0.0.0.0 --port 8000

Endpoints (INTERFACE.md):
    POST   /jobs                       submit a job (body = job.json without out_dir; out_dir = $DABAI_JOBS_DIR/<job_id>) -> 202 {job_id, status: QUEUED}
    GET    /jobs                       the known jobs (id, status, progress)
    GET    /jobs/{id}                  the status (= <out_dir>/status.json; QUEUED jobs also get queue_position)
    GET    /jobs/{id}/events?after=N   progress / status events with seq > N
    GET    /jobs/{id}/files/{kind}     video | subs | metrics | status | log
    DELETE /jobs/{id}                  cancel (queued: dropped; running: the pipeline is killed) -> status ERROR, error.code E_CANCELLED
    GET    /health                     200 when the node can take jobs (GPU visible, weights present, disk free), else 503
    GET    /version                    interface / pipeline / image versions
    GET    /schema/job, /schema/status the JSON schemas
Auth: when DABAI_API_TOKEN is set, every call except /health needs "Authorization: Bearer <token>" (the token is never logged or echoed).
Restart: jobs found RUNNING / QUEUED in $DABAI_JOBS_DIR at start are closed as ERROR E_WORKER_RESTART (retryable): no job is silently lost (TZ 12.4)."""
from __future__ import annotations
import json, os, pathlib, queue, re, shutil, subprocess, sys, threading, time, uuid
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dabai_job as dj  # noqa: E402
from fastapi import Body, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

JOBS = pathlib.Path(os.environ.get("DABAI_JOBS_DIR", "/data/jobs")).resolve(); JOBS.mkdir(parents=True, exist_ok=True)
INPUT_ROOTS = [pathlib.Path(x).resolve() for x in os.environ.get("DABAI_INPUT_ROOTS", "").split(":") if x]   # local inputs only under these (empty = any)
TOKEN = os.environ.get("DABAI_API_TOKEN") or None
MIN_FREE_GB = float(os.environ.get("DABAI_MIN_FREE_GB", "30"))
app = FastAPI(title="DabAI job API", version=dj.INTERFACE)
Q: "queue.Queue[str]" = queue.Queue(); ORDER: list[str] = []; CANCEL: dict[str, threading.Event] = {}; RUNNING: list[str] = []; LOCK = threading.Lock()


def _dir(jid: str) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", jid): raise HTTPException(404, "no such job")
    return JOBS / jid


def _status(jid: str) -> dict:
    p = _dir(jid) / "status.json"
    if not p.exists(): raise HTTPException(404, "no such job")
    for _ in range(3):
        try: return json.loads(p.read_text())
        except json.JSONDecodeError: time.sleep(0.05)   # written atomically (os.replace); a retry covers a reader on an old NFS
    raise HTTPException(500, "status unreadable")


@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.url.path != "/health" and request.headers.get("authorization") != f"Bearer {TOKEN}":
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _resident_wait(limit_s=float(os.environ.get("DABAI_RESIDENT_WAIT_S", "600"))):
    """r17 (VM load test 12.11: VRAM 31.6 of 32 GB): while the resident worker is LOADING (its KeySync models on the GPU), a cold job must not
    start beside it -> wait until it is ready (or the limit: then the job runs cold, as before)"""
    if os.environ.get("DABAI_RESIDENT") != "1": return
    t0 = time.time()
    while _resident().get("state") == "starting" and time.time() - t0 < limit_s: time.sleep(2)


def _worker():
    while True:
        jid = Q.get()
        _resident_wait()
        with LOCK:
            if jid in ORDER: ORDER.remove(jid)
            ev = CANCEL.get(jid)
            if ev is None or ev.is_set(): continue   # cancelled while queued (already closed)
            RUNNING.append(jid)
        try:
            job = json.loads((_dir(jid) / "job.json").read_text()); dj.run_job(job, cancel=ev)
        except Exception as x:   # never kill the worker thread
            print(f"worker: job {jid} crashed the wrapper: {type(x).__name__}", flush=True)
        finally:
            with LOCK:
                RUNNING.remove(jid); CANCEL.pop(jid, None)


def _close(jid: str, code: str, detail: str | None = None):
    d = _dir(jid); st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"schema": dj.INTERFACE, "job_id": jid}
    s_, en, ru, retry = dj.ERRORS[code]
    st.update(status=s_, error={"code": code, "message": en, "message_ru": ru, "retryable": retry, "detail": detail}, refund_suggested=True,
              finished_at=dj.now_iso(), step="end", step_label="Done", progress_pct=100.0, eta_s=0)
    dj._write(d / "status.json", st); dj.Events(d / "events.jsonl").emit("done", status=s_, error=code, progress_pct=100.0)


def _kill_orphan(st: dict):
    """the pipeline of a job that was RUNNING when the service died may still run (own session): kill it before the next job takes the GPU"""
    pg = st.get("pipeline_pgid")
    try:
        if pg and b"e2e.sh" in pathlib.Path(f"/proc/{int(pg)}/cmdline").read_bytes(): os.killpg(int(pg), 9); return True
    except (OSError, ValueError): pass
    return False


@app.on_event("startup")
def _startup():   # TZ 12.4: a restart loses no job: RUNNING ones end as E_WORKER_RESTART (retryable), QUEUED ones run, in submit order
    queued = []
    for p in JOBS.glob("*/status.json"):
        try: s = json.loads(p.read_text())
        except Exception: continue
        if s.get("status") == "RUNNING":
            k = _kill_orphan(s); _close(p.parent.name, "E_WORKER_RESTART", "the worker restarted while the job ran" + (" (its pipeline was killed)" if k else ""))
        elif s.get("status") == "QUEUED":
            if (p.parent / "job.json").exists(): queued.append((s.get("submitted_ts") or (p.parent / "job.json").stat().st_mtime, p.parent.name))
            else: _close(p.parent.name, "E_WORKER_RESTART", "the job file is missing")
    for _, jid in sorted(queued): CANCEL[jid] = threading.Event(); ORDER.append(jid); Q.put(jid)
    threading.Thread(target=_worker, daemon=True, name="dabai-worker").start()


@app.post("/jobs", status_code=202)
def submit(job: dict = Body(...)):
    job = dict(job); jid = job.get("job_id") or uuid.uuid4().hex[:16]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(jid)): raise HTTPException(422, {"code": "E_BAD_REQUEST", "errors": ["job_id: [A-Za-z0-9_-]{1,64}"]})
    if "out_dir" in job and os.environ.get("DABAI_ALLOW_OUT_DIR") != "1": raise HTTPException(422, {"code": "E_BAD_REQUEST", "errors": ["out_dir is set by the service"]})
    d = JOBS / jid; job.update(job_id=jid, out_dir=str(job.get("out_dir") or d))
    p, errs = dj.validate(job)
    src = str(job.get("input") or "")
    if not errs and not re.match(r"^https?://", src):
        ps = pathlib.Path(src).resolve()
        if INPUT_ROOTS and not any(ps == r or r in ps.parents for r in INPUT_ROOTS): errs.append("input: local path outside DABAI_INPUT_ROOTS")
        elif not ps.is_file(): errs.append("input: file not found")
    if errs: raise HTTPException(422, {"code": "E_BAD_REQUEST", "errors": errs})
    with LOCK:
        if (d / "status.json").exists() or jid in CANCEL: raise HTTPException(409, f"job {jid} exists")
        d.mkdir(parents=True, exist_ok=True); dj._write(d / "job.json", job)
        st = {"schema": dj.INTERFACE, "job_id": jid, "status": "QUEUED", "step": "queued", "step_label": "Queued", "progress_pct": 0.0, "eta_s": None,
              "params": {**p, "input": dj.redact(p["input"])}, "versions": dj.versions(), "submitted_at": dj.now_iso(), "submitted_ts": time.time(), "error": None}
        dj._write(d / "status.json", st); dj.Events(d / "events.jsonl").emit("status", status="QUEUED", progress_pct=0.0)
        CANCEL[jid] = threading.Event(); ORDER.append(jid); Q.put(jid); pos = len(ORDER)
    return {"job_id": jid, "status": "QUEUED", "queue_position": pos, "status_url": f"/jobs/{jid}"}


@app.post("/jobs/sync")
def submit_sync(job: dict = Body(...)):
    """r17 (Vast Serverless, TZ 10.1): ONE request = one whole job, for a router that sends every request of a client to ANY worker (the async
    POST /jobs + GET /jobs/{id} pair would poll the wrong worker). Body = a POST /jobs body + optional "output_put_url" (a presigned PUT URL:
    the result video is uploaded there; the URL is never logged or stored) + optional "output_put_subs_url". Blocks until SUCCESS / PARTIAL /
    ERROR (one job at a time per worker: the queue applies) and returns the final status (+ "uploaded": {video, subs})."""
    import urllib.request
    job = dict(job); put = {"video": job.pop("output_put_url", None), "subs": job.pop("output_put_subs_url", None)}
    jid = submit(job)["job_id"]
    while True:
        s = _status(jid)
        if s.get("status") in ("SUCCESS", "PARTIAL_SUCCESS", "ERROR"): break
        time.sleep(1.0)
    up = {}
    for kind, url in put.items():
        f = s.get("output") if kind == "video" else s.get("subs_ass")
        if not url or not f or not pathlib.Path(f).is_file(): continue
        try:
            with open(f, "rb") as fh:
                rq = urllib.request.Request(url, data=fh, method="PUT", headers={"content-length": str(pathlib.Path(f).stat().st_size),
                                                                                 "content-type": "video/mp4" if kind == "video" else "text/plain"})
                with urllib.request.urlopen(rq, timeout=3600) as r: up[kind] = r.status in (200, 201, 204)
        except Exception as e:
            up[kind] = False; s.setdefault("warnings", []).append({"code": "W_UPLOAD_FAILED", "message": f"{kind} upload failed: {type(e).__name__}"})
    return {**s, "uploaded": up}


@app.get("/jobs")
def list_jobs(limit: int = 100):
    rows = []
    for p in sorted(JOBS.glob("*/status.json"), key=lambda q: q.stat().st_mtime, reverse=True)[:max(1, min(limit, 1000))]:
        try: s = json.loads(p.read_text()); rows.append({k: s.get(k) for k in ("job_id", "status", "step", "progress_pct", "eta_s", "finished_at")})
        except Exception: pass
    return {"jobs": rows, "queue": list(ORDER), "running": list(RUNNING)}


@app.get("/jobs/{jid}")
def get_job(jid: str):
    s = _status(jid)
    if s.get("status") == "QUEUED":
        with LOCK: s["queue_position"] = ORDER.index(jid) + 1 if jid in ORDER else None
    return s


@app.get("/jobs/{jid}/events")
def events(jid: str, after: int = 0):
    p = _dir(jid) / "events.jsonl"
    if not p.exists(): raise HTTPException(404, "no such job")
    ev = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {"events": [e for e in ev if e.get("seq", 0) > after]}


@app.get("/jobs/{jid}/files/{kind}")
def files(jid: str, kind: str):
    s = _status(jid); d = _dir(jid)
    f = {"video": s.get("output"), "subs": s.get("subs_ass"), "metrics": s.get("metrics_json"), "status": str(d / "status.json"), "log": str(d / "job.log")}.get(kind)
    if f is None and kind not in ("video", "subs", "metrics", "status", "log"): raise HTTPException(404, "kind: video | subs | metrics | status | log")
    if not f or not pathlib.Path(f).is_file() or d.resolve() not in pathlib.Path(f).resolve().parents: raise HTTPException(404, f"no {kind} file (yet)")
    return FileResponse(f, filename=pathlib.Path(f).name)


@app.delete("/jobs/{jid}", status_code=202)
def cancel(jid: str):
    s = _status(jid)
    if s.get("status") not in ("QUEUED", "RUNNING"): return {"job_id": jid, "status": s.get("status"), "detail": "already finished"}
    with LOCK:
        ev = CANCEL.get(jid)
        if ev is not None: ev.set()
        queued = jid in ORDER
        if queued: ORDER.remove(jid)
    if queued or ev is None: _close(jid, "E_CANCELLED")
    return {"job_id": jid, "status": "CANCELLING" if not queued and ev is not None else "ERROR"}


def _gpu() -> dict | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        n, u, t, dv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": n, "memory_used_mib": int(u), "memory_total_mib": int(t), "driver": dv}
    except Exception: return None


def _weights() -> list[str]:
    """required weight dirs / files of the profile (IMAGE manifest) that are missing"""
    m = dj.ROOT / "models"; need = ["whisper-large-v3", "qwen2.5-14b-instruct-gguf", "chatterbox", "latentsync", "sd-vae-ft-mse", "wav2vec2-base-960h", "mediapipe",
                                    "realesrgan", "spleeter"]
    miss = [n for n in need if not (m / n).exists()]
    if os.environ.get("DUB_PROFILE") == "current":   # the NC parts of the current profile (linked by the entrypoint from the NC volume)
        miss += [p for p in (str(dj.ROOT / "third_party/CodeFormer/weights/CodeFormer/codeformer.pth"), "/workspace/dub/sep_pylib/demucs",
                             "/workspace/dub/quarantine_nc/torch/hub/checkpoints/955717e8-8726e21a.th") if not pathlib.Path(p).exists()]
    if not pathlib.Path(dj.STUDENT_LORA).is_file(): miss.append("student LoRA (mode fast)")
    return miss


def _resident():
    if os.environ.get("DABAI_RESIDENT") != "1": return {"state": "off"}
    f = pathlib.Path(os.environ.get("DABAI_RESIDENT_DIR", "/data/resident")) / "ready"
    try: return {"state": "ready", **json.loads(f.read_text())}
    except Exception: return {"state": "starting"}


@app.get("/health")
def health():
    g = _gpu(); miss = _weights(); free = shutil.disk_usage(JOBS).free / 1e9
    ok = g is not None and not [x for x in miss if not x.startswith("student")] and free >= MIN_FREE_GB
    body = {"ok": ok, "busy": bool(RUNNING), "queue_depth": len(ORDER), "gpu": g, "missing_weights": miss, "disk_free_gb": round(free, 1), "versions": dj.versions(),
            "resident": _resident()}   # r17: off | starting | ready (+ its load timings): jobs run cold until ready
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/version")
def version(): return dj.versions()


@app.get("/schema/{name}")
def schema(name: str):
    f = pathlib.Path(__file__).resolve().parent / f"{name}.schema.json"
    if name not in ("job", "status") or not f.exists(): raise HTTPException(404, "job | status")
    return json.loads(f.read_text())
