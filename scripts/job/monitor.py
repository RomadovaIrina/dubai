#!/usr/bin/env python3
"""DabAI worker monitoring (r18, TZ 10.4): worker liveness, queue depth, stuck jobs, GPU hours, VRAM peak, job wall per video minute.

Used by job_api.py (GET /health adds a "monitor" block, GET /metrics = the Prometheus text format) and testable without FastAPI / a GPU.

Data sources (nothing new runs on the GPU):
  - the API's own state: the queue (ORDER), the running job (RUNNING), the worker thread + its heartbeat;
  - <job dir>/status.json (rewritten every ~3 s by a live job = its heartbeat), events.jsonl (the step start times), metrics.json (final numbers);
  - a ledger $DABAI_MONITOR_DIR/ledger.jsonl (default $DABAI_JOBS_DIR/.monitor): one line per finished job, so the totals (GPU hours, video
    minutes) survive the deletion of job directories. At start the ledger is back-filled from the finished jobs already on disk.

Stuck job = a RUNNING job with one of (all thresholds by env):
  - "stale":    status.json not rewritten for DABAI_STUCK_STALE_S (120 s; a live job rewrites it every ~3 s -> its monitor loop is dead);
  - "orphan":   RUNNING on disk but not the worker's current job (a job the worker lost), older than 30 s;
  - "timeout":  running past its own timeout_s + DABAI_STUCK_TIMEOUT_GRACE_S (300 s): the kill at the timeout did not work;
  - "step":     the current step runs longer than max(DABAI_STUCK_STEP_MIN_S = 1800, DABAI_STUCK_STEP_FACTOR = 4 x its expected time from
                dabai_job.STEPS: fixed + per video minute). A hung stage keeps status.json fresh, so only this rule sees it before the timeout
                (which is 1200 s per video minute, i.e. 3.3 h on a 10-min video).
A QUEUED job on disk that is in neither the queue nor RUNNING (older than 30 s) is "lost_queued".
"""
from __future__ import annotations
import json, math, os, pathlib, threading, time
from datetime import datetime

FINAL = ("SUCCESS", "PARTIAL_SUCCESS", "ERROR", "TIMEOUT")
NORM_GPU_S_PER_VMIN = 300.0   # TZ: <= 300 s of processing per video minute (0.5 / block 0)


def _envf(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except ValueError: return default


def _read_json(p: pathlib.Path):
    for _ in range(3):
        try: return json.loads(p.read_text())
        except json.JSONDecodeError: time.sleep(0.02)   # atomic writes (os.replace); a retry covers an old NFS
        except OSError: return None
    return None


def _ts(iso) -> float | None:
    """dabai_job.now_iso() ("%Y-%m-%dT%H:%M:%S%z") -> epoch s"""
    if not iso: return None
    try: return datetime.strptime(str(iso), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError: return None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


# ------------------------------------------------------------------ the ledger (finished jobs)
class Ledger:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path); self._lock = threading.Lock(); self._cache: tuple | None = None

    @staticmethod
    def row(job_dir: pathlib.Path, st: dict | None = None, finished_ts: float | None = None) -> dict | None:
        st = st if st is not None else _read_json(job_dir / "status.json")
        if not st or st.get("status") not in FINAL: return None
        mt = _read_json(job_dir / "metrics.json") or {}
        nv = mt.get("gpu_nvml") or {}
        vmin = _num(mt.get("video_minutes")) or _num(st.get("video_minutes"))
        wall = _num(mt.get("wall_s")) or _num(st.get("wall_s"))
        ts = finished_ts or _ts(st.get("finished_at"))
        if ts is None:
            try: ts = (job_dir / "status.json").stat().st_mtime
            except OSError: ts = time.time()
        return {"job_id": st.get("job_id") or job_dir.name, "status": st.get("status"), "error": (st.get("error") or {}).get("code"),
                "finished_ts": round(ts, 1), "video_minutes": vmin, "wall_s": wall,
                "gpu_busy_s": _num(mt.get("gpu_busy_s")) or _num(st.get("gpu_busy_s")),   # NVML device busy (exact on a dedicated GPU)
                "gpu_weighted_s": _num(nv.get("gpu_weighted_s")),                         # the job's own processes, weighted by their share
                "vram_peak_mib": _num(mt.get("vram_peak_mib")), "cold_start": bool(st.get("cold_start")),
                "degraded_ratio": _num((mt.get("lipsync") or {}).get("degraded_ratio"))}

    def record(self, job_dir: pathlib.Path, st: dict | None = None) -> dict | None:
        r = self.row(pathlib.Path(job_dir), st, finished_ts=time.time())
        if r is None: return None
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    def rows(self) -> dict[str, dict]:
        """job_id -> its last row (a re-recorded job counts once), cached by the file's size + mtime"""
        try: s = self.path.stat(); key = (s.st_size, s.st_mtime_ns)
        except OSError: return {}
        with self._lock:
            if self._cache and self._cache[0] == key: return self._cache[1]
            out: dict[str, dict] = {}
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try: r = json.loads(line)
                except json.JSONDecodeError: continue   # a torn last line (power loss) costs one job, not the ledger
                if isinstance(r, dict) and r.get("job_id"): out[str(r["job_id"])] = r
            self._cache = (key, out); return out

    def backfill(self, jobs_dir: pathlib.Path) -> int:
        """finished jobs on disk that the ledger does not know (jobs from before r18, or closed while the API was down)"""
        known = self.rows(); n = 0; new = []
        for p in sorted(pathlib.Path(jobs_dir).glob("*/status.json")):
            if p.parent.name in known or p.parent.name.startswith("."): continue
            r = self.row(p.parent)
            if r: new.append(r); n += 1
        if new:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    for r in sorted(new, key=lambda r: r["finished_ts"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return n


# ------------------------------------------------------------------ stuck jobs
def _last_step(job_dir: pathlib.Path) -> tuple[str | None, float | None]:
    p = job_dir / "events.jsonl"; step = t = None
    try: lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError: return None, None
    for line in lines:
        try: e = json.loads(line)
        except json.JSONDecodeError: continue
        if e.get("type") == "step" and e.get("step"): step, t = e["step"], _num(e.get("t"))
    return step, t


def _expected_s(step: str, vmin: float, steps: dict | None) -> float:
    s = (steps or {}).get(step)
    return float(s[2] + s[3] * max(vmin, 0.05)) if s else 300.0


def stuck_jobs(jobs_dir: pathlib.Path, running: list[str] | None = None, queue: list[str] | None = None,
               now: float | None = None, steps: dict | None = None) -> list[dict]:
    now = now or time.time(); out = []
    stale_s = _envf("DABAI_STUCK_STALE_S", 120); grace = _envf("DABAI_STUCK_TIMEOUT_GRACE_S", 300)
    step_min = _envf("DABAI_STUCK_STEP_MIN_S", 1800); factor = _envf("DABAI_STUCK_STEP_FACTOR", 4)
    for p in pathlib.Path(jobs_dir).glob("*/status.json"):
        jid = p.parent.name
        if jid.startswith("."): continue
        try: age = now - p.stat().st_mtime
        except OSError: continue
        if age < 0: age = 0.0
        if age > 7 * 86400: continue   # a week-old leftover is history, not an alert (it is in /jobs)
        st = _read_json(p) or {}; s = st.get("status")
        if s == "QUEUED":
            if queue is not None and jid not in queue and jid not in (running or []) and age > 30:
                out.append({"job_id": jid, "reason": "lost_queued", "detail": "QUEUED on disk but not in the worker's queue", "age_s": round(age)})
            continue
        if s != "RUNNING": continue
        reasons = []
        if age > stale_s: reasons.append(("stale", f"status.json not updated for {age:.0f} s"))
        if running is not None and jid not in running and age > 30: reasons.append(("orphan", "RUNNING on disk but not the worker's job"))
        vmin = _num(st.get("video_minutes")) or 0.0; t0 = _ts(st.get("started_at"))
        if t0:
            to = _num((st.get("params") or {}).get("timeout_s")) or max(1800.0, 1200.0 * max(vmin, 0.5))
            if st.get("cold_start") and not _num((st.get("params") or {}).get("timeout_s")): to += _envf("DABAI_COLD_EXTRA_S", 2400)
            if now - t0 > to + grace: reasons.append(("timeout", f"running {now - t0:.0f} s, its timeout is {to:.0f} s"))
        step, ts = _last_step(p.parent)
        if step and ts and step != "end":
            lim = max(step_min, factor * _expected_s(step, vmin, steps))
            if now - ts > lim: reasons.append(("step", f"step {step} runs {now - ts:.0f} s (limit {lim:.0f} s)"))
        if reasons:
            out.append({"job_id": jid, "reason": reasons[0][0], "reasons": [r for r, _ in reasons], "detail": "; ".join(d for _, d in reasons),
                        "step": st.get("step"), "age_s": round(age)})
    return sorted(out, key=lambda x: x["job_id"])


# ------------------------------------------------------------------ the snapshot (/health "monitor" + /metrics)
def snapshot(jobs_dir: pathlib.Path, ledger: Ledger, running: list[str], queue: list[str], worker: dict | None = None,
             now: float | None = None, steps: dict | None = None, recent_n: int | None = None) -> dict:
    now = now or time.time(); jobs_dir = pathlib.Path(jobs_dir); recent_n = int(recent_n or _envf("DABAI_MONITOR_RECENT_N", 20))
    rows = sorted(ledger.rows().values(), key=lambda r: r.get("finished_ts") or 0)
    by_status = {s: 0 for s in FINAL}
    for r in rows: by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    tot = lambda k, rs=rows: round(sum(_num(r.get(k)) or 0.0 for r in rs), 3)
    recent = rows[-recent_n:]
    ok_r = [r for r in recent if r.get("status") in ("SUCCESS", "PARTIAL_SUCCESS") and _num(r.get("video_minutes"))]
    vm_ok = sum(r["video_minutes"] for r in ok_r)
    waits = []
    for jid in queue:
        st = _read_json(jobs_dir / jid / "status.json") or {}
        t = _num(st.get("submitted_ts")) or _ts(st.get("submitted_at"))
        if t: waits.append(now - t)
    cur = None
    if running:
        st = _read_json(jobs_dir / running[0] / "status.json") or {}; t0 = _ts(st.get("started_at"))
        cur = {"job_id": running[0], "step": st.get("step"), "progress_pct": st.get("progress_pct"), "eta_s": st.get("eta_s"),
               "elapsed_s": round(now - t0) if t0 else None, "video_minutes": st.get("video_minutes")}
    last = rows[-1] if rows else None
    lastv = {k: last.get(k) for k in ("job_id", "status", "finished_ts", "video_minutes", "wall_s", "gpu_busy_s", "vram_peak_mib")} if last else None
    if lastv:
        vm = _num(last.get("video_minutes"))
        lastv["wall_s_per_video_min"] = round(last["wall_s"] / vm, 1) if vm and _num(last.get("wall_s")) else None
        lastv["gpu_busy_s_per_video_min"] = round(last["gpu_busy_s"] / vm, 1) if vm and _num(last.get("gpu_busy_s")) else None
    vr = [r["vram_peak_mib"] for r in recent if _num(r.get("vram_peak_mib"))]
    return {
        "worker": worker or {}, "queue_depth": len(queue), "queue_oldest_wait_s": round(max(waits)) if waits else 0, "running": cur,
        "stuck": stuck_jobs(jobs_dir, running, queue, now, steps),
        "totals": {"jobs": by_status, "video_minutes": tot("video_minutes"), "wall_s": tot("wall_s"), "gpu_busy_s": tot("gpu_busy_s"),
                   "gpu_busy_hours": round(tot("gpu_busy_s") / 3600, 3), "gpu_weighted_s": tot("gpu_weighted_s")},
        "last_job": lastv,
        "recent": {"n": len(recent), "error_ratio": round(sum(r.get("status") in ("ERROR", "TIMEOUT") for r in recent) / len(recent), 3) if recent else 0.0,
                   "partial_ratio": round(sum(r.get("status") == "PARTIAL_SUCCESS" for r in recent) / len(recent), 3) if recent else 0.0,
                   "wall_s_per_video_min": round(sum(r.get("wall_s") or 0 for r in ok_r) / vm_ok, 1) if vm_ok else None,
                   "gpu_busy_s_per_video_min": round(sum(r.get("gpu_busy_s") or 0 for r in ok_r) / vm_ok, 1) if vm_ok else None,
                   "vram_peak_mib_max": max(vr) if vr else None},
        "norm_gpu_s_per_video_min": NORM_GPU_S_PER_VMIN,
    }


# ------------------------------------------------------------------ Prometheus text format (0.0.4)
def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def prom_text(snap: dict, health: dict | None = None) -> str:
    health = health or {}; L: list[str] = []

    def m(name, typ, help_, samples):
        L.append(f"# HELP {name} {help_}"); L.append(f"# TYPE {name} {typ}")
        for labels, v in samples:
            v = _num(v) if not isinstance(v, bool) else float(v)
            if v is None: continue
            lab = "{" + ",".join(f'{k}="{_esc(x)}"' for k, x in labels.items()) + "}" if labels else ""
            f = float(v); L.append(f"{name}{lab} {int(f) if f.is_integer() else repr(round(f, 4))}")   # full precision (timestamps, totals)

    w = snap.get("worker") or {}; t = snap["totals"]; lj = snap.get("last_job") or {}; rc = snap["recent"]
    g = health.get("gpu") or {}; res = (health.get("resident") or {}).get("state")
    ver = health.get("versions") or {}
    m("dabai_up", "gauge", "The API answers (always 1 when scraped).", [({}, 1)])
    m("dabai_info", "gauge", "Versions of this worker.", [({k: ver.get(k, "") for k in ("interface", "pipeline", "image", "profile")}, 1)])
    m("dabai_health_ok", "gauge", "1 when the node can take jobs (GPU visible, weights present, disk free): the /health verdict.",
      [({}, health.get("ok")) ] if "ok" in health else [])
    m("dabai_worker_alive", "gauge", "1 when the job worker thread is alive and its heartbeat is fresh.", [({}, w.get("alive"))])
    m("dabai_worker_heartbeat_age_seconds", "gauge", "Seconds since the idle worker loop last ticked (0 while a job runs).", [({}, w.get("heartbeat_age_s"))])
    m("dabai_uptime_seconds", "gauge", "Seconds since the API started (a serverless node is paid for its uptime).", [({}, w.get("uptime_s"))])
    m("dabai_worker_busy", "gauge", "1 while a job runs.", [({}, 1 if snap.get("running") else 0)])
    m("dabai_queue_depth", "gauge", "Jobs waiting in this worker's FIFO queue (TZ 3.7: scale out above 3).", [({}, snap["queue_depth"])])
    m("dabai_queue_oldest_wait_seconds", "gauge", "Wait of the oldest queued job.", [({}, snap["queue_oldest_wait_s"])])
    m("dabai_running_job_elapsed_seconds", "gauge", "Elapsed time of the running job.", [({}, (snap.get("running") or {}).get("elapsed_s"))])
    m("dabai_running_job_progress_percent", "gauge", "Progress of the running job.", [({}, (snap.get("running") or {}).get("progress_pct"))])
    m("dabai_jobs_stuck", "gauge", "Stuck jobs (stale / orphan / timeout / step / lost_queued; ids in /health monitor.stuck).", [({}, len(snap["stuck"]))])
    m("dabai_jobs_finished_total", "counter", "Finished jobs by final status (TZ 11.3: PARTIAL_SUCCESS is recorded here).",
      [({"status": s}, n) for s, n in sorted(t["jobs"].items())])
    m("dabai_video_minutes_total", "counter", "Video minutes of the finished jobs.", [({}, t["video_minutes"])])
    m("dabai_job_wall_seconds_total", "counter", "Wall seconds of the finished jobs.", [({}, t["wall_s"])])
    m("dabai_gpu_busy_seconds_total", "counter", "GPU busy seconds of the finished jobs (NVML device busy; /3600 = GPU hours).", [({}, t["gpu_busy_s"])])
    m("dabai_gpu_busy_hours_total", "counter", "GPU hours of the finished jobs (= dabai_gpu_busy_seconds_total / 3600).", [({}, t["gpu_busy_hours"])])
    m("dabai_gpu_weighted_seconds_total", "counter", "GPU seconds of the jobs' own processes (NVML, weighted by their share).", [({}, t["gpu_weighted_s"])])
    m("dabai_last_job_wall_seconds_per_video_minute", "gauge", "Wall seconds per video minute of the last finished job.", [({}, lj.get("wall_s_per_video_min"))])
    m("dabai_last_job_gpu_seconds_per_video_minute", "gauge", "GPU busy seconds per video minute of the last finished job (norm 300).", [({}, lj.get("gpu_busy_s_per_video_min"))])
    m("dabai_last_job_vram_peak_mib", "gauge", "VRAM peak of the last finished job (MiB; the card has 32607).", [({}, lj.get("vram_peak_mib"))])
    m("dabai_last_job_finished_timestamp_seconds", "gauge", "When the last job finished (epoch s).", [({}, lj.get("finished_ts"))])
    m("dabai_recent_jobs", "gauge", "Finished jobs in the recent window.", [({}, rc["n"])])
    m("dabai_recent_error_ratio", "gauge", "Share of ERROR + TIMEOUT among the recent jobs.", [({}, rc["error_ratio"])])
    m("dabai_recent_partial_ratio", "gauge", "Share of PARTIAL_SUCCESS among the recent jobs.", [({}, rc["partial_ratio"])])
    m("dabai_recent_wall_seconds_per_video_minute", "gauge", "Wall s per video minute over the recent successful jobs.", [({}, rc["wall_s_per_video_min"])])
    m("dabai_recent_gpu_seconds_per_video_minute", "gauge", "GPU busy s per video minute over the recent successful jobs.", [({}, rc["gpu_busy_s_per_video_min"])])
    m("dabai_recent_vram_peak_mib_max", "gauge", "Highest job VRAM peak in the recent window (MiB).", [({}, rc["vram_peak_mib_max"])])
    m("dabai_norm_gpu_seconds_per_video_minute", "gauge", "The TZ norm (300 s per video minute), for dashboards.", [({}, snap.get("norm_gpu_s_per_video_min"))])
    m("dabai_gpu_memory_used_mib", "gauge", "GPU memory in use now (nvidia-smi).", [({}, g.get("memory_used_mib"))])
    m("dabai_gpu_memory_total_mib", "gauge", "GPU memory of the card (nvidia-smi).", [({}, g.get("memory_total_mib"))])
    m("dabai_gpu_visible", "gauge", "1 when nvidia-smi sees the GPU.", [({}, 1 if g else 0)] if health else [])
    m("dabai_disk_free_gb", "gauge", "Free disk of the jobs directory (GB).", [({}, health.get("disk_free_gb"))])
    m("dabai_missing_weights", "gauge", "Missing required weight entries.", [({}, len(health.get("missing_weights") or []))] if health else [])
    m("dabai_resident_state", "gauge", "Resident worker state (off / starting / ready).", [({"state": s}, 1 if s == res else 0) for s in ("off", "starting", "ready")] if res else [])
    return "\n".join(L) + "\n"
