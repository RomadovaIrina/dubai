#!/usr/bin/env python3
"""r16 TZ 12.4 on KeySync: the REAL worker service (scripts/job/job_api.py + the real e2e pipeline, one RTX 5090) under a burst:
N jobs posted at once -> every POST accepted (202), never two pipelines at the same time (polled every 3 s), FIFO, none lost, every job
ends with its expected status; throughput = video minutes / hour and jobs / hour from the first POST to the last finish; per job queue wait,
run time, GPU-s, status. The API runs in its own session and is stopped (process group) at the end.
    python3 tz_load.py --out DIR --api-py /venv/api/bin/python IN1.mp4 IN2.mp4 ..."""
import argparse, json, os, pathlib, signal, subprocess, sys, threading, time, urllib.request

def call(url, data=None, method=None):
    r = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as f: return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e: return e.code, {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--api-py", default="/venv/api/bin/python")
    ap.add_argument("--port", type=int, default=18800); ap.add_argument("inputs", nargs="+"); a = ap.parse_args()
    out = pathlib.Path(a.out); jobs_dir = out / "jobs"; jobs_dir.mkdir(parents=True, exist_ok=True); H = f"http://127.0.0.1:{a.port}"
    env = dict(os.environ, DABAI_JOBS_DIR=str(jobs_dir), DABAI_MIN_FREE_GB="20")
    api = subprocess.Popen([a.api_py, "-m", "uvicorn", "job_api:app", "--app-dir", "scripts/job", "--host", "127.0.0.1", "--port", str(a.port)],
                           env=env, stdout=open(out / "api.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(120):
            try:
                if call(H + "/health")[0] == 200: break
            except Exception: pass
            time.sleep(0.5)
        ids = [f"ld{i:02d}_{pathlib.Path(p).stem[:20]}" for i, p in enumerate(a.inputs)]; sub = {}
        def post(i, p):
            t = time.time(); c, b = call(H + "/jobs", {"input": p, "job_id": ids[i], "target_lang": "en", "face": "keysync"}); sub[ids[i]] = (t, c, b.get("status"))
        th = [threading.Thread(target=post, args=(i, p)) for i, p in enumerate(a.inputs)]; t0 = time.time()
        [x.start() for x in th]; [x.join() for x in th]
        print("posted:", {k: v[1:] for k, v in sub.items()}, flush=True)
        seen_run, done, max_run, polls = {}, {}, 0, 0
        while len(done) < len(ids) and time.time() - t0 < 6 * 3600:
            c, b = call(H + "/jobs"); rows = {r["job_id"]: r for r in b.get("jobs", []) if r["job_id"] in ids}; polls += 1
            running = [j for j, r in rows.items() if r["status"] == "RUNNING"]; max_run = max(max_run, len(running)); now = time.time()
            for j in running: seen_run.setdefault(j, now)
            for j, r in rows.items():
                if r["status"] not in ("QUEUED", "RUNNING") and j not in done: done[j] = now; print(f"{j} {r['status']} at +{now - t0:.0f} s", flush=True)
            time.sleep(3)
        res = {}
        for j in ids:
            c, s = call(f"{H}/jobs/{j}")
            res[j] = {"status": s.get("status"), "error": (s.get("error") or {}).get("code"), "warnings": [w.get("code") for w in s.get("warnings") or []],
                      "video_minutes": s.get("video_minutes"), "gpu_busy_s": s.get("gpu_busy_s"), "wall_s": s.get("wall_s"),
                      "queue_wait_s": round(seen_run.get(j, done.get(j, t0)) - sub[j][0], 1), "finished_at_s": round(done.get(j, 0) - t0, 1)}
        order = sorted(seen_run, key=seen_run.get); vm = sum(r["video_minutes"] or 0 for r in res.values()); el = max(done.values()) - t0 if done else None
        gb = sum(r["gpu_busy_s"] or 0 for r in res.values())
        summ = {"jobs": len(ids), "posted_202": sum(v[1] == 202 for v in sub.values()), "all_done": len(done) == len(ids), "statuses": {j: r["status"] for j, r in res.items()},
                "max_running_at_once": max_run, "polls": polls, "fifo": order == [j for j in ids if j in seen_run], "elapsed_s": round(el, 1) if el else None,
                "video_minutes": round(vm, 2), "throughput_video_min_per_h": round(vm / el * 3600, 1) if el else None, "jobs_per_h": round(len(ids) / el * 3600, 1) if el else None,
                "gpu_s_per_video_min_all": round(gb / vm, 1) if vm else None}
        (out / "tz_load.json").write_text(json.dumps({"summary": summ, "jobs": res}, indent=1))
        print("TZ_LOAD", json.dumps(summ), flush=True)
    finally:
        try: os.killpg(api.pid, signal.SIGTERM)
        except Exception: pass


if __name__ == "__main__":
    main()
