#!/usr/bin/env python3
"""r18 TZ 10.4: CPU tests of monitor.py + alert.py (no GPU, no FastAPI; ~1 s).   python3 scripts/job/test_monitor.py -v"""
from __future__ import annotations
import http.server, io, json, os, pathlib, re, sys, tempfile, threading, time, unittest
from contextlib import redirect_stdout
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import monitor as mon  # noqa: E402
import alert  # noqa: E402

STEPS = {"audio": ("x", 30, 60, 120), "face_keysync": ("x", 50, 300, 260), "mux": ("x", 1, 2, 3)}
ISO = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))
# the real 10-min job of 12.11 (C:\repos\chkpts\r17_docker\long10_1211\data\jobs\long10\metrics.json)
LONG10 = {"video_minutes": 10.176, "wall_s": 4934.6, "gpu_busy_s": 3064.1, "vram_peak_mib": 30517, "lipsync": {"degraded_ratio": 0.0008},
          "gpu_nvml": {"gpu_weighted_s": 1736.1}}


def job(root: pathlib.Path, jid: str, status: str, age: float = 0, metrics: dict | None = None, events: list | None = None, **kw) -> pathlib.Path:
    d = root / jid; d.mkdir(parents=True, exist_ok=True)
    st = {"schema": "dabai-job/2", "job_id": jid, "status": status, **kw}
    (d / "status.json").write_text(json.dumps(st))
    if metrics is not None: (d / "metrics.json").write_text(json.dumps(metrics))
    if events is not None: (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    t = time.time() - age; os.utime(d / "status.json", (t, t))
    return d


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = pathlib.Path(self.tmp.name) / "jobs"; self.root.mkdir()
        self.led = mon.Ledger(self.root / ".monitor" / "ledger.jsonl")

    def tearDown(self): self.tmp.cleanup()

    # ---------------- ledger
    def test_ledger_record_backfill_dedup(self):
        d = job(self.root, "long10", "SUCCESS", metrics=LONG10, finished_at=ISO(time.time() - 100), cold_start=False)
        job(self.root, "e1", "ERROR", error={"code": "E_NO_SPEECH"}, video_minutes=0.5, wall_s=40)
        job(self.root, "q1", "QUEUED"); job(self.root, "r1", "RUNNING")
        self.assertEqual(self.led.backfill(self.root), 2)                 # finished only
        self.assertEqual(self.led.backfill(self.root), 0)                 # idempotent
        self.led.record(d)                                                # re-recorded -> counted once
        rows = self.led.rows(); self.assertEqual(sorted(rows), ["e1", "long10"])
        r = rows["long10"]
        self.assertEqual((r["video_minutes"], r["gpu_busy_s"], r["gpu_weighted_s"], r["vram_peak_mib"]), (10.176, 3064.1, 1736.1, 30517))
        self.assertEqual(rows["e1"]["error"], "E_NO_SPEECH"); self.assertEqual(rows["e1"]["wall_s"], 40)   # status fallback without metrics.json
        with open(self.led.path, "a") as f: f.write('{"job_id": "torn", "sta')                          # a torn last line
        self.assertEqual(sorted(self.led.rows()), ["e1", "long10"])
        self.assertIsNone(self.led.record(self.root / "q1"))              # not final -> no row

    # ---------------- stuck jobs
    def test_stuck_rules(self):
        now = time.time()
        ev = lambda step, ago: [{"seq": 1, "t": now - 900, "type": "status"}, {"seq": 2, "t": now - ago, "type": "step", "step": step}]
        job(self.root, "fine", "RUNNING", age=2, events=ev("audio", 100), started_at=ISO(now - 200), video_minutes=1.0)
        job(self.root, "stale", "RUNNING", age=500, started_at=ISO(now - 600), video_minutes=1.0)
        job(self.root, "orph", "RUNNING", age=60, started_at=ISO(now - 70))
        job(self.root, "late", "RUNNING", age=1, started_at=ISO(now - 4000), video_minutes=0.3, params={"timeout_s": 600})
        job(self.root, "hung", "RUNNING", age=1, events=ev("mux", 1850), started_at=ISO(now - 1900), video_minutes=1.0)
        job(self.root, "lostq", "QUEUED", age=120); job(self.root, "okq", "QUEUED", age=120)
        job(self.root, "old", "RUNNING", age=8 * 86400); job(self.root, "done", "SUCCESS", age=9999)
        running = ["fine", "stale", "late", "hung"]
        s = {x["job_id"]: x for x in mon.stuck_jobs(self.root, running, ["okq"], now, STEPS)}
        self.assertEqual(sorted(s), ["hung", "late", "lostq", "orph", "stale"])
        self.assertEqual(s["stale"]["reason"], "stale"); self.assertEqual(s["orph"]["reason"], "orphan"); self.assertEqual(s["late"]["reason"], "timeout")
        self.assertEqual(s["hung"]["reason"], "step"); self.assertIn("mux", s["hung"]["detail"]); self.assertEqual(s["lostq"]["reason"], "lost_queued")
        # the step rule uses max(1800 s, 4 x expected): face_keysync of a 10-min video expects 300 + 2600 s -> 11600 s is fine at 3 h
        job(self.root, "ks", "RUNNING", age=1, events=ev("face_keysync", 3 * 3600), started_at=ISO(now - 3.2 * 3600), video_minutes=10.0)
        self.assertNotIn("ks", {x["job_id"] for x in mon.stuck_jobs(self.root, running + ["ks"], [], now, STEPS)})
        # without the API state (a CLI job dir) orphan / lost_queued are not judged
        self.assertNotIn("lostq", {x["job_id"] for x in mon.stuck_jobs(self.root, None, None, now, STEPS)})

    # ---------------- snapshot + Prometheus
    def _fill(self):
        now = time.time()
        job(self.root, "long10", "SUCCESS", metrics=LONG10, finished_at=ISO(now - 500))
        job(self.root, "p1", "PARTIAL_SUCCESS", metrics={"video_minutes": 0.5, "wall_s": 300, "gpu_busy_s": 100, "vram_peak_mib": 31582}, finished_at=ISO(now - 50))
        job(self.root, "e1", "ERROR", video_minutes=0.2, wall_s=10, finished_at=ISO(now - 400))
        job(self.root, "run", "RUNNING", age=1, started_at=ISO(now - 30), step="audio", progress_pct=12.5)
        job(self.root, "w1", "QUEUED", submitted_ts=now - 90); job(self.root, "w2", "QUEUED", submitted_ts=now - 10)
        self.led.backfill(self.root)
        return mon.snapshot(self.root, self.led, ["run"], ["w1", "w2"], {"alive": True, "thread_alive": True, "heartbeat_age_s": 0, "uptime_s": 3600},
                            now=now, steps=STEPS)

    def test_snapshot(self):
        s = self._fill()
        self.assertEqual(s["queue_depth"], 2); self.assertAlmostEqual(s["queue_oldest_wait_s"], 90, delta=2)
        self.assertEqual(s["running"]["job_id"], "run"); self.assertEqual(s["stuck"], [])
        t = s["totals"]; self.assertEqual(t["jobs"], {"SUCCESS": 1, "PARTIAL_SUCCESS": 1, "ERROR": 1, "TIMEOUT": 0})
        self.assertAlmostEqual(t["gpu_busy_s"], 3164.1); self.assertAlmostEqual(t["gpu_busy_hours"], 3164.1 / 3600, places=3)
        self.assertAlmostEqual(t["video_minutes"], 10.876)
        lj = s["last_job"]; self.assertEqual(lj["job_id"], "p1"); self.assertEqual(lj["vram_peak_mib"], 31582)
        self.assertEqual(lj["wall_s_per_video_min"], 600.0); self.assertEqual(lj["gpu_busy_s_per_video_min"], 200.0)
        rc = s["recent"]; self.assertEqual(rc["n"], 3); self.assertAlmostEqual(rc["error_ratio"], 0.333, places=3)
        self.assertEqual(rc["wall_s_per_video_min"], round((4934.6 + 300) / 10.676, 1))     # successful jobs only, weighted by minutes
        self.assertEqual(rc["vram_peak_mib_max"], 31582)

    def test_prom_text(self):
        health = {"ok": False, "gpu": None, "missing_weights": ["latentsync"], "disk_free_gb": 12.5, "resident": {"state": "ready"},
                  "versions": {"interface": "dabai-job/2", "pipeline": "abc", "image": 'x"y', "profile": "clean"}}
        txt = mon.prom_text(self._fill(), health)
        names, samples = set(), {}
        for line in txt.splitlines():
            if line.startswith("# TYPE "): _, _, n, typ = line.split(" "); self.assertIn(typ, ("gauge", "counter")); names.add(n); continue
            if line.startswith("#") or not line: continue
            m = re.fullmatch(r'([a-z_]+)(\{[^}]*\})? (-?[0-9.]+(?:e[+-]?\d+)?)', line); self.assertIsNotNone(m, line)
            self.assertIn(m.group(1), names, f"sample before its TYPE: {line}"); samples[m.group(1) + (m.group(2) or "")] = float(m.group(3))
        self.assertEqual(samples["dabai_queue_depth"], 2); self.assertEqual(samples["dabai_jobs_stuck"], 0)
        self.assertEqual(samples['dabai_jobs_finished_total{status="PARTIAL_SUCCESS"}'], 1)
        self.assertAlmostEqual(samples["dabai_gpu_busy_seconds_total"], 3164.1)
        self.assertEqual(samples["dabai_last_job_vram_peak_mib"], 31582); self.assertEqual(samples["dabai_health_ok"], 0)
        self.assertEqual(samples["dabai_gpu_visible"], 0); self.assertEqual(samples["dabai_missing_weights"], 1)
        self.assertEqual(samples['dabai_resident_state{state="ready"}'], 1); self.assertEqual(samples['dabai_resident_state{state="off"}'], 0)
        self.assertIn('image="x\\"y"', txt)                                                  # label escaping
        ts = samples["dabai_last_job_finished_timestamp_seconds"]; self.assertGreater(ts, 1.7e9)   # full precision, not 1.79e+09
        self.assertNotIn("nan", txt.lower()); self.assertNotIn("None", txt)

    # ---------------- alerts
    def test_alert_rules(self):
        ok = {"ok": True, "gpu": {"name": "RTX 5090"}, "missing_weights": [], "disk_free_gb": 200,
              "monitor": {"worker": {"alive": True, "thread_alive": True}, "queue_depth": 1, "queue_oldest_wait_s": 5, "stuck": [],
                          "last_job": {"job_id": "a", "vram_peak_mib": 28500}, "recent": {"n": 6, "error_ratio": 0.0, "gpu_busy_s_per_video_min": 234,
                                                                                         "wall_s_per_video_min": 484}}}
        self.assertEqual(alert.rules("w", 200, ok, None), [])
        bad = json.loads(json.dumps(ok)); bad.update(ok=False, gpu=None, disk_free_gb=3)
        bad["monitor"].update(queue_depth=5, queue_oldest_wait_s=4000, stuck=[{"job_id": "j9", "reason": "step", "detail": "step mux runs 1850 s"}])
        bad["monitor"]["worker"] = {"alive": False, "thread_alive": True, "heartbeat_age_s": 300}
        bad["monitor"]["last_job"]["vram_peak_mib"] = 31582
        bad["monitor"]["recent"].update(error_ratio=0.5, gpu_busy_s_per_video_min=301.1, wall_s_per_video_min=950)
        got = {k.split(":", 1)[1]: (l, t) for l, k, t in alert.rules("w", 503, bad, None)}
        self.assertEqual(sorted(got), ["errors", "gpu_norm", "queue", "queue_wait", "stuck:j9", "unhealthy", "vram", "wall", "worker_stalled"])
        self.assertEqual(got["unhealthy"][0], "CRIT"); self.assertIn("GPU not visible", got["unhealthy"][1]); self.assertIn("disk 3", got["unhealthy"][1])
        self.assertEqual(got["vram"][0], "WARN"); self.assertEqual(got["stuck:j9"][0], "CRIT")
        self.assertEqual(alert.rules("w", None, None, "URLError: refused")[0][:2], ("CRIT", "w:down"))
        few = json.loads(json.dumps(ok)); few["monitor"]["recent"].update(n=2, error_ratio=1.0)   # < 5 jobs: no error-ratio alert
        self.assertEqual(alert.rules("w", 200, few, None), [])

    def test_alert_cli_end_to_end(self):
        body = {"ok": True, "gpu": {"name": "g"}, "missing_weights": [], "disk_free_gb": 100,
                "monitor": {"worker": {"alive": True, "thread_alive": True}, "queue_depth": 0, "stuck": [{"job_id": "x1", "reason": "stale", "detail": "d"}],
                            "last_job": None, "recent": {"n": 0}}}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                b = json.dumps(body).encode(); self.send_response(200 if self.path == "/health" else 404)
                self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(b)
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H); th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
        url = f"http://127.0.0.1:{srv.server_port}"
        try:
            buf = io.StringIO()
            with redirect_stdout(buf): rc = alert.main(["--url", url])
            self.assertEqual(rc, 1); self.assertIn("CRIT", buf.getvalue()); self.assertIn("job x1 stuck (stale)", buf.getvalue())
            body["monitor"]["stuck"] = []; buf = io.StringIO()
            with redirect_stdout(buf): rc = alert.main(["--url", url, "--json"])
            self.assertEqual(rc, 0); self.assertEqual(json.loads(buf.getvalue()), [])
        finally: srv.shutdown(); srv.server_close()
        with redirect_stdout(io.StringIO()): self.assertEqual(alert.main(["--url", "http://127.0.0.1:9"]), 1)   # nothing listens: CRIT down


if __name__ == "__main__":
    unittest.main()
