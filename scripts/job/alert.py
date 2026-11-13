#!/usr/bin/env python3
"""DabAI worker alerts (r18, TZ 10.4). Standard library only: runs on any host (cron, a sidecar, the backend box).

It reads GET /health of each worker (no token needed; the r18 "monitor" block is in it) and prints one line per problem.

    python3 alert.py --url http://worker:8000                 # one check; exit 0 = OK, 1 = alerts, 2 = bad arguments
    python3 alert.py --url http://w1:8000 --url http://w2:8000 --every 60    # a loop: notifies on CHANGE only (new / cleared)
    DABAI_ALERT_WEBHOOK=https://hooks.slack.com/...  python3 alert.py ...      # also POST {"text": ...} (Slack / Mattermost / any JSON hook)
    DABAI_ALERT_TG_TOKEN=... DABAI_ALERT_TG_CHAT=...  python3 alert.py ...     # also send to a Telegram chat

Rules (thresholds by env; CRIT = the node cannot work, WARN = look at it):
  CRIT down          the worker does not answer / answers garbage
  CRIT unhealthy     /health 503: GPU not visible, weights missing, disk below DABAI_MIN_FREE_GB, the worker thread is dead
  CRIT worker_stalled the worker thread lives but its idle loop has not ticked for > 60 s
  CRIT stuck         a stuck job (stale / orphan / timeout / step / lost_queued; monitor.py) — one alert per job id
  WARN queue         queue depth > DABAI_ALERT_QUEUE (3: the TZ 3.7 scale-out trigger)
  WARN queue_wait    the oldest queued job waits > DABAI_ALERT_QUEUE_WAIT_S (1800 s)
  WARN vram          the last job's VRAM peak > DABAI_ALERT_VRAM_MIB (31500 MiB of 32607: the open 31.6 GB risk)
  WARN gpu_norm      the recent GPU busy s per video minute > DABAI_ALERT_GPU_PER_VMIN (300: the TZ norm)
  WARN wall          the recent wall s per video minute > DABAI_ALERT_WALL_PER_VMIN (900; measured 484 on a 10-min video)
  WARN errors        ERROR + TIMEOUT share of the recent jobs > DABAI_ALERT_ERROR_RATIO (0.2), at least 5 jobs
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.parse, urllib.request


def _f(name, d):
    try: return float(os.environ.get(name, d))
    except ValueError: return float(d)


def fetch(url: str, timeout: float = 15.0) -> tuple[int | None, dict | None, str | None]:
    rq = urllib.request.Request(url.rstrip("/") + "/health", headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r: return r.status, json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:   # 503 = unhealthy, still a JSON body
        try: return e.code, json.loads(e.read().decode()), None
        except Exception: return e.code, None, f"HTTP {e.code}"
    except Exception as e: return None, None, f"{type(e).__name__}: {e}"


def rules(name: str, code: int | None, h: dict | None, err: str | None) -> list[tuple[str, str, str]]:
    """-> [(level, key, text)]; key identifies the alert across checks (for 'on change' notifications)"""
    if h is None: return [("CRIT", f"{name}:down", f"{name}: worker unreachable ({err or code})")]
    out = []; m = h.get("monitor") or {}; w = m.get("worker") or {}
    if code == 503 or not h.get("ok"):
        why = []
        if not h.get("gpu"): why.append("GPU not visible")
        miss = [x for x in h.get("missing_weights") or [] if not str(x).startswith("student")]
        if miss: why.append(f"missing weights: {', '.join(map(str, miss[:5]))}")
        mf = _f("DABAI_MIN_FREE_GB", 30)
        if (h.get("disk_free_gb") or 0) < mf: why.append(f"disk {h.get('disk_free_gb')} GB free (< {mf:g})")
        if w and not w.get("thread_alive"): why.append("the worker thread is dead")
        out.append(("CRIT", f"{name}:unhealthy", f"{name}: /health {code}: " + ("; ".join(why) or "not ok")))
    if w.get("thread_alive") and not w.get("alive"):
        out.append(("CRIT", f"{name}:worker_stalled", f"{name}: the worker loop has not ticked for {w.get('heartbeat_age_s')} s"))
    if "error" in m: out.append(("WARN", f"{name}:monitor", f"{name}: the monitor block failed: {m['error']}"))
    for s in m.get("stuck") or []:
        out.append(("CRIT", f"{name}:stuck:{s.get('job_id')}", f"{name}: job {s.get('job_id')} stuck ({s.get('reason')}): {s.get('detail')}"))
    q = m.get("queue_depth", h.get("queue_depth")) or 0
    if q > _f("DABAI_ALERT_QUEUE", 3): out.append(("WARN", f"{name}:queue", f"{name}: queue depth {q} (> {_f('DABAI_ALERT_QUEUE', 3):g}: add a node, TZ 3.7)"))
    qw = m.get("queue_oldest_wait_s") or 0
    if qw > _f("DABAI_ALERT_QUEUE_WAIT_S", 1800): out.append(("WARN", f"{name}:queue_wait", f"{name}: the oldest queued job waits {qw:.0f} s"))
    lj = m.get("last_job") or {}; rc = m.get("recent") or {}
    v = lj.get("vram_peak_mib")
    if v and v > _f("DABAI_ALERT_VRAM_MIB", 31500):
        out.append(("WARN", f"{name}:vram", f"{name}: job {lj.get('job_id')} VRAM peak {v} MiB (> {_f('DABAI_ALERT_VRAM_MIB', 31500):g}: OOM risk)"))
    gp = rc.get("gpu_busy_s_per_video_min")
    if gp and gp > _f("DABAI_ALERT_GPU_PER_VMIN", 300):
        out.append(("WARN", f"{name}:gpu_norm", f"{name}: {gp} GPU-s per video minute over the last {rc.get('n')} jobs (norm {_f('DABAI_ALERT_GPU_PER_VMIN', 300):g})"))
    wl = rc.get("wall_s_per_video_min")
    if wl and wl > _f("DABAI_ALERT_WALL_PER_VMIN", 900):
        out.append(("WARN", f"{name}:wall", f"{name}: {wl} wall s per video minute over the last {rc.get('n')} jobs"))
    er = rc.get("error_ratio") or 0
    if (rc.get("n") or 0) >= 5 and er > _f("DABAI_ALERT_ERROR_RATIO", 0.2):
        out.append(("WARN", f"{name}:errors", f"{name}: {er:.0%} of the last {rc.get('n')} jobs ended ERROR / TIMEOUT"))
    return out


def notify(lines: list[str]):
    if not lines: return
    text = "\n".join(lines)
    hook = os.environ.get("DABAI_ALERT_WEBHOOK")
    if hook:
        try: urllib.request.urlopen(urllib.request.Request(hook, data=json.dumps({"text": text}).encode(), headers={"content-type": "application/json"}), timeout=15)
        except Exception as e: print(f"alert: webhook failed: {type(e).__name__}", file=sys.stderr)
    tg, chat = os.environ.get("DABAI_ALERT_TG_TOKEN"), os.environ.get("DABAI_ALERT_TG_CHAT")
    if tg and chat:
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{tg}/sendMessage",
                                   data=urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode(), timeout=15)
        except Exception as e: print(f"alert: telegram failed: {type(e).__name__}", file=sys.stderr)   # (the token is never printed)


def check(urls: list[str]) -> list[tuple[str, str, str]]:
    out = []
    for u in urls:
        name = urllib.parse.urlparse(u).netloc or u
        out += rules(name, *fetch(u))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", default=[], help="a worker base URL (repeatable; or DABAI_ALERT_URLS=a,b)")
    ap.add_argument("--every", type=float, default=0, help="loop every N s and notify on change (0 = one check)")
    ap.add_argument("--json", action="store_true", help="print the alerts as JSON")
    a = ap.parse_args(argv)
    urls = a.url or [u for u in os.environ.get("DABAI_ALERT_URLS", "").split(",") if u]
    if not urls: ap.print_usage(sys.stderr); return 2
    if not a.every:
        al = check(urls)
        if a.json: print(json.dumps([{"level": l, "key": k, "text": t} for l, k, t in al], ensure_ascii=False))
        else:
            for l, _, t in al: print(f"{l} {t}")
            if not al: print("OK " + ", ".join(urls))
        notify([f"{l} {t}" for l, _, t in al]); return 1 if al else 0
    active: dict[str, str] = {}
    while True:
        al = {k: f"{l} {t}" for l, k, t in check(urls)}
        new = [t for k, t in al.items() if k not in active]; gone = [f"CLEARED {active[k]}" for k in active if k not in al]
        for t in new + gone: print(time.strftime("%H:%M:%S"), t, flush=True)
        notify(new + gone); active = al
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
