#!/usr/bin/env python3
"""r10 speed: NVML GPU sampler (100 ms) for honest GPU-time accounting (spec: <= 300 GPU-seconds per video-minute on one RTX 5090).

Every --interval s it writes one JSON line: t (epoch), util (device SM util %, NVML's own averaging window), mem_mib, and per-process
SM util since the previous sample (nvmlDeviceGetProcessUtilization), each process tagged with the job it belongs to: the R10_JOB=<id>
of its environment (/proc/<pid>/environ; e2e.sh exports it, every child inherits it), else its command name. With two jobs in flight
(scripts/r10/worker.py) the device-busy seconds are split between jobs by their per-process SM share.
    python scripts/r10/gpu_sampler.py --out nv.jsonl [--interval 0.1] [--stop-file F]        (ends on SIGTERM / SIGINT or when F exists)
    python scripts/r10/gpu_sampler.py --summary nv.jsonl --stages stages.tsv [--vmin X]        per-stage wall / busy / util-weighted s
"""
import argparse, json, os, signal, sys, time


def job_of(pid, cache):
    if pid in cache: return cache[pid]
    tag = None
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        for e in env:
            if e.startswith(b"R10_JOB="): tag = e[8:].decode(errors="replace"); break
        if tag is None: tag = "proc:" + open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        tag = f"pid:{pid}"
    cache[pid] = tag
    return tag


def sample(a):
    import pynvml as N
    N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(int(os.environ.get("R10_GPU_INDEX", "0")))
    run = [True]
    for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, lambda *_: run.__setitem__(0, False))
    last_ts = 0; cache = {}; f = open(a.out, "a", buffering=1); nxt = time.time()
    while run[0] and not (a.stop_file and os.path.exists(a.stop_file)):
        t = time.time(); u = N.nvmlDeviceGetUtilizationRates(h); m = N.nvmlDeviceGetMemoryInfo(h)
        procs = {}
        try:
            for p in N.nvmlDeviceGetProcessUtilization(h, last_ts):
                if p.timeStamp > last_ts: last_ts = p.timeStamp
                j = job_of(p.pid, cache); procs[j] = max(procs.get(j, 0), p.smUtil)
        except N.NVMLError:
            pass   # NVML_ERROR_NOT_FOUND: no process sample since last_ts
        pm = {}
        try:
            for p in N.nvmlDeviceGetComputeRunningProcesses(h):
                j = job_of(p.pid, cache); pm[j] = pm.get(j, 0) + (p.usedGpuMemory or 0) // 2**20
        except N.NVMLError:
            pass
        f.write(json.dumps({"t": round(t, 3), "util": u.gpu, "mem_mib": m.used // 2**20, "proc_sm": procs, "proc_mem_mib": pm}) + "\n")
        if len(cache) > 4096: cache.clear()
        nxt += a.interval; time.sleep(max(0.0, nxt - time.time()))


def load(path):
    out = []
    for l in open(path):
        try: out.append(json.loads(l))
        except ValueError: pass
    return out


def busy(samples, t0, t1, job=None):
    """(busy_s, util_weighted_s) over [t0, t1]: busy = device util > 10 %; with job: the device-busy time scaled by the job's share of the
    per-process SM util of that interval (0 when the job has no GPU process then)"""
    b = w = 0.0
    for s, n in zip(samples, samples[1:]):
        lo, hi = max(s["t"], t0), min(n["t"], t1)
        if hi <= lo: continue
        share = 1.0
        if job is not None:
            ps = n["proc_sm"] or s["proc_sm"]; tot = sum(ps.values())
            share = (ps.get(job, 0) / tot) if tot else (1.0 if job in (n["proc_mem_mib"] or {}) and len(n["proc_mem_mib"]) == 1 else 0.0)
        if s["util"] > 10: b += (hi - lo) * share
        w += (hi - lo) * s["util"] / 100 * share
    return b, w


def summary(a):
    smp = load(a.summary)
    rows = [l.rstrip("\n").split("\t") for l in open(a.stages) if "\t" in l]
    res = {"stages": {}, "interval_ms": round(1000 * (smp[-1]["t"] - smp[0]["t"]) / max(1, len(smp) - 1), 1) if len(smp) > 1 else None}
    for (n, t0), (_, t1) in zip(rows, rows[1:]):
        b, w = busy(smp, float(t0), float(t1), a.job)
        st = res["stages"].setdefault(n, {"wall_s": 0.0, "gpu_busy_s": 0.0, "gpu_weighted_s": 0.0})
        st["wall_s"] += float(t1) - float(t0); st["gpu_busy_s"] += b; st["gpu_weighted_s"] += w
    tot = {k: sum(v[k] for v in res["stages"].values()) for k in ("wall_s", "gpu_busy_s", "gpu_weighted_s")}
    res["total"] = tot
    if a.vmin:
        for v in list(res["stages"].values()) + [tot]:
            for k in ("wall_s", "gpu_busy_s", "gpu_weighted_s"): v[k.replace("_s", "_s_per_vmin")] = round(v[k] / a.vmin, 1)
    for v in list(res["stages"].values()) + [tot]:
        for k in ("wall_s", "gpu_busy_s", "gpu_weighted_s"): v[k] = round(v[k], 1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--interval", type=float, default=0.1); ap.add_argument("--stop-file")
    ap.add_argument("--summary"); ap.add_argument("--stages"); ap.add_argument("--vmin", type=float); ap.add_argument("--job")
    a = ap.parse_args()
    summary(a) if a.summary else sample(a)
