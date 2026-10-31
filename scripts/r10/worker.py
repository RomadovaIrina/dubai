#!/usr/bin/env python3
"""r10 speed: a queue worker for scripts/r9/e2e.sh that overlaps the CPU-bound face composite of job N with the GPU stages of job N+1.

Policy (spec 3.5: two heavy GPU jobs on one card = OOM):
  - the heavy GPU stages of every job (audio + sep, face_gen = LatentSync) run under ONE exclusive lock, R10_HEAVY_LOCK (e2e.sh wraps them
    when the variable is set; a separate file from /workspace/dub/gpu.lock, which the stages' inner scripts take themselves);
  - face_comp (CF-detail / layer SR on the GPU, ~12 GB peak; the rest CPU: repair, fallback, fps_composite, encodes) runs outside the lock but
    waits for the worker's go (R10_COMP_GATE: work/comp_ready -> work/comp_go), given once no other job is before / in its audio stage
    (~30 GB peak: audio + composite would not fit 32 GB); so job N's composite overlaps job N+1's face_gen (~14 GB);
  - at most --inflight jobs at a time (2): job N+1 is admitted when job N has entered face_gen (it then waits on the lock for N's face_gen),
    so the order of the heavy stages is the queue order: A(N) G(N) A(N+1) [G(N+1) || C(N)] A(N+2) ...
One NVML sampler for the whole queue (R10_NVML, per-process SM util tagged R10_JOB=<id>): the queue's GPU-busy s per video-minute counts
the device busy time ONCE, however the jobs overlap.
    python scripts/r10/worker.py --queue q.txt --out DIR [--inflight 2] [--env K=V ...]      q.txt: "<id> <input path>" per line
  -> DIR/<id>/ (the e2e outputs), DIR/queue.json (per job wall / stage times, queue wall, GPU-busy, throughput), DIR/nvml.jsonl
"""
import argparse, json, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
PRE_GEN = ("", "precheck", "upright", "wait_heavy", "audio", "sep")   # a job here needs (or is about to need) the ~30 GB audio stage
ADMIT_AFTER = ("face_gen", "wait_comp", "face_comp", "mux", "metrics", "end")


def stages(out):
    p = out / "work" / "stages.tsv"
    if not p.exists(): return []
    return [l.split("\t") for l in p.read_text().splitlines() if "\t" in l]


def last(out):
    s = stages(out)
    return s[-1][0] if s else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True); ap.add_argument("--out", required=True); ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--env", nargs="*", default=[]); ap.add_argument("--lock", default="/workspace/dub/r10_heavy.lock")
    ap.add_argument("--gen-worker", action="store_true", help="one resident face-generator worker for the queue (scripts/r10/gen_worker.sh)")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    jobs = [l.split(None, 1) for l in open(a.queue).read().splitlines() if l.strip() and not l.startswith("#")]
    env = dict(os.environ); env.update(dict(kv.split("=", 1) for kv in a.env))
    env["R10_HEAVY_LOCK"] = a.lock; env["R10_COMP_GATE"] = "1";env["R10_NVML"] = str(out / "nvml.jsonl"); env["E2E_FRAME_GATE"] = env.get("E2E_FRAME_GATE", "off")
    (out / "nvml.jsonl").unlink(missing_ok=True)
    if a.gen_worker:   # r10: one resident generator for the whole queue (LatentSync + compiled graphs loaded once); counted in the queue's GPU time
        sp = out / "gen_spool"; env["R10_GEN_SPOOL"] = str(sp)
        subprocess.run(["bash", str(ROOT / "scripts/r10/gen_worker.sh"), "start", str(sp)], env=env, check=True)
    nm = subprocess.Popen(["/venv/main/bin/python", str(HERE / "gpu_sampler.py"), "--out", str(out / "nvml.jsonl")], stdout=subprocess.DEVNULL, stderr=open(out / "gpu_sampler.log", "w"))
    t0 = time.time(); run = {}; done = {}; pending = list(jobs); log = open(out / "worker.log", "a", buffering=1)
    def say(s): log.write(f"{time.strftime('%T')} {s}\n"); print(s, flush=True)
    say(f"worker start: {len(jobs)} jobs, inflight {a.inflight}, lock {a.lock}, env {a.env}")
    while pending or run:
        for i, (p, o, ts) in list(run.items()):
            if p.poll() is not None:
                done[i] = {"rc": p.returncode, "start": ts, "end": time.time()}; del run[i]; say(f"done {i} rc {p.returncode} {done[i]['end'] - ts:.0f} s")
        for i, (p, o, _) in run.items():   # the composite of a job (~12 GB) starts only when no other job is before / in its audio stage (~30 GB)
            if (o / "work" / "comp_ready").exists() and not (o / "work" / "comp_go").exists() and not (pending and len(run) < a.inflight) and \
                    not any(last(o2) in PRE_GEN for j, (_, o2, _) in run.items() if j != i):
                (o / "work" / "comp_go").touch(); say(f"comp go {i}")
        gate = all(last(o) in ADMIT_AFTER for _, o, _ in run.values())   # admit the next job once every running job is past its audio stage
        if pending and len(run) < a.inflight and gate:
            i, inp = pending.pop(0); o = out / i; o.mkdir(exist_ok=True)
            e = dict(env, ID=i)
            p = subprocess.Popen(["bash", str(ROOT / "scripts/r9/e2e.sh"), inp.strip(), str(o)], env=e, cwd=ROOT, stdout=open(o / "worker_e2e.out", "w"), stderr=subprocess.STDOUT)
            run[i] = (p, o, time.time()); say(f"start {i} ({len(run)} in flight)")
        time.sleep(1)
    t1 = time.time()
    if a.gen_worker: subprocess.run(["bash", str(ROOT / "scripts/r10/gen_worker.sh"), "stop", env["R10_GEN_SPOOL"]], env=env)
    nm.terminate(); nm.wait()
    sys.path.insert(0, str(HERE)); import gpu_sampler as G
    smp = G.load(out / "nvml.jsonl"); res = {"jobs": {}, "queue_wall_s": round(t1 - t0, 1), "env": a.env, "inflight": a.inflight}
    vtot = 0.0
    for i, _ in jobs:
        tj = out / i / "timing.json"; t = json.loads(tj.read_text()) if tj.exists() else {}
        vmin = t.get("video_min") or 0; vtot += vmin; st = stages(out / i); per = {}
        for (n, s0), (_, s1) in zip(st, st[1:]):
            b, w = G.busy(smp, float(s0), float(s1), i); x = per.setdefault(n, {"wall_s": 0, "gpu_busy_s": 0, "gpu_weighted_s": 0})
            x["wall_s"] += float(s1) - float(s0); x["gpu_busy_s"] += b; x["gpu_weighted_s"] += w
        res["jobs"][i] = {"rc": done.get(i, {}).get("rc"), "video_min": vmin, "job_wall_s": round(done[i]["end"] - done[i]["start"], 1) if i in done else None,
                          "status": (json.loads((out / i / "manifest.json").read_text()).get("status") if (out / i / "manifest.json").exists() else None),
                          "stages": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in per.items()}}
    b, w = G.busy(smp, t0, t1)
    res.update({"video_min": round(vtot, 3), "queue_gpu_busy_s": round(b, 1), "queue_gpu_weighted_s": round(w, 1),
                "wall_s_per_vmin": round((t1 - t0) / vtot, 1) if vtot else None, "gpu_busy_s_per_vmin": round(b / vtot, 1) if vtot else None,
                "gpu_weighted_s_per_vmin": round(w / vtot, 1) if vtot else None,
                "gpu_busy_fraction_of_wall": round(b / (t1 - t0), 3)})
    (out / "queue.json").write_text(json.dumps(res, indent=1))
    say(f"QUEUE_DONE {len(jobs)} jobs {vtot:.2f} vmin: wall {res['wall_s_per_vmin']} s/vmin, GPU-busy {res['gpu_busy_s_per_vmin']} s/vmin "
        f"(weighted {res['gpu_weighted_s_per_vmin']}), GPU busy {100 * res['gpu_busy_fraction_of_wall']:.0f} % of the wall")


if __name__ == "__main__":
    main()
