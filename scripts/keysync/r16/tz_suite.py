#!/usr/bin/env python3
"""r16 TZ 12.3 / 0.4 / 0.5 on KeySync: ONE worker process (scripts/job/dabai_job.run_job, the live backend entry) runs a list of jobs
sequentially and records, per job: status / flags / user warnings / refund, wall, gpu_busy_s and video_minutes from the status, GPU-s per
video minute, and resources sampled every 2 s while the job runs: system RAM in use above the pre-job baseline (peak), /workspace disk in use
above the baseline (peak), GPU memory (peak; the GPU is shared with Q, so a sample is only clean when no foreign process is on it).
    python scripts/keysync/r16/tz_suite.py --out DIR --job name=input.mp4[,key=val...] ...   (keys: lang, subs, wm, face, keep, E.<ENV>=<val> = debug_env)
e.g. --job long10=/workspace/dub/r16tz/in/long_10min.mp4,keep=1 --job no_face=/.../no_face.mp4"""
import argparse, json, os, pathlib, shutil, subprocess, sys, threading, time
R = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R / "scripts" / "job"))
import dabai_job  # noqa: E402


def mem_used():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return (m["MemTotal"] - m["MemAvailable"]) * 1024


def disk_used(): return shutil.disk_usage("/workspace").used


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        mb, ut = [int(x) for x in o.strip().split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return mb, ut, [l.split(",")[0].strip() for l in apps.strip().splitlines() if l.strip()]
    except Exception: return 0, 0, []


def mine(pid):   # is this GPU pid a descendant of this worker? (else Q's process shares the GPU)
    try:
        while pid > 1:
            if pid == os.getpid(): return True
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except Exception: pass
    return False


class Sampler(threading.Thread):
    def __init__(s):
        super().__init__(daemon=True); s.stop = False; s.m0, s.d0 = mem_used(), disk_used(); s.pm = s.pd = s.pg = 0; s.foreign = 0; s.n = 0
    def run(s):
        while not s.stop:
            s.pm = max(s.pm, mem_used() - s.m0); s.pd = max(s.pd, disk_used() - s.d0); mb, ut, pids = gpu(); s.pg = max(s.pg, mb); s.n += 1
            s.foreign += any(not mine(int(p)) for p in pids if p.isdigit()); time.sleep(2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--job", action="append", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); res = {}
    if (out / "tz_suite.json").exists(): res = json.loads((out / "tz_suite.json").read_text())
    for spec in a.job:
        name, rest = spec.split("=", 1); inp, *kv = rest.split(","); kv = dict(x.split("=", 1) for x in kv)
        if name in res and res[name].get("status") not in (None, "WORKER_CRASH"): print(f"skip {name} (done)", flush=True); continue
        job = {"job_id": f"tz_{name}", "input": inp, "out_dir": str(out / name), "target_lang": kv.get("lang", "en"), "face": kv.get("face", "keysync"),
               "output": {"keep_intermediate": kv.get("keep") == "1"}}
        if kv.get("lipsync") == "off": job["lipsync"] = "off"   # audio only (+ mux): reproducibility runs
        de = {k[2:]: v for k, v in kv.items() if k.startswith("E.")}   # E.KEY=VAL -> the job's debug_env (fault injection / golden dub reuse)
        if de: job["debug_env"] = de
        if kv.get("subs"): job["subs"] = {"enabled": True, "style": kv["subs"], "lang": "dub", "karaoke": True}
        if kv.get("wm"): job["watermark"] = {"enabled": True, "text": kv["wm"]}
        smp = Sampler(); smp.start(); t0 = time.time()
        try: st = dabai_job.run_job(job)
        except Exception as e: st = {"status": "WORKER_CRASH", "error": repr(e)}
        smp.stop = True; smp.join(5); wall = time.time() - t0
        vm, gb = st.get("video_minutes"), st.get("gpu_busy_s")
        res[name] = {"input": inp, "job": {k: v for k, v in job.items() if k not in ("out_dir",)}, "status": st.get("status"), "flags": st.get("flags"),
                     "warnings": [w.get("code") for w in st.get("warnings") or []], "error": st.get("error"), "refund_suggested": st.get("refund_suggested"),
                     "video_status": (st.get("spec") or {}).get("video_status"), "wall_s": round(wall, 1), "video_minutes": vm, "gpu_busy_s": gb,
                     "gpu_s_per_video_min": round(gb / vm, 1) if vm and gb else None, "wall_s_per_video_min": round(wall / vm, 1) if vm else None,
                     "peak_ram_gb_above_baseline": round(smp.pm / 1e9, 2), "peak_disk_gb_above_baseline": round(smp.pd / 1e9, 2), "peak_gpu_mem_mb": smp.pg,
                     "gpu_shared_samples": f"{smp.foreign}/{smp.n}", "output": st.get("output"), "status_json": str(out / name / "status.json")}
        print(f"{name}: {json.dumps(res[name], ensure_ascii=False, default=str)}", flush=True)
        (out / "tz_suite.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print("TZ_SUITE_DONE", len(res))


if __name__ == "__main__":
    main()
