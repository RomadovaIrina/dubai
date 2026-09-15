#!/usr/bin/env python3
"""Pilot 0.5 — GPU-minutes per minute of source video, CodeFormer disabled.

This is intentionally a command harness, not a second pipeline implementation. Point it
at the real clean-pipeline command once that command exists. The harness measures how long
this process tree holds GPU memory (billing-like GPU occupancy) and normalizes that by
source duration. CPU-only gaps do not count unless the pipeline keeps models resident,
which is exactly what would occupy the GPU in production.

Example:
  python scripts/pilot/benchmark_05_gpu_coefficient.py test_videos \
    --command-template 'python -m dabai.pipeline --input {input} --output {output} --no-codeformer' \
    --repeat 2
"""
from __future__ import annotations
import argparse, json, os, pathlib, shlex, subprocess, sys, threading, time, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, append_jsonl, write_json, sh, gpu_info

OUT = PILOT / "0.5_gpu_coefficient.jsonl"
WORK = PILOT / "0.5_work"


def probe_duration(p: pathlib.Path) -> float:
    x = sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{p}"', timeout=30)
    return float(x)


def descendants(pid:int)->set[int]:
    out={pid}; q=[pid]
    while q:
        x=q.pop()
        try:
            for f in pathlib.Path(f"/proc/{x}/task").glob("*/children"):
                for c in f.read_text().split():
                    c=int(c)
                    if c not in out: out.add(c); q.append(c)
        except Exception: pass
    return out


def gpu_pids()->set[int]:
    raw=sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits", timeout=10)
    out=set()
    for x in raw.splitlines():
        try: out.add(int(x.strip()))
        except Exception: pass
    return out


class Occupancy(threading.Thread):
    def __init__(self,pid:int,interval:float=.2):
        super().__init__(daemon=True); self.pid=pid; self.interval=interval
        self.stop_evt=threading.Event(); self.samples=0; self.active=0
    def run(self):
        while not self.stop_evt.is_set():
            mine=descendants(self.pid); gp=gpu_pids(); self.samples+=1
            if mine & gp: self.active+=1
            self.stop_evt.wait(self.interval)
    def stop(self): self.stop_evt.set(); self.join(5)
    @property
    def gpu_seconds(self): return self.active*self.interval


def run_once(video:pathlib.Path, template:str, idx:int, interval:float)->dict:
    WORK.mkdir(parents=True, exist_ok=True)
    out=WORK/f"{video.stem}_r{idx}.mp4"
    cmd=template.format(input=shlex.quote(str(video)), output=shlex.quote(str(out)), stem=video.stem)
    t0=time.perf_counter(); p=subprocess.Popen(cmd,shell=True,cwd=str(ROOT))
    mon=Occupancy(p.pid,interval); mon.start(); rc=p.wait(); mon.stop(); wall=time.perf_counter()-t0
    dur=probe_duration(video)
    return {"video":video.name,"duration_s":dur,"run":idx,"returncode":rc,
            "wall_s":round(wall,3),"gpu_occupied_s":round(mon.gpu_seconds,3),
            "gpu_min_per_video_min":round(mon.gpu_seconds/dur,4),
            "samples":mon.samples,"active_samples":mon.active,"command":cmd,
            "output":str(out),"at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("videos", help="video file or directory")
    ap.add_argument("--command-template", required=True,
                    help="must contain {input} and {output}; MUST disable CodeFormer")
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--interval",type=float,default=.2)
    ap.add_argument("--clear",action="store_true")
    a=ap.parse_args()
    if "{input}" not in a.command_template or "{output}" not in a.command_template:
        ap.error("--command-template must contain {input} and {output}")
    p=pathlib.Path(a.videos)
    vids=[p] if p.is_file() else sorted([x for x in p.iterdir() if x.suffix.lower() in {'.mp4','.mov','.mkv','.avi'}])
    if not vids: raise SystemExit("no videos")
    if a.clear and OUT.exists(): OUT.unlink()
    rows=[]
    for v in vids:
        for i in range(1,a.repeat+1):
            print(f"== {v.name} run {i}/{a.repeat} ==",flush=True)
            r=run_once(v,a.command_template,i,a.interval); rows.append(r); append_jsonl(OUT,r); print(r)
            if r["returncode"]!=0: print("pipeline failed; keeping measurement",file=sys.stderr)
    ok=[r for r in rows if r["returncode"]==0]
    coeff=[r["gpu_min_per_video_min"] for r in ok]
    g=gpu_info(); target_gpu=g.get("capability")=="sm_120"
    summary={"task":"0.5","runs":len(rows),"successful_runs":len(ok),"gpu":g,
             "mean_gpu_min_per_video_min":round(statistics.mean(coeff),4) if coeff else None,
             "median_gpu_min_per_video_min":round(statistics.median(coeff),4) if coeff else None,
             "min":min(coeff) if coeff else None,"max":max(coeff) if coeff else None,
             "target_ratio":5.0,"closed":bool(ok) and target_gpu,"provisional":bool(ok) and not target_gpu,
             "method":"GPU process-tree occupancy sampled via nvidia-smi; CodeFormer must be disabled by command template"}
    write_json(PILOT/"0.5_gpu_coefficient.json",summary)
    md=["# Pilot 0.5 — base GPU coefficient", "",
        f"Successful runs: **{len(ok)}/{len(rows)}**  ",
        f"Mean: **{summary['mean_gpu_min_per_video_min']} GPU-min/video-min**  ",
        f"Median: **{summary['median_gpu_min_per_video_min']} GPU-min/video-min**  ","",
        "| video | duration s | wall s | GPU occupied s | GPU-min/video-min | rc |","|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r['video']} | {r['duration_s']:.2f} | {r['wall_s']:.2f} | {r['gpu_occupied_s']:.2f} | {r['gpu_min_per_video_min']:.3f} | {r['returncode']} |" for r in rows]
    (PILOT/"0.5_gpu_coefficient.md").write_text("\n".join(md)+"\n")
    print("\n".join(md)); return 0 if ok else 1

if __name__=="__main__": raise SystemExit(main())
