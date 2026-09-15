#!/usr/bin/env python3
"""Pilot 0.10 — generic cold-start stopwatch for target serverless/NVMe infra.

The formal metric is only valid on the target topology: multi-stage image + weights on
network NVMe. This harness intentionally does not hard-code RunPod/Vast APIs. Supply the
provider commands that start a cold node, test readiness, and tear it down.

Example shape:
  python scripts/pilot/benchmark_10_cold_start.py \
    --start-command './infra/start_node.sh' \
    --ready-command './infra/is_worker_ready.sh' \
    --stop-command './infra/stop_node.sh' --repeat 3

ready-command must exit 0 only when the worker AND required model weights are usable.
"""
from __future__ import annotations
import argparse, pathlib, statistics, subprocess, sys, time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json


def call(cmd:str,timeout:float|None=None)->subprocess.CompletedProcess:
    return subprocess.run(cmd,shell=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)


def one(a,idx:int)->dict:
    print(f"== cold start {idx}/{a.repeat} ==",flush=True)
    t0=time.perf_counter(); s=call(a.start_command,timeout=a.start_timeout)
    if s.returncode!=0:
        return {"run":idx,"status":"START_FAIL","seconds":round(time.perf_counter()-t0,3),"start_output":s.stdout[-3000:]}
    deadline=time.monotonic()+a.timeout; attempts=0; last=""
    while time.monotonic()<deadline:
        attempts+=1
        try:
            r=call(a.ready_command,timeout=min(a.ready_timeout,a.poll_interval))
            last=r.stdout[-2000:]
            if r.returncode==0:
                sec=time.perf_counter()-t0
                row={"run":idx,"status":"PASS","seconds":round(sec,3),"poll_attempts":attempts,
                     "start_output":s.stdout[-2000:],"ready_output":last}
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(a.poll_interval)
    else:
        row={"run":idx,"status":"TIMEOUT","seconds":round(time.perf_counter()-t0,3),
             "poll_attempts":attempts,"ready_output":last}
    if a.stop_command:
        try: row["stop_output"]=call(a.stop_command,timeout=a.stop_timeout).stdout[-2000:]
        except Exception as e: row["stop_output"]=f"{type(e).__name__}: {e}"
    if idx<a.repeat and a.cooldown: time.sleep(a.cooldown)
    return row


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--start-command',required=True); ap.add_argument('--ready-command',required=True)
    ap.add_argument('--stop-command'); ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--timeout',type=float,default=300); ap.add_argument('--poll-interval',type=float,default=2)
    ap.add_argument('--start-timeout',type=float,default=60); ap.add_argument('--ready-timeout',type=float,default=2)
    ap.add_argument('--stop-timeout',type=float,default=60); ap.add_argument('--cooldown',type=float,default=5)
    ap.add_argument('--target-seconds',type=float,default=60)
    ap.add_argument('--confirm-target-topology',action='store_true', help='required for formal closure: this run really used the target multi-stage image + network NVMe')
    ap.add_argument('--topology-note',default='MUST be run on target multi-stage image + network NVMe')
    a=ap.parse_args(); rows=[one(a,i) for i in range(1,a.repeat+1)]
    ok=[r['seconds'] for r in rows if r['status']=='PASS']
    rep={"task":"0.10","topology_note":a.topology_note,"target_seconds":a.target_seconds,"rows":rows,
         "mean_seconds":round(statistics.mean(ok),3) if ok else None,
         "median_seconds":round(statistics.median(ok),3) if ok else None,
         "max_seconds":max(ok) if ok else None,
         "within_target_all":all(x<=a.target_seconds for x in ok) if len(ok)==a.repeat else False,
         "closed":len(ok)==a.repeat and a.confirm_target_topology,"provisional":len(ok)==a.repeat and not a.confirm_target_topology}
    write_json(PILOT/'0.10_cold_start.json',rep)
    md=["# Pilot 0.10 — cold start","",f"Topology: {a.topology_note}  ",
        f"Target: **{a.target_seconds:.1f}s**  ",f"Median: **{rep['median_seconds']}s**  ",
        f"All runs <= target: **{rep['within_target_all']}**","",
        "| run | status | seconds | polls |","|---:|---|---:|---:|"]
    md += [f"| {r['run']} | {r['status']} | {r['seconds']} | {r.get('poll_attempts','-')} |" for r in rows]
    (PILOT/'0.10_cold_start.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))
    return 0 if rep['closed'] else 1
if __name__=='__main__': raise SystemExit(main())
