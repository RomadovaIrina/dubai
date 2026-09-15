#!/usr/bin/env python3
"""Pilot 0.8 — SyncNet on the five real customer videos/outputs.

The spec says the real-content value becomes an acceptance threshold but does not define
how five values must be reduced to one number. This script therefore reports all raw
values plus mean/median/min/max and does NOT silently choose the contractual threshold.
"""
from __future__ import annotations
import argparse, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json
from syncnet_common import eval_syncnet

EXT={'.mp4','.mov','.mkv','.avi','.webm'}

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("videos",help="directory containing the 5 real generated videos, or one video")
    ap.add_argument("--expect",type=int,default=5)
    a=ap.parse_args(); p=pathlib.Path(a.videos).resolve()
    vids=[p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix.lower() in EXT)
    if not vids: raise SystemExit("no videos found")
    rows=[]
    for i,v in enumerate(vids,1):
        print(f"[{i}/{len(vids)}] {v.name}",flush=True)
        try:
            r=eval_syncnet(v); r["name"]=v.name; r["status"]="PASS"; rows.append(r); print(r)
        except Exception as e:
            rows.append({"name":v.name,"status":"FAIL","error":f"{type(e).__name__}: {e}"}); print(rows[-1])
    ok=[r for r in rows if r["status"]=="PASS"]
    conf=[r["confidence"] for r in ok]; offs=[abs(r["av_offset_frames"]) for r in ok]
    summary={"task":"0.8","expected_videos":a.expect,"found_videos":len(vids),"successful":len(ok),
             "confidence":{"mean":round(statistics.mean(conf),4) if conf else None,
                           "median":round(statistics.median(conf),4) if conf else None,
                           "min":min(conf) if conf else None,"max":max(conf) if conf else None},
             "abs_av_offset_frames":{"mean":round(statistics.mean(offs),4) if offs else None,
                                      "median":statistics.median(offs) if offs else None,
                                      "max":max(offs) if offs else None},
             "rows":rows,"closed":len(ok)==a.expect==len(vids),
             "contract_threshold":"UNDECIDED — spec does not define aggregation rule",
             "at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(PILOT/"0.8_syncnet_real_content.json",summary)
    md=["# Pilot 0.8 — SyncNet on real content","",
        f"Successful: **{len(ok)}/{len(vids)}** (expected {a.expect})  ",
        f"Confidence mean/median/min/max: **{summary['confidence']['mean']} / {summary['confidence']['median']} / {summary['confidence']['min']} / {summary['confidence']['max']}**  ",
        f"Max |AV offset|: **{summary['abs_av_offset_frames']['max']} frames**  ",
        "**Contract threshold is intentionally not auto-selected:** the source spec does not define whether it should be min/mean/median/etc.","",
        "| video | confidence | AV offset frames | status |","|---|---:|---:|---|"]
    for r in rows: md.append(f"| {r['name']} | {r.get('confidence','-')} | {r.get('av_offset_frames','-')} | {r['status']} |")
    (PILOT/"0.8_syncnet_real_content.md").write_text("\n".join(md)+"\n")
    print("\n".join(md)); return 0 if summary["closed"] else 2

if __name__=="__main__": raise SystemExit(main())
