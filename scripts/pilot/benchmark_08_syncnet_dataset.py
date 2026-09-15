#!/usr/bin/env python3
"""Pilot 0.8 — SyncNet on the five real customer videos/outputs.

The spec says the real-content value becomes an acceptance threshold but does not define
how five values must be reduced to one number. This script therefore reports all raw
values plus mean/median/min/max and does NOT silently choose the contractual threshold.
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json, gpu_info
from syncnet_common import eval_syncnet, DEFAULT_MODEL
from benchmark_06_codeformer import probe

EXT={'.mp4','.mov','.mkv','.avi','.webm'}

def _key(p:pathlib.Path)->str:
    """'latentsync_04.mp4' and '04.mp4' -> '04' so outputs can be paired with originals."""
    s=p.stem.lower()
    return s[len("latentsync_"):] if s.startswith("latentsync_") else s

def _eval_rows(vids:list[pathlib.Path],tag:str)->list[dict]:
    rows=[]
    for i,v in enumerate(vids,1):
        print(f"[{tag} {i}/{len(vids)}] {v.name}",flush=True)
        try:
            r=eval_syncnet(v); r["name"]=v.name; r["key"]=_key(v); r["status"]="PASS"; r["meta"]=probe(v); rows.append(r)
            print({k:v for k,v in r.items() if k!="stdout_tail"})
        except Exception as e:
            rows.append({"name":v.name,"key":_key(v),"status":"FAIL","error":f"{type(e).__name__}: {e}"}); print(rows[-1])
    return rows

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("videos",help="directory containing the 5 real generated videos, or one video")
    ap.add_argument("--expect",type=int,default=5)
    ap.add_argument("--originals",default=None,help="directory with the untouched source videos -> reference baseline + deltas")
    ap.add_argument("--audio-note",default="original source audio, not translated/TTS")
    ap.add_argument("--prep-json",default=None,help="reports/pilot/0.8_latentsync_prep.json from prepare_08_latentsync.py (config provenance)")
    a=ap.parse_args(); p=pathlib.Path(a.videos).resolve()
    vids=[p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix.lower() in EXT)
    if not vids: raise SystemExit("no videos found")
    rows=_eval_rows(vids,"latentsync")
    orig_rows=[]
    if a.originals:
        op=pathlib.Path(a.originals).resolve()
        ovids=sorted(x for x in op.iterdir() if x.suffix.lower() in EXT)
        orig_rows=_eval_rows(ovids,"original")
        by_key={r["key"]:r for r in orig_rows if r["status"]=="PASS"}
        for r in rows:
            o=by_key.get(r["key"])
            if o and r["status"]=="PASS":
                r["original_confidence"]=o["confidence"]; r["original_av_offset_frames"]=o["av_offset_frames"]
                r["confidence_delta"]=round(r["confidence"]-o["confidence"],4)
                r["abs_offset_delta_frames"]=abs(r["av_offset_frames"])-abs(o["av_offset_frames"])
    ok=[r for r in rows if r["status"]=="PASS"]
    conf=[r["confidence"] for r in ok]; offs=[abs(r["av_offset_frames"]) for r in ok]
    summary={"task":"0.8","expected_videos":a.expect,"found_videos":len(vids),"successful":len(ok),
             "confidence":{"mean":round(statistics.mean(conf),4) if conf else None,
                           "median":round(statistics.median(conf),4) if conf else None,
                           "min":min(conf) if conf else None,"max":max(conf) if conf else None},
             "abs_av_offset_frames":{"mean":round(statistics.mean(offs),4) if offs else None,
                                      "median":statistics.median(offs) if offs else None,
                                      "max":max(offs) if offs else None},
             "worst_by_confidence":min(ok,key=lambda r:r["confidence"])["name"] if ok else None,
             "worst_by_abs_offset":max(ok,key=lambda r:abs(r["av_offset_frames"]))["name"] if ok else None,
             "rows":rows,"originals":orig_rows,"closed":len(ok)==a.expect==len(vids),
             "gpu":gpu_info(),"syncnet":{"evaluator":"third_party/latentsync/eval/eval_sync_conf.py (upstream)","model":str(DEFAULT_MODEL)},
             "prep":json.load(open(a.prep_json)) if a.prep_json and pathlib.Path(a.prep_json).exists() else None,
             "semantics":{"dataset":"real customer/content videos","lipsync_processing":"real LatentSync","audio":a.audio_note,
                          "caveat":"result suitable for pilot technical characterization; final contractual acceptance threshold should be confirmed on production-like dubbed outputs when TTS pipeline exists"},
             "contract_threshold":"UNDECIDED — spec does not define aggregation rule",
             "at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(PILOT/"0.8_syncnet_real_content.json",summary)
    md=["# Pilot 0.8 — SyncNet on real content","",
        f"Successful: **{len(ok)}/{len(vids)}** (expected {a.expect})  ",
        f"Confidence mean/median/min/max: **{summary['confidence']['mean']} / {summary['confidence']['median']} / {summary['confidence']['min']} / {summary['confidence']['max']}**  ",
        f"Max |AV offset|: **{summary['abs_av_offset_frames']['max']} frames**  ",
        "**Contract threshold is intentionally not auto-selected:** the source spec does not define whether it should be min/mean/median/etc.","",
        "| video | original conf | original offset | LatentSync conf | LatentSync offset | Δconf | Δ\\|offset\\| | status |","|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows: md.append(f"| {r['name']} | {r.get('original_confidence','-')} | {r.get('original_av_offset_frames','-')} | {r.get('confidence','-')} | {r.get('av_offset_frames','-')} | {r.get('confidence_delta','-')} | {r.get('abs_offset_delta_frames','-')} | {r['status']} |")
    (PILOT/"0.8_syncnet_real_content.md").write_text("\n".join(md)+"\n")
    print("\n".join(md)); return 0 if summary["closed"] else 2

if __name__=="__main__": raise SystemExit(main())
