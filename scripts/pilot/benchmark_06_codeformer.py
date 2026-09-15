#!/usr/bin/env python3
"""Pilot 0.6 — CodeFormer timing on real video.

Measures CodeFormer frame throughput and normalizes to seconds per source-video minute.
Optionally combines the result with a LatentSync seconds/video-minute value to test the
300 s/video-min budget from the pilot spec.
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, statistics, subprocess, sys, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, THIRD_PARTY, append_jsonl, sh, write_json, gpu_info
from measure_vram import measure_command

JSONL=PILOT/"0.6_codeformer.jsonl"


def probe(p:pathlib.Path)->dict:
    raw=sh(f'ffprobe -v error -print_format json -show_format -show_streams "{p}"',timeout=60)
    d=json.loads(raw)
    v=next(x for x in d["streams"] if x.get("codec_type")=="video")
    rate=v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    n,den=rate.split('/'); fps=float(n)/float(den) if float(den) else 0
    dur=float(d.get("format",{}).get("duration") or v.get("duration") or 0)
    frames=int(v.get("nb_frames") or round(dur*fps))
    return {"duration_s":dur,"fps":fps,"frames":frames,"width":v.get("width"),"height":v.get("height"),"codec":v.get("codec_name")}


def run_one(src:pathlib.Path,w:float,idx:int,keep:bool)->dict:
    info=probe(src)
    work=pathlib.Path(tempfile.mkdtemp(prefix="dabai-codeformer-"))
    cf=THIRD_PARTY/"CodeFormer"
    cmd=["/venv/dabai/bin/python","inference_codeformer.py","-i",str(src),"-o",str(work),
         "-w",str(w),"-s","1","--detection_model","retinaface_resnet50",
         "--save_video_fps",f"{info['fps'] or 25:g}"]
    row=measure_command(f"pilot0.6/codeformer/{src.stem}/r{idx}",cmd,
                        note=f"w={w}",cwd=str(cf))
    sec=float(row["seconds"]); dur=info["duration_s"]; frames=info["frames"]
    row.update(task="0.6",video=src.name,source=info,fidelity_weight=w,
               frames_per_s=round(frames/sec,4) if sec else None,
               seconds_per_frame=round(sec/frames,6) if frames else None,
               seconds_per_1000_frames=round(sec/frames*1000,3) if frames else None,
               codeformer_s_per_video_min=round(sec/dur*60,3) if dur else None,
               estimate_1500_frames_s=round(sec/frames*1500,3) if frames else None,
               workdir=str(work) if keep else None)
    append_jsonl(JSONL,row)
    if not keep: shutil.rmtree(work,ignore_errors=True)
    return row


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="one or more real videos")
    ap.add_argument("--w",type=float,default=.5,help="benchmark baseline only; 0.7 selects final weight")
    ap.add_argument("--repeat",type=int,default=1)
    ap.add_argument("--latentsync-s-per-min",type=float,default=None)
    ap.add_argument("--budget-s-per-min",type=float,default=300.0)
    ap.add_argument("--keep-work",action="store_true")
    ap.add_argument("--clear",action="store_true")
    a=ap.parse_args()
    if a.clear and JSONL.exists(): JSONL.unlink()
    rows=[]
    for s in a.videos:
        src=pathlib.Path(s).resolve()
        if not src.exists(): print(f"missing {src}",file=sys.stderr); continue
        for i in range(1,a.repeat+1):
            print(f"== CodeFormer {src.name} run {i}/{a.repeat} ==",flush=True)
            rows.append(run_one(src,a.w,i,a.keep_work))
    ok=[r for r in rows if r.get("ok")]
    vals=[r["codeformer_s_per_video_min"] for r in ok if r.get("codeformer_s_per_video_min") is not None]
    med=statistics.median(vals) if vals else None
    combined=(med+a.latentsync_s_per_min) if med is not None and a.latentsync_s_per_min is not None else None
    g=gpu_info(); target_gpu=g.get("capability")=="sm_120"
    summary={"task":"0.6","successful_runs":len(ok),"runs":len(rows),"gpu":g,
             "median_codeformer_s_per_video_min":round(med,3) if med is not None else None,
             "latentsync_s_per_video_min":a.latentsync_s_per_min,
             "combined_video_contour_s_per_video_min":round(combined,3) if combined is not None else None,
             "budget_s_per_video_min":a.budget_s_per_min,
             "within_budget":combined<=a.budget_s_per_min if combined is not None else None,
             "closed":bool(ok) and combined is not None and target_gpu,"provisional":bool(ok) and combined is not None and not target_gpu,
             "note":"0.6 formal closure needs LatentSync+CodeFormer combined timing on target Blackwell hardware."}
    write_json(PILOT/"0.6_codeformer.json",summary)
    md=["# Pilot 0.6 — CodeFormer timing","",
        f"Median CodeFormer: **{summary['median_codeformer_s_per_video_min']} s/video-min**  ",
        f"LatentSync supplied: **{summary['latentsync_s_per_video_min']} s/video-min**  ",
        f"Combined: **{summary['combined_video_contour_s_per_video_min']} s/video-min**  ",
        f"Budget: **{a.budget_s_per_min:.1f} s/video-min**  ",
        f"Verdict: **{('PASS' if summary['within_budget'] else 'FAIL') if summary['within_budget'] is not None else 'PENDING LATENTSYNC NUMBER'}**","",
        "| video | dur | frames | fps | time | fps(proc) | CF s/video-min | peak device MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s=r["source"]; md.append(f"| {r['video']} | {s['duration_s']:.2f} | {s['frames']} | {s['fps']:.2f} | {r['seconds']:.2f} | {r['frames_per_s']} | {r['codeformer_s_per_video_min']} | {r['peak_device_mib']} |")
    (PILOT/"0.6_codeformer.md").write_text("\n".join(md)+"\n")
    print("\n".join(md)); return 0 if ok else 1

if __name__=="__main__": raise SystemExit(main())
