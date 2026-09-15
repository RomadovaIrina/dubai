#!/usr/bin/env python3
"""Pilot 0.7 — SyncNet before/after CodeFormer fidelity sweep.

Input must be a lip-synced video (normally LatentSync output). The script measures the
official LatentSync SyncNet confidence/AV offset BEFORE restoration, runs CodeFormer at a
weight grid, re-measures each result, and chooses the most aggressive (lowest w) candidate
that satisfies the user-supplied non-degradation tolerance.

The pilot spec does not define a numerical tolerance, so the default is strict: no
confidence drop and no worsening of absolute AV offset. Change explicitly if the team
agrees a tolerance.
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, THIRD_PARTY, write_json, sh, gpu_info
from syncnet_common import eval_syncnet
from benchmark_06_codeformer import probe  # ffprobe metadata: CodeFormer may change geometry (0.6: 464x848 -> 512x936)


def fps_of(p:pathlib.Path)->float:
    x=sh(f'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "{p}"')
    n,d=x.split('/'); return float(n)/float(d)


def codeformer(src:pathlib.Path,w:float,work:pathlib.Path)->pathlib.Path:
    cf=THIRD_PARTY/"CodeFormer"; out=work/f"w_{w:g}"; out.mkdir(parents=True,exist_ok=True)
    cmd=["/venv/dabai/bin/python","inference_codeformer.py","-i",str(src),"-o",str(out),
         "-w",str(w),"-s","1","--detection_model","retinaface_resnet50",
         "--save_video_fps",f"{fps_of(src):g}"]
    p=subprocess.run(cmd,cwd=str(cf),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/"codeformer.log").write_text(p.stdout,encoding="utf-8")
    if p.returncode: raise RuntimeError(f"CodeFormer w={w} failed:\n"+'\n'.join(p.stdout.splitlines()[-30:]))
    vids=list(out.rglob("*.mp4"))
    if not vids: raise RuntimeError(f"CodeFormer w={w} produced no mp4 in {out}")
    return max(vids,key=lambda x:x.stat().st_mtime)


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("input",help="LatentSync output with audio")
    ap.add_argument("--weights",nargs="+",type=float,default=[0.3,0.5,0.7,0.9])
    ap.add_argument("--max-confidence-drop",type=float,default=0.0,
                    help="explicit allowed drop vs BEFORE; spec itself defines no tolerance")
    ap.add_argument("--max-offset-worsening-frames",type=int,default=0)
    ap.add_argument("--keep-work",action="store_true")
    ap.add_argument("--reuse",nargs="*",default=[],metavar="W:MP4",
                    help="existing CodeFormer output for a weight (same input/config), e.g. 0.5:/tmp/x/latentsync_04.mp4")
    ap.add_argument("--work-dir",default=None,help="where CodeFormer outputs go (default: mkdtemp under /tmp)")
    a=ap.parse_args(); src=pathlib.Path(a.input).resolve()
    if not src.exists(): raise SystemExit(f"missing {src}")
    reuse={float(x.split(":",1)[0]):pathlib.Path(x.split(":",1)[1]) for x in a.reuse}
    work=pathlib.Path(a.work_dir) if a.work_dir else pathlib.Path(tempfile.mkdtemp(prefix="dabai-0.7-"))
    work.mkdir(parents=True,exist_ok=True)
    src_meta=probe(src)
    try:
        print("== SyncNet BEFORE ==",flush=True); before=eval_syncnet(src)
        rows=[]
        for w in sorted(a.weights):
            print(f"\n== CodeFormer w={w} ==",flush=True)
            if w in reuse and reuse[w].exists():
                out=reuse[w]; cf_seconds=None; print(f"reusing {out}")
            else:
                t0=time.perf_counter(); out=codeformer(src,w,work); cf_seconds=round(time.perf_counter()-t0,3)
            meta=probe(out)
            after=eval_syncnet(out)
            delta=after["confidence"]-before["confidence"]
            offset_worse=abs(after["av_offset_frames"])-abs(before["av_offset_frames"])
            passes=(delta>=-a.max_confidence_drop and offset_worse<=a.max_offset_worsening_frames)
            rows.append({"w":w,"output":str(out) if (a.keep_work or w in reuse) else None,
                         "reused":w in reuse,"codeformer_seconds":cf_seconds,"output_meta":meta,
                         "geometry_changed":(meta["width"],meta["height"])!=(src_meta["width"],src_meta["height"]),
                         "confidence":after["confidence"],"av_offset_frames":after["av_offset_frames"],
                         "abs_offset_before":abs(before["av_offset_frames"]),"abs_offset_after":abs(after["av_offset_frames"]),
                         "confidence_delta":round(delta,4),"offset_abs_worsening_frames":offset_worse,
                         "syncnet_seconds":after["seconds"],"syncnet_stdout_tail":after["stdout_tail"],
                         "non_degrading":passes})
            print({k:v for k,v in rows[-1].items() if k!="syncnet_stdout_tail"})
        candidates=[r for r in rows if r["non_degrading"]]
        selected=min(candidates,key=lambda r:r["w"])["w"] if candidates else None
        rep={"task":"0.7","input":str(src),"input_meta":src_meta,"gpu":gpu_info(),"before":before,"weights":rows,
             "codeformer_config":{"upscale":1,"detection_model":"retinaface_resnet50","bg_upsampler":None,"face_upsample":False},
             "policy":{"max_confidence_drop":a.max_confidence_drop,
                       "max_offset_worsening_frames":a.max_offset_worsening_frames,
                       "selection":"lowest w that passes = strongest restoration under stated sync tolerance"},
             "selected_fidelity_weight":selected,"closed":selected is not None,
             "verdict":"PASS" if selected is not None else "NO_WEIGHT_PASSES_STRICT_TOLERANCE",
             "at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
        write_json(PILOT/"0.7_codeformer_syncnet.json",rep)
        md=["# Pilot 0.7 — CodeFormer vs SyncNet","",
            f"Before: confidence **{before['confidence']:.3f}**, AV offset **{before['av_offset_frames']} frames**  ",
            f"Allowed confidence drop: **{a.max_confidence_drop}**, offset worsening: **{a.max_offset_worsening_frames} frames**  ",
            f"Selected fidelity weight: **{selected if selected is not None else 'NONE'}**","",
            "| w | conf before | conf after | Δconf | offset before | offset after | Δ\\|offset\\| | strict pass | output geometry |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        md += [f"| {r['w']} | {before['confidence']:.2f} | {r['confidence']:.2f} | {r['confidence_delta']:+.2f} | {before['av_offset_frames']} | {r['av_offset_frames']} | {r['offset_abs_worsening_frames']:+d} | {r['non_degrading']} | {r['output_meta']['width']}x{r['output_meta']['height']} {r['output_meta']['frames']}f {r['output_meta']['fps']:g}fps {r['output_meta']['duration_s']:.2f}s |" for r in rows]
        (PILOT/"0.7_codeformer_syncnet.md").write_text("\n".join(md)+"\n")
        print("\n".join(md)); return 0 if selected is not None else 2
    finally:
        if not a.keep_work: shutil.rmtree(work,ignore_errors=True)

if __name__=="__main__": raise SystemExit(main())
