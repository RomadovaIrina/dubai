#!/usr/bin/env python3
"""r15 W: WHERE the mouth pops are (scripts/r14/flicker_metric.py's jitter above the source's p99), per video, and how many fall within
+-2 output frames of a generator segment / window seam (the gate segment starts of a candidate_video_stage manifest, master 25 fps).
    python scripts/r15/pop_times.py --lm source_lm_F.npz --video src=in.mp4 q2=a.mp4 tb=b.mp4 [--manifest gen_25fps.manifest.json]"""
import argparse, json, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r14"))
import flicker_metric as fm

ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--video", nargs="+", required=True); ap.add_argument("--manifest")
a = ap.parse_args()
vids = dict(x.split("=", 1) for x in a.video); W, H, fps = fm.probe(next(iter(vids.values()))); x0, y0, s = fm.box(a.lm, W, H)
J = {k: fm.jitter(fm.frames(v, x0, y0, s)) for k, v in vids.items()}; thr = float(np.percentile(J["src"][1:-1], 99))
seams = []
if a.manifest:
    m = json.load(open(a.manifest)); segs = (m.get("lipsync") or {}).get("segments") or m.get("segments") or []
    seams = sorted({sg["start_frame"] / 25.0 for sg in segs if sg.get("action") == "LATENT_SYNC"} | {sg["end_frame"] / 25.0 for sg in segs if sg.get("action") == "LATENT_SYNC"})
for k, j in J.items():
    t = np.flatnonzero(j > thr) / fps
    near = sum(1 for x in t if any(abs(x - y) <= 2 / fps for y in seams)) if seams else None
    print(f"{k:8s} pops {len(t):3d}" + (f", {near} within +-2 frames of a segment seam ({len(seams)} seams)" if seams else "") + ": " + " ".join(f"{x:.2f}" for x in t))
