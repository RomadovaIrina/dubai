#!/usr/bin/env python3
"""r11 FACE (KeySync composite): the gate's SyncNet number (r4_gate add_sync: mean conf at offset 0 over the dub-speech seconds of a 25 fps
rendition) for ANY video, so each composite stage can be scored alone.
    sync_speech.py --video V.mp4 --audio dubbed_16k.wav [--label x] [--native]   (--native: the video is already 25 fps, no fps=25 re-encode)"""
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np

Q = pathlib.Path(__file__).resolve().parents[1] / "quality"
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--label", default="")
ap.add_argument("--native", action="store_true"); a = ap.parse_args()
with tempfile.TemporaryDirectory() as td:
    v25 = a.video if a.native else str(pathlib.Path(td) / "v25.mp4"); sp = pathlib.Path(td) / "sp.json"
    if not a.native:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video, "-vf", "fps=25", "-c:v", "libx264", "-crf", "12", "-an", v25], check=True)
    subprocess.run([sys.executable, str(Q / "sync_perframe.py"), "--video", v25, "--audio", a.audio, "--out", str(sp)], capture_output=True, text=True)
    j = json.loads(sp.read_text())
ps = j["per_second"]; sp_ = [x["conf_at0"] for x in ps if x.get("conf_at0") is not None and (x.get("dub_act") or 0) >= 0.5]
print(f"== sync_speech {a.label or pathlib.Path(a.video).name}: {np.mean(sp_):.3f} over {len(sp_)} s; global offset {j.get('global', {}).get('av_offset')}", flush=True)
