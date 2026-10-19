#!/usr/bin/env python3
"""Full-length LatentSync-raw video for fps_composite.py from decompose window dirs: master frames everywhere, each window's S0_ls_raw.mp4
spliced in at its master indices (round 2, track B: source-fps tests on the shaky windows only; outside the windows the "LatentSync layer"
is the master itself, so only frames inside the windows are measured).
    python scripts/quality/r2_splice_ls.py --master WORK/master_25fps.mp4 --win W1 [W2 ...] --out S0_full.mp4"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import composite_register as cr
ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--win", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); M = cr.read_video(a.master)
for w in a.win:
    w = pathlib.Path(w); A, B = json.loads((w / "window_manifest.json").read_text())["window"]; L = cr.read_video(w / "S0_ls_raw.mp4")
    for i, f in enumerate(L): M[A + i] = f
    print(w.name, A, A + len(L))
cr.write_video(a.out, M)
