#!/usr/bin/env python3
"""Exact frame-range cut (sequential PyAV decode, no seeking) -> lossless clip. OpenCV's CAP_PROP_POS_FRAMES seek is not frame-exact on
these H.264 files (VFR 05, different GOPs of source and outputs): the common metric's frame pairing drifted by several frames in some
windows (upper-face 'error' 0.6-6 px on untouched source pixels). Measure the cut clips from frame 0 instead.
    python scripts/quality/r2_cut_frames.py IN.mp4 A:B OUT.mp4 [--npz LM.npz --npz-out LMwin.npz]"""
import argparse, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("--npz"); ap.add_argument("--npz-out")
a = ap.parse_args(); A, B = (int(x) for x in a.frames.split(":"))
import av, composite_register as cr
c = av.open(a.inp); fr = []
for i, f in enumerate(c.decode(c.streams.video[0])):
    if i >= B: break
    if i >= A: fr.append(f.to_ndarray(format="bgr24"))
c.close(); cr.write_video(a.out, fr)
if a.npz: z = np.load(a.npz); np.savez_compressed(a.npz_out, lm=z["lm"][A:B])
print(a.out, len(fr))
