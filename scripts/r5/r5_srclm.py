#!/usr/bin/env python3
"""MediaPipe landmarks on every SOURCE frame + its timestamp (no master timing needed) -> npz {lm, t}: the r5_twitch.py --lm input.
    python scripts/r5/r5_srclm.py --src SRC --out <id>_src_lm.npz"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import clean_face_stack as cfs
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
out = pathlib.Path(a.out); raw = out.with_name(out.stem + "_mp.npz"); lm = np.load(cfs.run_mediapipe(pathlib.Path(a.src), raw))["lm"]
import av
c = av.open(a.src); s = c.streams.video[0]; tb = float(s.time_base); t = np.sort(np.array([p.pts for p in c.demux(s) if p.pts is not None], np.int64))[:len(lm)] * tb; c.close()
np.savez_compressed(out, lm=lm, t=t - t[0]); print(a.src, len(lm), "frames, face in", int(np.isfinite(lm[:, 0, 0]).sum()))
