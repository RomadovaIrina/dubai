#!/usr/bin/env python3
"""MediaPipe landmarks on every SOURCE frame (not interpolated from the 25 fps master) + each source frame's master position at its true
time (mu = tau^-1(t_k), r3_master_timing.py). -> <out>.npz {lm (n_src, P, 3), mu (n_src,), t (n_src,)}: the format r2_slide_fps.sh expects.
Note: the master is `-r 25` = a subset of the source frames (master m = source j[m]), so master landmarks could also be taken from this
single pass (lm[j]) instead of a second MediaPipe run on the master.
    python scripts/quality/r3_source_lm.py --src SRC --timing timing/<id>_master_timing.json --out timing/<id>_source_lm.npz"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import clean_face_stack as cfs
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--timing", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
raw = out.with_name(out.stem + "_mp.npz"); z = np.load(cfs.run_mediapipe(pathlib.Path(a.src), raw)); lm = z["lm"]
import av
c = av.open(a.src); s = c.streams.video[0]; tb = float(s.time_base); t = np.array([p.pts for p in c.demux(s) if p.pts is not None], np.int64); c.close()
t = np.sort(t)[:len(lm)] * tb
tm = json.loads(pathlib.Path(a.timing).read_text()); tau = np.asarray(tm["tau_s"], np.float64)
mu = np.interp(t, tau, np.arange(len(tau), dtype=np.float64))
np.savez_compressed(out, lm=lm, mu=mu, t=t, j=np.asarray(tm["j"], np.int64))
print(a.src, "source frames", len(lm), "with face", int(np.isfinite(lm[:, 0, 0]).sum()), "mu range", round(float(mu[0]), 2), round(float(mu[-1]), 2))
