#!/usr/bin/env python3
"""r15 Z: F's pop metric (explode.pop_at) of finals vs the source around given times (the m03 snap list): max excess within +-win s.
    snap_at.py --src SRC --src-lm source_lm_F.npz --final lab=V ... --times 11.6 11.75 ... [--win 0.2]"""
import argparse, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from explode import pop_at
ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--src-lm"); ap.add_argument("--final", action="append"); ap.add_argument("--times", nargs="+", type=float)
ap.add_argument("--win", type=float, default=0.2); a = ap.parse_args()
slm = np.load(a.src_lm)["lm"].astype(np.float64); fps = cv2.VideoCapture(a.src).get(cv2.CAP_PROP_FPS)
need = sorted({k for t in a.times for k in range(int((t - a.win) * fps) - 1, int((t + a.win) * fps) + 2)})


def grab(p, shape=None):
    c = cv2.VideoCapture(p); out = {}; i = 0; last = max(need) + 1
    while i <= last:
        ok, f = c.read()
        if not ok: break
        if i in need or i - 1 in need or i + 1 in need:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); out[i] = cv2.resize(g, (shape[1], shape[0])) if shape and g.shape != shape else g
        i += 1
    return out


S = grab(a.src); shp = next(iter(S.values())).shape
class L(dict):
    def __len__(self): return 10 ** 9
    def __getitem__(self, k): return dict.__getitem__(self, k)
fins = [(s.split("=", 1)[0], L(grab(s.split("=", 1)[1], shp))) for s in a.final]
print("t      " + "  ".join(f"{lab:>12s}" for lab, _ in fins))
for t in a.times:
    ks = [k for k in range(int((t - a.win) * fps), int((t + a.win) * fps) + 1)]
    ps = {k: pop_at(L(S), slm, k) for k in ks}
    row = [max((pop_at(F, slm, k) - ps[k]) for k in ks if np.isfinite(ps[k])) for _, F in fins]
    print(f"{t:6.2f} " + "  ".join(f"{v:12.2f}" for v in row))
