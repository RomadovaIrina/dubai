#!/usr/bin/env python3
"""r23 (Irina 17.11 on new_1755: "полная жесть, где-то губы тупо не открылись"): the "doesn't fit" gate. KeySync draws the mouth inside ONE
square crop per face track (constant side: k2_prep_track / ks_prep). When the face is much larger than that crop (camera brought close: the chin
and the mouth at its edge), cut by the frame edge, or much smaller than it, the generator draws garbage (new_1755 14.4-15.8 s: a red smear,
flattened lips low on the chin). On such frames the paste fades to the SOURCE face, like the cup gate (ks_occl): per master frame
  face height fh = landmark y-extent; side = the crop side of that frame (box_<f>.json "side", "xs" / "ys" when tracked);
  too big:   fh > --big x side   or the face width > --big x side
  cut:       a landmark of the lower face (chin, jaw, mouth) outside the frame or outside the crop box by > --edge x side
  too small: fh < --small x side
runs shorter than --min-run frames are dropped, then a smoothstep ramp of --ramp s on both sides. --merge: take the max with another gate
json (ks_occl.py --dump: the cup / hand gate), so ks_plain --gate-w gets one file. Writes {"w": [...], "fit": {...}}; prints KS_FITGATE.
    ks_fitgate.py --lm lm_f0.npz --box data/box_f0.json --out gate.json [--merge occl.json --big 0.95 --small 0.3 --ramp 0.25 --fps 25]"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--box", required=True); ap.add_argument("--out", required=True); ap.add_argument("--merge", default="")
ap.add_argument("--big", type=float, default=0.95); ap.add_argument("--small", type=float, default=0.30); ap.add_argument("--edge", type=float, default=0.02)
ap.add_argument("--ramp", type=float, default=0.25); ap.add_argument("--min-run", type=int, default=3); ap.add_argument("--fps", type=float, default=25.0)
a = ap.parse_args()
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM)); n = len(J)
B = json.load(open(a.box)); side = float(B["side"]); W, H = float(B.get("W", 1e9)), float(B.get("H", 1e9))
xs = B.get("xs"); ys = B.get("ys")
LOW = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397, 288, 361, 61, 291, 17, 0, 13, 14]
bad = np.zeros(n, bool); why = {"big": 0, "cut": 0, "small": 0, "no_face": 0}
for k in range(n):
    p = LM[min(int(J[k]), len(LM) - 1)]
    if not np.isfinite(p).all(): why["no_face"] += 1; continue   # (no face: the paste already skips it)
    fh = float(p[:, 1].max() - p[:, 1].min()); fw = float(p[:, 0].max() - p[:, 0].min())
    bx = float(xs[min(k, len(xs) - 1)]) if xs else float(B["x"]); by = float(ys[min(k, len(ys) - 1)]) if ys else float(B["y"])
    q = p[LOW]; e = a.edge * side
    cut = (q[:, 0].min() < -e or q[:, 1].min() < -e or q[:, 0].max() > W + e or q[:, 1].max() > H + e or
           q[:, 0].min() < bx - e or q[:, 1].min() < by - e or q[:, 0].max() > bx + side + e or q[:, 1].max() > by + side + e)
    if fh > a.big * side or fw > a.big * side: bad[k] = True; why["big"] += 1
    elif cut: bad[k] = True; why["cut"] += 1
    elif fh < a.small * side: bad[k] = True; why["small"] += 1
d = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]])); runs = [(int(s), int(e)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if e - s >= a.min_run]
w = np.zeros(n, np.float32); R = max(1, int(round(a.ramp * a.fps)))
for s, e in runs:
    w[s:e] = 1.0
    for i in range(1, R + 1):
        u = 1.0 - i / (R + 1.0); v = u * u * (3 - 2 * u)
        if s - i >= 0: w[s - i] = max(w[s - i], v)
        if e - 1 + i < n: w[e - 1 + i] = max(w[e - 1 + i], v)
info = {"frames": n, "side": side, "gated_frames": int((w > 0.999).sum()), "ramp_frames": int(((w > 0) & (w <= 0.999)).sum()), "why": why,
        "runs_s": [[round(s / a.fps, 2), round(e / a.fps, 2)] for s, e in runs], "big": a.big, "small": a.small}
out = {"w": [round(float(v), 3) for v in w], "fit": info}
if a.merge:
    try:
        M = json.load(open(a.merge)); m = np.asarray(M["w"], np.float32); k_ = min(len(m), n); w2 = w.copy(); w2[:k_] = np.maximum(w2[:k_], m[:k_])
        out["w"] = [round(float(v), 3) for v in w2]; info["merged"] = a.merge; info["merged_gated_frames"] = int((w2 > 0.999).sum())
        if M.get("wm"): out["wm"] = M["wm"]
    except Exception as ex: info["merge_failed"] = str(ex)[:120]
json.dump(out, open(a.out, "w")); print("KS_FITGATE " + json.dumps(info), flush=True)
