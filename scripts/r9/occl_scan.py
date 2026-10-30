#!/usr/bin/env python3
"""r9: "the mouth slid onto the son's ear" = something in front of / next to the speaker's lower face (another head, an ear, a hand, a phone)
lies inside the area the generator repaints and the composite pastes back, so it gets overwritten with her face.
Per source frame: in a box of +-R IOD around her mouth, the pixels OUTSIDE her outer-lip hull (dilated 0.2 IOD) and inside the lower-face region:
  ring_diff  = mean |output - source| luma there (normal frames: small, the composite keeps the source skin; an overwritten occluder: large)
  ring_p95   = 95th percentile of the same
Top frames -> a contact sheet (source row / output row), plus a per-second summary.
    python occl_scan.py --src SRC --out OUT.mp4 --lm SOURCE_LM.npz [--top 24] --sheet sheet.jpg --json scan.json"""
import argparse, json, pathlib, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r8"))
from r8_mouth_flicker import LIPS_OUTER   # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lm", required=True)
ap.add_argument("--r", type=float, default=1.1); ap.add_argument("--top", type=int, default=24); ap.add_argument("--sheet", required=True); ap.add_argument("--json", required=True)
a = ap.parse_args()
lm = np.load(a.lm)["lm"]; cs, co = cv2.VideoCapture(a.src), cv2.VideoCapture(a.out); fps = cs.get(cv2.CAP_PROP_FPS)
rows, keep = [], {}
k = 0
while True:
    ok1, fs = cs.read(); ok2, fo = co.read()
    if not (ok1 and ok2) or k >= len(lm): break
    p = lm[k]
    if np.isfinite(p[LIPS_OUTER + [33, 263, 1, 152]]).all():
        iod = np.linalg.norm(p[33, :2] - p[263, :2]); c = p[LIPS_OUTER, :2].mean(0); R = int(a.r * iod)
        y0, y1, x0, x1 = max(0, int(c[1] - R)), min(fs.shape[0], int(c[1] + R)), max(0, int(c[0] - R)), min(fs.shape[1], int(c[0] + R))
        m = np.ones((y1 - y0, x1 - x0), np.uint8)
        lip = np.zeros_like(m); cv2.fillPoly(lip, [np.round(p[LIPS_OUTER, :2] - [x0, y0]).astype(np.int32)], 1)
        lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.2 * iod) + 1,) * 2)); m[lip > 0] = 0
        m[: max(0, int(p[1, 1] - y0)), :] = 0   # below the nose tip only (the lower face)
        Ys = cv2.cvtColor(fs[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32); Yo = cv2.cvtColor(fo[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        dd = np.abs(Yo - Ys)[m > 0]
        if dd.size > 50:
            rows.append({"frame": k, "t": round(k / fps, 2), "ring_diff": round(float(dd.mean()), 2), "ring_p95": round(float(np.percentile(dd, 95)), 1)})
            keep[k] = (cv2.resize(fs[y0:y1, x0:x1], (180, 180)), cv2.resize(fo[y0:y1, x0:x1], (180, 180)))
            if len(keep) > 400:   # memory: keep only the current worst 200
                worst = set(r["frame"] for r in sorted(rows, key=lambda r: -r["ring_p95"])[:200]); keep = {f: v for f, v in keep.items() if f in worst}
    k += 1
rd = np.array([r["ring_p95"] for r in rows]); med = float(np.median(rd))
top = sorted(rows, key=lambda r: -r["ring_p95"]); sel = []
for r in top:   # at most one frame per 0.5 s in the sheet
    if all(abs(r["t"] - s["t"]) >= 0.5 for s in sel) and r["frame"] in keep: sel.append(r)
    if len(sel) >= a.top: break
sel.sort(key=lambda r: r["t"]); tiles = []
for r in sel:
    s, o = keep[r["frame"]]; t = np.vstack([s, o]); cv2.putText(t, f"{r['t']:.2f}s p95 {r['ring_p95']:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1); tiles.append(t)
rowsz = 8; lines = [np.hstack(tiles[i:i + rowsz] + [np.zeros_like(tiles[0])] * (rowsz - len(tiles[i:i + rowsz]))) for i in range(0, len(tiles), rowsz)]
cv2.imwrite(a.sheet, np.vstack(lines))
sec = {}
for r in rows: sec.setdefault(int(r["t"]), []).append(r["ring_p95"])
res = {"frames": len(rows), "ring_p95_median": med, "frames_over_2x_median": int((rd > 2 * med).sum()), "frames_over_3x_median": int((rd > 3 * med).sum()),
       "per_second_max": {s: round(max(v), 1) for s, v in sorted(sec.items())}, "top": sel}
pathlib.Path(a.json).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "top"}))
