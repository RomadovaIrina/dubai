#!/usr/bin/env python3
"""r15 K: where does the cfhf chain break the good raw KeySync crop? A still sheet, one row per time (2 frames: t and t + 0.2 s, to see motion),
columns = source master | raw KeySync crop (the frame shown at t after the AV shift) | ks_layer | ks_fix | ks_tsmooth | cfhf final | plain paste,
every panel = the KeySync box (the crop region), 240 px, labelled.
    python k_stagesheet.py --id m03 --final-dir final/m03_cfhf_fast2clamp --raw R.mp4 --cfhf F.mp4 --plain P.mp4 --master M.mp4 --box box.json --shift 1 --times 3 7 12 --out sheet.jpg"""
import argparse, json
import numpy as np, cv2

ap = argparse.ArgumentParser()
for k in ("final_dir", "raw", "cfhf", "plain", "master", "box", "out"): ap.add_argument("--" + k.replace("_", "-"), required=True)
ap.add_argument("--shift", type=int, default=1); ap.add_argument("--times", type=float, nargs="+", required=True); ap.add_argument("--px", type=int, default=240)
a = ap.parse_args(); b = json.load(open(a.box)); x, y, s = b["x"], b["y"], b["side"]; P = a.px


def grab(path, t, crop=True):
    c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 25; c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = c.read(); c.release()
    if not ok: return np.zeros((P, P, 3), np.uint8)
    if crop: f = f[y:y + s, x:x + s]
    return cv2.resize(f, (P, P), interpolation=cv2.INTER_AREA)


cols = [("source", a.master, True, 0), ("raw KS crop", a.raw, False, -a.shift / 25.0), ("ks_layer", f"{a.final_dir}/layer.mp4", True, 0),
        ("ks_fix", f"{a.final_dir}/layer_fix.mp4", True, 0), ("tsmooth a0.8", f"{a.final_dir}/layer_ts.mp4", True, 0), ("cfhf final", a.cfhf, True, 0), ("plain paste", a.plain, True, 0)]
rows = []
for t in a.times:
    for dt in (0.0, 0.2):
        tiles = []
        for name, path, crop, off in cols:
            g = grab(path, max(0.0, t + dt + off), crop); cv2.putText(g, f"{name} {t + dt:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(g, f"{name} {t + dt:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA); tiles.append(g)
        rows.append(np.hstack(tiles))
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("sheet ->", a.out)
