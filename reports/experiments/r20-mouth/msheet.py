#!/usr/bin/env python3
"""mouth sheet: rows = videos (512 crops, frame-aligned), cols = frames. usage: msheet.py out.jpg f0,f1,... name=path[:box.json:master] ...
a path with :box.json = a master video cropped by the box to 512. --zoom lm.npz crops around the mouth (from that npz, 478-pt) x3."""
import sys, json, cv2, numpy as np
out = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(",")]; rows = []; zoom = None
args = sys.argv[3:]
if args and args[0].startswith("--zoom="): zoom = np.load(args[0][7:])["lm"][..., :2]; args = args[1:]
for a in args:
    name, p = a.split("=", 1); box = None
    if ":" in p: p, bj = p.split(":", 1); box = json.load(open(bj))
    c = cv2.VideoCapture(p); tiles = []
    for k in frames:
        c.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = c.read()
        if not ok: f = np.zeros((512, 512, 3), np.uint8)
        if box is not None:
            x = box["xs"][min(k, len(box["xs"]) - 1)] if "xs" in box else box["x"]; y = box["ys"][min(k, len(box["ys"]) - 1)] if "ys" in box else box["y"]; s = box["side"]
            x, y, s = int(round(x)), int(round(y)), int(round(s)); pad = cv2.copyMakeBorder(f, s, s, s, s, cv2.BORDER_REPLICATE)
            f = cv2.resize(pad[y + s:y + 2 * s, x + s:x + 2 * s], (512, 512), interpolation=cv2.INTER_AREA if s > 512 else cv2.INTER_CUBIC)
        if f.shape[0] != 512: f = cv2.resize(f, (512, 512))
        if zoom is not None and k < len(zoom) and np.isfinite(zoom[k]).all():
            m = zoom[k][[61, 291, 0, 17, 13, 14]]; cx, cy = m.mean(0); h = max(48, 0.75 * np.linalg.norm(zoom[k][61] - zoom[k][291]))
            x0, y0 = int(max(0, cx - h)), int(max(0, cy - h)); f = cv2.resize(f[y0:y0 + int(2 * h), x0:x0 + int(2 * h)], (320, 320), interpolation=cv2.INTER_CUBIC)
        else: f = cv2.resize(f, (320, 320))
        cv2.putText(f, "%s %d %.2fs" % (name, k, k / 25), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA); tiles.append(f)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", out)
