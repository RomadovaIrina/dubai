#!/usr/bin/env python3
"""r13 LS lip-corner defect (management: "небольшой дефект с кончиками губ"): close-ups of both lip corners on the master frames where the generated
mouth's width differs most from the source's, three rows per frame: SOURCE master | generated LAYER (the composite's input) | FINAL (k5).
A defect present in LAYER = generator; only in FINAL = composite (texture transfer / mouth zone edges).
    python scripts/r12/corner_sheet.py <id> [--n 8] [--out sheet.jpg]"""
import argparse, json
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default=""); a = ap.parse_args()
B = "/workspace/dub/artifacts/r4m"; W = f"{B}/r9g_{a.id}/work_{a.id}_r9g_{a.id}"; P = f"{W}/{a.id}_r9g_{a.id}_25fps"
lay = P + "_ls_cfd_ia04_srt1_rep2s.mp4"; lm_m = np.load(W + "/video/master_mediapipe.npz")["lm"]; lm_l = np.load(lay[:-4] + ".mediapipe.npz")["lm"]
fin = f"{B}/r9f_{a.id}/{a.id}_r9f_{a.id}.mp4"; tj = json.load(open(f"{B}/r9g_{a.id}/timing/{a.id}_master_timing_F.json")); jj = np.asarray(tj["j"])   # master m = source frame j[m]
n = min(len(lm_m), len(lm_l)); iod = np.linalg.norm(lm_m[:n, 33, :2] - lm_m[:n, 263, :2], axis=1)
ws = np.linalg.norm(lm_m[:n, 61, :2] - lm_m[:n, 291, :2], axis=1) / iod; wl = np.linalg.norm(lm_l[:n, 61, :2] - lm_l[:n, 291, :2], axis=1) / iod
d = np.nan_to_num(np.abs(wl - ws)); pick = []
for k in np.argsort(-d):
    if all(abs(k - q) > 25 for q in pick): pick.append(int(k))
    if len(pick) >= a.n: break
pick.sort()
def frames(path, ks):
    c = cv2.VideoCapture(path); o = {}; k = 0; mx = max(ks)
    while k <= mx:
        ok, f = c.read()
        if not ok: break
        if k in ks: o[k] = f
        k += 1
    return o
S, L = frames(W + "/video/master_25fps.mp4", set(pick)), frames(lay, set(pick)); F = frames(fin, {int(jj[k]) for k in pick})
tiles = []
for k in pick:
    p = lm_m[k]; r = int(0.32 * iod[k]); col = []
    for img, lab in ((S[k], "src"), (L[k], "layer"), (F[int(jj[k])], "final")):
        pair = []
        for c_ in (61, 291):
            x, y = p[c_, :2].astype(int); pair.append(cv2.resize(img[max(0, y - r):y + r, max(0, x - r):x + r], (120, 120), interpolation=cv2.INTER_CUBIC))
        t = np.concatenate(pair, 1); cv2.putText(t, f"{lab} m{k} dw{d[k]:.2f}", (2, 12), 0, 0.38, (0, 255, 255), 1); col.append(t)
    tiles.append(np.concatenate(col, 0))
out = a.out or f"/workspace/dub/r12/corners_{a.id}.jpg"; cv2.imwrite(out, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("==", out, "frames", pick)
