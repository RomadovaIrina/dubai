#!/usr/bin/env python3
"""r14 F (mouth flicker on LS q2): per-stage temporal irregularity at user timestamps.
Stages on the 25 fps master grid: source master, raw LatentSync layer (ls_raw), CodeFormer detail (cfd), layer SR (srt1), layer repair (rep2s);
plus the FINAL at the source fps (mapped to master time). Per frame k (master grid):
  mouth[k] = mean |f_k - (w_{k-1} + w_{k+1}) / 2| on the lips hull + 0.3 IOD (k+-1 flow-warped onto k), the mouth_pop residual;
  low[k]   = the same on the lower face (jaw / cheeks, lips + 0.3 IOD excluded): a blinking lower face;
  tone[k]  = |mean grey change k-1 -> k| of the lower face minus the source's (a brightness blink).
Events per master frame: 16-frame window index (k - segment start) mod 16 (0 = a window start), bridge seam (+-4), repair-replaced frame,
fallback weight > 0.5, sanitize replacement (from the composite log, when given).
Output: a per-window table (each user window +-0.5 s): per stage the max of mouth / low / tone and the frame where it peaks, the events there.
    python scripts/r12/stage_diag.py <id> <tag of the gen run, e.g. r9g_old03q2> <final mp4> "3-4,5-6,20.8-21,24,27-28" [--out table.md]"""
import argparse, json, pathlib, re
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("gt"); ap.add_argument("final"); ap.add_argument("wins"); ap.add_argument("--out", default="")
ap.add_argument("--pad", type=float, default=0.5); a = ap.parse_args()
B = "/workspace/dub/artifacts/r4m"; W = f"{B}/{a.gt}/work_{a.id}_{a.gt}"; P = f"{W}/{a.id}_{a.gt}_25fps"
stages = {"master": W + "/video/master_25fps.mp4", "ls_raw": P + "_ls_raw.mp4", "cfd": P + "_ls_cfd_ia04.mp4", "srt1": P + "_ls_cfd_ia04_srt1.mp4", "rep2s": P + "_ls_cfd_ia04_srt1_rep2s.mp4"}
stages = {k: v for k, v in stages.items() if pathlib.Path(v).exists()}
lm = np.load(W + "/video/master_mediapipe.npz")["lm"]; n = len(lm)
man = json.loads(pathlib.Path(P + ".manifest.json").read_text())
wins = []
for w in a.wins.split(","):
    t0, t1 = (map(float, w.split("-")) if "-" in w else (float(w), float(w)))
    wins.append((w, max(0.0, t0 - a.pad), t1 + a.pad))
need = sorted({k for _, t0, t1 in wins for k in range(int(t0 * 25) - 1, int(t1 * 25) + 2) if 0 <= k < n})
# events
seg_start = np.full(n, -1); win_idx = np.full(n, -1)
for s in man["lipsync"]["segments"]:
    if s["action"] == "LATENT_SYNC":
        for k in range(s["start_frame"], min(n, s["end_frame"])): seg_start[k] = s["start_frame"]; win_idx[k] = (k - s["start_frame"]) % 16
rep = set()
rj = pathlib.Path(P + "_ls_cfd_ia04_srt1_rep2s.repair.json")
if rj.exists():
    for b, e in json.loads(rj.read_text()).get("repaired", []): rep.update(range(b, e + 1))
fbw = np.zeros(n)
for f in pathlib.Path(W).glob(f"{a.id}_{a.gt}_25fps_fb_fb7s_*rep2s.npz"):
    z = np.load(f)["fb"]; fbw[:len(z)] = np.maximum(fbw[:len(z)], z[:n])


def read(path, ks):
    c = cv2.VideoCapture(path); o = {}; k = 0; mx = max(ks)
    while k <= mx:
        ok, f = c.read()
        if not ok: break
        if k in ks: o[k] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        k += 1
    return o


dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def masks(k, shape):
    p = lm[k]
    if not np.isfinite(p[LIPS + LOW + [33, 263]]).all(): return None
    iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); H, W_ = shape
    mo = np.zeros((H, W_), np.uint8); cv2.fillPoly(mo, [np.round(cv2.convexHull(p[LIPS, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    mo = cv2.dilate(mo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 * iod) + 1,) * 2)) > 0
    lo = np.zeros((H, W_), np.uint8); cv2.fillPoly(lo, [np.round(cv2.convexHull(p[LOW, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    lo = (lo > 0) & ~mo; ys, xs = np.nonzero(mo | lo); pad = 8
    return mo, lo, (max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W_, xs.max() + pad))


def series(F):
    out = {}
    for k in need:
        if k - 1 not in F or k + 1 not in F or k not in F: continue
        mm = masks(k, F[k].shape)
        if mm is None: continue
        mo, lo, (y0, y1, x0, x1) = mm; A, Bk, C = [np.ascontiguousarray(F[j][y0:y1, x0:x1]) for j in (k - 1, k, k + 1)]
        gy, gx = np.mgrid[0:Bk.shape[0], 0:Bk.shape[1]].astype(np.float32)
        fa = dis.calc(Bk, A, None); fc = dis.calc(Bk, C, None)
        wa = cv2.remap(A, gx + fa[..., 0], gy + fa[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        wc = cv2.remap(C, gx + fc[..., 0], gy + fc[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        r = np.abs(Bk.astype(np.float32) - 0.5 * (wa + wc)); mo_, lo_ = mo[y0:y1, x0:x1], lo[y0:y1, x0:x1]
        tone = abs(float(Bk[lo_].mean()) - float(A[lo_].mean()))
        out[k] = (float(r[mo_].mean()), float(r[lo_].mean()), tone)
    return out


# the final at the source fps -> master grid by time (source pts / timing): nearest source frame of each master time
tj = json.loads(pathlib.Path(f"{B}/{a.gt}/timing/{a.id}_master_timing_F.json").read_text()); jj = np.asarray(tj["j"])
S = {}
for name, path in stages.items(): S[name] = series(read(path, set(need) | {k - 1 for k in need} | {k + 1 for k in need}))
cf = cv2.VideoCapture(a.final); Ff = {}; ks_src = {int(jj[k]): k for k in need if k < len(jj)}; kk = 0; mx = max(ks_src) + 2
want = set(ks_src) | {j - 1 for j in ks_src} | {j + 1 for j in ks_src}
while kk <= mx:
    ok, f = cf.read()
    if not ok: break
    if kk in want: Ff[kk] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    kk += 1
Fm = {}
for j, k in ks_src.items():
    for d in (-1, 0, 1):
        if j + d in Ff: Fm[k + d] = Ff[j + d]
S["final"] = series(Fm)
rows = ["| window | stage | mouth max (frame) | lowface max (frame) | tone max (frame) |", "|---|---|---|---|---|"]
for w, t0, t1 in wins:
    ks = [k for k in need if t0 * 25 <= k <= t1 * 25]
    for name in list(stages) + ["final"]:
        d = {k: S[name][k] for k in ks if k in S[name]}
        if not d: continue
        cell = []
        for i in range(3):
            k = max(d, key=lambda q: d[q][i]); ev = []
            if win_idx[k] == 0: ev.append("win-start")
            elif win_idx[k] in (1, 15): ev.append(f"win{win_idx[k]}")
            if k in rep: ev.append("repaired")
            if fbw[k] > 0.5: ev.append("fallback")
            if win_idx[k] < 0: ev.append("not-LS")
            cell.append(f"{d[k][i]:.2f} (m{k} {k / 25:.2f}s{' ' + ','.join(ev) if ev else ''})")
        rows.append(f"| {w} | {name} | " + " | ".join(cell) + " |")
txt = "\n".join(rows); print(txt)
if a.out: pathlib.Path(a.out).write_text(txt + "\n")
json.dump({n_: {str(k): v for k, v in s.items()} for n_, s in S.items()}, open((a.out or "/tmp/stage_diag") + ".json", "w"))
