#!/usr/bin/env python3
"""r18: per-band numbers for the seam candidates (the autopsy's pic_cmp.py measures, on the job's own landmarks of the lip-synced face).
Bands from the source landmarks (master grid): lips (hull + 0.15 IOD), jaw (jaw line, 0.12 IOD thick), cheekL / cheekR (ear -> jaw corner, 0.10 IOD;
L = image left), neck (0.10-0.40 IOD below the jaw), inner (the oval eroded 0.12 IOD, lips out), out (> 0.5 IOD from the oval).
Per video and band: jitter = median / p95 over frames of mean |frame - previous frame| (lower = calmer; the source has its own, from head motion);
to_ref = median of mean |video - the FIRST video|. Windows (--win name=t0-t1) are reported beside the whole clip.
    r18_bands.py --work /data/jobs/<id>/e2e/work --vid 0811=ref.mp4 --vid restored=r.mp4 --vid a=a.mp4 --vid source=in.mp4 [--win cheek=19-23] --out bands.json"""
import argparse, glob, json, os, subprocess
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--vid", action="append", required=True); ap.add_argument("--win", action="append", default=[])
ap.add_argument("--out", default=""); a = ap.parse_args()
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAWLOW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]; CHL = [234, 93, 132, 58, 172]; CHR = [454, 323, 361, 288, 397]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]; BANDS = ["lips", "jaw", "cheekL", "cheekR", "neck", "inner", "out"]
kw = os.path.join(a.work, "ks"); mst = os.path.join(kw, "master_25fps.mp4")
MW, MH = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", mst], capture_output=True, text=True).stdout.strip().split(",")[:2]]
N = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", mst], capture_output=True, text=True).stdout.strip())
LM = np.full((N, 478, 2), np.nan, np.float32)
segs = json.load(open(os.path.join(kw, "shots.json")))["segments"] if os.path.exists(os.path.join(kw, "shots.json")) else []
def put(d, k0, k1):
    f = sorted(glob.glob(os.path.join(d, "lm_f*.npz")))
    if not f: return
    z = np.load(f[0]); lm = z["lm"][..., :2].astype(np.float32); m = min(len(lm), k1 - k0, N - k0); LM[k0:k0 + m] = lm[:m, :478]
if len(segs) > 1:
    for i, sg in enumerate(segs): put(os.path.join(a.work, "shots", f"s{i}", "ks"), int(sg[0]), int(sg[1]))
else: put(kw, 0, N)
w, h = MW // 2, MH // 2; sc = np.array([w / MW, h / MH], np.float32)
hull = lambda p: cv2.convexHull(np.round(p).astype(np.int32)).reshape(-1, 1, 2)
el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(int(round(r)), 1) + 1,) * 2)
pl = lambda p: np.round(p).astype(np.int32).reshape(-1, 1, 2)
def bands(p):
    q = p * sc; iod = float(np.linalg.norm(q[33] - q[263])); M = {}; z = lambda: np.zeros((h, w), np.uint8)
    m = z(); cv2.fillPoly(m, [hull(q[LIPS])], 1); M["lips"] = cv2.dilate(m, el(0.15 * iod))
    m = z(); cv2.polylines(m, [pl(q[JAWLOW])], False, 1, max(1, int(0.12 * iod))); M["jaw"] = m
    for nm, idx in (("cheekL", CHL), ("cheekR", CHR)):
        m = z(); cv2.polylines(m, [pl(q[idx])], False, 1, max(1, int(0.10 * iod))); M[nm] = m
    m = z(); cv2.fillPoly(m, [pl(np.concatenate([q[JAWLOW] + [0, 0.10 * iod], q[JAWLOW][::-1] + [0, 0.40 * iod]]))], 1); M["neck"] = m
    m = z(); cv2.fillPoly(m, [hull(q[OVAL])], 1); inner = cv2.erode(m, el(0.12 * iod)); inner[cv2.dilate(M["lips"], el(0.10 * iod)) > 0] = 0; M["inner"] = inner
    M["out"] = 1 - cv2.dilate(m, el(0.5 * iod)); return {k: v > 0 for k, v in M.items()}
def rd(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-vf", f"fps=25,scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE, bufsize=w * h * 8)
    while True:
        b = pr.stdout.read(w * h)
        if len(b) < w * h: break
        yield np.frombuffer(b, np.uint8).reshape(h, w).astype(np.float32)
V = [x.split("=", 1) for x in a.vid]; nv = len(V)
J = np.full((nv, N, len(BANDS)), np.nan, np.float32); D = np.full((nv, N, len(BANDS)), np.nan, np.float32); prev = None
for k, fs in enumerate(zip(*[rd(p) for _, p in V])):
    if k >= N: break
    if np.isfinite(LM[k]).all():
        B = bands(LM[k])
        for bi, bn in enumerate(BANDS):
            m = B[bn]
            if m.sum() < 20: continue
            for vi in range(nv):
                if prev is not None: J[vi, k, bi] = np.abs(fs[vi][m] - prev[vi][m]).mean()
                D[vi, k, bi] = np.abs(fs[vi][m] - fs[0][m]).mean()
    prev = fs
W = [("all", 0.0, N / 25.0)] + [(x.split("=")[0], float(x.split("=")[1].split("-")[0]), float(x.split("=")[1].split("-")[1])) for x in a.win]
res = {"videos": [l for l, _ in V], "bands": BANDS, "frames": N, "windows": {}}
for nm, t0, t1 in W:
    s = slice(int(t0 * 25), int(t1 * 25)); r = {}
    for vi, (l, _) in enumerate(V):
        with np.errstate(all="ignore"):
            r[l] = {"jitter_med": [round(float(x), 2) for x in np.nanmedian(J[vi, s], 0)], "jitter_p95": [round(float(x), 2) for x in np.nanpercentile(J[vi, s], 95, 0)],
                    "to_ref_med": [round(float(x), 2) for x in np.nanmedian(D[vi, s], 0)]}
    res["windows"][nm] = r
    print(f"R18_BANDS {nm} [{t0:.1f}-{t1:.1f} s]   bands: {' '.join(BANDS)}")
    for l in r: print(f"  {l:>9}  jitter {' '.join('%5.2f' % x for x in r[l]['jitter_med'])}   to {V[0][0]} {' '.join('%5.2f' % x for x in r[l]['to_ref_med'])}")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
