#!/usr/bin/env python3
"""r11 FACE (KeySync old03 'the upper lip vanishes', 'the chin is a soft pink smear'): per-frame lip / chin measures on several 25 fps
renditions of the same master + a contact sheet of fixed (or the worst) frames.
  Renditions: src (the master, source landmarks), ls (LatentSync layer, its own mediapipe), and any --rend name=video.mp4 (KeySync-derived,
  measured with the KeySync layer's mediapipe --ks-lm).
    edge    = mean |grad L| across the upper outer lip contour (the lip line)
    da      = Lab a* (redness) of the upper-lip band minus the moustache band just above it
    must / must_hf = moustache band L / Laplacian std with the SOURCE landmarks on every rendition (same pose -> the same pixels)
    inL     = mean L inside the inner lip polygon when open > 0.08
    chin_hf = Laplacian std between the lower lip and the jaw line (own landmarks) / the source's at the same frame
    chin_da = chin Lab a* minus the source chin a* at the same frame (+ = pinker)
    upper_lip.py --src M.mp4 --src-lm F.npz --ls LS.mp4 --ls-lm L.npz --ks-lm K.npz --rend ks=LAYER.mp4 [--rend fix=OUT.mp4 ...]
                 [--frames 224,440,...] --out DIR [--tag name]"""
import argparse, pathlib, subprocess
import numpy as np, cv2

UO = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]                    # upper outer lip, left -> right
UI = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]                   # upper inner
LO = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]                  # lower outer, right -> left
JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
INNER = UI + [324, 318, 402, 317, 14, 87, 178, 88, 95]

ap = argparse.ArgumentParser()
for k in ["src", "src_lm", "ls", "ls_lm", "ks_lm", "out"]: ap.add_argument("--" + k.replace("_", "-"), required=True)
ap.add_argument("--rend", action="append", default=[]); ap.add_argument("--frames", default=""); ap.add_argument("--tag", default="sheet")
ap.add_argument("--worst", type=int, default=10); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
sz = np.load(a.src_lm); SL = sz["lm"][sz["j"]].astype(np.float32)
KL = np.load(a.ks_lm)["lm"][..., :2].astype(np.float32); LL = np.load(a.ls_lm)["lm"][..., :2].astype(np.float32)
W, H = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.src],
                                        capture_output=True, text=True).stdout.strip().split(",")]
N = min(len(SL), len(KL), len(LL))
rends = [("src", a.src, SL), ("ls", a.ls, LL)] + [(r.split("=", 1)[0], r.split("=", 1)[1], KL) for r in a.rend]

def reader(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8); n = W * H * 3
    while True:
        r = pr.stdout.read(n)
        if len(r) < n: return
        yield np.frombuffer(r, np.uint8).reshape(H, W, 3)

def poly(pts, shape, ero=0):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if ero > 0: m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=ero)
    return m.astype(bool)

def bands(L):
    mw = np.linalg.norm(L[291] - L[61]) + 1e-3
    up = L[2] - L[13]; up = up / (np.linalg.norm(up) + 1e-3)
    uo = L[UO]; must = np.concatenate([uo, (uo + up * 0.22 * mw)[::-1]])
    return mw, up, uo, np.concatenate([uo, L[UI][::-1]]), must

def measure(img, L, Ls, srcref):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); Lc = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0); sh = Lc.shape
    mw, up, uo, lipp, mustp = bands(L); gy, gx = np.gradient(Lc); gs = []
    for i in range(len(uo) - 1):
        for t in np.linspace(0, 1, 6, endpoint=False):
            p = uo[i] * (1 - t) + uo[i + 1] * t; x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < sh[1] and 0 <= y < sh[0]: gs.append(abs(gx[y, x] * up[0] + gy[y, x] * up[1]))
    lm_, mm_ = poly(lipp, sh), poly(mustp, sh); ms_ = poly(bands(Ls)[4], sh)
    lap = cv2.Laplacian(lab[..., 0], cv2.CV_32F); op = np.linalg.norm(L[14] - L[13]) / mw; inn = poly(L[INNER], sh)
    ch = poly(np.concatenate([L[LO], L[JAW]]), sh, max(1, int(0.04 * mw)))
    r = dict(edge=float(np.mean(gs)) if gs else np.nan, da=float(lab[..., 1][lm_].mean() - lab[..., 1][mm_].mean()) if lm_.any() and mm_.any() else np.nan,
             must=float(lab[..., 0][ms_].mean()), must_hf=float(lap[ms_].std()), open=float(op),
             inL=float(lab[..., 0][inn].mean()) if (op > 0.08 and inn.sum() > 20) else np.nan,
             chin_std=float(lap[ch].std()) if ch.sum() > 50 else np.nan, chin_a=float(lab[..., 1][ch].mean()) if ch.sum() > 50 else np.nan)
    if srcref is not None:
        r["chin_hf"] = r["chin_std"] / srcref["chin_std"] if srcref["chin_std"] else np.nan; r["chin_da"] = r["chin_a"] - srcref["chin_a"]
    else: r["chin_hf"], r["chin_da"] = 1.0, 0.0
    return r

fixed = [int(v) for v in a.frames.split(",") if v.strip()]
gens = [reader(p) for _, p, _ in rends]; R = {nm: [] for nm, _, _ in rends}; keep = {}
for i in range(N):
    ims = [next(g) for g in gens]; ref = None
    for (nm, _, L), im in zip(rends, ims):
        r = measure(im, L[i], SL[i], ref); R[nm].append(r)
        if nm == "src": ref = r
    if (fixed and i in fixed) or (not fixed and i % 2 == 0): keep[i] = ims
    if i % 500 == 0: print("..", i, flush=True)

T = {nm: {key: np.array([r[key] for r in R[nm]]) for key in R[nm][0]} for nm in R}
np.savez(out / f"{a.tag}.npz", **{f"{nm}_{key}": v for nm in T for key, v in T[nm].items()})
med = lambda v: float(np.nanmedian(v))
print(f"== {a.tag} medians (da = upper lip a* - moustache a*; chin_hf = chin texture / source; chin_da = chin a* - source, + pinker; inL = open interior L)")
for nm in T:
    t = T[nm]; print(f"  {nm:8s} edge {med(t['edge']):5.2f}  da {med(t['da']):5.2f}  must {med(t['must']):6.1f} hf {med(t['must_hf']):5.2f}  inL {med(t['inL']):5.1f}  "
                     f"chin_hf {med(t['chin_hf']):4.2f}  chin_da {med(t['chin_da']):+5.2f}")
first = rends[2][0] if len(rends) > 2 else "ls"
if fixed: pick = fixed
else:
    sc = T[first]["da"]; order = [i for i in np.argsort(sc) if i in keep and np.isfinite(sc[i])]; pick = []
    for i in order:
        if all(abs(i - p) >= 12 for p in pick): pick.append(i)
        if len(pick) >= a.worst: break
rows = []
for i in sorted(pick):
    c = SL[i][13]; mw = np.linalg.norm(SL[i][291] - SL[i][61]); hw = int(1.1 * mw); x0, y0 = int(c[0] - hw), int(c[1] - hw); tiles = []
    for (nm, _, _), im in zip(rends, keep[i]):
        crop = cv2.copyMakeBorder(im, hw, hw, hw, hw, cv2.BORDER_REPLICATE)[y0 + hw:y0 + 3 * hw, x0 + hw:x0 + 3 * hw]
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, nm, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(crop, f"da {T[nm]['da'][i]:.1f} chin {T[nm]['chin_hf'][i]:.2f}", (6, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(crop)
    lab = np.zeros((256, 110, 3), np.uint8)
    for k_, t in enumerate([f"f{i}", f"t{i/25:.2f}s", f"open {T['src']['open'][i]:.2f}"]): cv2.putText(lab, t, (6, 30 + 40 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.concatenate([lab] + tiles, 1))
cv2.imwrite(str(out / f"{a.tag}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"== {a.tag} sheet frames {sorted(pick)} -> {out / (a.tag + '.jpg')}")
