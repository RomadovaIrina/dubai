#!/usr/bin/env python3
"""r24 (Irina 17.11, new_1755 / VID_20261117: soap, smeared mouths on close-ups, mush when far): a SCALE-ADAPTIVE crop for KeySync's input.
The tracked / static crop has ONE side per face track (1.8 x the p90 face height, capped at the frame), so when the camera comes close the face
is larger than the crop (chin cut off, the mouth at the edge: garbage) and when it goes far the face is a third of the input (mush). Here the
crop side follows the face per frame (k x the face height, smoothed over time) and may extend past the frame (border replicate), so KeySync
always sees the whole face at ~1/k of its input. Everything after KeySync keeps the CONSTANT box (post layer, ECC, guard, seam, gates, paste):
the render is resampled back into the constant box's 512 geometry ("back"); where the variable crop does not cover the constant box (a far face)
the constant crop's own pixels (the padded source crop) fill in, so nothing changes there.
  make:  ks_scrop.py make --master master_25fps.mp4 --lm lm_f0.npz --box data/box_f0.json --pad 60 --out-video data/vpad_f0.mp4 --out-box data/vbox_f0.json [--k 1.8 --sigma 6]
         (the output is already padded like pad_<f>.mp4: the last --pad frames mirrored without repeating the seam frame)
  back:  ks_scrop.py back --raw out/f0/x.mp4 --vbox data/vbox_f0.json --box data/box_f0.json --pad-src data/pad_f0.mp4 --out out/f0/x.mp4.c"""
import argparse, json, subprocess, sys
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["make", "back"])
for k in ("master", "lm", "box", "out-video", "out-box", "raw", "vbox", "pad-src", "out"): ap.add_argument("--" + k, default="")
ap.add_argument("--pad", type=int, default=60); ap.add_argument("--k", type=float, default=1.8); ap.add_argument("--sigma", type=float, default=6.0); ap.add_argument("--min-side", type=float, default=96.0)
ap.add_argument("--crf", default="10")
a = ap.parse_args(); S = 512
def reader(p, w, h):
    q = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    while True:
        b = q.stdout.read(w * h * 3)
        if len(b) < w * h * 3: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
def writer(p, w, h):
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", a.crf, "-pix_fmt", "yuv420p", p], stdin=subprocess.PIPE)
def wh(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip().split(",")
    return int(r[0]), int(r[1])
B = json.load(open(a.box)); s_c = float(B["side"]); XS, YS = B.get("xs"), B.get("ys")
cbox = lambda k: (float(XS[min(k, len(XS) - 1)]) if XS else float(B["x"]), float(YS[min(k, len(YS) - 1)]) if YS else float(B["y"]))
if a.cmd == "make":
    z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float64); J = z["j"] if "j" in z.files else np.arange(len(LM)); W, H = wh(a.master); n = len(J)
    C = np.full((n, 2), np.nan); FH = np.full(n, np.nan)
    for k in range(n):
        p = LM[min(int(J[k]), len(LM) - 1)]
        if np.isfinite(p).all() and p.max() > 0: mn, mx = p.min(0), p.max(0); C[k] = (mn + mx) / 2; FH[k] = max(mx[1] - mn[1], 0.8 * (mx[0] - mn[0]))
    ok = np.isfinite(FH)
    if ok.sum() < 2: print("KS_SCROP no face: nothing to do", flush=True); sys.exit(2)
    for d in (0, 1): C[:, d] = np.interp(np.arange(n), np.flatnonzero(ok), C[ok, d])
    lf = np.interp(np.arange(n), np.flatnonzero(ok), np.log(FH[ok]))   # smoothed in log: a zoom is multiplicative
    r = int(3 * a.sigma); g = np.exp(-0.5 * (np.arange(-r, r + 1) / a.sigma) ** 2); g /= g.sum(); sm = lambda v: np.convolve(np.pad(v, r, mode="edge"), g, mode="valid")
    Cs = np.stack([sm(C[:, 0]), sm(C[:, 1])], 1); side = np.maximum(a.min_side, a.k * np.exp(sm(lf)))
    xs = Cs[:, 0] - side / 2; ys = Cs[:, 1] - side / 2 + 0.05 * side   # (the tracker's own small downward shift: the mouth nearer the centre)
    w = writer(a.out_video, S, S); F = []
    for k, f in enumerate(reader(a.master, W, H)):
        if k >= n: break
        sc = S / side[k]; M = np.float32([[sc, 0, -xs[k] * sc], [0, sc, -ys[k] * sc]])
        F.append(cv2.warpAffine(f, M, (S, S), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE))
    F = F + F[-2:-2 - a.pad:-1]
    for f in F: w.stdin.write(f.tobytes())
    w.stdin.close(); w.wait()
    json.dump({"xs": np.round(xs, 2).tolist(), "ys": np.round(ys, 2).tolist(), "sides": np.round(side, 2).tolist(), "n": n, "pad": a.pad, "k": a.k, "sigma": a.sigma, "const_side": s_c}, open(a.out_box, "w"))
    rel = side / s_c; out = (xs < 0) | (ys < 0) | (xs + side > W) | (ys + side > H)
    print("KS_SCROP " + json.dumps({"frames": n, "padded": len(F), "side_px": [round(float(side.min())), round(float(np.median(side))), round(float(side.max()))], "const_side": s_c,
                                    "side_over_const": [round(float(rel.min()), 2), round(float(rel.max()), 2)], "past_frame_edge_frames": int(out.sum())}), flush=True)
else:
    V = json.load(open(a.vbox)); n = int(V["n"]); xv, yv, sv = np.array(V["xs"]), np.array(V["ys"]), np.array(V["sides"]); w = writer(a.out, S, S); m1 = np.ones((S, S), np.float32); cov = []
    for i, (fr, fp) in enumerate(zip(reader(a.raw, S, S), reader(a.pad_src, S, S))):
        k = i if i < n else max(0, n - 2 - (i - n))   # the padded tail mirrors frames n-2, n-3, ...
        xc, yc = cbox(k); sc_ = S / s_c   # constant 512 (u, v) -> frame (xc + u / sc_, ...) -> variable 512: u' = (X - xv) * 512 / sv
        a_ = s_c / sv[k]; M = np.float32([[a_, 0, (xc - xv[k]) * S / sv[k]], [0, a_, (yc - yv[k]) * S / sv[k]]])
        g = cv2.warpAffine(fr, M, (S, S), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        mk = cv2.warpAffine(m1, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mk = cv2.GaussianBlur(cv2.erode(mk, np.ones((5, 5), np.uint8)), (0, 0), 2.0)[..., None]; cov.append(float(mk.mean()))
        w.stdin.write(np.clip(g * mk + fp * (1 - mk) + 0.5, 0, 255).astype(np.uint8).tobytes())
    w.stdin.close(); w.wait()
    print("KS_SCROP_BACK " + json.dumps({"frames": len(cov), "covered_share_median": round(float(np.median(cov)), 3), "covered_share_min": round(float(np.min(cov)), 3)}), flush=True)
