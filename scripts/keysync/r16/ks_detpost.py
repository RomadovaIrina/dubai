#!/usr/bin/env python3
"""r22 (Irina 16.11 on the full 03 A_1: "вклейка очень-очень мелко дрожит, мерцает, раз 10 за секунду"): the source-detail transfer as a POST step
on the pasted video, with LOOK-AHEAD, so everything it estimates per frame is smoothed over time (ks_plain works frame by frame and cannot):
  1. flow: where the drawn face sits relative to the source (DIS on blurred half-size frames, the field blurred by --flow x IOD), then a
     temporal binomial window of --tw frames -> the source detail follows the drawn jaw WITHOUT the sub-pixel frame-to-frame jitter of a
     per-frame flow (r21 --det-flow: own frame-to-frame noise 1.63 vs 1.43 without it, source 1.21);
  2. flicker: D = lowpass_--lf(paste - jaw-aligned source) is the drawn patch's brightness / colour difference FIELD; its fast part
     (D minus its --tw-frame temporal average) is removed, the slow part stays (no lock to the source: r21 --det-lf froze the chin shading);
  3. detail: finer than --det px from the (aligned) source where the paste agrees with it at low frequency (--lo / --hi), as r21 --det.
The lips zone (the source lips + --lip x IOD, and wherever the paste really differs) keeps the generator's picture. Frames with no paste pass
through. Audio is stream-copied from the paste.
    ks_detpost.py --master master_25fps.mp4 --paste paste.mp4 --lm lm_f0.npz --out out.mp4 [--det 8 --flow 0.12 --tw 5 --lf 12]"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ks_cms
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("master", "paste", "lm", "out")]
ap.add_argument("--det", type=float, default=8.0); ap.add_argument("--flow", type=float, default=0.12); ap.add_argument("--tw", type=int, default=5); ap.add_argument("--lf", type=float, default=12.0)
ap.add_argument("--lo", type=float, default=5.0); ap.add_argument("--hi", type=float, default=14.0); ap.add_argument("--lip", type=float, default=0.06); ap.add_argument("--lip-lf", type=float, default=0.12)
ap.add_argument("--lm-fps", type=float, default=0.0, help="the landmarks' frame rate when the videos run at another (the native-fps final: 25); 0 = one landmark row per frame")
ap.add_argument("--chin", type=float, default=0.0, help="r22 (Irina: the mouth moves while the face around it stands still): 0..1 = how much the zone from the lips down to the chin is left to the generator's own brightness motion (no temporal smoothing there), feathered; 0 = off")
ap.add_argument("--flicker", type=float, default=1.0, help="share of the fast brightness difference removed (0 = off)"); ap.add_argument("--crf", default=os.environ.get("KS_PLAIN_CRF", "16"))
a = ap.parse_args()
def probe(p):
    r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", p], capture_output=True, text=True).stdout)["streams"][0]
    return int(r["width"]), int(r["height"]), r["r_frame_rate"], int(r["nb_read_packets"])
W, H, fps, n = probe(a.paste); Wm, Hm, _, nm = probe(a.master); assert (W, H) == (Wm, Hm), "master and paste differ in size"; n = min(n, nm)
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM))
FPSV = eval(fps); lmk = lambda k: LM[min(int(J[min(k if not a.lm_fps else int(k * a.lm_fps / FPSV + 0.5), len(J) - 1)]), len(LM) - 1)]
ok = np.array([np.isfinite(lmk(k)).all() for k in range(n)])
if not ok.any(): print("KS_DETPOST no face landmarks: the paste is copied"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.paste, "-c", "copy", a.out], check=True); sys.exit(0)
P_ = np.stack([lmk(k) for k in np.flatnonzero(ok)]); iodm = float(np.median(np.linalg.norm(P_[:, 33] - P_[:, 263], axis=1)))
x0, y0 = np.floor(P_.reshape(-1, 2).min(0) - 0.6 * iodm).astype(int); x1, y1 = np.ceil(P_.reshape(-1, 2).max(0) + 0.8 * iodm).astype(int)
x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1); x1 -= (x1 - x0) % 2; y1 -= (y1 - y0) % 2; w, h = x1 - x0, y1 - y0
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
rd = lambda p: subprocess.Popen(["ffmpeg", "-v", "error", "-i", p] + ks_cms.dec(None, "rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
pm, pp = rd(a.master), rd(a.paste)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", a.paste, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium"] + ks_cms.enc() + ["-c:a", "copy", "-frames:v", str(n), a.out], stdin=subprocess.PIPE)
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GX, GY = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
LUM = np.float32([0.299, 0.587, 0.114]); h2, w2 = max(16, h // 2), max(16, w // 2)
bw = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [1, 6, 15, 20, 15, 6, 1], 9: [1, 8, 28, 56, 70, 56, 28, 8, 1]}.get(a.tw, [1]); R = len(bw) // 2; bw = np.float32(bw)
def u8(x): return cv2.resize(cv2.cvtColor(np.clip(cv2.GaussianBlur(x, (0, 0), 2.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (w2, h2), interpolation=cv2.INTER_AREA)
def lips(p, ex, iod):
    m = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull((p[LIPS] - np.float32([x0, y0])).astype(np.int32)), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(ex * iod)) + 1,) * 2)); return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.03 * iod))
CHIN = [61, 291, 176, 148, 152, 377, 400, 149, 378, 17, 84, 314]
def chinz(p, iod):
    m = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull((p[CHIN] - np.float32([x0, y0])).astype(np.int32)), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(0.03 * iod)) + 1,) * 2)); return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.07 * iod))
buf = []; ST = {"act": 0, "flow": [], "fl_fast": [], "d_fast": []}
def analyse(k, fm, fp):
    e = {"f": fp, "act": False}
    if not ok[k]: return e
    S = fm[y0:y1, x0:x1].astype(np.float32); O = fp[y0:y1, x0:x1].astype(np.float32); df = np.abs(O - S).max(2)
    if float(df.mean()) < 0.25: return e   # no paste on this frame
    p = lmk(k); iod = float(np.linalg.norm(p[33] - p[263]))
    pres = cv2.dilate((cv2.GaussianBlur(df, (0, 0), 3.0) > 0.8).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); pres = cv2.GaussianBlur(pres.astype(np.float32), (0, 0), 5.0)
    if a.flow > 0:
        fl = DIS.calc(u8(O), u8(S), None); fl = cv2.GaussianBlur(fl, (0, 0), max(1.0, a.flow * iod / 2)); fl = np.clip(cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0, -0.2 * iod, 0.2 * iod) * pres[..., None]
    else: fl = np.zeros((h, w, 2), np.float32)
    Sw = cv2.remap(S, GX + fl[..., 0], GY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    e.update({"act": True, "S": S, "O": O, "fl": fl, "D": cv2.GaussianBlur(O - Sw, (0, 0), a.lf), "pres": pres, "p": p, "iod": iod}); return e
def finish(c):   # the centre entry buf[c] with its neighbours buf[c-R .. c+R]
    e = buf[c]
    if not e["act"]: return e["f"]
    js = [j for j in range(c - R, c + R + 1) if 0 <= j < len(buf) and buf[j]["act"]]; ww = np.float32([bw[j - c + R] for j in js]); ww /= ww.sum()
    fl = sum(wi * buf[j]["fl"] for wi, j in zip(ww, js)); Db = sum(wi * buf[j]["D"] for wi, j in zip(ww, js))
    S, O, p, iod, pres = e["S"], e["O"], e["p"], e["iod"], e["pres"]
    Sw = cv2.remap(S, GX + fl[..., 0], GY + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    fast = e["D"] - Db; wl = pres * (1.0 - lips(p, a.lip_lf, iod)) * (1.0 - np.clip((np.abs(Db @ LUM) - a.hi) / a.hi, 0, 1))
    if a.chin > 0: wl = wl * (1.0 - a.chin * chinz(p, iod))
    O2 = O - a.flicker * wl[..., None] * fast
    lr, lc = cv2.GaussianBlur(Sw, (0, 0), a.det), cv2.GaussianBlur(O2, (0, 0), a.det); dl = np.abs((lc - lr) @ LUM)
    wd = (1.0 - np.clip((dl - a.lo) / max(a.hi - a.lo, 1e-3), 0, 1)) * pres * (1.0 - lips(p, a.lip, iod)); wd = cv2.GaussianBlur(wd.astype(np.float32), (0, 0), max(1.0, 0.5 * a.det))
    O3 = np.clip(O2 + wd[..., None] * ((Sw - lr) - (O2 - lc)) + 0.5, 0, 255).astype(np.uint8)
    ST["act"] += 1; ST["flow"].append(float(np.abs(fl).max())); ST["fl_fast"].append(float(np.abs(e["fl"] - fl).mean())); ST["d_fast"].append(float(np.abs(fast @ LUM)[wl > 0.5].mean()) if (wl > 0.5).sum() > 50 else 0.0)
    out = e["f"].copy(); out[y0:y1, x0:x1] = O3; return out
done = 0; base = 0   # buf[i] = frame base + i
for k in range(n):
    bm, bp = pm.stdout.read(W * H * 3), pp.stdout.read(W * H * 3)
    if len(bm) < W * H * 3 or len(bp) < W * H * 3: n = k; break
    buf.append(analyse(k, np.frombuffer(bm, np.uint8).reshape(H, W, 3), np.frombuffer(bp, np.uint8).reshape(H, W, 3)))
    while done + R <= k:   # the frame `done` has its R look-ahead frames
        wr.stdin.write(finish(done - base).tobytes()); done += 1
        while base < done - R: buf.pop(0); base += 1
for d_ in range(done, n): wr.stdin.write(finish(d_ - base).tobytes())
wr.stdin.close(); wr.wait(); pm.kill(); pp.kill()
md = lambda v: float(np.median(v)) if v else 0.0
print("KS_DETPOST " + json.dumps({"frames": n, "active": ST["act"], "roi": [int(x0), int(y0), int(w), int(h)], "det": a.det, "flow": a.flow, "tw": a.tw, "lf": a.lf, "flow_max_px_median": round(md(ST["flow"]), 2),
                                  "flow_jitter_removed_px": round(md(ST["fl_fast"]), 3), "flicker_removed_levels": round(md(ST["d_fast"]), 3)}), flush=True)
