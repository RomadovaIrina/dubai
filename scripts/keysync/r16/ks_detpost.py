#!/usr/bin/env python3
"""r22 (Irina 16.11 on the full 03: "вклейка очень-очень мелко дрожит, мерцает, раз 10 за секунду"): source detail + flicker removal as a POST step
on the finished face video, with LOOK-AHEAD, so everything it estimates per frame is smoothed over time (ks_plain works frame by frame and cannot).
Run it on the picture the viewer gets, i.e. AFTER the native-fps retime (fps_native_merge2: each source frame gets the nearest 25 fps take
shifted in time, a pattern that repeats 5 times a second and puts the ripple back: cheeks 7-12 Hz share 0.10 at 25 fps -> 0.29 after it) and
BEFORE the subtitle burn. Per face track:
  1. flow: where the drawn face sits relative to the source (DIS on blurred half-size frames, the field blurred by --flow x IOD), then a
     temporal binomial window of --tw frames: the source detail follows the drawn jaw without the sub-pixel jitter of a per-frame flow;
  2. flicker: D = lowpass_--lf(paste - jaw-aligned source) is the drawn patch's brightness / colour difference FIELD; its fast part
     (D minus its --tw-frame temporal average) is removed, the slow part stays (no lock to the source: r21 --det-lf froze the chin shading);
  3. detail: finer than --det px from the aligned source where the paste agrees with it at low frequency (--lo / --hi).
The lips zone (the source lips + --lip x IOD, and wherever the paste really differs) keeps the generator's picture; --chin 1 also leaves the
lips-to-chin zone out of step 2. Frames / faces with no paste pass through. Audio is stream-copied from the paste.
    ks_detpost.py --master SRC.mp4 --paste face.mp4 --lm lm_f0.npz[:first_frame] [--lm ...] --out out.mp4 [--lm-fps 25 --lm-size WxH]
--lm: MediaPipe landmarks of one face track ("lm" [n, 478, 2+], optional "j"); first_frame = the track's first frame on the landmarks' own
frame grid (a shot's offset in the master). --lm-fps: that grid's rate when the videos run at another (25 for the native-fps final).
--lm-size: the frame size the landmarks were measured on (the 25 fps master) when the videos are larger."""
import argparse, collections, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ks_cms
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("master", "paste", "out")]; ap.add_argument("--lm", action="append", required=True)
ap.add_argument("--det", type=float, default=8.0); ap.add_argument("--flow", type=float, default=0.12); ap.add_argument("--tw", type=int, default=9); ap.add_argument("--lf", type=float, default=8.0)
ap.add_argument("--lo", type=float, default=5.0); ap.add_argument("--hi", type=float, default=14.0); ap.add_argument("--lip", type=float, default=0.06); ap.add_argument("--lip-lf", type=float, default=0.12)
ap.add_argument("--lm-fps", type=float, default=0.0); ap.add_argument("--lm-size", default=""); ap.add_argument("--chin", type=float, default=0.0)
ap.add_argument("--sharp", type=float, default=0.0, help="r22 (\"если мылить не будет\"): unsharp mask of this amount (sigma --sharp-s px) ONLY where the patch kept the generator's own detail (the lips zone, real changes): there is no source detail to use there, so the drawn mouth is brought to the sharpness of the skin around it"); ap.add_argument("--sharp-s", type=float, default=1.2)
ap.add_argument("--flicker", type=float, default=1.0, help="share of the fast brightness difference removed (0 = off)"); ap.add_argument("--crf", default=os.environ.get("KS_PLAIN_CRF", "16")); ap.add_argument("--preset", default="medium")
ap.add_argument("--threads", type=int, default=min(16, max(2, (os.cpu_count() or 8) // 2)))
a = ap.parse_args(); t_start = time.time()
def probe(p):
    r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", p], capture_output=True, text=True).stdout)["streams"][0]
    return int(r["width"]), int(r["height"]), r["r_frame_rate"], int(r["nb_read_packets"])
W, H, fps, n = probe(a.paste); Wm, Hm, _, nm = probe(a.master); assert (W, H) == (Wm, Hm), "master and paste differ in size"; n = min(n, nm); FPSV = eval(fps)
SX, SY = (W / float(a.lm_size.split("x")[0]), H / float(a.lm_size.split("x")[1])) if a.lm_size else (1.0, 1.0)
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]; CHIN = [61, 291, 176, 148, 152, 377, 400, 149, 378, 17, 84, 314]
LUM = np.float32([0.299, 0.587, 0.114]); TL = threading.local()
bw = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [1, 6, 15, 20, 15, 6, 1], 9: [1, 8, 28, 56, 70, 56, 28, 8, 1], 11: [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]}.get(a.tw, [1]); R = len(bw) // 2; bw = np.float32(bw)
def dis():
    if not hasattr(TL, "d"): TL.d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return TL.d
class Track:
    def __init__(s, spec):
        path, _, k0 = spec.partition(":"); z = np.load(path); s.LM = z["lm"][..., :2].astype(np.float32) * np.float32([SX, SY]); s.J = z["j"] if "j" in z.files else np.arange(len(s.LM)); s.k0 = int(k0 or 0); s.name = os.path.basename(path)
        s.ok = np.array([s.lmk(k) is not None for k in range(n)]); s.live = bool(s.ok.any())
        if not s.live: return
        P_ = np.stack([s.lmk(k) for k in np.flatnonzero(s.ok)]); iodm = float(np.median(np.linalg.norm(P_[:, 33] - P_[:, 263], axis=1)))
        x0, y0 = np.floor(P_.reshape(-1, 2).min(0) - 0.6 * iodm).astype(int); x1, y1 = np.ceil(P_.reshape(-1, 2).max(0) + 0.8 * iodm).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1); s.x0, s.y0, s.x1, s.y1 = x0, y0, x1, y1; s.w, s.h = x1 - x0, y1 - y0; s.live = s.w > 32 and s.h > 32
        if s.live: s.GX, s.GY = np.meshgrid(np.arange(s.w, dtype=np.float32), np.arange(s.h, dtype=np.float32)); s.h2, s.w2 = max(16, s.h // 2), max(16, s.w // 2)
    def lmk(s, k):
        r = (k if not a.lm_fps else int(k * a.lm_fps / FPSV + 0.5)) - s.k0
        if r < 0 or r >= len(s.J): return None
        p = s.LM[min(int(s.J[r]), len(s.LM) - 1)]
        return p if np.isfinite(p).all() else None
    def u8(s, x): return cv2.resize(cv2.cvtColor(np.clip(cv2.GaussianBlur(x, (0, 0), 2.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (s.w2, s.h2), interpolation=cv2.INTER_AREA)
    def zone(s, p, idx, ex, feather):
        m = np.zeros((s.h, s.w), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull((p[idx] - np.float32([s.x0, s.y0])).astype(np.int32)), 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(ex)) + 1,) * 2)); return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, feather))
    def analyse(s, k, fm, fp):
        if not s.ok[k]: return None
        S = fm[s.y0:s.y1, s.x0:s.x1].astype(np.float32); O = fp[s.y0:s.y1, s.x0:s.x1].astype(np.float32); df = np.abs(O - S).max(2)
        if float(df.mean()) < 0.25: return None   # no paste on this face in this frame
        p = s.lmk(k); iod = float(np.linalg.norm(p[33] - p[263]))
        pres = cv2.dilate((cv2.GaussianBlur(df, (0, 0), 3.0) > 0.8).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); pres = cv2.GaussianBlur(pres.astype(np.float32), (0, 0), 5.0)
        if a.flow > 0:
            fl = dis().calc(s.u8(O), s.u8(S), None); fl = cv2.GaussianBlur(fl, (0, 0), max(1.0, a.flow * iod / 2)); fl = np.clip(cv2.resize(fl, (s.w, s.h), interpolation=cv2.INTER_LINEAR) * 2.0, -0.2 * iod, 0.2 * iod) * pres[..., None]
        else: fl = np.zeros((s.h, s.w, 2), np.float32)
        Sw = cv2.remap(S, s.GX + fl[..., 0], s.GY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return {"S": S, "O": O, "fl": fl, "D": cv2.GaussianBlur(O - Sw, (0, 0), a.lf), "pres": pres, "p": p, "iod": iod}
    def finish(s, ents):   # ents: this track's entries of frames c-R .. c+R (None = no paste / outside the video); the centre is ents[R]
        e = ents[R]
        if e is None: return None
        js = [j for j in range(len(ents)) if ents[j] is not None]; ww = np.float32([bw[j] for j in js]); ww /= ww.sum()
        fl = sum(wi * ents[j]["fl"] for wi, j in zip(ww, js)); Db = sum(wi * ents[j]["D"] for wi, j in zip(ww, js))
        S, O, p, iod, pres = e["S"], e["O"], e["p"], e["iod"], e["pres"]
        Sw = cv2.remap(S, s.GX + fl[..., 0], s.GY + fl[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        fast = e["D"] - Db; wl = pres * (1.0 - s.zone(p, LIPS, a.lip_lf * iod, 0.03 * iod)) * (1.0 - np.clip((np.abs(Db @ LUM) - a.hi) / a.hi, 0, 1))
        if a.chin > 0: wl = wl * (1.0 - a.chin * s.zone(p, CHIN, 0.03 * iod, 0.07 * iod))
        O2 = O - a.flicker * wl[..., None] * fast
        lr, lc = cv2.GaussianBlur(Sw, (0, 0), a.det), cv2.GaussianBlur(O2, (0, 0), a.det); dl = np.abs((lc - lr) @ LUM)
        wd = (1.0 - np.clip((dl - a.lo) / max(a.hi - a.lo, 1e-3), 0, 1)) * pres * (1.0 - s.zone(p, LIPS, a.lip * iod, 0.03 * iod)); wd = cv2.GaussianBlur(wd.astype(np.float32), (0, 0), max(1.0, 0.5 * a.det))
        O3 = O2 + wd[..., None] * ((Sw - lr) - (O2 - lc))
        if a.sharp > 0: O3 = O3 + a.sharp * (pres * (1.0 - wd))[..., None] * (O3 - cv2.GaussianBlur(O3, (0, 0), a.sharp_s))
        O3 = np.clip(O3 + 0.5, 0, 255).astype(np.uint8)
        ST["flow"].append(float(np.abs(fl).max())); ST["fl_fast"].append(float(np.abs(e["fl"] - fl).mean())); ST["d_fast"].append(float(np.abs(fast @ LUM)[wl > 0.5].mean()) if (wl > 0.5).sum() > 50 else 0.0)
        return O3
ST = {"flow": [], "fl_fast": [], "d_fast": [], "act": 0}
TR = [t for t in (Track(sp) for sp in a.lm) if t.live]
if not TR: print("KS_DETPOST " + json.dumps({"frames": n, "active": 0, "note": "no face landmarks: the paste is copied"})); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.paste, "-c", "copy", a.out], check=True); sys.exit(0)
rd = lambda p: subprocess.Popen(["ffmpeg", "-v", "error", "-i", p] + ks_cms.dec(None, "rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
pm, pp = rd(a.master), rd(a.paste)
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", a.paste, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(a.crf), "-preset", a.preset] + ks_cms.enc() + ["-c:a", "copy", "-frames:v", str(n), a.out], stdin=subprocess.PIPE)
PA, PF = ThreadPoolExecutor(a.threads), ThreadPoolExecutor(a.threads); FR = {}; AN = {}; OUT = collections.deque()   # frames, analyse futures by frame index, finish futures in order
def fin(c, frame, futs):   # futs[t][j] = track t's analyse future of frame c-R+j (None outside the video)
    out = None
    for t, fs in zip(TR, futs):
        o3 = t.finish([f.result() if f is not None else None for f in fs])
        if o3 is not None:
            if out is None: out = frame.copy()
            out[t.y0:t.y1, t.x0:t.x1] = o3
    if out is not None: ST["act"] += 1
    return frame if out is None else out
def submit(c, last):
    futs = [[AN[j][ti] if 0 <= j <= last and j in AN else None for j in range(c - R, c + R + 1)] for ti in range(len(TR))]
    OUT.append(PF.submit(fin, c, FR.pop(c), futs)); AN.pop(c - R, None)
def drain(block_over):
    while OUT and (OUT[0].done() or len(OUT) > block_over): wr.stdin.write(OUT.popleft().result().tobytes())
done = 0
for k in range(n):
    bm, bp = pm.stdout.read(W * H * 3), pp.stdout.read(W * H * 3)
    if len(bm) < W * H * 3 or len(bp) < W * H * 3: n = k; break
    fm, fp = np.frombuffer(bm, np.uint8).reshape(H, W, 3), np.frombuffer(bp, np.uint8).reshape(H, W, 3); FR[k] = fp; AN[k] = [PA.submit(t.analyse, k, fm, fp) for t in TR]
    while done + R <= k: submit(done, n - 1); done += 1
    drain(4 * a.threads)
for d_ in range(done, n): submit(d_, n - 1)
drain(0); wr.stdin.close(); wr.wait(); pm.kill(); pp.kill()
md = lambda v: float(np.median(v)) if v else 0.0
print("KS_DETPOST " + json.dumps({"frames": n, "active": ST["act"], "tracks": [[t.name, t.k0, int(t.ok.sum()), [int(t.x0), int(t.y0), int(t.w), int(t.h)]] for t in TR], "det": a.det, "flow": a.flow, "tw": a.tw, "lf": a.lf, "chin": a.chin,
                                  "flow_max_px_median": round(md(ST["flow"]), 2), "flow_jitter_removed_px": round(md(ST["fl_fast"]), 3), "flicker_removed_levels": round(md(ST["d_fast"]), 3), "wall_s": round(time.time() - t_start, 1), "threads": a.threads}), flush=True)
