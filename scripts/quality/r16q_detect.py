#!/usr/bin/env python3
"""r16 Q: face-defect detectors for a dubbed video vs its source. Each detector prints / writes FLAGGED TIME RANGES (seconds) + a per-frame series.
Both videos are decoded on the same 25 fps grid (ffmpeg fps=25, same size). Geometry comes from MediaPipe (Face Landmarker, VIDEO mode):
up to 3 faces on the source frame (the target = the largest face, tracked by its centre; the rest = "others"), then a smoothed square crop
around the target (2.0 x face width -> 512 px), and 478 landmarks on the source crop (lmS) and on our crop (lmO). IOD = eye-corner distance.

Detectors (all on the 512 crop unless noted; the mouth is excluded where the dub legitimately differs):
  black    black mouth cavity: our mouth open (inner gap / width > 0.15) and the inner-lip hull mean L < 15 with < 2 % teeth-bright pixels
  swap     the source face shown instead of ours (old ks_fix guard): mouth |ours - source| < 0.4 x its median; cross-fade floats at the edges
  slide    face slide / mask lag vs the head: dense flow source_t -> ours_t on the eyes-nose / left cheek / right cheek / chin regions
           (mouth + 0.3 IOD excluded); the max regional median displacement in % IOD. A perfect paste = ~0; a 1-frame lag = the head speed.
  overlap  pasted face over another face (full frame): share of the other faces' hulls (minus the target oval) where |ours - source| > 10 grey
  texture  texture loss (waxy patch, lost stubble): high-frequency energy |g - blur(g)| of ours / source on the jaw-cheek region minus the lips
  explode  mouth explosion: a 1-2 frame snap of our mouth shape (inner gap and width in IOD units, 2nd difference) or of the mouth pixels
           (frame-difference spike vs its +-12-frame median AND vs the source's own change)
  edge     visible mask edge: gradient of blur(ours - source) (Lab) in a band around the lower face oval, minus the same in the cheek centres
  flicker  lip flicker: the lip ring's colour (Lab) 2nd temporal difference of ours vs the source's own, 13-frame mean
    /venv/mediapipe/bin/python r16q_detect.py --src SOURCE --ours OURS --out OUT.json [--npz cache.npz] [--t0 s --t1 s] [--only black,slide]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time
import numpy as np, cv2

FPS = 25.0
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LOWOVAL = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
REG = {"upper": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 1, 4, 98, 327],
       "lcheek": [116, 123, 147, 187, 205, 36, 100, 117], "rcheek": [345, 352, 376, 411, 425, 266, 329, 346],
       "chin": [152, 148, 176, 377, 400, 200, 199, 175, 171, 396]}
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
CROP = 512
MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
ALL = ["black", "slide", "swap", "overlap", "texture", "explode", "edge", "flicker"]
# thresholds (calibrated r16 on m03 / m18 / old03, see reports/experiments/r16-qa/DETECTORS.md)
THR = dict(black_L=15.0, black_bright=0.02, black_open=0.15, slide=4.0, overlap=0.03, texture=0.6, explode_shape=0.10, explode_px=3.5,
           edge=3.0, flicker=2.0, swap=0.4)
MINLEN = dict(black=3, slide=2, swap=3, overlap=3, texture=8, explode=1, edge=5, flicker=6)


def probe(v):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
                        "-of", "json", v], capture_output=True, text=True).stdout
    j = json.loads(o); s = j["streams"][0]; w, h = s["width"], s["height"]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd: rot = int(sd["rotation"])
    if abs(rot) % 180 == 90: w, h = h, w
    return w, h, float(j["format"].get("duration", 0) or 0)


def reader(v, W, H, t0, t1):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(t0)] if t0 > 0 else []) + ["-i", v] + (["-t", str(t1 - t0)] if t1 < 1e8 else []) + \
          ["-vf", f"fps={FPS},scale={W}:{H}:flags=bicubic", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 2); n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.stdout.close(); p.wait()


def landmarker(n):
    import mediapipe as mp
    if not MODEL.exists():
        import urllib.request; MODEL.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(URL, MODEL)
    o = mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)), running_mode=mp.tasks.vision.RunningMode.VIDEO,
                                              num_faces=n, min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
    return mp, mp.tasks.vision.FaceLandmarker.create_from_options(o)


def detect(mp, lk, img, k):
    r = lk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img[..., ::-1])), int(round(k * 1000 / FPS)))
    h, w = img.shape[:2]
    return [np.array([[q.x * w, q.y * h] for q in f], np.float32) for f in r.face_landmarks]


def hull_mask(shape, pts, dil=0):
    m = np.zeros(shape, np.uint8)
    if pts is None or not np.isfinite(pts).all(): return m
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dil) + 1, 2 * int(dil) + 1)))
    elif dil < 0: m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(-dil) + 1, 2 * int(-dil) + 1)))
    return m


def poly_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    if pts is None or not np.isfinite(pts).all(): return m
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1); return m


def runs(flag, val, minlen, maxgap=2):
    """True-runs of flag (gaps <= maxgap merged), length >= minlen -> [[t0, t1, peak]]"""
    idx = np.flatnonzero(flag); out = []
    if not len(idx): return out
    s = p = idx[0]
    for i in list(idx[1:]) + [None]:
        if i is not None and i - p <= maxgap + 1: p = i; continue
        if p - s + 1 >= minlen:
            v = val[s:p + 1]; v = v[np.isfinite(v)]
            out.append([round(s / FPS, 2), round((p + 1) / FPS, 2), round(float(v.max() if len(v) else np.nan), 3)])
        if i is not None: s = p = i
    return out


def smooth_track(c, k=9):
    c = c.copy(); ok = np.isfinite(c).all(1)
    if not ok.any(): return c
    for d in range(c.shape[1]): c[~ok, d] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok, d])
    from scipy.ndimage import median_filter, gaussian_filter1d
    return np.stack([gaussian_filter1d(median_filter(c[:, d], size=k, mode="nearest"), 2.0) for d in range(c.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--ours", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None, help="cache of the geometry pass (reused if it exists)")
    ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9); ap.add_argument("--only", default=",".join(ALL))
    ap.add_argument("--label", default="")
    a = ap.parse_args(); only = a.only.split(","); T0 = time.time()
    W, H, dur = probe(a.src); sc = min(1.0, 1920 / max(W, H)); W, H = int(round(W * sc / 2) * 2), int(round(H * sc / 2) * 2)
    dsc = min(1.0, 960 / max(W, H)); DW, DH = int(W * dsc) // 2 * 2, int(H * dsc) // 2 * 2

    # ---- pass 1: faces on the source (full frame, downscaled) -> target track + others
    cache = pathlib.Path(a.npz) if a.npz else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True); cen, fw, others = z["cen"], z["fw"], list(z["others"]); lmS, lmO = z["lmS"], z["lmO"]; boxes = z["boxes"]
        have_crop_lm = True
    else:
        have_crop_lm = False
        mp, lk = landmarker(3); cen, fw, others = [], [], []; prev = None
        for k, f in enumerate(reader(a.src, W, H, a.t0, a.t1)):
            fs = cv2.resize(f, (DW, DH), interpolation=cv2.INTER_AREA); faces = [p / dsc for p in detect(mp, lk, fs, k)]
            if faces:
                ws = [np.linalg.norm(p[234] - p[454]) for p in faces]; cs = [p[OVAL].mean(0) for p in faces]
                if prev is None: j = int(np.argmax(ws))
                else: j = int(np.argmin([np.linalg.norm(c - prev) - 0.5 * w for c, w in zip(cs, ws)]))
                prev = cs[j]; cen.append(cs[j]); fw.append(ws[j]); others.append(np.stack([p for i, p in enumerate(faces) if i != j]) if len(faces) > 1 else np.zeros((0, 478, 2), np.float32))
            else:
                cen.append(np.array([np.nan, np.nan], np.float32)); fw.append(np.nan); others.append(np.zeros((0, 478, 2), np.float32))
        lk.close(); cen = np.array(cen, np.float64); fw = np.array(fw, np.float64)
    N = len(cen)
    sm = smooth_track(np.concatenate([cen, fw[:, None]], 1), 9); side = np.maximum(2.0 * sm[:, 2], 64)
    if not have_crop_lm:
        boxes = np.stack([sm[:, 0] - side / 2, sm[:, 1] - side / 2, side], 1)
    def crop(f, k):
        x0, y0, s = boxes[k]; s_ = CROP / s
        M = np.array([[s_, 0, -x0 * s_], [0, s_, -y0 * s_]], np.float32)
        return cv2.warpAffine(f, M, (CROP, CROP), flags=cv2.INTER_AREA if s_ < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---- pass 2: crops of both videos (+ landmarks on the crops), all per-frame measures
    if not have_crop_lm:
        mp, lkS = landmarker(1); _, lkO = landmarker(1); lmS = np.full((N, 478, 2), np.nan, np.float32); lmO = lmS.copy()
    nanv = lambda: np.full(N, np.nan)
    M = {k: nanv() for k in ["iod", "open_o", "cavL_o", "cavB_o", "open_s", "cavL_s", "cavB_s", "slide", "slide_up", "slide_lc", "slide_rc", "slide_ch",
                             "overlap", "mdiff", "hf_o", "hf_s", "gap_o", "mw_o", "mpx_o", "mpx_s", "edge", "edge_in", "lipL_o", "lipA_o", "lipL_s", "lipA_s"]}
    gS, gO, prevO, prevS = None, None, None, None
    for k, (fS, fO) in enumerate(zip(reader(a.src, W, H, a.t0, a.t1), reader(a.ours, W, H, a.t0, a.t1))):
        if k >= N: break
        cS, cO = crop(fS, k), crop(fO, k)
        if not have_crop_lm:
            d = detect(mp, lkS, cS, k); lmS[k] = d[0] if d else np.nan
            d = detect(mp, lkO, cO, k); lmO[k] = d[0] if d else np.nan
        pS, pO = lmS[k], lmO[k]
        gS, gO = cv2.cvtColor(cS, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cO, cv2.COLOR_BGR2GRAY)
        if not np.isfinite(pS).all():
            prevO, prevS = None, None; continue
        iod = float(np.linalg.norm(pS[33] - pS[263])); M["iod"][k] = iod; sh = gS.shape
        mouthX = hull_mask(sh, pS[LIPS], dil=0.3 * iod)
        if np.isfinite(pO).all(): mouthX |= hull_mask(sh, pO[LIPS], dil=0.3 * iod)
        # black (ours + source as the reference)
        for tag, p, img in (("o", pO, cO), ("s", pS, cS)):
            if not np.isfinite(p).all(): continue
            gap, mw = float(np.linalg.norm(p[13] - p[14])), float(np.linalg.norm(p[61] - p[291]))
            M["open_" + tag][k] = gap / max(mw, 1e-6)
            if tag == "o": M["gap_o"][k], M["mw_o"][k] = gap / iod, mw / iod
            if gap / max(mw, 1e-6) > THR["black_open"]:
                mk = cv2.erode(hull_mask(sh, p[INNER]), np.ones((3, 3), np.uint8))
                if mk.sum() > 30:
                    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0][mk > 0].astype(np.float32) * 100 / 255
                    M["cavL_" + tag][k] = L.mean(); M["cavB_" + tag][k] = (L > 55).mean()
        # slide: flow source_t -> ours_t on stable regions
        if "slide" in only:
            fl = cv2.calcOpticalFlowFarneback(gS, gO, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            vals = []
            for rn, ix in REG.items():
                mk = hull_mask(sh, pS[ix]) & (1 - mouthX)
                if mk.sum() < 50: vals.append(np.nan); continue
                v = np.median(fl[mk > 0], 0); vals.append(float(np.linalg.norm(v)) / iod * 100)
            M["slide_up"][k], M["slide_lc"][k], M["slide_rc"][k], M["slide_ch"][k] = vals; M["slide"][k] = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
        # texture: HF energy on the jaw / cheeks minus the lips
        if "texture" in only:
            mk = poly_mask(sh, pS[LOW]) & (1 - mouthX)
            if mk.sum() > 200:
                hf = lambda g: np.abs(g.astype(np.float32) - cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32))[mk > 0].mean()
                M["hf_o"][k], M["hf_s"][k] = hf(gO), hf(gS)
        mk = hull_mask(sh, pS[LIPS], dil=0.15 * iod)
        if mk.sum() > 50: M["mdiff"][k] = float(np.abs(gO.astype(np.float32) - gS.astype(np.float32))[mk > 0].mean())
        # explode (pixel part): mouth ROI frame difference, ours and source
        if "explode" in only:
            mk = hull_mask(sh, pS[LIPS], dil=0.15 * iod)
            if prevO is not None and mk.sum() > 50:
                M["mpx_o"][k] = float(np.abs(gO.astype(np.float32) - prevO)[mk > 0].mean()); M["mpx_s"][k] = float(np.abs(gS.astype(np.float32) - prevS)[mk > 0].mean())
        prevO, prevS = gO.astype(np.float32), gS.astype(np.float32)
        # edge: gradient of blur(ours - source) in a band around the lower oval, vs the cheek centres
        if "edge" in only:
            D = cv2.GaussianBlur(cv2.cvtColor(cO, cv2.COLOR_BGR2LAB).astype(np.float32) - cv2.cvtColor(cS, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 3)
            gm = np.sqrt(sum(cv2.Sobel(D[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(D[..., c], cv2.CV_32F, 0, 1) ** 2 for c in range(3))) / 8
            ov = poly_mask(sh, pS[OVAL]); r = max(2, int(0.15 * iod)); el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            band = (cv2.dilate(ov, el) - cv2.erode(ov, el)) & (poly_mask(sh, np.concatenate([pS[LOWOVAL[:5]], pS[[1]]])) | poly_mask(sh, np.concatenate([pS[LOWOVAL[-5:]], pS[[1]]]))) .astype(np.uint8)
            band = cv2.dilate(band, np.ones((3, 3), np.uint8)) & (1 - mouthX)
            inner = (hull_mask(sh, pS[REG["lcheek"]]) | hull_mask(sh, pS[REG["rcheek"]])) & (1 - mouthX)
            if band.sum() > 100 and inner.sum() > 50:
                M["edge"][k] = float(np.percentile(gm[band > 0], 90)); M["edge_in"][k] = float(np.percentile(gm[inner > 0], 90))
        # flicker: the lip ring colour
        if "flicker" in only:
            for tag, p, img in (("o", pO, cO), ("s", pS, cS)):
                if not np.isfinite(p).all(): continue
                ring = hull_mask(sh, p[LIPS]) & (1 - hull_mask(sh, p[INNER], dil=2))
                if ring.sum() > 50:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[ring > 0]
                    M["lipL_" + tag][k], M["lipA_" + tag][k] = lab[:, 0].mean() * 100 / 255, lab[:, 1].mean() - 128
        # overlap (full frame, downscaled): the other faces
        if "overlap" in only and len(others[k]):
            s2 = 0.5; g1 = cv2.resize(cv2.cvtColor(fS, cv2.COLOR_BGR2GRAY), None, fx=s2, fy=s2).astype(np.int16)
            g2 = cv2.resize(cv2.cvtColor(fO, cv2.COLOR_BGR2GRAY), None, fx=s2, fy=s2).astype(np.int16)
            Dd = cv2.GaussianBlur(np.abs(g2 - g1).astype(np.uint8), (0, 0), 1.5)
            tgt = np.zeros(g1.shape, np.uint8)
            x0, y0, s = boxes[k]; pf = pS * (s / CROP) + np.array([x0, y0]); tgt = hull_mask(g1.shape, pf[OVAL] * s2, dil=0.15 * iod * s / CROP * s2)
            sh_ = []
            for q in others[k]:
                om = hull_mask(g1.shape, q[OVAL] * s2, dil=-2) & (1 - tgt)
                if om.sum() > 100: sh_.append(float((Dd[om > 0] > 10).mean()))
            if sh_: M["overlap"][k] = max(sh_)
    if not have_crop_lm:
        lkS.close(); lkO.close()
        if cache: np.savez_compressed(cache, cen=cen, fw=fw, others=np.array(others, dtype=object), lmS=lmS, lmO=lmO, boxes=boxes)

    # ---- per-detector flags
    R = {"label": a.label, "src": a.src, "ours": a.ours, "frames": int(N), "t0": a.t0, "fps": FPS, "size": [W, H],
         "face_frames": int(np.isfinite(M["iod"]).sum()), "iod_px_crop": round(float(np.nanmedian(M["iod"])), 1) if np.isfinite(M["iod"]).any() else None, "det": {}}
    def put(name, flag, val, extra):
        flag = np.nan_to_num(flag.astype(float)) > 0
        R["det"][name] = {"flag_frames": int(flag.sum()), "flag_share": round(float(flag.mean()), 4), "ranges": runs(flag, val, MINLEN[name]), **extra}
    if "black" in only:
        bo = (M["cavL_o"] < THR["black_L"]) & (M["cavB_o"] < THR["black_bright"]); bs = (M["cavL_s"] < THR["black_L"]) & (M["cavB_s"] < THR["black_bright"])
        no, ns = int(np.isfinite(M["cavL_o"]).sum()), int(np.isfinite(M["cavL_s"]).sum())
        put("black", bo, 100 - np.nan_to_num(M["cavL_o"], nan=100), {"open_frames_ours": no, "black_share_ours": round(bo.sum() / max(no, 1), 3),
            "open_frames_src": ns, "black_share_src": round(bs.sum() / max(ns, 1), 3)})
    if "slide" in only:
        s = M["slide"]; put("slide", s > THR["slide"], s, {"median_pct_iod": round(float(np.nanmedian(s)), 2), "p95": round(float(np.nanpercentile(s, 95)), 2),
            "regions_p95": {r: round(float(np.nanpercentile(M["slide_" + c], 95)), 2) for r, c in (("upper", "up"), ("lcheek", "lc"), ("rcheek", "rc"), ("chin", "ch"))}})
    if "swap" in only:   # the source face shown instead of the dub (the old ks_fix guard): mouth |ours - source| collapses; the cross-fades = floats
        from scipy.ndimage import median_filter
        md = M["mdiff"]; base = np.nanmedian(md); rel = md / max(base, 1e-6); sw = np.nan_to_num(rel, nan=1) < THR["swap"]
        put("swap", sw, 1 - np.nan_to_num(rel, nan=1), {"mouth_diff_median": round(float(base), 2), "note": "range = the source face shown; floats at its edges"})
    if "overlap" in only:
        o = M["overlap"]; put("overlap", o > THR["overlap"], o, {"frames_with_others": int(np.isfinite(o).sum()), "p95": round(float(np.nanpercentile(o, 95)), 3) if np.isfinite(o).any() else None})
    if "texture" in only:
        from scipy.ndimage import uniform_filter1d
        ratio = M["hf_o"] / M["hf_s"]; rr = uniform_filter1d(np.nan_to_num(ratio, nan=1.0), 13)
        put("texture", rr < THR["texture"], 1 - rr, {"median_ratio": round(float(np.nanmedian(ratio)), 3), "p05_ratio": round(float(np.nanpercentile(ratio, 5)), 3)})
    if "explode" in only:
        from scipy.ndimage import median_filter
        def d2(x):
            y = np.full_like(x, np.nan); y[1:-1] = np.abs(x[2:] - 2 * x[1:-1] + x[:-2]); return y
        shp = np.fmax(np.fmax(d2(M["gap_o"]), d2(M["mw_o"])), d2(M["open_o"]) * THR["explode_shape"] / 0.15)   # open ratio snap 0.15 = the m03 8.68 s LatentSync tear
        loc = median_filter(np.nan_to_num(M["mpx_o"], nan=0), size=25, mode="nearest")
        pz = M["mpx_o"] / np.maximum(loc, 0.5); rel = M["mpx_o"] / np.maximum(M["mpx_s"], 1.0)
        px = (pz > THR["explode_px"]) & (rel > 2.5) & (M["mpx_o"] > 6.0)
        put("explode", (shp > THR["explode_shape"]) | px, np.fmax(np.nan_to_num(shp) / THR["explode_shape"], np.nan_to_num(pz) / THR["explode_px"]),
            {"shape_snaps": int(np.nansum(shp > THR["explode_shape"])), "pixel_snaps": int(px.sum())})
    if "edge" in only:
        e = M["edge"] - M["edge_in"]; put("edge", e > THR["edge"], e, {"median": round(float(np.nanmedian(e)), 2), "p95": round(float(np.nanpercentile(e, 95)), 2)})
    if "flicker" in only:
        from scipy.ndimage import uniform_filter1d
        def d2c(x):
            y = np.full_like(x, np.nan); y[1:-1] = np.abs(x[1:-1] - (x[2:] + x[:-2]) / 2); return y
        fo = np.fmax(d2c(M["lipL_o"]), d2c(M["lipA_o"])); fs = np.fmax(d2c(M["lipL_s"]), d2c(M["lipA_s"]))
        wo, ws = uniform_filter1d(np.nan_to_num(fo), 13), uniform_filter1d(np.nan_to_num(fs), 13)
        put("flicker", (wo > THR["flicker"] * np.maximum(ws, 0.4)) & (wo > 0.9), wo / np.maximum(ws, 0.4),
            {"ours_mean": round(float(np.nanmean(fo)), 3), "src_mean": round(float(np.nanmean(fs)), 3)})
    R["sec"] = round(time.time() - T0, 1)
    np.savez_compressed(pathlib.Path(a.out).with_suffix(".series.npz"), **M)
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps({"label": a.label, **{k: [v["flag_frames"], v["ranges"][:12]] for k, v in R["det"].items()}, "sec": R["sec"]}))


if __name__ == "__main__":
    main()
