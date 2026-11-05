#!/usr/bin/env python3
"""r15 Z: PER-STAGE JITTER / SLIDE DECOMPOSITION of the mouth ("worms on a frying pan"). MediaPipe 478 landmarks of each stage (the cached
scripts/r14/mouth_open.py pass, <video>.mp_lm.npz), the lip contour (outer + inner ring) taken relative to a rigid head anchor (eye corners
33 / 263 + nose bridge 168 / 6), in units of the median inter-ocular distance (IOD) x 1000:
  HF  jitter = mean |p - median5(p)| over the lip points (the residual after a robust 5-frame smoother: frame-to-frame shake)
  LF  slide  = std over time of the 1 s moving average of the mouth-corner midpoint (61 / 291) relative to the anchor, MINUS nothing:
               reported raw and as EXCESS over the source (std of LP_stage(t) - LP_source(t), time-aligned): low-frequency drift
  anchor HF  = the head anchor's own HF residual (landmark noise floor; in crop space = how much the CROP itself shakes)
  lip colour = mean |dL|, |da*| (Lab) inside the outer lip polygon between consecutive frames, the excess over the source and the count of
               jumps above the source's p99
Stages (frame space unless noted): a source | b LatentSync input crops (crop space: the source face after the affine) + the affine placement
(a fixed crop-space mouth point mapped back to the frame by the inverse affine, relative to the source anchor) | c generated crops (crop
space) | d raw layer pasted to the frame (25 fps) | e after CodeFormer / CF-detail / SR | f after the layer repair | g final (source fps).
    python scripts/r15/stage_jitter.py --src in.mp4 --stage d_raw=ls_pre.mp4 e_cf=ls_raw.mp4 ... --crops <LS_DUMP_DIR> --master-lm master.npz --json out.json"""
import argparse, glob, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r14"))
from mouth_open import landmarks as landmarks_video

MODE = {"lm": "img"}


def landmarks(v):
    """IMAGE mode (default, --lm img): independent per-frame detections (no tracking smoother) -> <video>.mp_lm_img.npz;
    --lm video: the VIDEO-mode cache of scripts/r14/mouth_open.py (temporally smoothed: hides frame-to-frame jitter)"""
    if MODE["lm"] == "video": return landmarks_video(v)
    import subprocess
    out = pathlib.Path(v).with_suffix(".mp_lm_img.npz")
    if not out.exists():
        subprocess.run(["/venv/mediapipe/bin/python", str(pathlib.Path(__file__).with_name("mp_landmarks_img.py")), "--video", v, "--out", str(out)], check=True,
                       stdout=subprocess.DEVNULL)
    z = np.load(out); return z["lm"][..., :2].astype(np.float64), z["ok"].astype(bool)

LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
       78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OUTER = LIP[:20]; ANCHOR = [33, 263, 168, 6]; CORNERS = [61, 291]


def probe(v):
    import subprocess
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=p=0", v],
                       capture_output=True, text=True).stdout.strip().split(",")
    n, d = o[2].split("/"); return int(o[0]), int(o[1]), float(n) / float(d)


def med5(x):
    p = np.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]]); return np.median(np.stack([p[i:i + len(x)] for i in range(5)]), 0)


def lp(x, w):
    k = np.ones(w) / w; p = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w - 1 - w // 2, 0)])
    return np.stack([np.convolve(p[:, j], k, "valid") for j in range(x.shape[1])], 1)


def geom(lm, ok, fps):
    """lm (F, 478, 2) -> dict: hf (lip), anchor_hf, lp (F, 2) corner-midpoint low-pass, times, iod"""
    lm = lm.astype(np.float64); ok = ok.copy(); iod = np.median(np.linalg.norm(lm[ok][:, 33] - lm[ok][:, 263], axis=1))
    if not ok.all():   # hold the last good frame (the smoother needs a contiguous series)
        last = None
        for i in range(len(lm)):
            if ok[i]: last = lm[i]
            elif last is not None: lm[i] = last
    anc = lm[:, ANCHOR].mean(1); rel = lm[:, LIP] - anc[:, None]
    hf = np.abs(rel - med5(rel)).mean((1, 2)) / iod; ahf = np.abs(anc - med5(anc)).mean(1) / iod
    c = rel[:, [LIP.index(k) for k in CORNERS]].mean(1); L = lp(c, max(3, int(round(fps))))
    return {"hf": hf[ok], "anchor_hf": ahf[ok], "lp": L / iod, "t": np.arange(len(lm)) / fps, "ok": ok, "iod": iod}


def colour(v, lm, ok):
    import cv2
    cap = cv2.VideoCapture(v); dL, dA, prev = [], [], None; i = 0
    W, H = int(cap.get(3)), int(cap.get(4)); P = lm if lm.max() > 1.5 else lm * np.array([W, H])
    while True:
        r, f = cap.read()
        if not r or i >= len(P): break
        cur = None
        if ok[i]:
            poly = P[i, OUTER].astype(np.int32); x0, y0 = np.maximum(poly.min(0) - 2, 0); x1, y1 = np.minimum(poly.max(0) + 3, [W, H])
            if x1 > x0 + 2 and y1 > y0 + 2:
                m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [poly - [x0, y0]], 1)
                lab = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2Lab).astype(np.float32); sel = m > 0
                if sel.sum() > 10: cur = (lab[..., 0][sel].mean() * 100 / 255, lab[..., 1][sel].mean() - 128)
        if cur is not None and prev is not None: dL.append(abs(cur[0] - prev[0])); dA.append(abs(cur[1] - prev[1]))
        prev = cur; i += 1
    return np.array(dL), np.array(dA)


def crops_geom(d):
    """crop-space stages from the LS_DUMP_DIR segments: b_in (input crops), c_out (generated crops), b_affine (placement)"""
    out = {}
    for name in ("in", "out"):
        H, A, Lp = [], [], []
        for v in sorted(glob.glob(f"{d}/seg_*_{name}.mp4")):
            lm, ok = landmarks(v)
            if ok.sum() < 8: continue
            g = geom(lm, ok, 25.0); H.append(g["hf"]); A.append(g["anchor_hf"]); Lp.append(np.std(g["lp"][ok], 0).mean())
        if H: out[name] = {"hf": float(np.concatenate(H).mean() * 1000), "anchor_hf": float(np.concatenate(A).mean() * 1000), "lf_std": float(np.mean(Lp) * 1000)}
    return out


def placement(d, master_lm):
    """the affine's mouth placement: the input crops' median mouth-corner midpoint (crop space) mapped to the frame by the inverse affine
    of each frame, relative to the source anchor at that master frame (master MediaPipe), in IOD x 1000"""
    z = np.load(master_lm); M = z[next(k for k in ("lm", "lm_all", "landmarks") if k in z.files)]
    M = (M[:, 0] if M.ndim == 4 else M)[..., :2].astype(np.float64)
    if np.nanmax(M) <= 1.5:
        W, H = (int(x) for x in str(z["size"]).split(",")) if "size" in z.files else (None, None)
    hf, lf = [], []
    import re
    for npz in sorted(f for f in glob.glob(f"{d}/seg_*.npz") if re.fullmatch(r"seg_\d+\.npz", pathlib.Path(f).name)):   # not the landmark caches
        s = np.load(npz); A = s["affine"]; st = int(s["start"]); n = len(A)
        lm, ok = landmarks(npz.replace(".npz", "_in.mp4"))
        if ok.sum() < 8 or st + n > len(M): continue
        cc = np.median(lm[ok][:, CORNERS].mean(1), 0)                                   # crop-space mouth point
        P = np.stack([np.linalg.solve(np.vstack([a, [0, 0, 1]]), np.array([cc[0], cc[1], 1.0]))[:2] for a in A])   # frame space
        src = M[st:st + n]; good = np.isfinite(src).all((1, 2))
        if np.nanmax(src) <= 1.5: continue                                               # normalised master landmarks: skip (no size)
        anc = src[:, ANCHOR].mean(1); iod = np.median(np.linalg.norm(src[good][:, 33] - src[good][:, 263], axis=1)); rel = (P - anc)
        rel = rel[good]
        if len(rel) < 8: continue
        hf.append(np.linalg.norm(rel - med5(rel), axis=1) / iod); lf.append(np.std(lp(rel, 25), 0).mean() / iod)
    return ({"hf": float(np.concatenate(hf).mean() * 1000), "lf_std": float(np.mean(lf) * 1000)} if hf else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--stage", nargs="*", default=[]); ap.add_argument("--crops")
    ap.add_argument("--master-lm"); ap.add_argument("--json"); ap.add_argument("--no-colour", action="store_true"); ap.add_argument("--lm", choices=["img", "video"], default="img")
    a = ap.parse_args(); MODE["lm"] = a.lm
    res = {}; vids = [("a_source", a.src)] + [tuple(x.split("=", 1)) for x in a.stage]; G = {}
    for name, v in vids:
        lm, ok = landmarks(v); W, H, fps = probe(v); g = geom(lm, ok, fps); G[name] = g
        r = {"fps": round(fps, 2), "hf_x1000": round(float(g["hf"].mean() * 1000), 2), "anchor_hf_x1000": round(float(g["anchor_hf"].mean() * 1000), 2),
             "lf_std_x1000": round(float(np.std(g["lp"][g["ok"]], 0).mean() * 1000), 2)}
        if name != "a_source":   # LF excess: the stage's low-passed mouth position minus the source's at the same time
            S = G["a_source"]; d_ = g["lp"] - np.stack([np.interp(g["t"], S["t"], S["lp"][:, j]) for j in range(2)], 1)
            r["lf_excess_x1000"] = round(float(np.std(d_[g["ok"]], 0).mean() * 1000), 2)
        if not a.no_colour:
            dL, dA = colour(v, lm, ok); r["dL"] = round(float(dL.mean()), 3); r["da"] = round(float(dA.mean()), 3); r["_dL"] = dL; r["_dA"] = dA
        res[name] = r
    if not a.no_colour:
        tL, tA = np.percentile(res["a_source"]["_dL"], 99), np.percentile(res["a_source"]["_dA"], 99)
        for name in res:
            r = res[name]; r["dL_excess"] = round(r["dL"] - res["a_source"]["dL"], 3); r["da_excess"] = round(r["da"] - res["a_source"]["da"], 3)
            r["colour_jumps"] = int((r["_dL"] > tL).sum() + (r["_dA"] > tA).sum()); del r["_dL"], r["_dA"]
    if a.crops:
        cg = crops_geom(a.crops)
        if "in" in cg: res["b_input_crop"] = {k: round(v, 2) for k, v in cg["in"].items()}
        if "out" in cg: res["c_generated_crop"] = {k: round(v, 2) for k, v in cg["out"].items()}
        if a.master_lm:
            p = placement(a.crops, a.master_lm)
            if p: res["b_affine_placement"] = {k: round(v, 2) for k, v in p.items()}
    for k, v in res.items(): print(f"{k:22s} " + "  ".join(f"{m} {x}" for m, x in v.items()))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
