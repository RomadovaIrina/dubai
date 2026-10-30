#!/usr/bin/env python3
"""r9 F: lip colour matched to the source on the final output (bake-off on m18: "lips over-saturated red, lipstick-like"). Measured on m18: the
mean lip colour is right (a* within +0.5), but the lips' redness CONTRAST to the surrounding skin is 1.2x the source's and the red OUTLINE is
~1.9x sharper (CodeFormer detail + SR + the composite's lip sharpening). Per output frame (same frames as the source), chroma only (Lab a*, b*;
lightness untouched):
  1. contrast: inside the lip polygon, (a*, b*) - skin ring mean are scaled by the source's contrast / the output's (clamped 0.6..1);
  2. outline: in a band around the outer lip contour, (a*, b*) are blended toward a Gaussian-blurred copy (sigma = --edge-sigma x IOD).
    python scripts/r9/r9_lipcolour.py --video OUT.mp4 --src SRC.mp4 --lm SOURCE_LM.npz --out OUT_lipc.mp4"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r8"))
from r8_mouth_flicker import LIPS_OUTER, LIPS_INNER


def masks(shape, p, iod):
    lip = np.zeros(shape, np.uint8); cv2.fillPoly(lip, [np.round(p[LIPS_OUTER, :2]).astype(np.int32)], 1)
    inner = np.zeros(shape, np.uint8); cv2.fillPoly(inner, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 1)
    ring = cv2.erode(lip * (1 - inner), np.ones((3, 3), np.uint8))
    skin = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.25 * iod) * 2 + 1,) * 2)) - cv2.dilate(lip, np.ones((5, 5), np.uint8))
    return lip, ring, skin


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--contrast", choices=["on", "off"], default="off", help="on = also scale the lip-to-skin chroma contrast to the source (m18: over-corrects together with the outline step)")
    ap.add_argument("--edge-sigma", type=float, default=0.025); ap.add_argument("--band", type=float, default=0.06, help="half width of the outline band (IOD)"); ap.add_argument("--crf", type=int, default=14)
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    lm = np.load(a.lm)["lm"]; cv_ = cv2.VideoCapture(a.video); cs = cv2.VideoCapture(a.src)
    ci = av.open(a.video); rate = ci.streams.video[0].average_rate; ci.close()
    o = av.open(a.out, "w"); st = None; k = 0; scales = []
    while True:
        ok1, f = cv_.read(); ok2, g = cs.read()
        if not (ok1 and ok2): break
        if st is None:
            st = o.add_stream("libx264", rate=rate); st.height, st.width = f.shape[:2]; st.pix_fmt = "yuv420p"; st.options = {"crf": str(a.crf), "preset": "medium"}
        p = lm[k] if k < len(lm) else None; k += 1
        if p is not None and np.isfinite(p[LIPS_OUTER + LIPS_INNER + [33, 263]]).all():
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); lip, ring, skin = masks(f.shape[:2], p, iod)
            if ring.sum() >= 20 and skin.sum() >= 20:
                lo = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); ls = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)
                for ch in ((1, 2) if a.contrast == "on" else ()):
                    so, ss = lo[..., ch][skin > 0].mean(), ls[..., ch][skin > 0].mean()
                    co, cs_ = lo[..., ch][ring > 0].mean() - so, ls[..., ch][ring > 0].mean() - ss
                    s = float(np.clip(cs_ / co, 0.6, 1.0)) if abs(co) > 1e-3 and np.sign(co) == np.sign(cs_) else 1.0
                    if ch == 1: scales.append(s)
                    wl = cv2.GaussianBlur(lip.astype(np.float32), (0, 0), max(1.0, 0.02 * iod))
                    lo[..., ch] = so + (lo[..., ch] - so) * (1 - wl * (1 - s))
                band = cv2.morphologyEx(lip, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(a.band * iod) * 2 + 1,) * 2)).astype(np.float32)
                band = cv2.GaussianBlur(band, (0, 0), max(1.0, 0.03 * iod))[..., None]; sg = max(0.8, a.edge_sigma * iod)
                ab = lo[..., 1:3].copy(); m0 = ab[ring > 0].mean(0)
                lo[..., 1:3] = ab * (1 - band) + cv2.GaussianBlur(ab, (0, 0), sg) * band
                wl = cv2.GaussianBlur(lip.astype(np.float32), (0, 0), max(1.0, 0.02 * iod))[..., None]   # keep the lips' own mean colour (the outline blur pulls skin colour in)
                lo[..., 1:3] += (m0 - lo[..., 1:3][ring > 0].mean(0)) * wl
                f = cv2.cvtColor(np.clip(lo, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        for pk in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close()
    rep = {"frames": k, "contrast_scale_a_median": round(float(np.median(scales)), 3) if scales else None, "seconds": round(time.perf_counter() - t0, 1)}
    pathlib.Path(a.out).with_suffix(".lipc.json").write_text(json.dumps(rep)); print("== lipcolour", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
