#!/usr/bin/env python3
"""Slide of the pasted lower face relative to the source (track A, round 2, 2026-10-17). CPU, deterministic, no learned weights.

The output frame t and the 25 fps master frame t show the same head at the same place (the upper face IS the source), so any
displacement of rigid lower-face structures between them is a registration error of the generated lower face. Per LatentSync frame:
DIS optical flow (MEDIUM preset, finest scale 0) master(t) -> output(t) on a grey crop around the lower face; the displacement of a region = the median flow
over its pixels that lie inside the composite mask (alpha > 0.7, the ellipse the composite used, from the recorded mask boxes; without
them every region pixel counts). Regions from the MASTER's MediaPipe landmarks (articulating lips / jaw excluded):
  nosebase   alae / nostrils / subnasale / top of the philtrum  (what the user saw: "the nose looks displaced")
  cheeks     discs (0.18 IOD) at the nasolabial cheeks (MediaPipe 205 / 425, 36 / 266)
  upper      brows / eyes (sanity: must be ~0, the composite keeps the source there)
Reported per region (px at master resolution, and in IOD %): abs = |d_t| (static misregistration), slide = |d_t - d_(t-1)| over consecutive
LS frames (what the eye sees as sliding), and the same for the x (left-right) component; corr_shake = correlation of d_x with the
high-passed rigid head displacement (anchor_study rigid npz): > 0 = the generated face lags the head on shake.

    python scripts/quality/slide_metric.py --video OUT.mp4 --master M/03/master_25fps.mp4 --npz M/03/master_mediapipe.npz \
        --offset F0 --frames A:B [--boxes masks.npz] [--manifest window.manifest.json] [--rigid R/03_rigid.npz] --out x.json
--offset: master frame index of the video's frame 0 (window clips). Frames A:B are MASTER indices.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))

NOSEBASE = [48, 278, 64, 294, 98, 327, 2, 97, 326, 164, 165, 391, 167, 393, 240, 460, 75, 305]
CHEEKS = [205, 425, 36, 266]
UPPER = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 362, 263, 6, 168, 197]


def read_range(p, a, b):
    import cv2
    cap = cv2.VideoCapture(str(p)); cap.set(cv2.CAP_PROP_POS_FRAMES, a); out = []
    for _ in range(b - a):
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def ellipse_alpha(bbox, H, W, top=0.52, feather=0.08):
    """Same mask as candidate_video_stage.MouthOnlyRunner.mask, full frame."""
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]; w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    cy = y1 + int(top * h); cx = (x1 + x2) // 2
    axes = (int(w * 0.55), int((y2 + int(0.06 * h) - cy) * 0.5 + 1)); centre = (cx, (cy + y2 + int(0.06 * h)) // 2)
    k = max(3, int(feather * w) | 1); m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, centre, axes, 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (k, k), 0, borderType=cv2.BORDER_CONSTANT)


def region_masks(lm, H, W, iod):
    import cv2
    out = {}
    m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(lm[NOSEBASE].astype(np.int32)), 1); out["nosebase"] = m.astype(bool)
    m = np.zeros((H, W), np.uint8)
    for k in CHEEKS: cv2.circle(m, tuple(int(v) for v in lm[k]), max(3, int(0.18 * iod)), 1, -1)
    out["cheeks"] = m.astype(bool)
    m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(lm[UPPER].astype(np.int32)), 1); out["upper"] = m.astype(bool)
    return out


def stats(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return {"mean": round(float(x.mean()), 4), "p95": round(float(np.percentile(x, 95)), 4), "n": int(len(x))} if len(x) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--master", required=True); ap.add_argument("--npz", required=True)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--frames", default=None); ap.add_argument("--boxes", default=None)
    ap.add_argument("--manifest", default=None); ap.add_argument("--rigid", default=None); ap.add_argument("--out", required=True); ap.add_argument("--series", action="store_true")
    a = ap.parse_args()
    import cv2
    import anchor_filters as af
    lm = np.load(a.npz)["lm"].astype(np.float64); T = len(lm)
    A, B = (int(x) for x in a.frames.split(":")) if a.frames else (a.offset, T)
    ls = np.ones(T, bool)
    if a.manifest:
        man = json.loads(pathlib.Path(a.manifest).read_text()); ls[:] = False
        for sg in man["lipsync"]["segments"]:
            if sg["action"] == "LATENT_SYNC": ls[sg["start_frame"]:sg["end_frame"]] = True
    boxes = None
    if a.boxes:
        z = np.load(a.boxes); boxes = np.full((T, 4), np.nan); boxes[z["frames"]] = z["boxes"]
    vid = read_range(a.video, A - a.offset, B - a.offset); ref = read_range(a.master, A, A + len(vid)); B = A + len(vid)
    Hh, Ww = ref[0].shape[:2]; dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); dis.setFinestScale(0); dis.setVariationalRefinementIterations(10)
    regs = ("nosebase", "cheeks", "upper"); d = {r: np.full((B - A, 2), np.nan) for r in regs}
    for k in range(B - A):
        t = A + k
        if not ls[t] or not np.isfinite(lm[t, 0, 0]): continue
        L = lm[t]; iod = float(np.linalg.norm(L[33] - L[263]))
        x0, x1 = int(max(0, L[:, 0].min() - 0.3 * iod)), int(min(Ww, L[:, 0].max() + 0.3 * iod)); y0, y1 = int(max(0, L[:, 1].min() - 0.3 * iod)), int(min(Hh, L[:, 1].max() + 0.3 * iod))
        g0 = np.ascontiguousarray(cv2.cvtColor(ref[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)); g1 = np.ascontiguousarray(cv2.cvtColor(vid[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
        F = dis.calc(g0, g1, None)
        rm = region_masks(L, Hh, Ww, iod)
        alpha = ellipse_alpha(boxes[t], Hh, Ww) if boxes is not None and np.isfinite(boxes[t, 0]) else None
        for r in regs:
            m = rm[r][y0:y1, x0:x1]
            if alpha is not None and r != "upper": m = m & (alpha[y0:y1, x0:x1] > 0.7)
            if m.sum() < 20: continue
            d[r][k] = np.median(F[m], axis=0)
    iod_med = float(np.nanmedian(np.linalg.norm(lm[A:B, 33] - lm[A:B, 263], axis=1)))
    hp = None
    if a.rigid:   # high-passed rigid head displacement (integrated velocity at the eye midpoint), for corr_shake
        H = np.load(a.rigid)["H"]; cen = 0.5 * (lm[:, 33] + lm[:, 263]); v = np.zeros(T)
        for t in range(1, T):
            if np.isfinite(cen[t - 1, 0]): v[t] = af._apply(H[t], cen[t - 1][None])[0, 0] - cen[t - 1, 0]
        pos = np.cumsum(v); hp = (pos - af.gauss1d(pos, 2.0))[A:B]
    res = {"video": a.video, "frames": [A, B], "ls_frames": int(ls[A:B].sum()), "iod_px": round(iod_med, 1), "mask_from_boxes": boxes is not None, "regions": {}}
    for r in regs:
        x = d[r]; ok = np.isfinite(x[:, 0]); dd = np.full_like(x, np.nan); dd[1:] = x[1:] - x[:-1]
        e = {"abs_px": stats(np.linalg.norm(x, axis=1)), "abs_x_px": stats(np.abs(x[:, 0])), "slide_px": stats(np.linalg.norm(dd, axis=1)),
             "slide_x_px": stats(np.abs(dd[:, 0])), "slide_iod_pct_mean": None}
        if e["slide_px"]: e["slide_iod_pct_mean"] = round(100 * e["slide_px"]["mean"] / iod_med, 3)
        if hp is not None and ok.sum() > 10:
            xm = x[ok, 0] - x[ok, 0].mean(); hm = hp[ok] - hp[ok].mean()
            e["corr_shake_x"] = round(float((xm * hm).sum() / max(np.sqrt((xm ** 2).sum() * (hm ** 2).sum()), 1e-9)), 3)
            e["gain_vs_shake_x"] = round(float((xm * hm).sum() / max((hm ** 2).sum(), 1e-9)), 3)
        res["regions"][r] = e
        if a.series: res.setdefault("series", {})[r] = np.round(x, 3).tolist()
    if boxes is not None and a.rigid:   # composite mask (ellipse) centre vs the rigid head motion: does the mask follow the same trajectory?
        cxy = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + 0.76 * (boxes[:, 3] - boxes[:, 1])], 1)   # ~ellipse centre (mask_top 0.52)
        r = []
        for t in range(max(A, 1), B):
            if ls[t] and ls[t - 1] and np.isfinite(cxy[t, 0]) and np.isfinite(cxy[t - 1, 0]):
                head = af._apply(H[t], cxy[t - 1][None])[0] - cxy[t - 1]; r.append(np.linalg.norm((cxy[t] - cxy[t - 1]) - head))
        res["mask_centre_resid_vs_head_px"] = stats(r)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    nb = res["regions"]["nosebase"]; ch = res["regions"]["cheeks"]; up = res["regions"]["upper"]
    print(f"slide {pathlib.Path(a.video).name}: nosebase abs {nb['abs_px']} slide {nb['slide_px']} slide_x {nb['slide_x_px']} corr {nb.get('corr_shake_x')} | "
          f"cheeks slide {ch['slide_px']} | upper abs {up['abs_px']}")


if __name__ == "__main__":
    main()
