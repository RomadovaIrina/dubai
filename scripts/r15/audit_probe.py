#!/usr/bin/env python3
"""r15 Z: CPU probe of the GENERATOR-side items of the face-path code audit (coordination/audit_r15_face.md), on one arm's work dir
(the master_25fps.mp4 + master_mediapipe.npz + the manifest frame_map + timing/<id>_master_timing_F.json):
  A  frame_map.py greedy thumbnail match (manifest master_to_source) vs r3_master_timing j: disagreeing master frames, max |diff|
  B  anchor_filters.rigid_motion: RANSAC inlier share per frame pair (failed / identity fits, < 0.3), and the LatentSync crop anchors
     (brow_l / brow_r / nose centroids of the replayed 106 via MAP106, filtered flow:4) with the rigid track measured on UPPER (brows +
     forehead, production) vs UPPER without brows / forehead: their difference (mean / p99 / max, IOD x 1000) and each one's HF
  N  the nose anchor (106 74/77/83/86 = MediaPipe 5/64/294/4: two nostril wings) vs a rigid nose (5/4/1/195) relative to the eye corners:
     HF and LF (1 s) motion, IOD x 1000
  C  CodeFormer 5-point alignment (LM106_TO_5 on the replayed 106): eye centroids with the iris points (production) vs without, LMEDS vs
     least squares, source mouth corners vs rigid mouth corners (the per-video median in the eyes + nose frame): HF of the aligned crop
     (template mouth centre / eye mid mapped back to the frame, |p - median5(p)|) and of its scale, IOD x 1000
    python scripts/r15/audit_probe.py --work <arm work dir> [--json out.json]"""
import argparse, glob, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in ("scripts/pilot", "scripts/optim"): sys.path.insert(0, str(ROOT / p))
import anchor_filters as af
from clean_face_stack import MAP106, ANCHORS

UPPER_PROD = list(af.UPPER)
BROW_FOREHEAD = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 9, 8, 151, 108, 337, 69, 299}
UPPER_NOBROW = [k for k in UPPER_PROD if k not in BROW_FOREHEAD]
NOSE_RIGID = [5, 4, 1, 195]
TPL = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
IRIS = {468, 469, 470, 471, 472, 473, 474, 475, 476, 477}


def med5(x):
    p = np.concatenate([x[:1], x[:1], x, x[-1:], x[-1:]]); return np.median(np.stack([p[i:i + len(x)] for i in range(5)]), 0)


def lp(x, w):
    k = np.ones(w) / w; p = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w - 1 - w // 2, 0)])
    return np.stack([np.convolve(p[:, j], k, "valid") for j in range(x.shape[1])], 1)


class Seq:
    """sequential lazy reader (rigid_motion reads frames 0, 1, 2, ... in order)"""
    def __init__(self, path):
        import cv2
        self.cap = cv2.VideoCapture(str(path)); self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.i = -1; self.f = None; self.f0 = None
    def __len__(self): return self.n
    def __getitem__(self, t):
        if t == 0 and self.f0 is not None: return self.f0
        while self.i < t:
            ok, f = self.cap.read()
            if not ok: raise IndexError(t)
            self.i += 1; self.f = f
            if self.i == 0: self.f0 = f
        return self.f


def runs_hf(x, ok):
    """mean |x - med5(x)| over contiguous runs (x (T, D)), per frame"""
    out = []
    for s, e in af.runs(ok):
        if e - s >= 5: out.append(np.linalg.norm(x[s:e] - med5(x[s:e]), axis=-1).reshape(e - s, -1).mean(1))
    return np.concatenate(out) if out else np.zeros(0)


def anchors106(lm):   # (T, 478, 2) -> (T, 3, 2): LatentSync's affine anchors (brow_l, brow_r, nose centroids of the replayed 106)
    return np.stack([lm[:, [MAP106[i] for i in ANCHORS[k]]].mean(1) for k in ("brow_l", "brow_r", "nose")], 1)


def lsq_sim(src, dst):
    """least-squares similarity src -> dst (2x3), closed form"""
    A = []; b = []
    for (x, y), (u, v) in zip(src, dst): A += [[x, -y, 1, 0], [y, x, 0, 1]]; b += [u, v]
    a_, b_, tx, ty = np.linalg.lstsq(np.array(A, np.float64), np.array(b, np.float64), rcond=None)[0]
    return np.array([[a_, -b_, tx], [b_, a_, ty]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--json"); ap.add_argument("--skip-rigid", action="store_true")
    a = ap.parse_args(); W = pathlib.Path(a.work); V = W / "video"; res = {}
    z = np.load(V / "master_mediapipe.npz"); lm = z["lm"].astype(np.float64); ok = z["presence"].astype(bool) & np.isfinite(lm[:, 0, 0])
    if np.nanmax(lm) <= 1.5: lm = lm * z["wh"][None, None, :]
    lm[~ok] = np.nan; iod = float(np.nanmedian(np.linalg.norm(lm[ok][:, 33] - lm[ok][:, 263], axis=1))); res["iod_px"] = round(iod, 1); res["frames"] = int(len(lm))
    # ---- A: frame maps
    man = [m for m in glob.glob(str(W / "*_25fps.manifest.json"))]
    tim = glob.glob(str(W.parent / "timing" / "*_master_timing_F.json"))
    if man and tim:
        fm = np.array(json.load(open(man[0]))["frame_map"]["master_to_source"]); j = np.array(json.load(open(tim[0]))["j"]); n = min(len(fm), len(j))
        d = fm[:n] - j[:n]; res["A_frame_map"] = {"master_frames": n, "disagree": int((d != 0).sum()), "max_abs": int(np.abs(d).max()),
                                                  "examples": [[int(k), int(fm[k]), int(j[k])] for k in np.nonzero(d)[0][:8]]}
    # ---- N: nose anchor, wings vs rigid, relative to the eye corners (head-relative motion only)
    eyes = lm[:, [33, 263]].mean(1)
    dw = lm[:, [MAP106[i] for i in ANCHORS["nose"]]].mean(1) - lm[:, NOSE_RIGID].mean(1)   # the wings' own (non-rigid) part of the anchor
    res["N_nose"] = {"wings_minus_rigid": {"hf_x1000": round(float(runs_hf(dw, ok).mean() / iod * 1000), 2),
                                           "std_x1000": round(float(np.nanstd(dw[ok], 0).mean() / iod * 1000), 2),
                                           "p99_dev_x1000": round(float(np.percentile(np.linalg.norm(dw[ok] - np.nanmedian(dw[ok], 0), axis=1), 99) / iod * 1000), 2)}}
    for name, idx in (("nose_wings_prod", [MAP106[i] for i in ANCHORS["nose"]]), ("nose_rigid", NOSE_RIGID)):
        rel = lm[:, idx].mean(1) - eyes
        res.setdefault("N_nose", {})[name] = {"idx": idx, "hf_x1000": round(float(runs_hf(rel, ok).mean() / iod * 1000), 2),
                                              "lf_std_x1000": round(float(np.nanstd(lp(np.nan_to_num(rel), 25)[ok], 0).mean() / iod * 1000), 2)}
    # ---- C: CodeFormer 5-point alignment variants on the replayed 106
    l106 = lm[:, MAP106]
    def eye_pts(rng, iris):
        return [i for i in rng if iris or MAP106[i] not in IRIS]
    def lm5(iris):
        return np.stack([l106[:, eye_pts(range(33, 43), iris)].mean(1), l106[:, eye_pts(range(87, 97), iris)].mean(1),
                         l106[:, 77:85].mean(1), l106[:, 52], l106[:, 61]], 1)
    import cv2
    def fit(P, method, rigid_mouth):
        P = P.copy(); T = len(P); M = np.full((T, 2, 3), np.nan)
        if rigid_mouth:   # mouth corners in the eyes + nose frame: per-video median, mapped back per frame (follows the head, not the speech)
            loc = np.full((T, 2, 2), np.nan)
            for t in np.nonzero(ok)[0]:
                S = lsq_sim(P[t, :3], TPL[:3]); loc[t] = P[t, 3:] @ S[:, :2].T + S[:, 2]
            med = np.nanmedian(loc, 0)
            for t in np.nonzero(ok)[0]:
                Si = af._inv(lsq_sim(P[t, :3], TPL[:3])); P[t, 3:] = med @ Si[:, :2].T + Si[:, 2]
        for t in np.nonzero(ok)[0]:
            M[t] = (cv2.estimateAffinePartial2D(P[t].astype(np.float32), TPL, method=cv2.LMEDS)[0] if method == "lmeds" else lsq_sim(P[t], TPL))
        return M
    probe_pts = np.array([[257.0, 371.0], [256.0, 240.0]])   # template mouth centre, eye mid
    for name, iris, method, rm in (("prod_iris_lmeds_srcmouth", True, "lmeds", False), ("noiris_lmeds_srcmouth", False, "lmeds", False),
                                   ("noiris_lsq_srcmouth", False, "lsq", False), ("noiris_lsq_rigidmouth", False, "lsq", True)):
        M = fit(lm5(iris), method, rm); good = ok & np.isfinite(M[:, 0, 0])
        Mi = np.stack([af._inv(m) if np.isfinite(m).all() else np.full((2, 3), np.nan) for m in M])
        P = np.einsum("tij,pj->tpi", Mi[:, :, :2], probe_pts) + Mi[:, None, :, 2]; sc = np.sqrt(np.abs(np.linalg.det(Mi[:, :, :2])))[:, None]
        res.setdefault("C_codeformer_align", {})[name] = {"mouth_hf_x1000": round(float(runs_hf(P[:, :1], good).mean() / iod * 1000), 2),
                                                          "eye_hf_x1000": round(float(runs_hf(P[:, 1:], good).mean() / iod * 1000), 2),
                                                          "scale_hf_x1000": round(float(runs_hf(sc / np.nanmedian(sc), good).mean() * 1000), 2)}
    # ---- B: rigid motion inliers + UPPER with / without brows -> LatentSync crop anchors
    if not a.skip_rigid:
        lm32 = lm.astype(np.float32); out = {}
        for name, up in (("upper_prod", UPPER_PROD), ("upper_nobrow", UPPER_NOBROW)):
            af.UPPER = up; H, info = af.rigid_motion(Seq(V / "master_25fps.mp4"), lm32, return_info=True)
            pair = ok[1:] & ok[:-1]; inl = info[1:, 1][pair]; nsamp = info[1:, 0][pair]
            A3 = anchors106(af.filter_landmarks(lm32, H, "flow:4").astype(np.float64))
            out[name] = {"H": H, "A3": A3, "stats": {"pairs": int(pair.sum()), "failed_or_lt12": int((nsamp < 12).sum() + ((nsamp >= 12) & (inl == 0)).sum()),
                                                      "inlier_lt_0.3": int((inl < 0.3).sum()), "inlier_lt_0.5": int((inl < 0.5).sum()),
                                                      "inlier_p1": round(float(np.percentile(inl, 1)), 3), "inlier_median": round(float(np.median(inl)), 3),
                                                      "anchor_hf_x1000": round(float(runs_hf(A3, ok).mean() / iod * 1000), 2)}}
        af.UPPER = UPPER_PROD
        dd = np.linalg.norm(out["upper_prod"]["A3"] - out["upper_nobrow"]["A3"], axis=-1).mean(1)[ok] / iod * 1000
        res["B_rigid"] = {k: v["stats"] for k, v in out.items()} | {"anchor_diff_x1000": {"mean": round(float(dd.mean()), 2), "p99": round(float(np.percentile(dd, 99)), 2),
                                                                                         "max": round(float(dd.max()), 2)}}
    print(json.dumps(res, indent=1))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
