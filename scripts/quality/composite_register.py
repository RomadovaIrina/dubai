#!/usr/bin/env python3
"""Lower-face POSITION on the composite side (track B, round 2, 2026-10-17): register LatentSync's generated lower face onto the source
frame before blending. CPU, deterministic, no learned weights (OpenCV DIS flow + RANSAC; MediaPipe landmarks already computed).

Why: the user sees the pasted lower face (mouth, base of the nose, cheeks) slide left / right relative to the untouched upper face,
most on camera shake. LatentSync generates the masked lower half of its 512 crop in CROP coordinates (its mask starts at the nose anchor:
alae / nostrils / subnasale / cheeks / mouth are generated, the nose bridge and everything above are the source through the VAE), and the
crop follows the head only partly (landmark smoothing, the affine's bias EMA, fp16): after the inverse warp the generated lower face sits
where the CROP was, not where the head is. The upper face is the source, so the error shows as the lower face sliding against the eyes.

Per LatentSync master frame t (L = LatentSync raw frame, M = master frame; both full frames, same size):
  register  DIS flow M -> L on a grey crop around the lower face; correspondences on the STABLE generated parts only (nose base hull
            alae / nostrils / subnasale, cheek discs at the nasolabial cheeks, the nasolabial bands; lips, chin and jaw never count,
            mouth hull dilated by --lip-guard mouth widths). --reg trans: median displacement; --reg sim: RANSAC similarity (cv2
            estimateAffinePartial2D, threshold --ransac-px, fallback to trans when < 40 % inliers or |scale-1| / angle too large).
            L is resampled onto M's geometry: A(p) = L(S(p)) (bicubic) -> the generated nose base / cheeks land on the source's.
  ring      (--ring on) residual DIS flow M -> A, kept on non-lip skin near the mask edge (the paste ring) and the stable parts,
            smoothed by normalised convolution (sigma --ring-sigma IOD), faded to 0 towards the lips (the lip shapes stay exactly
            LatentSync's) and applied as a remap: removes the residual shear between pasted and source skin at the seam.
  mask      --mask clean0: clean0's lower-face ellipse (MouthOnlyRunner geometry on the ±2-smoothed MediaPipe detections, box ±3 inside each
            LatentSync piece = exactly clean0 / decompose S2nc). --mask nosecut: the same ellipse built on the CURRENT frame's landmarks
            (no smoothing lag) with an explicit --edge-px (10-90 %) edge, and cut from above in face coordinates so that alpha < 0.1 at and
            above the subnasale / alar bases (no nose in the patch); the cut ramps 0.1 -> 0.9 over --edge-px below that line
            (on small faces the line moves up towards the nose tip so the upper lip still gets alpha >= 0.9).
  blend     out = M + w (alpha (A - M)), w = the clean0 crossfade weight of the piece (decompose's rule).
--variant clean0 reproduces decompose's S3nc_final_nocf bit for bit (two roundings, cv2.addWeighted crossfade) as a pipeline check.

Writes <win>/<name>.mp4 (lossless, window frames, same layout as the decompose stages: mouth_jitter_hr / window_syncnet / slide_metric
work unchanged), <win>/<name>.reg.json (per-frame registration: dx, dy, angle, scale, inliers, residual) and <win>/<name>.boxes.npz.

    python scripts/quality/composite_register.py --win DECOMP_DIR --npz WORK/master_mediapipe.npz --name V2 --reg sim --ring on --mask clean0
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
FPS = 25
NOSEBASE = [48, 278, 64, 294, 98, 327, 2, 97, 326, 164, 165, 391, 167, 393, 240, 460, 75, 305]   # = slide_metric.NOSEBASE
CHEEKS = [205, 425, 36, 266, 50, 280, 187, 411]
NASOLABIAL = [[203, 206, 216], [423, 426, 436]]   # band from the ala towards the mouth corner, each side
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
EYE_L = [33, 133]; EYE_R = [362, 263]


def read_video(p):
    import cv2
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def write_video(path, frames_bgr):
    import imageio, cv2
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264rgb", macro_block_size=None, pixelformat="rgb24",
                           ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
    for f in frames_bgr: w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    w.close()


def face_axes(L):
    eL, eR = L[EYE_L].mean(0), L[EYE_R].mean(0); d = eR - eL; iod = float(np.linalg.norm(d)); u = d / max(iod, 1e-6)
    return u, np.array([-u[1], u[0]]), iod   # v points down the face (image y grows downwards)


def stable_mask(L, shape, iod, lip_guard):
    """Stable generated parts (nose base, cheeks, nasolabial bands) minus a dilated lip hull; bool, frame coordinates."""
    import cv2
    H, W = shape; m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(L[NOSEBASE].astype(np.int32)), 1)
    r = max(3, int(round(0.16 * iod)))
    for k in CHEEKS: cv2.circle(m, tuple(int(round(v)) for v in L[k]), r, 1, -1)
    for band in NASOLABIAL: cv2.polylines(m, [L[band].astype(np.int32)], False, 1, max(2, int(round(0.12 * iod))))
    mw = float(np.linalg.norm(L[61] - L[291])); lip = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(lip, cv2.convexHull(L[LIPS_OUTER].astype(np.int32)), 1)
    g = int(round(lip_guard * mw))
    if g > 0: lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    return (m > 0) & (lip == 0), lip > 0


def lip_distance(lip, shape):
    import cv2
    return cv2.distanceTransform((~lip).astype(np.uint8), cv2.DIST_L2, 5)


class Dis:
    def __init__(self):
        import cv2
        self.d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); self.d.setFinestScale(0); self.d.setVariationalRefinementIterations(10)

    def __call__(self, a, b):
        return self.d.calc(np.ascontiguousarray(a), np.ascontiguousarray(b), None)


def gray(x):
    import cv2
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def nosecut_alpha(L, shape, edge_px, top=0.52):
    """Ellipse of clean0's geometry on the CURRENT landmarks' LatentSync bbox, explicit edge, cut above the nose base (face coordinates)."""
    import cv2, clean_face_stack as cfs
    from fps_composite import ellipse_alpha
    H, W = shape; lm106 = cfs.lm106_from_mp(L, True); bb = cfs.ls_bbox(lm106, W, H)
    if bb is None: return None, None
    al, roi = ellipse_alpha(bb, H, W, edge_px, top)
    if al is None: return None, None
    ya, yb, xa, xb = roi; u, v, iod = face_axes(L)
    # cut line: through the lowest of subnasale / alar bases along v; alpha_cut = Phi((s - s0) / sig) with 0.1 at s0 and 0.9 at s0 + edge
    # s0 = where alpha_cut = 0.1: on the nose-base line (subnasale / alar bases) when the gap to the upper lip holds the whole edge, else moved
    # up just enough that alpha_cut reaches 0.9 one pixel above the upper-lip top (small faces: 01 / 05 have 7-10 px), never above the nose tip
    s_nb = max(float(L[k] @ v) for k in (2, 98, 327, 64, 294)); s_tip = float(L[1] @ v); s_lip = float(L[0] @ v)
    s_line = max(s_tip, min(s_nb, s_lip - edge_px - 1.0)); sig = edge_px / 2.563; s0 = s_line + 1.2816 * sig   # Phi(-1.28) = 0.1
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32); s = xx * v[0] + yy * v[1]
    cut = 0.5 * (1 + _erf((s - s0) / (sig * math.sqrt(2))))
    return (al * cut).astype(np.float32), roi


def _erf(z):
    # Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7), vectorised
    s = np.sign(z); z = np.abs(z); t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return s * y


def main() -> int:
    import cv2
    import clean_face_stack as cfs, face_aware_latentsync_accel as faa
    from candidate_video_stage import MouthOnlyRunner
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--win", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--variant", choices=["clean0", "reg"], default="reg"); ap.add_argument("--reg", choices=["none", "trans", "sim"], default="sim")
    ap.add_argument("--ring", choices=["off", "on"], default="off"); ap.add_argument("--ring-sigma", type=float, default=0.12); ap.add_argument("--ring-width", type=float, default=0.35)
    ap.add_argument("--mask", choices=["clean0", "nosecut"], default="clean0"); ap.add_argument("--edge-px", type=float, default=12.0)
    ap.add_argument("--lip-guard", type=float, default=0.12); ap.add_argument("--ransac-px", type=float, default=1.0)
    ap.add_argument("--max-shift-iod", type=float, default=0.15, help="larger estimated shifts are treated as failures (registration skipped for that frame)")
    a = ap.parse_args()
    win = pathlib.Path(a.win); wm = json.loads((win / "window_manifest.json").read_text()); A, B = wm["window"]
    t_all = time.perf_counter()
    M = read_video(win / "master.mp4"); Lr = read_video(win / "S0_ls_raw.mp4"); n = min(len(M), len(Lr)); H, W = M[0].shape[:2]
    mp = np.load(a.npz)["lm"].astype(np.float64)
    dets_all = cfs.ls_detections(mp.astype(np.float32), W, H, calib=True)
    # clean0 boxes + crossfade weights per piece (decompose_video_stage: dets ±2 inside the piece, MouthOnlyRunner.boxes ±3 inside the piece)
    runner = MouthOnlyRunner(type("I", (), {"config": {}, "calls": 0, "steps": 0, "guidance": 0, "seed": 0, "load_s": 0})(), wm["settings"]["mask_top"],
                             wm["settings"]["mask_feather"], wm["settings"]["mask_smooth"], defer=True)
    boxes = [None] * n; wts = np.zeros(n, np.float32)
    for s in wm["pieces"]:
        st, en = s["start"], min(s["end"], A + n); d = faa.smooth_detections(list(dets_all[st:en]), wm["settings"]["ls_landmark_smooth"])
        bx = runner.boxes(d, en - st); L_ = en - st
        for k in range(L_):
            boxes[st - A + k] = bx[k]; w = 1.0
            if s["xf_in"] and k < s["xf_in"]: w = (k + 1) / (s["xf_in"] + 1)
            if s["xf_out"] and k >= L_ - s["xf_out"]: w = min(w, (L_ - k) / (s["xf_out"] + 1))
            wts[st - A + k] = w
    dis = Dis(); out = []; reg = []; tsec = 0.0; box_rec = np.full((n, 4), np.nan)
    for i in range(n):
        t = A + i; m = M[i]
        if wts[i] <= 0 or boxes[i] is None: out.append(m); continue
        box_rec[i] = boxes[i]
        if a.variant == "clean0":   # exact decompose S2nc + crossfade
            al, roi = runner.mask(boxes[i], H, W); ya, yb, xa, xb = roi; r = m.copy()
            r[ya:yb, xa:xb] = (Lr[i][ya:yb, xa:xb].astype(np.float32) * al + r[ya:yb, xa:xb].astype(np.float32) * (1 - al)).round().clip(0, 255).astype(np.uint8)
            out.append(cv2.addWeighted(r, float(wts[i]), m, 1.0 - float(wts[i]), 0.0) if wts[i] < 1.0 else r); continue
        t0 = time.perf_counter(); Lm = mp[t]
        if not np.isfinite(Lm).all(): out.append(m); continue
        u, v, iod = face_axes(Lm)
        if a.mask == "clean0":
            al, roi = runner.mask(boxes[i], H, W); al = al[..., 0]
        else:
            al, roi = nosecut_alpha(Lm, (H, W), a.edge_px)
        if al is None: out.append(m); continue
        ya, yb, xa, xb = roi
        # registration crop: lower face + margin (mask roi grown by 0.25 IOD, plus the nose base)
        g = int(0.25 * iod); y0 = max(0, min(ya, int(Lm[NOSEBASE, 1].min())) - g); y1 = min(H, yb + g); x0 = max(0, xa - g); x1 = min(W, xb + g)
        gm, gl = gray(m[y0:y1, x0:x1]), gray(Lr[i][y0:y1, x0:x1])
        stab, lip = stable_mask(Lm, (H, W), iod, a.lip_guard); stab = stab[y0:y1, x0:x1]; lip = lip[y0:y1, x0:x1]
        rec = {"t": t, "w": float(wts[i])}
        Lk = Lr[i][y0:y1, x0:x1]
        if a.reg != "none":
            F = dis(gm, gl); ys, xs = np.nonzero(stab); sel = slice(None, None, 2)
            P = np.stack([xs[sel], ys[sel]], 1).astype(np.float32); Q = P + F[ys[sel], xs[sel]]
            med = np.median(Q - P, axis=0); S = np.array([[1, 0, med[0]], [0, 1, med[1]]], np.float64); rec.update(n_pts=int(len(P)), kind="trans")
            if a.reg == "sim" and len(P) >= 30:
                Ms, inl = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=a.ransac_px, maxIters=2000, confidence=0.995, refineIters=10)
                if Ms is not None:
                    sc = math.hypot(Ms[0, 0], Ms[1, 0]); ang = math.degrees(math.atan2(Ms[1, 0], Ms[0, 0])); fr = float(inl.mean())
                    rec.update(inliers=round(fr, 3), scale=round(sc, 5), angle=round(ang, 4))
                    if fr >= 0.4 and abs(sc - 1) < 0.04 and abs(ang) < 3.0: S = Ms.astype(np.float64); rec["kind"] = "sim"
            # S maps master-crop coords -> LatentSync-crop coords; shift measured at the nose base centre
            c = Lm[NOSEBASE].mean(0) - [x0, y0]; d = S[:, :2] @ c + S[:, 2] - c; rec.update(dx=round(float(d[0]), 3), dy=round(float(d[1]), 3))
            if np.linalg.norm(d) > a.max_shift_iod * iod: rec["kind"] = "rejected"
            else: Lk = cv2.warpAffine(Lr[i], np.array([[S[0, 0], S[0, 1], S[0, 2] + x0 - (S[0, 0] * x0 + S[0, 1] * y0)],
                                                        [S[1, 0], S[1, 1], S[1, 2] + y0 - (S[1, 0] * x0 + S[1, 1] * y0)]]),
                                      (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)[y0:y1, x0:x1]
        if a.ring == "on":
            F2 = dis(gm, gray(Lk)); al_c = np.zeros((y1 - y0, x1 - x0), np.float32)
            al_c[ya - y0:yb - y0, xa - x0:xb - x0] = al
            ringw = int(max(3, a.ring_width * iod)); band = ((al_c > 0.02) & (al_c < 0.98)).astype(np.uint8)
            band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ringw + 1, 2 * ringw + 1))) > 0
            wgt = ((band | stab) & ~lip).astype(np.float32)
            sg = a.ring_sigma * iod; den = cv2.GaussianBlur(wgt, (0, 0), sg) + 1e-4
            Fs = np.stack([cv2.GaussianBlur(F2[..., c_] * wgt, (0, 0), sg) / den for c_ in range(2)], -1)
            dl = lip_distance(lip, lip.shape); fade = np.clip(dl / max(0.25 * iod, 1.0), 0, 1).astype(np.float32)   # 0 on the lips -> 1 at 0.25 IOD
            Fs *= fade[..., None] * (den > 0.02)[..., None]
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
            Lk = cv2.remap(Lk, xx + Fs[..., 0], yy + Fs[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rec["ring_mean_px"] = round(float(np.linalg.norm(Fs, axis=-1)[al_c > 0.02].mean()), 3)
        # blend inside the mask roi
        o = m.copy(); sk = m[ya:yb, xa:xb].astype(np.float32); lk = Lk[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
        aw = (al * wts[i])[..., None]; o[ya:yb, xa:xb] = np.clip(sk + aw * (lk - sk), 0, 255).round().astype(np.uint8)
        out.append(o); tsec += time.perf_counter() - t0; reg.append(rec)
    out += M[n:]
    write_video(win / f"{a.name}.mp4", out)
    np.savez_compressed(win / f"{a.name}.boxes.npz", frames=np.flatnonzero(np.isfinite(box_rec[:, 0])) + A, boxes=box_rec[np.isfinite(box_rec[:, 0])])
    ls_frames = int((wts > 0).sum())
    summ = {"name": a.name, "args": vars(a), "window": [A, B], "ls_frames": ls_frames, "composite_s": round(tsec, 2),
            "composite_s_per_ls_video_min": round(tsec / max(ls_frames / FPS / 60, 1e-9), 1), "total_s": round(time.perf_counter() - t_all, 2)}
    if reg and "dx" in reg[0]:
        dxy = np.array([[r["dx"], r["dy"]] for r in reg if "dx" in r]); kinds = [r["kind"] for r in reg if "kind" in r]
        summ["est_misreg_px"] = {"abs_mean": round(float(np.linalg.norm(dxy, axis=1).mean()), 3), "abs_p95": round(float(np.percentile(np.linalg.norm(dxy, axis=1), 95)), 3),
                                 "dx_std": round(float(dxy[:, 0].std()), 3), "frame_to_frame_mean": round(float(np.linalg.norm(np.diff(dxy, axis=0), axis=1).mean()), 3)}
        summ["kinds"] = {k: kinds.count(k) for k in set(kinds)}
    (win / f"{a.name}.reg.json").write_text(json.dumps({"summary": summ, "frames": reg}, indent=0))
    print(json.dumps(summ)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
