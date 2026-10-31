#!/usr/bin/env python3
"""r10 FACE: the two artefacts management sees on every video, measured the way a viewer sees them (output vs the source, same frame grid).

LIP FLICKER (frame-to-frame pops of the lip / teeth detail and brightness)
  Per frame k, in a box around the mouth: the neighbours k-1 and k+1 are DIS-flow-aligned to k and the second temporal difference
  D2 = I_k - (W(I_k-1) + W(I_k+1)) / 2 is taken (smooth motion cancels; a 1-frame pop A-B-A does not). Inside the lip / inner-mouth mask
  (outer-lip hulls of the source AND the output landmarks, dilated 0.08 IOD):
     hf_k = mean |D2 - Gauss(D2, 0.03 IOD)|    (detail pop: lip line, teeth, texture)
     br_k = |mean D2|                          (brightness pop)
  The same on the source gives its natural level (codec noise, real articulation the flow misses). Rows: flicker ratio = mean hf out / src,
  brightness ratio; events = runs of frames where hf_out - hf_src > EV x s (s = the source's median hf), strong > 2 EV x s
  (brightness: br_out - br_src > EV / 2 x s_br ...). Events per touched minute.

CHIN SLIDE (the re-rendered lower face drifts / swims relative to the upper face and the jaw / neck line)
  a) flow: per frame, DIS flow source_k -> output_k (the upper face IS the source, so this is the lower face's displacement relative to it)
     in a chin band (below the lower lip to the chin, between the jaw lines; source landmarks) -> median vector, in face axes / IOD:
     lat (sideways: never articulation) and vert. vert is regressed on the jaw-opening difference out - src (landmarks 13 / 14: legitimate
     dub articulation) and only the residual counts. slide_k = |d_k - d_k-1| (lat, vert residual) at 30 fps; events: runs with slide > 0.012
     IOD/frame (strong > 0.025), swim RMS = RMS of d minus its 0.5 s moving median (the low-frequency wander), drift = RMS of d.
  b) landmarks: MediaPipe on the output; chin / jaw points (152, 148, 377, 176, 400, 149, 378) minus the source's, after removing the
     upper-face points' mean offset (tracker noise common to both), face axes / IOD; same velocity / events / RMS rows.
    python scripts/r10/face_metrics.py --video OUT.mp4 --src SRC.mp4 --lm-src SRC_lm.npz [--lm-out OUT.mediapipe.npz] --out x.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "quality"))
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
UPPER = [33, 133, 362, 263, 168, 6, 197, 195, 5, 4, 70, 300, 105, 334]   # eyes, nose bridge / tip, brows
CHIN = [152, 148, 377, 176, 400, 149, 378]
JAW_L, JAW_R = [172, 136, 150, 149, 176, 148], [397, 365, 379, 378, 400, 377]


def read(p, every=1):
    import av
    c = av.open(str(p)); s = c.streams.video[0]; s.thread_type = "AUTO"; fps = float(s.average_rate)
    F = [f.to_ndarray(format="bgr24") for f in c.decode(s)]; c.close(); return F, fps


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)


def axes(p):
    u = p[263, :2] - p[33, :2]; iod = float(np.linalg.norm(u)); u = u / max(iod, 1e-6); v = np.array([-u[1], u[0]]); return u, v, iod


def runs(b):
    out = []; i = 0; n = len(b)
    while i < n:
        if not b[i]: i += 1; continue
        j = i
        while j < n and b[j]: j += 1
        out.append((i, j)); i = j
    return out


def mov_median(x, w):
    y = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w):i + w + 1]; seg = seg[np.isfinite(seg)]
        if len(seg): y[i] = np.median(seg)
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm-src", required=True)
    ap.add_argument("--lm-out", default=None, help="MediaPipe npz of the output (computed + cached next to --out when missing); 'none' = skip b)")
    ap.add_argument("--out", required=True); ap.add_argument("--ev", type=float, default=1.5); ap.add_argument("--frames", default=None)
    ap.add_argument("--slide-ev", type=float, default=0.012); ap.add_argument("--slide-strong", type=float, default=0.025)
    ap.add_argument("--per-frame", action="store_true", help="also store the per-frame series")
    a = ap.parse_args(); t0 = time.perf_counter()
    O, fps = read(a.video); S, fps_s = read(a.src); n = min(len(O), len(S))
    lms = np.load(a.lm_src)["lm"].astype(np.float64); lms = np.concatenate([lms, np.full((max(0, n - len(lms)),) + lms.shape[1:], np.nan)])[:n]
    lmo = None
    if a.lm_out != "none":
        import clean_face_stack as cfs
        p = pathlib.Path(a.lm_out) if a.lm_out else pathlib.Path(a.out).with_suffix(".out_mp.npz")
        lmo = np.load(cfs.run_mediapipe(pathlib.Path(a.video), p))["lm"].astype(np.float64)
        lmo = np.concatenate([lmo, np.full((max(0, n - len(lmo)),) + lmo.shape[1:], np.nan)])[:n]
    H, W = S[0].shape[:2]; k0, k1 = (int(x) for x in a.frames.split(":")) if a.frames else (0, n)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    tscale = fps / 30.0   # per-frame velocities -> per 1/30 s
    hf = np.full((n, 2), np.nan); br = np.full((n, 2), np.nan); touched = np.zeros(n, bool)
    dflow = np.full((n, 2), np.nan); dopen = np.full(n, np.nan); dlm = np.full((n, 2), np.nan)
    brr = np.full((n, 2), np.nan); chb = np.full((n, 2), np.nan); chh = np.full((n, 2), np.nan); shade = np.full(n, np.nan); shade_sd = np.full(n, np.nan)
    def ok(p, idx): return p is not None and np.isfinite(p[idx]).all()
    for k in range(max(1, k0), min(n - 1, k1)):
        p = lms[k]
        if not ok(p, LIPS_OUTER + UPPER + CHIN + JAW_L + JAW_R): continue
        u, v, iod = axes(p); q = p[LIPS_OUTER, :2]; cx, cy = q.mean(0); r = int(0.85 * iod)
        y0, y1, x0, x1 = max(0, int(cy - 0.7 * r)), min(H, int(p[152, 1] + 0.25 * iod)), max(0, int(cx - r)), min(W, int(cx + r))
        if y1 - y0 < 16 or x1 - x0 < 16: continue
        so, ss = O[k][y0:y1, x0:x1], S[k][y0:y1, x0:x1]
        touched[k] = float(np.abs(so.astype(np.int16) - ss.astype(np.int16)).mean()) > 0.5
        # lip / inner-mouth mask: hulls of the source + output lips, dilated
        m = np.zeros((y1 - y0, x1 - x0), np.uint8); pts = [q] + ([lmo[k][LIPS_OUTER, :2]] if lmo is not None and ok(lmo[k], LIPS_OUTER) else [])
        for qq in pts: cv2.fillPoly(m, [np.round(cv2.convexHull(qq.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        rd = max(1, int(0.08 * iod)); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1,) * 2)) > 0
        sg = max(0.8, 0.03 * iod)
        # lip ring without the cavity (inner-lip polygons of both): a brightness pop there is flicker, not the mouth opening differently
        cav = np.zeros_like(m, np.uint8)
        for P_ in [p] + ([lmo[k]] if lmo is not None and ok(lmo[k], LIPS_INNER) else []):
            if ok(P_, LIPS_INNER): cv2.fillPoly(cav, [np.round(P_[LIPS_INNER, :2] - [x0, y0]).astype(np.int32)], 1)
        cav = cv2.dilate(cav, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(0.03 * iod)) + 1,) * 2)) > 0
        ring = m & ~cav
        # a) chin band (below the lower lip to the chin, between the jaw lines; source landmarks)
        cb = np.zeros((y1 - y0, x1 - x0), np.uint8)
        poly = np.concatenate([p[JAW_L[::-1], :2], p[[152], :2], p[JAW_R, :2]], 0)
        top = p[17, :2] + 0.10 * iod * v   # just below the lower lip (17 = its bottom centre)
        band = np.concatenate([[top - 0.35 * iod * u], poly, [top + 0.35 * iod * u]], 0)
        cv2.fillPoly(cb, [np.round(cv2.convexHull(band.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        lipx = np.zeros_like(cb)
        for qq in pts: cv2.fillPoly(lipx, [np.round(cv2.convexHull(qq.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        lipx = cv2.dilate(lipx, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(0.12 * iod)) + 1,) * 2))
        cbm = (cb > 0) & (lipx == 0)
        for c_, V in enumerate((O, S)):   # flow-aligned 2nd temporal difference D2 (pops), on the lips, the lip ring and the chin band
            g = gray(V[k][y0:y1, x0:x1]); acc = np.zeros_like(g)
            for j in (k - 1, k + 1):
                gj = gray(V[j][y0:y1, x0:x1]); F = dis.calc(g.astype(np.uint8), gj.astype(np.uint8), None)
                gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
                acc += cv2.remap(gj, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            D2 = g - 0.5 * acc; h_ = D2 - cv2.GaussianBlur(D2, (0, 0), sg)
            hf[k, c_] = float(np.abs(h_[m]).mean()); br[k, c_] = abs(float(D2[m].mean()))
            if ring.sum() > 20: brr[k, c_] = abs(float(D2[ring].mean()))
            if cbm.sum() > 30: chb[k, c_] = abs(float(cv2.GaussianBlur(D2, (0, 0), 3 * sg)[cbm].mean())); chh[k, c_] = float(np.abs(h_[cbm]).mean())
        if cbm.sum() > 30:   # shading of the chin vs the source (low-pass difference, mean over the band); its wander over time = the 'swimming' shade
            dd_ = cv2.GaussianBlur(gray(so) - gray(ss), (0, 0), 0.1 * iod); shade[k] = float(dd_[cbm].mean()); shade_sd[k] = float(dd_[cbm].std())
        if cbm.sum() > 30:
            F = dis.calc(gray(ss).astype(np.uint8), gray(so).astype(np.uint8), None)   # ss(x) ~ so(x + F)
            d = np.median(F[cbm], 0); dflow[k] = [float(d @ u) / iod, float(d @ v) / iod]
        if lmo is not None and ok(lmo[k], CHIN + UPPER + [13, 14]) and ok(p, [13, 14]):
            off = (lmo[k][UPPER, :2] - p[UPPER, :2]).mean(0); dc = (lmo[k][CHIN, :2] - p[CHIN, :2]).mean(0) - off
            dlm[k] = [float(dc @ u) / iod, float(dc @ v) / iod]
            dopen[k] = (np.linalg.norm(lmo[k][13, :2] - lmo[k][14, :2]) - np.linalg.norm(p[13, :2] - p[14, :2])) / iod
    # ---- lip flicker rows
    res = {"video": a.video, "src": a.src, "fps": fps, "frames": n, "touched_frames": int(touched.sum())}
    T = touched & np.isfinite(hf).all(1); tmin = max(T.sum() / fps / 60, 1e-6)
    if T.sum() > 10:
        s_hf = float(np.median(hf[T, 1])); s_br = float(np.median(br[T, 1])) + 0.05
        ex = hf[:, 0] - hf[:, 1]; exb = br[:, 0] - br[:, 1]
        ev = runs(T & (ex > a.ev * s_hf)); st = runs(T & (ex > 2 * a.ev * s_hf)); evb = runs(T & (exb > 2 * a.ev * s_br)); stb = runs(T & (exb > 4 * a.ev * s_br))
        res["flicker"] = {"hf_out": round(float(hf[T, 0].mean()), 3), "hf_src": round(float(hf[T, 1].mean()), 3), "ratio": round(float(hf[T, 0].mean() / max(hf[T, 1].mean(), 1e-6)), 3),
                          "p95_ratio": round(float(np.percentile(hf[T, 0], 95) / max(np.percentile(hf[T, 1], 95), 1e-6)), 3),
                          "bright_ratio": round(float(br[T, 0].mean() / max(br[T, 1].mean(), 1e-6)), 3),
                          "events": len(ev), "strong": len(st), "bright_events": len(evb), "bright_strong": len(stb),
                          "strong_per_min": round(len(st) / tmin, 2), "events_per_min": round(len(ev) / tmin, 2),
                          "strong_frames": [int(s_) for s_, _ in st][:80], "bright_strong_frames": [int(s_) for s_, _ in stb][:80]}
        Tr = T & np.isfinite(brr).all(1)
        if Tr.sum() > 10:   # lip ring (cavity excluded): the brightness pops that are NOT the mouth opening
            s_r = float(np.median(brr[Tr, 1])) + 0.05; exr = brr[:, 0] - brr[:, 1]; str_ = runs(Tr & (exr > 4 * a.ev * s_r))
            res["flicker"].update({"ring_bright_ratio": round(float(brr[Tr, 0].mean() / max(brr[Tr, 1].mean(), 1e-6)), 3), "ring_bright_strong": len(str_),
                                   "ring_bright_strong_frames": [int(s_) for s_, _ in str_][:80]})
    Tc = T & np.isfinite(chb).all(1)
    if Tc.sum() > 10:   # chin band: brightness / texture pops and the shading wander vs the source
        s_c = float(np.median(chb[Tc, 1])) + 0.05; exc = chb[:, 0] - chb[:, 1]; stc = runs(Tc & (exc > 4 * a.ev * s_c))
        sh = shade.copy(); sh[~Tc] = np.nan; w_ = max(2, int(round(0.5 * fps))); shp = sh - mov_median(sh, w_)
        res["chin_shade"] = {"bright_ratio": round(float(chb[Tc, 0].mean() / max(chb[Tc, 1].mean(), 1e-6)), 3), "tex_ratio": round(float(chh[Tc, 0].mean() / max(chh[Tc, 1].mean(), 1e-6)), 3),
                             "bright_strong": len(stc), "bright_strong_per_min": round(len(stc) / tmin, 2), "shade_wander_rms": round(float(np.nanstd(sh)), 3),
                             "shade_hp_rms": round(float(np.sqrt(np.nanmean(shp ** 2))), 3), "shade_step_rms": round(float(np.sqrt(np.nanmean(np.diff(sh) ** 2))), 3),
                             "bright_strong_frames": [int(s_) for s_, _ in stc][:80]}
    # ---- chin slide rows
    def slide_rows(d, name):
        okf = T & np.isfinite(d).all(1)
        if okf.sum() < 10: return None
        dd = d.copy()
        o_ = okf & np.isfinite(dopen)
        if o_.sum() > 20 and np.std(dopen[o_]) > 1e-4:   # vert residual after the legitimate jaw articulation (opening difference)
            A = np.stack([dopen[o_], np.ones(o_.sum())], 1); cf_, *_ = np.linalg.lstsq(A, dd[o_, 1], rcond=None)
            dd[o_, 1] = dd[o_, 1] - A @ cf_; beta = float(cf_[0])
        else: beta = None
        dd[~okf] = np.nan
        vel = np.full(n, np.nan); vl = np.full(n, np.nan)
        for k in range(1, n):
            if okf[k] and okf[k - 1]: vel[k] = float(np.linalg.norm(dd[k] - dd[k - 1])) / tscale; vl[k] = abs(dd[k, 0] - dd[k - 1, 0]) / tscale
        w = max(2, int(round(0.25 * fps))); lf = np.stack([mov_median(dd[:, c_], w) for c_ in range(2)], 1); hp = dd - lf
        ev = runs(np.nan_to_num(vel) > a.slide_ev); st = runs(np.nan_to_num(vel) > a.slide_strong)
        wl = max(4, int(round(1.0 * fps))); lfl = np.stack([mov_median(dd[:, c_], wl) for c_ in range(2)], 1)
        return {"vel_rms": round(float(np.sqrt(np.nanmean(vel ** 2))), 5), "vel_lat_rms": round(float(np.sqrt(np.nanmean(vl ** 2))), 5),
                "swim_rms": round(float(np.sqrt(np.nanmean((hp ** 2).sum(1)))), 5), "swim_lat_rms": round(float(np.sqrt(np.nanmean(hp[:, 0] ** 2))), 5),
                "wander_rms": round(float(np.sqrt(np.nanmean(((lfl - np.nanmean(lfl, 0)) ** 2).sum(1)))), 5),
                "events": len(ev), "strong": len(st), "strong_per_min": round(len(st) / tmin, 2), "events_per_min": round(len(ev) / tmin, 2),
                "strong_frames": [int(s_) for s_, _ in st][:80], "open_beta": beta}
    res["chin_flow"] = slide_rows(dflow, "flow"); res["chin_lm"] = slide_rows(dlm, "lm") if lmo is not None else None
    res["seconds"] = round(time.perf_counter() - t0, 1)
    if a.per_frame:
        np.savez_compressed(pathlib.Path(a.out).with_suffix(".frames.npz"), hf=hf, br=br, dflow=dflow, dlm=dlm, dopen=dopen, touched=touched, brr=brr, chb=chb, chh=chh, shade=shade, shade_sd=shade_sd)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("== face_metrics", pathlib.Path(a.video).name, json.dumps({k: ({kk: vv for kk, vv in v.items() if not kk.endswith("frames")} if isinstance(v, dict) else v)
                                                                       for k, v in res.items() if k in ("flicker", "chin_flow", "chin_shade", "chin_lm", "touched_frames", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
