#!/usr/bin/env python3
"""r10 FACE (mgmt: 'the lips flicker'): temporal stabiliser of the mouth detail on the FINAL composite, at the source frame rate.

r9_hf_stabilise.py works on the 25 fps layer; the composite then adds its own frame-to-frame pops (25 -> 30 retime blends, lip sharpening,
registration). Here, per output frame k, inside the re-rendered mouth area only (pixels the composite changed vs the source, feathered, within a
box around the lips): neighbours k-R..k+R are DIS-flow-aligned to k; the frame keeps its own low frequencies (shape / opening / motion = the
lip-sync) and takes the per-pixel MEDIAN of the aligned high frequencies (sigma --sigma IOD). With --bright, the mouth area's mean luma is also
pulled to the median of the aligned neighbours' (a 1-frame brightness pop), capped at --bright-max grey levels.
The audio stream of the input is copied.
    python scripts/r10/r10_out_stab.py --video FINAL.mp4 --src SRC.mp4 --lm SRC_lm.npz --out FINAL_ostab.mp4 [--radius 2 --sigma 0.035 --bright on]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, time
import numpy as np, cv2

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=int, default=2); ap.add_argument("--radius-s", type=float, default=0.0, help="r10: radius in SECONDS (overrides --radius: round(fps x this), "
                    "so 25 / 30 / 60 fps sources get the same temporal support; 0 = use --radius)"); ap.add_argument("--sigma", type=float, default=0.035, help="HF / LF split, fraction of the IOD")
    ap.add_argument("--box", type=float, default=0.75, help="half size of the mouth box, IOD"); ap.add_argument("--dilate", type=float, default=0.18, help="lips hull dilation, IOD")
    ap.add_argument("--bright", choices=["on", "off"], default="off"); ap.add_argument("--bright-max", type=float, default=6.0)
    ap.add_argument("--hf", choices=["on", "off"], default="on", help="median of the aligned neighbours' high frequencies (detail pops)")
    ap.add_argument("--shade", choices=["on", "off"], default="off", help="r10: shading pops (LatentSync's per-frame brightness of the lips / chin skin): a SMOOTH "
                    "correction field = Gauss(median of the aligned neighbours' low frequencies - own, --shade-sigma IOD), capped at --shade-max; the shape "
                    "(edges, opening) is not touched, only slow shading")
    ap.add_argument("--keep-cavity", choices=["on", "off"], default="off", help="r10: no shading correction on the dark inner mouth (inside the lips hull, "
                    "luma < 0.6 x the lips' median): its darkness IS the articulation, and a failed flow at a fast opening would otherwise flatten it")
    ap.add_argument("--chin-reg", choices=["on", "off"], default="off", help="r10 (mgmt: 'the chin slides'): the re-rendered chin's SIDEWAYS displacement "
                    "vs the source (median DIS flow source -> output in the chin band, along the eye line; never articulation, the head is the source's) "
                    "is measured per frame, smoothed over +-1 frame and warped back, weight 1 on the chin band fading to 0 toward the lips and the paste edge")
    ap.add_argument("--chin-reg-vert", type=float, default=0.0, help="r10: also remove this fraction of the vertical displacement's deviation from its "
                    "0.5 s running median (slow vertical swim; the articulation rides on top at speech rate); 0 = lateral only")
    ap.add_argument("--shade-model", choices=["field", "poly0", "poly1"], default="field", help="r10: field = the smooth per-pixel correction (can move a "
                    "low-contrast lip EDGE where the flow misaligns: stick / slide rows); poly0 = one brightness offset per channel over the region; poly1 = "
                    "offset + linear gradient (x, y): a parametric shading change cannot move an edge")
    ap.add_argument("--shade-sigma", type=float, default=0.12); ap.add_argument("--shade-max", type=float, default=10.0)
    ap.add_argument("--region", choices=["lips", "paste"], default="lips", help="lips = near the lips (hull + --dilate); paste = every re-rendered pixel "
                    "from the nose base to below the chin (the chin shading too)")
    ap.add_argument("--crf", type=int, default=13); ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    c = av.open(a.video); s = c.streams.video[0]; s.thread_type = "AUTO"; rate = s.average_rate; tb = s.time_base
    O, pts = [], []
    for f in c.decode(s): O.append(f.to_ndarray(format="bgr24")); pts.append(f.pts)
    c.close()
    c = av.open(a.src); s2 = c.streams.video[0]; s2.thread_type = "AUTO"; S = [f.to_ndarray(format="bgr24") for f in c.decode(s2)]; c.close()
    n = min(len(O), len(S)); H, W = O[0].shape[:2]
    if a.radius_s > 0: a.radius = max(1, int(round(float(rate) * a.radius_s)))
    lm = np.load(a.lm)["lm"].astype(np.float64); lm = np.concatenate([lm, np.full((max(0, n - len(lm)),) + lm.shape[1:], np.nan)])[:n]
    touched = np.zeros(n, bool)
    for k in range(n): touched[k] = np.abs(O[k][::4, ::4].astype(np.int16) - S[k][::4, ::4].astype(np.int16)).max() > 3
    stats = {"stabilised": 0, "bright_mean_abs": []}

    JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]

    def chin_geom(k):   # (box, chin weight map, u, v, iod) of frame k from the source landmarks, or None
        p = lm[k]
        if not np.isfinite(p[LIPS_OUTER + JAW + [33, 263, 17]]).all(): return None
        u = p[263, :2] - p[33, :2]; iod = float(np.linalg.norm(u)); u = u / max(iod, 1e-6); v = np.array([-u[1], u[0]])
        cx = float(p[152, 0]); y0, y1 = max(0, int(p[17, 1] - 0.15 * iod)), min(H, int(p[152, 1] + 0.3 * iod)); x0, x1 = max(0, int(cx - 1.0 * iod)), min(W, int(cx + 1.0 * iod))
        if y1 - y0 < 12 or x1 - x0 < 12: return None
        poly = np.concatenate([p[JAW, :2], [p[17, :2] + 0.08 * iod * v + 0.4 * iod * u, p[17, :2] + 0.08 * iod * v - 0.4 * iod * u]], 0)
        cm = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(cm, [np.round(cv2.convexHull(poly.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        lipm = np.zeros_like(cm); cv2.fillPoly(lipm, [np.round(cv2.convexHull(p[LIPS_OUTER, :2].astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        dl = cv2.distanceTransform(1 - lipm, cv2.DIST_L2, 3); wl = np.clip((dl - 0.05 * iod) / (0.12 * iod), 0, 1)   # 0 at the lips -> 1 at 0.17 IOD below
        return (y0, y1, x0, x1), cm.astype(np.float32) * wl, u, v, iod

    cdisp = np.full((n, 2), np.nan)   # chin displacement source -> output, px, (along u, along v)
    if a.chin_reg == "on":
        def chin_d(k):
            if not touched[k]: return None
            g_ = chin_geom(k)
            if g_ is None: return None
            (y0, y1, x0, x1), cw, u, v, iod = g_
            dd = np.abs(O[k][y0:y1, x0:x1].astype(np.int16) - S[k][y0:y1, x0:x1].astype(np.int16)).max(2) > 2
            sel = (cw > 0.5) & dd
            if sel.sum() < 30: return None
            dis_ = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            F = dis_.calc(cv2.cvtColor(S[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(O[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None)
            d = np.median(F[sel], 0); return float(d @ u), float(d @ v)
        import concurrent.futures as cf0
        with cf0.ThreadPoolExecutor(max_workers=a.threads) as ex0:
            for k, r_ in zip(range(n), ex0.map(chin_d, range(n))):
                if r_ is not None: cdisp[k] = r_
        sm = cdisp.copy()
        for k in range(n):   # +-1 frame smoothing of the measurement (flow noise)
            nb = [cdisp[j] for j in (k - 1, k, k + 1) if 0 <= j < n and np.isfinite(cdisp[j]).all()]
            if np.isfinite(cdisp[k]).all() and nb: sm[k] = np.mean(nb, 0)
        if a.chin_reg_vert > 0:
            w_ = max(2, int(round(0.25 * float(rate)))); vv = sm[:, 1].copy(); base = np.full(n, np.nan)
            for k in range(n):
                seg = vv[max(0, k - w_):k + w_ + 1]; seg = seg[np.isfinite(seg)]
                if len(seg): base[k] = np.median(seg)
            sm[:, 1] = a.chin_reg_vert * (vv - base)
        else: sm[:, 1] = 0.0
        cdisp = sm; stats["chin_lat_px_rms"] = round(float(np.sqrt(np.nanmean(cdisp[:, 0] ** 2))), 3)

    def chin_fix(k, o):   # warp the output's chin back by the measured displacement (only where the composite changed pixels)
        if not np.isfinite(cdisp[k]).all(): return o
        g_ = chin_geom(k)
        if g_ is None: return o
        (y0, y1, x0, x1), cw, u, v, iod = g_
        dvec = cdisp[k, 0] * u + cdisp[k, 1] * v
        diff = cv2.GaussianBlur(np.abs(O[k][y0:y1, x0:x1].astype(np.float32) - S[k][y0:y1, x0:x1].astype(np.float32)).max(2), (0, 0), max(1.0, 0.05 * iod))
        wt = cv2.GaussianBlur(cw, (0, 0), max(1.0, 0.06 * iod)) * np.clip((diff - 1.0) / 3.0, 0, 1)
        gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        crop = o[y0:y1, x0:x1].astype(np.float32)
        wc = cv2.remap(crop, gx + wt * float(dvec[0]), gy + wt * float(dvec[1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        o = o.copy(); o[y0:y1, x0:x1] = np.clip(wc, 0, 255).round().astype(np.uint8); return o

    def one(k):
        p = lm[k]
        if not touched[k] or not np.isfinite(p[LIPS_OUTER + [33, 263]]).all(): return None
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); q = p[LIPS_OUTER, :2]; cx, cy = q.mean(0); r = int(a.box * iod)
        y0, y1, x0, x1 = max(0, int(cy - r)), min(H, int(cy + 1.1 * r)), max(0, int(cx - r)), min(W, int(cx + r))
        if a.region == "paste" and np.isfinite(p[152]).all():
            r = int(1.1 * iod); y0, y1, x0, x1 = max(0, int(cy - 0.75 * iod)), min(H, int(p[152, 1] + 0.35 * iod)), max(0, int(cx - r)), min(W, int(cx + r))
        if y1 - y0 < 16 or x1 - x0 < 16: return None
        ck = O[k][y0:y1, x0:x1].astype(np.float32)
        # where: the composite changed the pixels (|out - src| blurred) AND near the lips (hull dilated), feathered
        diff = cv2.GaussianBlur(np.abs(ck - S[k][y0:y1, x0:x1].astype(np.float32)).max(2), (0, 0), max(1.0, 0.05 * iod))
        hm = np.zeros(ck.shape[:2], np.uint8); cv2.fillPoly(hm, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        rd = max(1, int(a.dilate * iod)); hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1,) * 2)).astype(np.float32)
        m = (cv2.GaussianBlur(hm, (0, 0), max(1.0, 0.06 * iod)) if a.region == "lips" else 1.0) * np.clip((diff - 1.0) / 3.0, 0, 1)
        if a.region == "paste": m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.04 * iod))
        if m.max() < 0.05: return None
        sg = max(0.8, a.sigma * iod); lf = cv2.GaussianBlur(ck, (0, 0), sg); gk = cv2.cvtColor(O[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        lfs = [lf]; hfs = [ck - lf]; lums = [float((cv2.cvtColor(np.clip(lf, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) * m).sum() / max(m.sum(), 1e-3))]
        for j in range(k - a.radius, k + a.radius + 1):
            if j == k or j < 0 or j >= n or not touched[j]: continue
            cj = O[j][y0:y1, x0:x1]; fl = dis.calc(gk, cv2.cvtColor(cj, cv2.COLOR_BGR2GRAY), None)
            wj = cv2.remap(cj.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            lj = cv2.GaussianBlur(wj, (0, 0), sg); hfs.append(wj - lj); lfs.append(lj)
            lums.append(float((cv2.cvtColor(np.clip(lj, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) * m).sum() / max(m.sum(), 1e-3)))
        if len(hfs) < 3: return None
        new = lf + (np.median(np.stack(hfs), axis=0) if a.hf == "on" else hfs[0]); db = 0.0
        if a.shade == "on":   # smooth shading correction toward the aligned temporal median (no shape change)
            ssg = max(1.0, a.shade_sigma * iod); dl = np.median(np.stack(lfs), axis=0) - lf
            ms = m
            if a.keep_cavity == "on":
                hl = np.zeros(ck.shape[:2], np.uint8); cv2.fillPoly(hl, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
                hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(0.1 * iod)) + 1,) * 2)) > 0
                yk = cv2.cvtColor(np.clip(ck, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if hl.sum() > 20:
                    cavm = (hl & (yk < 0.6 * np.median(yk[hl]))).astype(np.float32); cavm = cv2.GaussianBlur(cv2.dilate(cavm, np.ones((3, 3), np.uint8)), (0, 0), max(1.0, 0.02 * iod))
                    ms = m * (1 - np.clip(cavm * 1.5, 0, 1))
            wsum = cv2.GaussianBlur(ms.astype(np.float32), (0, 0), ssg) + 1e-3   # normalised inside the mask: the untouched source around does not dilute it
            if a.shade_model == "field":
                fld = np.stack([cv2.GaussianBlur(dl[..., c_] * ms, (0, 0), ssg) for c_ in range(3)], -1) / wsum[..., None]
            else:   # weighted least squares of the per-pixel difference on a low-order model (subsampled), evaluated on the box
                yy_, xx_ = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32); xn = (xx_ - (cx - x0)) / iod; yn = (yy_ - (cy - y0)) / iod
                cols = [np.ones_like(xn)] + ([xn, yn] if a.shade_model == "poly1" else []); Xf = np.stack(cols, -1)
                sel = ms[::2, ::2] > 0.3
                if sel.sum() < 20: fld = np.zeros_like(dl)
                else:
                    Xs = Xf[::2, ::2][sel]; ws = np.sqrt(ms[::2, ::2][sel])[:, None]; coef = np.linalg.lstsq(Xs * ws, dl[::2, ::2][sel] * ws, rcond=None)[0]
                    fld = (Xf.reshape(-1, Xf.shape[-1]) @ coef).reshape(dl.shape).astype(np.float32)
            if a.keep_cavity == "on": fld = fld * (ms / np.maximum(m, 1e-3))[..., None]
            new = new + np.clip(fld, -a.shade_max, a.shade_max); db = float(np.abs(fld[m > 0.5]).mean()) if (m > 0.5).any() else 0.0
        if a.bright == "on": db = float(np.clip(np.median(lums) - lums[0], -a.bright_max, a.bright_max)); new = new + db
        o = O[k].copy(); o[y0:y1, x0:x1] = np.clip(ck + m[..., None] * (new - ck), 0, 255).round().astype(np.uint8)
        return o, db

    import concurrent.futures as cf
    out = list(O[:n])
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        for k, r_ in zip(range(n), ex.map(one, range(n))):
            if r_ is not None: out[k] = r_[0]; stats["stabilised"] += 1; stats["bright_mean_abs"].append(abs(r_[1]))
            if a.chin_reg == "on": out[k] = chin_fix(k, out[k])
    outp = pathlib.Path(a.out); vtmp = outp.with_name(".v_" + outp.name)
    o = av.open(str(vtmp), "w"); st = o.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.time_base = tb
    st.options = {"crf": str(a.crf), "preset": "medium"}; st.codec_context.time_base = tb
    for k, f in enumerate(out):
        fr = av.VideoFrame.from_ndarray(f, format="bgr24"); fr.pts = int(pts[k]); fr.time_base = tb
        for pk in st.encode(fr): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close()
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vtmp), "-i", a.video, "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", str(outp)])
    vtmp.unlink(missing_ok=True)
    rep = {"frames": n, "touched": int(touched.sum()), "stabilised": stats["stabilised"], "bright_mean_abs": round(float(np.mean(stats["bright_mean_abs"])), 3) if stats["bright_mean_abs"] else 0, "chin_lat_px_rms": stats.get("chin_lat_px_rms"),
           "seconds": round(time.perf_counter() - t0, 1), "args": vars(a)}
    outp.with_suffix(".ostab.json").write_text(json.dumps(rep, indent=1)); print("== out_stab", json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
