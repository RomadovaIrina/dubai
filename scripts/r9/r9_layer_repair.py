#!/usr/bin/env python3
"""r9 F: repair short mouth glitches IN the generated layer instead of swapping to the source mouth (user on m03: "the lips stick together and
unstick", "the mouth slid"). The m19b tight fallback (fb7) crossfades single flagged frames to the SOURCE mouth with 1-frame ramps; the
source mouth has another shape, so that swap is itself a 1-frame snap (4 of the 6 stick / unstick events in m03's output sit on fb7 peaks).
Flagged frames (LatentSync frames only):
  - the fallback's own flags (fb > --fb-thr),
  - 1-2-frame opening spikes / dips of the layer (s = o - median5(o), |s| > --spike, and the source's own s at t is < half of it),
  - lip-centre slide off the source lip centre (face coordinates) > --slide IOD.
Runs of <= --max-run flagged frames between good generated frames are re-drawn by flow interpolation of the two good neighbours (DIS flow,
bidirectional, blended by time); their landmarks are interpolated likewise. Longer runs keep the source fallback, with smoothed ramps
(Gaussian --ramp-sigma frames) -> --fb-out for the composite.
    python scripts/r9/r9_layer_repair.py --ls L.mp4 --lm L.mediapipe.npz --fb FB.npz --src-lm master_mediapipe.npz --out L_rep.mp4 --fb-out FB_rep.npz"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
_LRGB = __import__("os").environ.get("LS_LAYER_RGB", "0") == "1"  # r13 LS_LAYER_RGB=1: lossless RGB layer (each yuv420p round trip lost ~0.7 L / +0.4 b*)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r8"))
from r8_mouth_flicker import LIPS_OUTER


def opening(lm):
    fh = np.linalg.norm(lm[:, 10, :2] - lm[:, 152, :2], axis=1); return np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / fh


def lipxy(lm):
    iod = np.linalg.norm(lm[:, 33, :2] - lm[:, 263, :2], axis=1); ax = lm[:, 152, :2] - lm[:, 1, :2]; ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    perp = np.stack([-ax[:, 1], ax[:, 0]], 1); mc = lm[:, LIPS_OUTER, :2].mean(1) - lm[:, 1, :2]
    return (mc * perp).sum(1) / iod, (mc * ax).sum(1) / iod


def spike(o):
    s = np.full_like(o, np.nan)
    for t in range(2, len(o) - 2):
        w = o[t - 2:t + 3]
        if np.isfinite(w).sum() >= 4: s[t] = o[t] - np.nanmedian(w)
    return s


def red_stats(frames, lm):
    """Lip-colour (Lab a*) mass in a box around the SOURCE lip centre: weighted centroid offset (IOD) and lateral spread (IOD). A generated mouth
    that slid or doubled moves / spreads the red mass even when landmarks on the layer still follow the drawn mouth (m03 18.9 s)."""
    out = np.full((len(frames), 3), np.nan)
    for k, f in enumerate(frames):
        p = lm[k]
        if not np.isfinite(p[[13, 14, 33, 263, 61, 291]]).all(): continue
        iod = np.linalg.norm(p[33, :2] - p[263, :2]); cx, cy = p[[61, 291, 13, 14], :2].mean(0); r = int(0.8 * iod)
        y0, x0 = max(0, int(cy - 0.6 * r)), max(0, int(cx - r)); box = f[y0:max(y0 + 2, int(cy + 0.6 * r)), x0:max(x0 + 2, int(cx + r))]
        a = cv2.cvtColor(box, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32); w = np.clip(a - np.percentile(a, 60), 0, None); w /= w.sum() + 1e-6
        gy, gx = np.mgrid[0:a.shape[0], 0:a.shape[1]]; mx, my = (w * gx).sum(), (w * gy).sum()
        out[k] = ((mx + x0 - cx) / iod, (my + y0 - cy) / iod, np.sqrt((w * (gx - mx) ** 2).sum()) / iod)
    return out


def face_motion(frames, lm):
    """r15 F: per master frame k, the face's pixel motion from k-1 to k: p90 of the DIS flow magnitude inside the source face box (landmark bbox
    +20 %, half resolution), in IOD. From PIXELS on purpose: MediaPipe VIDEO-mode landmarks lag on a fast head move (m03 19 s: the nose x moves
    2 px while the face moves ~15 px), so the slide / lip-colour detectors fire on a correct generated mouth and the 2-neighbour flow
    interpolation then blends frames across the move (the doubled / lopsided mouth)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); m = np.zeros(len(frames)); prev = None
    for k, f in enumerate(frames):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
        if prev is not None and k < len(lm) and np.isfinite(lm[k][[33, 263]]).all():
            q = lm[k]; iod = np.linalg.norm(q[33, :2] - q[263, :2]); x0, y0 = np.nanmin(q[:, :2], 0); x1, y1 = np.nanmax(q[:, :2], 0); px, py = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            H, W = g.shape; a0, b0 = max(0, int((y0 - py) / 2)), max(0, int((x0 - px) / 2)); a1, b1 = min(H, int((y1 + py) / 2) + 1), min(W, int((x1 + px) / 2) + 1)
            if a1 - a0 > 8 and b1 - b0 > 8:
                fl = dis.calc(np.ascontiguousarray(prev[a0:a1, b0:b1]), np.ascontiguousarray(g[a0:a1, b0:b1]), None); m[k] = 2 * np.percentile(np.linalg.norm(fl, axis=2), 90) / max(iod, 1e-6)
        prev = g
    return m


def main():
    ap = argparse.ArgumentParser()
    for k in ("ls", "lm", "fb", "src-lm", "out", "fb-out"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--src-video", default="", help="the 25 fps master (enables the lip-colour slide detector)")
    ap.add_argument("--red-off", type=float, default=0.065, help="LATERAL lip-colour offset (IOD) that flags a slid mouth (m03: the slides 0.07-0.11, p99 of the video 0.045)"); ap.add_argument("--red-spread", type=float, default=99.0, help="off by default: the spread over-fires on a generator that does not copy the source shape")
    ap.add_argument("--fb-thr", type=float, default=0.5); ap.add_argument("--spike", type=float, default=0.025); ap.add_argument("--slide", type=float, default=0.06)
    ap.add_argument("--spikes", choices=["on", "off"], default="on", help="off = fx-lite: repair only breakages (fallback flags, slides, lip-colour slides), keep 1-2-frame opening spikes")
    ap.add_argument("--motion-thr", type=float, default=0.0, help="r15 F motion-aware repair (needs --src-video; 0 = off): where the face moves faster than this "
                    "(face_motion, IOD per frame, max over the run's span +-2 frames) the flow interpolation is faded out (weight 1 at the threshold, 0 at 2x it) and the generated frames kept")
    ap.add_argument("--interp-fb-err", type=float, default=0.0, help="r15 F interpolation confidence (0 = off): the forward-backward error (p90, IOD) of the p<->q flow in the "
                    "mouth box; above it the redraw fades out (weight 1 at the threshold, 0 at 2x it) and the generated frames are kept. m03 19 s (the doubled mouth): 0.168; "
                    "the good repairs of old03 / m03 / m18: <= 0.01 mostly")
    ap.add_argument("--max-run", type=int, default=3); ap.add_argument("--ramp-sigma", type=float, default=2.0); ap.add_argument("--crf", type=int, default=10); ap.add_argument("--preset", default="medium", help="x264 preset of this intermediate layer (r9 speed: ultrafast)")
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    c = av.open(a.ls); s = c.streams.video[0]; s.thread_type = "AUTO"; rate = s.average_rate
    F = [f.to_ndarray(format="bgr24") for f in c.decode(s)]; c.close(); n = len(F)
    lm = np.load(a.lm)["lm"][:n].astype(np.float64); src = np.load(a.src_lm)["lm"][:n]; fbz = np.load(a.fb); fb = fbz["fb"][:n]; lsm = fbz["ls_mask"][:n].astype(bool)
    pad = lambda v, fill: np.concatenate([v, np.full((n - len(v),) + v.shape[1:], fill, v.dtype)]) if len(v) < n else v   # scores / landmarks a frame or two shorter than the layer (m15)
    fb, lsm, src, lm = pad(fb, 0), pad(lsm, False), pad(src.astype(np.float64), np.nan), pad(lm, np.nan)
    o, so = opening(lm), opening(src); sp, ssp = spike(o), spike(so); x, y = lipxy(lm); sx, sy = lipxy(src)
    f_fb = lsm & (fb > a.fb_thr); f_slide = lsm & (np.hypot(x - sx, y - sy) > a.slide)
    f_spike = lsm & (np.abs(sp) > a.spike) & ~(np.abs(ssp) > 0.5 * np.abs(sp))
    if a.spikes == "off": f_spike[:] = False   # r9 fx-lite: a 1-2-frame opening spike can be a REAL consonant closure / opening (user: m03 r9fx lips look lazy)
    flag = f_fb | f_spike | f_slide
    flag &= np.isfinite(o) | (fb > a.fb_thr)
    nred = 0
    if a.src_video:
        cs = cv2.VideoCapture(a.src_video); SF = []
        for _ in range(n):
            ok, f = cs.read()
            if not ok: break
            SF.append(f)
        if a.motion_thr > 0:
            mot = face_motion(SF, src); mot = np.concatenate([mot, np.zeros(max(0, n - len(mot)))])[:n]
        Sr, Lr = red_stats(SF, src), red_stats(F[:len(SF)], src); m = min(len(Sr), n)
        off = np.abs(Lr[:m, 0] - Sr[:m, 0]); spr = Lr[:m, 2] / np.maximum(1e-6, Sr[:m, 2])   # lateral only: a different mouth SHAPE moves the colour centre up / down, a slide moves it sideways
        red = np.zeros(n, bool); red[:m] = lsm[:m] & np.nan_to_num((off > a.red_off) | (spr > a.red_spread)).astype(bool); nred = int(red.sum()); flag |= red; del SF
    runs, k = [], 0
    while k < n:
        if flag[k]:
            e = k
            while e + 1 < n and flag[e + 1]: e += 1
            runs.append((k, e)); k = e + 1
        else: k += 1
    if "mot" not in locals(): mot = np.zeros(n)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); fb_out = np.zeros(n, np.float32); rep = []; moved = []
    H, W = F[0].shape[:2]; gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    for (b, e) in runs:
        p, q = b - 1, e + 1
        if e - b + 1 <= a.max_run and p >= 0 and q < n and lsm[p] and lsm[q] and not flag[p] and not flag[q]:
            mv = float(mot[max(0, p - 2):min(n, q + 3)].max()) if a.motion_thr > 0 else 0.0
            wm = 1.0 if a.motion_thr <= 0 else float(np.clip(2.0 - mv / a.motion_thr, 0.0, 1.0))   # r15 F: 1 below the threshold, 0 at 2x it
            if wm < 1.0: moved.append([b, e, round(mv, 4), round(wm, 3)])
            if wm <= 0.0: continue   # fast move: keep the generated frames (the generator followed the move; the flags are stale-landmark false alarms)
            gp, gq = cv2.cvtColor(F[p], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[q], cv2.COLOR_BGR2GRAY)
            fpq, fqp = dis.calc(gp, gq, None), dis.calc(gq, gp, None)
            if a.interp_fb_err > 0:   # r15 F: the two anchors do not map onto each other in the mouth (head move, mouth reshaped): interpolation would ghost
                f_ = src[(b + e) // 2] if np.isfinite(src[(b + e) // 2][[33, 263, 61, 291, 13, 14]]).all() else lm[(b + e) // 2]
                if np.isfinite(f_[[33, 263, 61, 291, 13, 14]]).all():
                    iod_ = np.linalg.norm(f_[33, :2] - f_[263, :2]); cx_, cy_ = f_[[61, 291, 13, 14], :2].mean(0); r_ = int(0.9 * iod_)
                    ys, xs = slice(max(0, int(cy_ - r_)), max(1, int(cy_ + r_))), slice(max(0, int(cx_ - r_)), max(1, int(cx_ + r_)))
                    back = cv2.remap(fqp, gx + fpq[..., 0], gy + fpq[..., 1], cv2.INTER_LINEAR)
                    fbe = float(np.percentile(np.linalg.norm((fpq + back)[ys, xs], axis=2), 90) / max(iod_, 1e-6))
                    wc = float(np.clip(2.0 - fbe / a.interp_fb_err, 0.0, 1.0))
                    if wc < 1.0: moved.append([b, e, "fb_err", round(fbe, 4), round(wc, 3)])
                    wm = min(wm, wc)
                if wm <= 0.0: continue
            for k in range(b, e + 1):
                al = (k - p) / (q - p)
                wp = cv2.remap(F[p], gx + al * fqp[..., 0], gy + al * fqp[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                wq = cv2.remap(F[q], gx + (1 - al) * fpq[..., 0], gy + (1 - al) * fpq[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                F[k] = np.clip(wm * ((1 - al) * wp.astype(np.float32) + al * wq.astype(np.float32)) + (1 - wm) * F[k].astype(np.float32), 0, 255).astype(np.uint8)
                lm[k] = wm * ((1 - al) * lm[p] + al * lm[q]) + (1 - wm) * lm[k]
            rep.append([b, e])
        else:
            fb_out[b:e + 1] = 1.0   # a long broken run: the source mouth over the whole run, smooth ramps below
    if a.ramp_sigma > 0 and fb_out.any():   # smoothed ramps for the source fallback that remains (no 1-frame snap)
        k = np.arange(-int(3 * a.ramp_sigma), int(3 * a.ramp_sigma) + 1); g = np.exp(-0.5 * (k / a.ramp_sigma) ** 2)
        fb_out = np.maximum(fb_out, np.minimum(1.0, np.convolve(fb_out, g / g.max() * 0.999, mode="same")) * lsm).astype(np.float32)
    o2 = av.open(a.out, "w"); st = o2.add_stream("libx264rgb" if _LRGB else "libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "bgr24" if _LRGB else "yuv420p"; st.options = ({"qp": "0", "preset": a.preset} if _LRGB else {"crf": str(a.crf), "preset": a.preset})
    for f in F:
        for pk in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")): o2.mux(pk)
    for pk in st.encode(): o2.mux(pk)
    o2.close()
    np.savez_compressed(pathlib.Path(a.out).with_suffix(".mediapipe.npz"), **{**{k: v for k, v in np.load(a.lm).items()}, "lm": lm.astype(np.float32)})
    np.savez_compressed(a.fb_out, fb=fb_out, ls_mask=lsm)
    rp = {"frames": n, "ls_frames": int(lsm.sum()), "flagged": int(flag.sum()), "by_class": {"fallback": int(f_fb.sum()), "opening_spike": int(f_spike.sum()), "landmark_slide": int(f_slide.sum()), "lip_colour": nred}, "lip_colour_flags": nred, "runs": len(runs), "repaired_runs": len(rep), "repaired_frames": int(sum(e - b + 1 for b, e in rep)),
          "fallback_frames_left": int((fb_out > 0.5).sum()), "fb7_frames_gt05": int((fb > 0.5).sum()), "seconds": round(time.perf_counter() - t0, 1)}
    if a.motion_thr > 0: rp["motion"] = {"thr": a.motion_thr, "p95": round(float(np.percentile(mot, 95)), 4), "p99": round(float(np.percentile(mot, 99)), 4)}
    if a.motion_thr > 0 or a.interp_fb_err > 0: rp["faded_runs"] = len(moved); rp["interp_fb_err"] = a.interp_fb_err
    pathlib.Path(a.out).with_suffix(".repair.json").write_text(json.dumps({**rp, "repaired": rep, "motion_faded": moved}, indent=1)); print("== layer_repair", json.dumps(rp), flush=True)


if __name__ == "__main__":
    main()
