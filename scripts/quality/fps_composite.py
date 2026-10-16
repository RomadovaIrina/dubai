#!/usr/bin/env python3
"""Source-frame-rate composite (track B, P0a): the output keeps the SOURCE frames, frame count, timestamps (CFR or VFR) and resolution;
only the lip-synced lower face comes from LatentSync, retimed from the 25 fps master to every source timestamp.

Why: LatentSync runs on a 25 fps master (ffmpeg fps=25: one source frame in six dropped for a 30 fps source) and the pipeline muxed that
master -> 25 fps output, uneven head motion (judder) and spec 9.4 (source fps) violated.

Per source frame k at time t_k (master position m = 25 t_k, between master frames m0 = floor(m) and m1 = m0 + 1, alpha = m - m0):
  LatentSync layer   L(t_k) = flow interpolation between the LatentSync raw frames L[m0], L[m1] at alpha (DIS both ways, Super-SloMo
                     intermediate flows, occlusion-aware; flows cached per master pair); --boundary on first replaces the two LatentSync
                     frames around every 16-frame window boundary by t = 1/3, 2/3 interpolations (C2).
  LS weight          w(t_k) = the clean0 per-master-frame weight (1 inside LATENT_SYNC segments, the 4-frame crossfade ramps at switches,
                     0 elsewhere) linearly interpolated; w = 0 -> the output frame IS the source frame (pixels untouched before encoding).
  mask               clean0's lower-face ellipse (LatentSync bbox formula on the clean MediaPipe landmarks, landmarks interpolated to t_k,
                     smoothing +-2 / +-3 master frames) with an explicit edge: alpha goes 0.1 -> 0.9 over --edge-px output pixels (10-15).
  sharpen            (--ls-sharpen A) unsharp mask of the LatentSync layer: L += A (L - Gauss_sigma(L)), deterministic, no model.
  skin texture       (--skin-texture source) outside a lip mask, the high frequencies come from the SOURCE frame k itself, flow-aligned
                     to the LatentSync geometry (no extra frame, no extra model).
  dub-silent         (--silence still|orig) as temporal_mouth_stabilize.py: OPT1 grows the mask to the lower face, OPT2 passes the source
                     through, both eased over --silence-ramp master frames.
  robustness         any per-frame exception -> the previous valid output frame (or the source frame); counted; the run reports
                     fallback_fraction and exits 3 when it exceeds 2 %.
Output: H.264 crf 13 yuv420p with the source time base / pts (PyAV), then the audio stream of the run output (--manifest "output", the
final mix) COPIED in (md5 identity with the baseline).

    python scripts/quality/fps_composite.py --manifest CLEAN0.manifest.json --ls-raw S0_ls_raw.mp4 --out out.mp4 [--boundary on]
        [--skin-texture source] [--silence none|still|orig] [--edge-px 12] [--threads 12]
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, math, pathlib, subprocess, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
import temporal_mouth_stabilize as tms   # Flow, interp, poly_mask, lip / oval indices, silent_stretches, ramp   # noqa: E402
FPS_M = 25.0; P = 16


def decode_source(path: str):
    import av
    c = av.open(path); s = c.streams.video[0]; frames, pts = [], []
    for fr in c.decode(s):
        frames.append(fr.to_ndarray(format="bgr24")); pts.append(fr.pts)
    tb = s.time_base; rate = s.average_rate; c.close()
    return frames, np.array(pts, np.int64), tb, rate


def master_weights(man: dict, n: int) -> np.ndarray:
    """clean0's per-master-frame LS weight: 1 in LATENT_SYNC segments, the assembly crossfade ramps at LS <-> original switches."""
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"]); w = np.zeros(n, np.float32)
    for s in segs:
        if s["action"] != "LATENT_SYNC": continue
        xf = s.get("crossfade") or {}; L = s["end_frame"] - s["start_frame"]
        for k in range(L):
            a = 1.0
            if xf.get("in") and k < xf["in"]: a = (k + 1) / (xf["in"] + 1)
            if xf.get("out") and k >= L - xf["out"]: a = min(a, (L - k) / (xf["out"] + 1))
            if s["start_frame"] + k < n: w[s["start_frame"] + k] = a
    return w


def ellipse_alpha(bbox, H, W, edge_px: float, top: float = 0.52):
    """clean0's lower-face ellipse (MouthOnlyRunner.mask geometry) with an explicit edge width: Gaussian sigma = edge_px / 2.563 gives a
    0.1 -> 0.9 transition of edge_px pixels across a straight edge. Returns (alpha crop, roi)."""
    import cv2
    x1, y1, x2, y2 = [float(v) for v in bbox]; w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cy = y1 + top * h; cx = (x1 + x2) / 2; axes = (w * 0.55, (y2 + 0.06 * h - cy) * 0.5 + 1); centre = (cx, (cy + y2 + 0.06 * h) / 2)
    sig = edge_px / 2.563; pad = int(math.ceil(3 * sig)) + 2
    xa, xb = int(max(0, centre[0] - axes[0] - pad)), int(min(W, centre[0] + axes[0] + pad + 1)); ya, yb = int(max(0, centre[1] - axes[1] - pad)), int(min(H, centre[1] + axes[1] + pad + 1))
    if xb <= xa or yb <= ya: return None, None
    m = np.zeros((yb - ya, xb - xa), np.float32)
    cv2.ellipse(m, (int(round(centre[0] - xa)), int(round(centre[1] - ya))), (int(round(axes[0])), int(round(axes[1]))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), sig, borderType=cv2.BORDER_CONSTANT), (ya, yb, xa, xb)


def edge_width_px(alpha: np.ndarray) -> float | None:
    """Measured 10-90 % transition width of a soft mask: area(0.1 < a < 0.9) / perimeter of the a = 0.5 contour."""
    import cv2
    band = float(((alpha > 0.1) & (alpha < 0.9)).sum()); cs, _ = cv2.findContours((alpha >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return round(band / per, 2) if per > 0 else None


def main() -> int:
    import cv2
    import clean_face_stack as cfs, face_aware_latentsync_accel as faa
    cfs.guard_nc()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="clean0 run manifest (segments, crossfades, work dir with master_mediapipe.npz, audio)")
    ap.add_argument("--ls-raw", required=True, help="LatentSync raw frames on the 25 fps master (decompose_video_stage.py S0_ls_raw.mp4, full video)")
    ap.add_argument("--out", required=True); ap.add_argument("--boundary", choices=["on", "off"], default="on")
    ap.add_argument("--skin-texture", choices=["off", "source"], default="off"); ap.add_argument("--lip-texture", choices=["off", "on"], default="off")
    ap.add_argument("--tex-sigma", type=float, default=1.5); ap.add_argument("--tex-tau", type=float, default=12.0); ap.add_argument("--lip-dilate", type=float, default=0.12)
    ap.add_argument("--silence", choices=["none", "still", "orig"], default="none"); ap.add_argument("--silence-min-s", type=float, default=0.3); ap.add_argument("--silence-ramp", type=int, default=6)
    ap.add_argument("--gaps", default=None, help="residual-gap JSON (temporal_mouth_stabilize.load_gaps; e.g. track C's list); default: computed from the manifest")
    ap.add_argument("--retime", choices=["flow", "nearest"], default="flow", help="LatentSync layer at a source timestamp: flow interpolation between the two master "
                    "frames, or the nearest master frame (exact LatentSync shapes, the mouth repeats one frame in six for a 30 fps source)")
    ap.add_argument("--ls-sharpen", type=float, default=0.0, help="unsharp-mask amount on the LatentSync layer (0 = off), radius --ls-sharpen-sigma")
    ap.add_argument("--ls-sharpen-sigma", type=float, default=1.2)
    ap.add_argument("--ls-sharpen-scope", choices=["all", "lips"], default="all", help="lips = only inside the lip mask (the skin gets the source texture instead)")
    ap.add_argument("--ls-sharpen-lips", type=float, default=0.0, help="EXTRA unsharp amount inside the outer lip contour of the LatentSync frame (on top of --ls-sharpen)")
    ap.add_argument("--edge-px", type=float, default=12.0); ap.add_argument("--threads", type=int, default=12); ap.add_argument("--crf", type=int, default=13)
    ap.add_argument("--max-fallback", type=float, default=0.02)
    a = ap.parse_args()
    t_all = time.perf_counter(); times = {}
    man = json.loads(pathlib.Path(a.manifest).read_text()); work = pathlib.Path(man["work_dir"]); src = man["input"]
    t0 = time.perf_counter(); S, pts, tb, rate = decode_source(src); times["decode_source"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter(); L = tms.read_video(a.ls_raw); times["decode_ls"] = round(time.perf_counter() - t0, 2)
    H, W = S[0].shape[:2]; nm = len(L)
    if L[0].shape[:2] != (H, W): raise SystemExit(f"LatentSync frames {L[0].shape[:2]} != source {(H, W)}")
    mp = np.load(cfs.run_mediapipe(work / "master_25fps.mp4", work / "master_mediapipe.npz"))["lm"][:nm]
    wm = master_weights(man, nm)
    dets = cfs.ls_detections(mp, W, H, calib=True); dets = faa.smooth_detections(dets, 2)
    bb = np.array([d[0] if d[0] is not None else (np.nan,) * 4 for d in dets], np.float64)
    for k in range(4):   # mask smoothing +-3 (MouthOnlyRunner.boxes) on the master grid
        col = bb[:, k]; ok = np.isfinite(col); sm = col.copy()
        for i in np.flatnonzero(ok): nb = col[max(0, i - 3):i + 4]; sm[i] = np.nanmean(nb)
        bb[:, k] = sm
    silent = tms.silent_stretches(man, nm, 0, a.silence_min_s, gaps=tms.load_gaps(a.gaps) if a.gaps else None) if a.silence != "none" else np.zeros(nm, bool)
    sil_w = tms.ramp(silent, a.silence_ramp)
    mp_ls = np.load(cfs.run_mediapipe(pathlib.Path(a.ls_raw), pathlib.Path(a.ls_raw).with_suffix(".mediapipe.npz")))["lm"][:nm] if (a.skin_texture == "source" or a.ls_sharpen_scope == "lips" or a.ls_sharpen_lips > 0) else None
    # C2 on the LatentSync layer (master grid): two frames around every 16-frame window boundary of every LATENT_SYNC segment
    fl0 = tms.Flow(); nbnd = 0
    if a.boundary == "on":
        t0 = time.perf_counter()
        for s in man["lipsync"]["segments"]:
            if s["action"] != "LATENT_SYNC": continue
            for b in range(s["start_frame"] + P, s["end_frame"], P):
                if b >= nm: continue
                y, z = L[b - 1].astype(np.float32), L[b].astype(np.float32)
                L[b - 1] = np.clip(tms.interp(fl0, y, z, 1 / 3), 0, 255).astype(np.uint8); L[b] = np.clip(tms.interp(fl0, y, z, 2 / 3), 0, 255).astype(np.uint8); nbnd += 1
        times["c2_boundaries"] = round(time.perf_counter() - t0, 2)
    tk = pts.astype(np.float64) * float(tb); mpos = tk * FPS_M
    def lerp(arr, m):
        m0 = int(np.clip(math.floor(m), 0, nm - 1)); m1 = min(m0 + 1, nm - 1); al = float(m - math.floor(m))
        if not (np.isfinite(arr[m0]).all() and np.isfinite(arr[m1]).all()): return arr[m0] if np.isfinite(arr[m0]).all() else arr[m1]
        return (1 - al) * arr[m0] + al * arr[m1]
    import threading
    local = threading.local()
    def flow():
        if not hasattr(local, "f"): local.f = tms.Flow()
        return local.f
    edge_meas = []
    def one(k):
        m = mpos[k]; m0 = int(np.clip(math.floor(m), 0, nm - 1)); m1 = min(m0 + 1, nm - 1); al = float(m - math.floor(m))
        w = float((1 - al) * wm[m0] + al * wm[m1])
        if w <= 0: return S[k], 0, None
        box = lerp(bb, m)
        if not np.isfinite(box).all(): return S[k], 0, None
        alpha, roi = ellipse_alpha(box, H, W, a.edge_px)
        if alpha is None: return S[k], 0, None
        ya, yb, xa, xb = roi; f = flow()
        if a.retime == "nearest":
            mn = m0 if (al < 0.5 or wm[m1] <= 0) else m1; Lk = L[mn][ya:yb, xa:xb].astype(np.float32)
        elif al < 0.04 or m1 == m0 or wm[m1] <= 0: Lk = L[m0][ya:yb, xa:xb].astype(np.float32)
        elif al > 0.96 or wm[m0] <= 0: Lk = L[m1][ya:yb, xa:xb].astype(np.float32)
        else: Lk = tms.interp(f, L[m0][ya:yb, xa:xb].astype(np.float32), L[m1][ya:yb, xa:xb].astype(np.float32), al)
        sk = S[k][ya:yb, xa:xb].astype(np.float32)
        ex = None
        if a.skin_texture == "source" or (a.ls_sharpen > 0 and a.ls_sharpen_scope == "lips"):
            lm_s = lerp(mp, m); lm_l = lerp(mp_ls, m) if mp_ls is not None else None; mw_ = tms.mouth_width(lm_s)
            idx = tms.LIPS_INNER if a.lip_texture == "on" else tms.LIPS_OUTER
            ex = tms.poly_mask(lm_l[idx] if lm_l is not None else None, roi, a.lip_dilate * mw_, 0.08 * mw_)
            if a.lip_texture == "off": ex = np.maximum(ex, tms.poly_mask(lm_s[idx], roi, a.lip_dilate * mw_, 0.08 * mw_))
        if a.ls_sharpen > 0 or a.ls_sharpen_lips > 0:   # deterministic spatial unsharp mask of the (soft) LatentSync layer
            det = Lk - cv2.GaussianBlur(Lk, (0, 0), a.ls_sharpen_sigma); amt = np.full(Lk.shape[:2], a.ls_sharpen, np.float32)
            if a.ls_sharpen_scope == "lips": amt = a.ls_sharpen * ex
            if a.ls_sharpen_lips > 0:
                lm_l2 = lerp(mp_ls, m) if mp_ls is not None else lerp(mp, m); mw2 = tms.mouth_width(lm_l2)
                amt = amt + a.ls_sharpen_lips * tms.poly_mask(lm_l2[tms.LIPS_OUTER], roi, 0.08 * mw2, 0.08 * mw2)
            Lk = Lk + amt[..., None] * det
        if a.skin_texture == "source":
            Lk = tms.skin_texture_transfer(f, [Lk], [sk], [ex], a.tex_sigma, a.tex_tau)[0]
        aw = alpha * w
        if a.silence == "still" and (sil_w[m0] > 0 or sil_w[m1] > 0):
            g = (1 - al) * sil_w[m0] + al * sil_w[m1]; lm_s = lerp(mp, m)
            if np.isfinite(lm_s).all():
                oval = lm_s[tms.FACE_OVAL, :2]; ny = float(lm_s[tms.NOSE_BASE, 1].mean()); low = oval[oval[:, 1] >= ny - 1]
                hull = cv2.convexHull(np.concatenate([low, [[oval[:, 0].min(), ny], [oval[:, 0].max(), ny]]]).astype(np.float32))[:, 0]
                gm = tms.poly_mask(hull, roi, 0, 0) ; gm = cv2.GaussianBlur(gm, (0, 0), a.edge_px / 2.563) * g * w
                aw = np.maximum(aw, gm)
        if a.silence == "orig" and (sil_w[m0] > 0 or sil_w[m1] > 0):
            aw = aw * (1.0 - ((1 - al) * sil_w[m0] + al * sil_w[m1]))
        o = S[k].copy(); o[ya:yb, xa:xb] = np.clip(sk + aw[..., None] * (Lk - sk), 0, 255).round().astype(np.uint8)
        return o, 1, (alpha if k % 50 == 0 else None)
    t0 = time.perf_counter(); out = [None] * len(S); fails = 0; touched = 0
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = {ex.submit(one, k): k for k in range(len(S))}
        for fu in cf.as_completed(futs):
            k = futs[fu]
            try:
                o, t_, al_ = fu.result(); out[k] = o; touched += t_
                if al_ is not None: edge_meas.append(edge_width_px(al_))
            except Exception as e:   # never crash the worker: previous valid frame, counted
                out[k] = None; fails += 1
                if fails <= 3: print(f"  frame {k} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
    for k in range(len(out)):
        if out[k] is None: out[k] = out[k - 1] if k > 0 and out[k - 1] is not None else S[k]
    times["composite"] = round(time.perf_counter() - t0, 2)
    # encode with the source time base / pts, then copy the run's own final audio stream in
    import av
    t0 = time.perf_counter(); outp = pathlib.Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True); vtmp = outp.with_name(".v_" + outp.name)
    c = av.open(str(vtmp), "w"); st = c.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.time_base = tb
    st.options = {"crf": str(a.crf), "preset": "medium"}; st.codec_context.time_base = tb
    for k, o in enumerate(out):
        fr = av.VideoFrame.from_ndarray(o, format="bgr24"); fr.pts = int(pts[k]); fr.time_base = tb
        for p in st.encode(fr): c.mux(p)
    for p in st.encode(): c.mux(p)
    c.close()
    # audio: the run output's own (final-mix) audio stream, copied -> bit-identical to the baseline / clean0 output
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vtmp), "-i", str(man["output"]), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(outp)], check=True)
    vtmp.unlink(missing_ok=True); times["encode_mux"] = round(time.perf_counter() - t0, 2); times["total"] = round(time.perf_counter() - t_all, 2)
    dur = float(tk[-1] - tk[0] + (tk[-1] - tk[-2] if len(tk) > 1 else 0))
    em = [e for e in edge_meas if e]
    rep = {"input": src, "output": str(outp), "source_frames": len(S), "source_fps_avg": float(rate), "master_frames": nm, "ls_touched_frames": touched,
           "fallback_frames": fails, "fallback_fraction": round(fails / max(len(S), 1), 5), "boundaries_interpolated": nbnd, "edge_px_target": a.edge_px,
           "edge_px_measured": {"median": float(np.median(em)) if em else None, "min": min(em) if em else None, "max": max(em) if em else None, "n": len(em)},
           "silent_master_frames": int(silent.sum()), "args": vars(a), "seconds": times, "s_per_video_min": round(times["total"] / (dur / 60), 1)}
    outp.with_suffix(".fps_composite.json").write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("source_frames", "ls_touched_frames", "fallback_frames", "edge_px_measured", "s_per_video_min")}))
    return 3 if rep["fallback_fraction"] > a.max_fallback else 0


if __name__ == "__main__":
    raise SystemExit(main())
