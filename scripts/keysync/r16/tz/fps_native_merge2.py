#!/usr/bin/env python3
"""r18 TZ 9.4 / 5.6: the KeySync result at the SOURCE frame rate (and size) = "B4 retime", accepted by eye 13.11 (m18: as steady as A).
Why not B2 (fps_native_merge.py --mode flownear): the 25 fps master is the source with frames dropped, so at a 30 fps source 1 frame in 6 has
no KeySync take; B2's round(u) lands on .5 there and Python's half-to-even alternates prev / next take -> the mouth freeze jumps before /
after the motion every 0.2 s (Irina: "jitters more than A"). B2 also ADDED the delta, so the lips were over-sharp, and its lossy intermediate
+ the final encode shimmered (one extra generation).
retime: every source frame t gets the KeySync mouth at ITS audio time u = t * 25:
    k = floor(u), f = u - k;  f ~ 0 (|f| < --snap) -> take k as is;  else the NEARER take moved by the fractional KeySync->KeySync flow
    (DIS on the KeySync Y planes inside the change ROI, Super SloMo linear approximation F_f0 = -(1-f) f F01 + f^2 F10, F_f1 = (1-f)^2 F01 -
    f (1-f) F10), Lanczos resampling, no cross-fade (B1's softness)
    lo grid: composite(src_lo, take, M) with M = fps_native_merge's feathered change mask of that take
    out = hi + up(M * (take - down(hi)))     (hi size == lo size: exactly the composite; hi larger: the change only, the source detail stays)
The flow is the DUBBED mouth's own motion (not the original lips); the head pose of take k is the source frame the master picked, so the
head-time error is < 0.5 source frame at any rate (60 / 30 / 29.97 / 24 / 15 fps). The hi frames keep their pts (output fps = source fps).
Encoding: the output is the final picture unless subtitles re-encode it (then pass a near-lossless --crf, e.g. 10).
    python fps_native_merge2.py --hi SRC_NATIVE.mp4 --lo-src master_25fps.mp4 --lo-comp ks_face.mp4 --out OUT.mp4 [--crf 18 --preset medium] [--json s.json]
Exit 0 = OUT written."""
from __future__ import annotations
import argparse, json, pathlib, queue, sys, threading, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "r15"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hires_merge import yuv_frames, x264_colour   # noqa: E402


class Takes:
    """the KeySync takes: planes C + the feathered change mask M (fps_native_merge / hires_merge's mask: same parameters, built at half size
    for frames >= 400 px, then upsampled: it is feathered anyway, 4x less CPU), computed in a background thread (cv2 / numpy release the
    GIL: the mask work overlaps the warp + the encode), a few kept for the retime's k / k+1"""
    def __init__(s, lo_src, lo_comp, a):
        s.a = a; s.K = None; s.cache = {}; s.n_read = 0; s.end = None; s.q = queue.Queue(maxsize=8)
        threading.Thread(target=s._run, args=(lo_src, lo_comp), daemon=True).start()

    def _run(s, lo_src, lo_comp):
        try:
            for (_, sp), (_, cp) in zip(yuv_frames(lo_src), yuv_frames(lo_comp)): s.q.put(s.layer(sp, cp))
            s.q.put(None)
        except BaseException as e: s.q.put(e)

    def get(s, k):
        k = max(0, k)
        if s.end is not None and k >= s.end: k = s.end - 1
        while k not in s.cache and s.end is None:
            L = s.q.get()
            if isinstance(L, BaseException): raise L
            if L is None:
                s.end = s.n_read; k = min(k, s.end - 1); break
            s.cache[s.n_read] = L; s.n_read += 1
            for old in [i for i in s.cache if i < s.n_read - 3]: del s.cache[old]
        return s.cache.get(k), k

    def layer(s, sp, cp):
        a = s.a; h, w = sp[0].shape; ds = 2 if min(h, w) >= 400 else 1; hs, ws = h // ds, w // ds
        if s.K is None:
            Lm = max(h, w); kc = max(3, int(round(a.close * Lm / ds)) | 1); kd = max(1, int(round(a.dilate * Lm / ds)))
            s.K = dict(sg=max(1.0, a.feather * Lm / ds), sb=max(1.0, a.blur * Lm / ds), pad=int(round(a.dilate * Lm)) + int(3 * max(1.0, a.feather * Lm)) + 2,
                       Kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)), Kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1)))
        K = s.K; dY = np.abs(cp[0].astype(np.int16) - sp[0].astype(np.int16)).astype(np.float32)
        dC = np.maximum(np.abs(cp[1].astype(np.int16) - sp[1].astype(np.int16)), np.abs(cp[2].astype(np.int16) - sp[2].astype(np.int16))).astype(np.float32) * 1.5
        if ds > 1: dY = cv2.resize(dY, (ws, hs), interpolation=cv2.INTER_AREA)
        mc = dC if dC.shape == (hs, ws) else cv2.resize(dC, (ws, hs), interpolation=cv2.INTER_NEAREST)
        mag = cv2.GaussianBlur(np.maximum(dY, mc), (0, 0), K["sb"])
        sub = mag[::4 // ds, ::4 // ds]; med = float(np.median(sub)); mad = float(np.median(np.abs(sub - med))) * 1.4826
        m0 = (mag > max(a.thr, med + a.k * mad)).astype(np.uint8)
        if int(m0.sum()) * ds * ds < a.min_px: return {"roi": None, "Y": sp[0]}
        m0 = cv2.dilate(cv2.morphologyEx(m0, cv2.MORPH_CLOSE, K["Kc"]), K["Kd"]); ys, xs = np.nonzero(m0); p = K["pad"]
        y0, y1 = max(0, ys.min() * ds - p) & ~1, min(h, ys.max() * ds + p + 2) & ~1; x0, x1 = max(0, xs.min() * ds - p) & ~1, min(w, xs.max() * ds + p + 2) & ~1
        M = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), K["sg"])
        if ds > 1: M = cv2.resize(M, (w, h), interpolation=cv2.INTER_LINEAR)
        return {"roi": (y0, y1, x0, x1), "Y": sp[0], "C": [q.astype(np.float32) for q in cp], "M": M}


INTERP = cv2.INTER_LANCZOS4


def cut(L, box):
    y0, y1, x0, x1 = box
    return [L["C"][0][y0:y1, x0:x1], L["C"][1][y0 // 2:y1 // 2, x0 // 2:x1 // 2], L["C"][2][y0 // 2:y1 // 2, x0 // 2:x1 // 2]], L["M"][y0:y1, x0:x1]


def warp(L, F, box):
    """backward-warp take L's planes (Y full, U/V half) and mask by flow F (on box, Y px) -> planes + mask cut to box"""
    y0, y1, x0, x1 = box; gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32); mx, my = gx + F[..., 0], gy + F[..., 1]
    Y = cv2.remap(L["C"][0], mx, my, INTERP, borderMode=cv2.BORDER_REPLICATE)
    Mw = cv2.remap(L["M"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hw = ((x1 - x0) // 2, (y1 - y0) // 2); mxc, myc = cv2.resize(mx, hw) / 2, cv2.resize(my, hw) / 2
    UV = [cv2.remap(L["C"][i], mxc, myc, INTERP, borderMode=cv2.BORDER_REPLICATE) for i in (1, 2)]
    return [Y, *UV], Mw


def apply(hp, Cw, Mw, box, lo_hw):
    """out = hi + up(M (take - down(hi))) on box (lo Y px); same size -> the plain composite"""
    H, W = hp[0].shape; h, w = lo_hw; y0, y1, x0, x1 = box; planes = [q.copy() for q in hp]
    if (H, W) == (h, w):
        for q in range(3):
            s = 1 if q == 0 else 2; m = Mw if q == 0 else cv2.resize(Mw, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
            reg = planes[q][y0 // s:y1 // s, x0 // s:x1 // s].astype(np.float32); ch, cw_ = reg.shape
            planes[q][y0 // s:y1 // s, x0 // s:x1 // s] = np.clip(reg + (Cw[q][:ch, :cw_] - reg) * m[:ch, :cw_] + 0.5, 0, 255).astype(np.uint8)
        return planes
    sy, sx = H / h, W / w; Y0, Y1 = int(round(y0 * sy / 2)) * 2, min(H, int(round(y1 * sy / 2)) * 2); X0, X1 = int(round(x0 * sx / 2)) * 2, min(W, int(round(x1 * sx / 2)) * 2)
    for q in range(3):
        s = 1 if q == 0 else 2; lh, lw = Cw[q].shape; m = Mw if q == 0 else cv2.resize(Mw, (lw, lh), interpolation=cv2.INTER_AREA)
        reg = planes[q][Y0 // s:Y1 // s, X0 // s:X1 // s]; down = cv2.resize(reg, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
        D = (Cw[q] - down) * m[:lh, :lw]; up = cv2.resize(D, (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_CUBIC)
        planes[q][Y0 // s:Y1 // s, X0 // s:X1 // s] = np.clip(reg.astype(np.float32) + up + 0.5, 0, 255).astype(np.uint8)
    return planes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", required=True); ap.add_argument("--lo-src", required=True); ap.add_argument("--lo-comp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lo-fps", type=float, default=25.0)
    ap.add_argument("--thr", type=float, default=3.0); ap.add_argument("--k", type=float, default=6.0); ap.add_argument("--blur", type=float, default=0.004)
    ap.add_argument("--min-px", type=int, default=64); ap.add_argument("--close", type=float, default=0.015); ap.add_argument("--dilate", type=float, default=0.008)
    ap.add_argument("--feather", type=float, default=0.006); ap.add_argument("--flow-pad", type=int, default=24)
    ap.add_argument("--interp", choices=["cubic", "lanczos"], default="lanczos"); ap.add_argument("--snap", type=float, default=0.02, help="|u - k| below this (takes) -> take k unwarped")
    ap.add_argument("--flow-preset", choices=["ultrafast", "fast", "medium"], default="fast", help="DIS preset (m18: fast = medium by every number, flow 2.3x faster)")
    ap.add_argument("--crf", type=int, default=18); ap.add_argument("--preset", default="medium"); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--json", default="")
    a = ap.parse_args(); t0 = time.perf_counter()
    global INTERP; INTERP = cv2.INTER_LANCZOS4 if a.interp == "lanczos" else cv2.INTER_CUBIC
    import av
    Lz = Takes(a.lo_src, a.lo_comp, a); dis = cv2.DISOpticalFlow_create({"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                                                                    "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[a.flow_preset]); fc: dict = {}
    hi_i: dict = {}; G_hi = yuv_frames(a.hi, hi_i)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".v_" + out.name)
    pts0 = None; c = st = None; n = exact = warped = 0; t_flow = 0.0; lo_hw = None
    try:
        for pts, hp in G_hi:
            H, W = hp[0].shape; pts0 = pts if pts0 is None else pts0
            if c is None:
                if H % 2 or W % 2: raise SystemExit(f"odd source size {W}x{H}")
                c = av.open(str(tmp), "w"); st = c.add_stream("libx264", rate=hi_i["rate"]); st.width, st.height = W, H; st.pix_fmt = "yuv420p"
                st.time_base = hi_i["tb"]; st.codec_context.time_base = hi_i["tb"]; st.options = {"crf": str(a.crf), "preset": a.preset, "threads": str(a.threads)}
                col = x264_colour(a.lo_comp)
                if col: st.options = {**st.options, "x264-params": col}
            u = float((pts - pts0) * hi_i["tb"]) * a.lo_fps; k = int(np.floor(u + 1e-6)); f = u - k
            if f > 1 - a.snap: k, f = k + 1, 0.0
            A, ka = Lz.get(k); B, kb = (Lz.get(k + 1) if f >= a.snap else (None, ka))
            if B is not None and kb == ka: B = None   # past the last take
            lo_hw = lo_hw or A["Y"].shape
            planes = hp
            if B is not None and A["roi"] is not None and B["roi"] is not None:
                h, w = A["Y"].shape; p = a.flow_pad
                box = (max(0, min(A["roi"][0], B["roi"][0]) - p) & ~1, min(h, max(A["roi"][1], B["roi"][1]) + p) & ~1,
                       max(0, min(A["roi"][2], B["roi"][2]) - p) & ~1, min(w, max(A["roi"][3], B["roi"][3]) + p) & ~1)
                if fc.get("k") != (ka, box):
                    y0, y1, x0, x1 = box; ia = np.ascontiguousarray(A["C"][0][y0:y1, x0:x1].astype(np.uint8)); ib = np.ascontiguousarray(B["C"][0][y0:y1, x0:x1].astype(np.uint8))
                    tf = time.perf_counter(); fc = {"k": (ka, box), "F01": dis.calc(ia, ib, None), "F10": dis.calc(ib, ia, None)}; t_flow += time.perf_counter() - tf
                F01, F10 = fc["F01"], fc["F10"]
                Cw, Mw = warp(A, -(1 - f) * f * F01 + f * f * F10, box) if f <= 0.5 else warp(B, (1 - f) ** 2 * F01 - f * (1 - f) * F10, box)
                planes = apply(hp, Cw, Mw, box, lo_hw); warped += 1
            else:
                L = B if (B is not None and f > 0.5 and B["roi"] is not None) or A["roi"] is None else A
                if L is not None and L["roi"] is not None:
                    Cw, Mw = cut(L, L["roi"]); planes = apply(hp, Cw, Mw, L["roi"], lo_hw); exact += 1
            fr = av.VideoFrame.from_ndarray(np.concatenate([q.ravel() for q in planes]).reshape(H * 3 // 2, W), format="yuv420p")
            fr.pts = pts; fr.time_base = hi_i["tb"]
            for pk in st.encode(fr): c.mux(pk)
            n += 1
        if c is None: raise SystemExit("hi has no frames")
        for pk in st.encode(): c.mux(pk)
        c.close(); c = None; tmp.replace(out)
    finally:
        if c is not None: c.close()
        tmp.unlink(missing_ok=True)
    el = time.perf_counter() - t0
    s = {"mode": "retime", "frames": n, "exact": exact, "warped": warped, "untouched": n - exact - warped, "hi_fps": round(float(hi_i["rate"]), 3),
         "hi_size": [W, H], "lo_size": [lo_hw[1], lo_hw[0]] if lo_hw else None, "lo_frames": Lz.n_read, "flow_preset": a.flow_preset, "seconds": round(el, 1), "flow_s": round(t_flow, 1)}
    print("== fps_native_merge2", json.dumps(s), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except SystemExit as e:
        if e.code not in (0, None): print("== fps_native_merge2 FAILED:", e.code, flush=True); sys.exit(1)
        raise
