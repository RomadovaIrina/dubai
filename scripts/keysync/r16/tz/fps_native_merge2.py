#!/usr/bin/env python3
"""r18 TZ 5.6 / 9.4: B3 = KeySync at the source frame rate WITHOUT the 5 Hz mouth freeze of B2.
Why B2 jitters (m18, 30 fps): the 25 fps master is the source with every 6th frame dropped (ffmpeg fps=25 picks source frame j(k) ~ round(1.2 k)),
so 5 of 6 source frames have their OWN KeySync take (exact head pose) and the 6th has none. B2 shows the previous take again (moved by the
source flow): the mouth shape freezes one frame in six while the head keeps moving -> the mouth stutters against the face at 5 Hz.
A (25 fps) freezes the WHOLE frame, which reads as ordinary 25 fps.
B3: pass 1 matches every master frame k to its source frame j(k) (min MAD near 1.2 k), pass 2 per source frame i:
    j(k) == i        out = composite(hi_i, comp_k, M_k)                  (the exact take; no warp)
    j(k) < i < j(k+1), f = (i - j(k)) / (j(k+1) - j(k)):
       half   out = composite(hi_i, W(comp_k, F_f0), W(M_k))   comp_k moved f of the way along the KeySync flow k->k+1 (lips AND head morph,
                                                                 no cross-fade: stays sharp); nearer take used (k+1 with F_f1 when f > 0.5)
       blend  out = composite(hi_i, (1-f) W(comp_k) + f W(comp_k+1))   both takes moved to f and cross-faded (Super SloMo linear flow approx)
    composite(h, c, m) = h (1 - m) + c m (m: the feathered change mask of hires_merge; one source of truth with fps_native_merge)
Flow = DIS on the KeySync Y planes inside the change ROI (+pad): the DUBBED mouth's own motion, not the original lips.
    python fps_native_merge2.py --hi SRC.mp4 --lo-src master_25fps.mp4 --lo-comp ks_face.mp4 --out OUT.mp4 [--mode half|blend] [--json s.json]
Same resolution only (hi size == lo size); exit 0 = OUT written."""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "r15"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hires_merge import yuv_frames, x264_colour   # noqa: E402
from fps_native_merge import Layers   # noqa: E402


def small(Y, s=4): return cv2.resize(Y, (Y.shape[1] // s, Y.shape[0] // s), interpolation=cv2.INTER_AREA).astype(np.float32)


def match(hi, lo_src, ratio):
    """j[k] = the source frame index whose picture IS master frame k (monotone, min MAD in a +-3 window around ratio*k)"""
    S = [small(p[0]) for _, p in yuv_frames(hi)]; M = [small(p[0]) for _, p in yuv_frames(lo_src)]; j = []; last = -1; cost = []
    for k, m in enumerate(M):
        c0 = int(round(ratio * k)); cand = [i for i in range(max(last + 1, c0 - 3), min(len(S), c0 + 4))] or [min(len(S) - 1, last + 1)]
        d = [float(np.abs(S[i] - m).mean()) for i in cand]; b = cand[int(np.argmin(d))]; j.append(b); last = b; cost.append(min(d))
    return j, len(S), cost


class Takes(Layers):
    """Layers, but each take keeps the KeySync planes C and the full feathered mask M (the mask is fps_native_merge's, same parameters)"""
    def layer(s, sp, cp):
        a = s.a; h, w = sp[0].shape
        if s.K is None:
            Lm = max(h, w); kc = max(3, int(round(a.close * Lm)) | 1); kd = max(1, int(round(a.dilate * Lm)))
            s.K = dict(sg=max(1.0, a.feather * Lm), sb=max(1.0, a.blur * Lm), pad=kd + int(3 * max(1.0, a.feather * Lm)) + 2,
                       Kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)), Kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1)))
        K = s.K; d = [cq.astype(np.int16) - sq.astype(np.int16) for cq, sq in zip(cp, sp)]
        mc = cv2.resize(np.maximum(np.abs(d[1]), np.abs(d[2])).astype(np.float32) * 1.5, (w, h), interpolation=cv2.INTER_NEAREST)
        mag = cv2.GaussianBlur(np.maximum(np.abs(d[0]).astype(np.float32), mc), (0, 0), K["sb"])
        sub = mag[::4, ::4]; med = float(np.median(sub)); mad = float(np.median(np.abs(sub - med))) * 1.4826
        m0 = (mag > max(a.thr, med + a.k * mad)).astype(np.uint8)
        if int(m0.sum()) < a.min_px: return {"roi": None, "Y": sp[0]}
        m0 = cv2.dilate(cv2.morphologyEx(m0, cv2.MORPH_CLOSE, K["Kc"]), K["Kd"]); ys, xs = np.nonzero(m0); p = K["pad"]
        y0, y1 = max(0, ys.min() - p) & ~1, min(h, ys.max() + p + 2) & ~1; x0, x1 = max(0, xs.min() - p) & ~1, min(w, xs.max() + p + 2) & ~1
        M = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), K["sg"])
        return {"roi": (y0, y1, x0, x1), "Y": sp[0], "C": [q.astype(np.float32) for q in cp], "M": M}


INTERP = cv2.INTER_CUBIC


def warp_planes(C, M, F, box):
    """backward-warp KeySync planes C (Y full, U/V half) and mask M by flow F (defined on box = y0,y1,x0,x1 in Y px) -> planes cut to box"""
    y0, y1, x0, x1 = box; gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32); mx, my = gx + F[..., 0], gy + F[..., 1]
    Y = cv2.remap(C[0], mx, my, INTERP, borderMode=cv2.BORDER_REPLICATE)
    Mw = cv2.remap(M, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hw = ((x1 - x0) // 2, (y1 - y0) // 2); mxc, myc = cv2.resize(mx, hw) / 2, cv2.resize(my, hw) / 2
    UV = [cv2.remap(C[i], mxc, myc, INTERP, borderMode=cv2.BORDER_REPLICATE) for i in (1, 2)]
    return [Y] + UV, Mw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", required=True); ap.add_argument("--lo-src", required=True); ap.add_argument("--lo-comp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["half", "blend", "retime"], default="half"); ap.add_argument("--lo-fps", type=float, default=25.0)
    ap.add_argument("--thr", type=float, default=3.0); ap.add_argument("--k", type=float, default=6.0); ap.add_argument("--blur", type=float, default=0.004)
    ap.add_argument("--min-px", type=int, default=64); ap.add_argument("--close", type=float, default=0.015); ap.add_argument("--dilate", type=float, default=0.008)
    ap.add_argument("--feather", type=float, default=0.006); ap.add_argument("--flow-pad", type=int, default=24)
    ap.add_argument("--interp", choices=["cubic", "lanczos"], default="cubic"); ap.add_argument("--snap", type=float, default=0.02, help="retime: |u - nearest take| below this (in takes) -> the exact take, no warp")
    ap.add_argument("--crf", type=int, default=14); ap.add_argument("--preset", default="veryfast"); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--json", default="")
    a = ap.parse_args(); t0 = time.perf_counter()
    global INTERP; INTERP = cv2.INTER_LANCZOS4 if a.interp == "lanczos" else cv2.INTER_CUBIC
    import av
    info: dict = {}; G = yuv_frames(a.hi, info); next(G); G.close(); ratio = float(info["rate"]) / a.lo_fps
    j, n_src, cost = match(a.hi, a.lo_src, ratio); jinv = {}
    for k, i in enumerate(j): jinv.setdefault(i, k)
    Lz = Takes(a.lo_src, a.lo_comp, a)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); flows: dict = {}
    hi_i: dict = {}; G_hi = yuv_frames(a.hi, hi_i)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".v_" + out.name)
    pts_first = None; c = st = None; n = exact = filled = held = 0; t_flow = 0.0
    try:
        for i, (pts, hp) in enumerate(G_hi):
            H, W = hp[0].shape; pts_first = pts if pts_first is None else pts_first
            if c is None:
                c = av.open(str(tmp), "w"); st = c.add_stream("libx264", rate=hi_i["rate"]); st.width, st.height = W, H; st.pix_fmt = "yuv420p"
                st.time_base = hi_i["tb"]; st.codec_context.time_base = hi_i["tb"]; st.options = {"crf": str(a.crf), "preset": a.preset, "threads": str(a.threads)}
                col = x264_colour(a.lo_comp)
                if col: st.options = {**st.options, "x264-params": col}
            planes = hp; box = None
            if a.mode == "retime":
                u = float((pts - pts_first) * hi_i["tb"]) * a.lo_fps; ka = int(np.floor(u + 1e-6)); f = u - ka
                if f > 1 - a.snap and ka + 1 < len(j): ka, f = ka + 1, 0.0
                if f < a.snap or ka + 1 >= len(j): f = 0.0
            if a.mode == "retime" and f == 0.0 and ka < len(j):
                L, _ = Lz.get(ka)
                if L is not None and L["roi"] is not None:
                    y0, y1, x0, x1 = L["roi"]; Cw = [L["C"][0][y0:y1, x0:x1], L["C"][1][y0 // 2:y1 // 2, x0 // 2:x1 // 2], L["C"][2][y0 // 2:y1 // 2, x0 // 2:x1 // 2]]
                    Mw = L["M"][y0:y1, x0:x1]; box = (y0, y1, x0, x1); exact += 1
            elif a.mode != "retime" and i in jinv:
                L, _ = Lz.get(jinv[i])
                if L is not None and L["roi"] is not None:
                    y0, y1, x0, x1 = L["roi"]; Cw = [L["C"][0][y0:y1, x0:x1], L["C"][1][y0 // 2:y1 // 2, x0 // 2:x1 // 2], L["C"][2][y0 // 2:y1 // 2, x0 // 2:x1 // 2]]
                    Mw = L["M"][y0:y1, x0:x1]; box = (y0, y1, x0, x1); exact += 1
            else:
                if a.mode != "retime":
                    kb = next((k for k, jj in enumerate(j) if jj > i), None); ka = (kb - 1) if kb is not None else len(j) - 1
                else:
                    kb = ka + 1
                A, _ = Lz.get(ka); B, _ = Lz.get(kb) if kb is not None else (None, None)
                if A is not None and A["roi"] is not None and B is not None and B["roi"] is not None:
                    f = f if a.mode == "retime" else (i - j[ka]) / (j[kb] - j[ka]); h, w = A["Y"].shape; p = a.flow_pad
                    y0 = max(0, min(A["roi"][0], B["roi"][0]) - p) & ~1; y1 = min(h, max(A["roi"][1], B["roi"][1]) + p) & ~1
                    x0 = max(0, min(A["roi"][2], B["roi"][2]) - p) & ~1; x1 = min(w, max(A["roi"][3], B["roi"][3]) + p) & ~1; box = (y0, y1, x0, x1)
                    ia = np.ascontiguousarray(A["C"][0][y0:y1, x0:x1].astype(np.uint8)); ib = np.ascontiguousarray(B["C"][0][y0:y1, x0:x1].astype(np.uint8))
                    tf = time.perf_counter(); F01 = dis.calc(ia, ib, None); F10 = dis.calc(ib, ia, None); t_flow += time.perf_counter() - tf
                    Ft0 = -(1 - f) * f * F01 + f * f * F10; Ft1 = (1 - f) ** 2 * F01 - f * (1 - f) * F10
                    if a.mode in ("half", "retime"):
                        Cw, Mw = warp_planes(A["C"], A["M"], Ft0, box) if f <= 0.5 else warp_planes(B["C"], B["M"], Ft1, box)
                    else:
                        Ca, Ma = warp_planes(A["C"], A["M"], Ft0, box); Cb, Mb = warp_planes(B["C"], B["M"], Ft1, box)
                        Cw = [(1 - f) * qa + f * qb for qa, qb in zip(Ca, Cb)]; Mw = (1 - f) * Ma + f * Mb
                    filled += 1
                elif A is not None and A["roi"] is not None:
                    y0, y1, x0, x1 = A["roi"]; box = A["roi"]; held += 1
                    Cw = [A["C"][0][y0:y1, x0:x1], A["C"][1][y0 // 2:y1 // 2, x0 // 2:x1 // 2], A["C"][2][y0 // 2:y1 // 2, x0 // 2:x1 // 2]]; Mw = A["M"][y0:y1, x0:x1]
            if box is not None:
                y0, y1, x0, x1 = box; planes = [q.copy() for q in hp]
                for q in range(3):
                    s = 1 if q == 0 else 2; m = Mw if q == 0 else cv2.resize(Mw, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
                    reg = planes[q][y0 // s:y1 // s, x0 // s:x1 // s].astype(np.float32); cw = Cw[q][:reg.shape[0], :reg.shape[1]]; m = m[:reg.shape[0], :reg.shape[1]]
                    planes[q][y0 // s:y1 // s, x0 // s:x1 // s] = np.clip(reg * (1 - m) + cw * m + 0.5, 0, 255).astype(np.uint8)
            fr = av.VideoFrame.from_ndarray(np.concatenate([q.ravel() for q in planes]).reshape(H * 3 // 2, W), format="yuv420p")
            fr.pts = pts; fr.time_base = hi_i["tb"]
            for pk in st.encode(fr): c.mux(pk)
            n += 1
        for pk in st.encode(): c.mux(pk)
        c.close(); c = None; tmp.replace(out)
    finally:
        if c is not None: c.close()
        tmp.unlink(missing_ok=True)
    el = time.perf_counter() - t0; gaps = int(sum(1 for i in range(n_src) if i not in jinv))
    s = {"mode": a.mode, "frames": n, "exact": exact, "filled": filled, "held": held, "src_without_take": gaps, "lo_frames": len(j),
         "match_mad_max": round(max(cost), 2), "match_mad_mean": round(float(np.mean(cost)), 3), "seconds": round(el, 1), "flow_s": round(t_flow, 1)}
    print("== fps_native_merge2", json.dumps(s), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
