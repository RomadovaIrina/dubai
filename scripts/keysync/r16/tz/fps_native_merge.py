#!/usr/bin/env python3
"""r16 TZ 5.6 (Irina 09.11: "the frame rate just has to look good"): the KeySync result at the SOURCE frame rate. KeySync renders the face at
25 fps; here only the layer it CHANGED is carried onto the native-fps source frames, the rest of every frame stays the source (background,
camera, hands keep their own smoothness). Generalises scripts/r15/hires_merge.py (same change layer, any resolution AND any frame rate):
    D_k = M_k * (lo_comp_k - lo_src_k)       the change of 25 fps frame k (mask M_k: blurred |delta| over an adaptive threshold, closed, dilated,
                                             feathered; exactly hires_merge's), Y / U / V planes, low res
    out(t) = hi(t) + up(D(t)),  u = t * 25,  k = floor(u),  a = u - k
modes for D(t):
    nearest   D_round(u)                                  (the mouth jumps at 25 fps; its position lags the head by up to 20 ms)
    blend     (1 - a) D_k + a D_k+1                       (cross-fade; position half-way)
    flownear  W_t(D_round(u))                             (the nearest take, no mouth cross-fade, moved with the head)
    flow      (1 - a) W_t(D_k) + a W_t(D_k+1)             W_t = backward warp by the optical flow (DIS, CPU) from the native source frame
                                                          at t to the SOURCE frame under D_k (master k): real frames only, so the change
                                                          follows the true head motion; the mouth cross-fades between the two takes
The hi frames keep their pts / time base (the output = the source fps); no GPU. Exit 0 = OUT written.
    python fps_native_merge.py --hi SRC_NATIVE.mp4 --lo-src master_25fps.mp4 --lo-comp ks_face.mp4 --out OUT.mp4 [--mode flow] [--json s.json]"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "r15"))
from hires_merge import yuv_frames, x264_colour   # noqa: E402


class Layers:
    """the 25 fps change layers D_k (ROI + Y/U/V float planes) and the master Y plane, decoded lazily, two kept"""
    def __init__(s, lo_src, lo_comp, a):
        s.Gs, s.Gc, s.a = yuv_frames(lo_src), yuv_frames(lo_comp), a; s.cache = {}; s.n_read = 0; s.end = None; s.K = None

    def get(s, k):
        k = max(0, k)
        if s.end is not None and k >= s.end: k = s.end - 1
        while k not in s.cache and (s.end is None or s.n_read < s.end):
            try: _, sp = next(s.Gs); _, cp = next(s.Gc)
            except StopIteration:
                s.end = s.n_read; k = min(k, s.end - 1); break
            s.cache[s.n_read] = s.layer(sp, cp); s.n_read += 1
            for old in [i for i in s.cache if i < s.n_read - 3]: del s.cache[old]
        return s.cache.get(k), k

    def layer(s, sp, cp):
        a = s.a; h, w = sp[0].shape
        if s.K is None:
            L = max(h, w); kc = max(3, int(round(a.close * L)) | 1); kd = max(1, int(round(a.dilate * L)))
            s.K = dict(sg=max(1.0, a.feather * L), sb=max(1.0, a.blur * L), pad=kd + int(3 * max(1.0, a.feather * L)) + 2,
                       Kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)), Kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1)))
        K = s.K; d = [cq.astype(np.int16) - sq.astype(np.int16) for cq, sq in zip(cp, sp)]
        mc = cv2.resize(np.maximum(np.abs(d[1]), np.abs(d[2])).astype(np.float32) * 1.5, (w, h), interpolation=cv2.INTER_NEAREST)
        mag = cv2.GaussianBlur(np.maximum(np.abs(d[0]).astype(np.float32), mc), (0, 0), K["sb"])
        sub = mag[::4, ::4]; med = float(np.median(sub)); mad = float(np.median(np.abs(sub - med))) * 1.4826
        m0 = (mag > max(a.thr, med + a.k * mad)).astype(np.uint8)
        if int(m0.sum()) < a.min_px: return {"roi": None, "Y": sp[0]}
        m0 = cv2.dilate(cv2.morphologyEx(m0, cv2.MORPH_CLOSE, K["Kc"]), K["Kd"]); ys, xs = np.nonzero(m0); p = K["pad"]
        y0, y1 = max(0, ys.min() - p) & ~1, min(h, ys.max() + p + 2) & ~1; x0, x1 = max(0, xs.min() - p) & ~1, min(w, xs.max() + p + 2) & ~1
        M = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), K["sg"])   # full frame (the union ROI of two frames is cut later)
        D = [d[0].astype(np.float32) * M] + [d[i].astype(np.float32) * cv2.resize(M, (d[i].shape[1], d[i].shape[0]), interpolation=cv2.INTER_AREA) for i in (1, 2)]
        return {"roi": (y0, y1, x0, x1), "D": D, "Y": sp[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", required=True); ap.add_argument("--lo-src", required=True); ap.add_argument("--lo-comp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["nearest", "blend", "flow", "flownear"], default="flow"); ap.add_argument("--lo-fps", type=float, default=25.0)
    ap.add_argument("--thr", type=float, default=3.0); ap.add_argument("--k", type=float, default=6.0); ap.add_argument("--blur", type=float, default=0.004)
    ap.add_argument("--min-px", type=int, default=64); ap.add_argument("--close", type=float, default=0.015); ap.add_argument("--dilate", type=float, default=0.008)
    ap.add_argument("--feather", type=float, default=0.006); ap.add_argument("--flow-pad", type=int, default=24, help="low-res px of context around the ROI for the flow")
    ap.add_argument("--crf", type=int, default=14); ap.add_argument("--preset", default="veryfast"); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--json", default="")
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    hi_i: dict = {}; G_hi = yuv_frames(a.hi, hi_i); Lz = Layers(a.lo_src, a.lo_comp, a)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if a.mode in ("flow", "flownear") else None
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".v_" + out.name)
    c = st = None; n = changed = 0; pts0 = None; flow_mag = []; t_flow = 0.0
    try:
        for pts, hp in G_hi:
            H, W = hp[0].shape
            if c is None:
                if H % 2 or W % 2: raise SystemExit(f"odd source size {W}x{H}")
                c = av.open(str(tmp), "w"); st = c.add_stream("libx264", rate=hi_i["rate"]); st.width, st.height = W, H; st.pix_fmt = "yuv420p"
                st.time_base = hi_i["tb"]; st.codec_context.time_base = hi_i["tb"]; st.options = {"crf": str(a.crf), "preset": a.preset, "threads": str(a.threads)}
                col = x264_colour(a.lo_comp)
                if col: st.options = {**st.options, "x264-params": col}
            pts0 = pts if pts0 is None else pts0; t = float((pts - pts0) * hi_i["tb"]); u = t * a.lo_fps; k = int(np.floor(u + 1e-9)); al = u - k
            if a.mode in ("nearest", "flownear"): parts = [(Lz.get(int(round(u)))[0], 1.0)]   # flownear: the nearest take only (no mouth cross-fade), motion-compensated
            else:
                A, ka = Lz.get(k); B, kb = Lz.get(k + 1)
                parts = [(A, 1.0)] if kb == ka else [(A, 1.0 - al), (B, al)]
            parts = [(L, wgt) for L, wgt in parts if L is not None and L["roi"] is not None and wgt > 1e-4]
            planes = hp
            if parts:
                h, w = parts[0][0]["Y"].shape; sy, sx = H / h, W / w
                y0 = min(L["roi"][0] for L, _ in parts); y1 = max(L["roi"][1] for L, _ in parts); x0 = min(L["roi"][2] for L, _ in parts); x1 = max(L["roi"][3] for L, _ in parts)
                acc = [np.zeros((y1 - y0, x1 - x0), np.float32)] + [np.zeros(((y1 - y0) // 2, (x1 - x0) // 2), np.float32) for _ in (1, 2)]
                if a.mode in ("flow", "flownear"):
                    p = a.flow_pad; fy0, fy1, fx0, fx1 = max(0, y0 - p), min(h, y1 + p), max(0, x0 - p), min(w, x1 + p)
                    hy = cv2.resize(hp[0], (w, h), interpolation=cv2.INTER_AREA) if (h, w) != (H, W) else hp[0]
                    It = np.ascontiguousarray(hy[fy0:fy1, fx0:fx1])
                for L, wgt in parts:
                    if a.mode in ("flow", "flownear"):
                        tf = time.perf_counter(); F = dis.calc(It, np.ascontiguousarray(L["Y"][fy0:fy1, fx0:fx1]), None); t_flow += time.perf_counter() - tf
                        flow_mag.append(float(np.abs(F).mean()))
                        gy, gx = np.mgrid[fy0:fy1, fx0:fx1].astype(np.float32)
                        mx, my = gx + F[..., 0], gy + F[..., 1]
                        Y = cv2.remap(L["D"][0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[y0 - fy0:y1 - fy0, x0 - fx0:x1 - fx0]
                        mxc, myc = cv2.resize(mx, ((fx1 - fx0) // 2, (fy1 - fy0) // 2)) / 2, cv2.resize(my, ((fx1 - fx0) // 2, (fy1 - fy0) // 2)) / 2
                        cy0, cx0 = (y0 - fy0) // 2, (x0 - fx0) // 2
                        UV = [cv2.remap(L["D"][i], mxc, myc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[cy0:cy0 + (y1 - y0) // 2, cx0:cx0 + (x1 - x0) // 2] for i in (1, 2)]
                        cur = [Y] + UV
                    else:
                        cur = [L["D"][0][y0:y1, x0:x1], L["D"][1][y0 // 2:y1 // 2, x0 // 2:x1 // 2], L["D"][2][y0 // 2:y1 // 2, x0 // 2:x1 // 2]]
                    for i in range(3):
                        ch, cw = acc[i].shape; acc[i] += wgt * cur[i][:ch, :cw]
                Y0, Y1 = int(round(y0 * sy / 2)) * 2, min(H, int(round(y1 * sy / 2)) * 2); X0, X1 = int(round(x0 * sx / 2)) * 2, min(W, int(round(x1 * sx / 2)) * 2)
                planes = [q.copy() for q in hp]
                for i in range(3):
                    f = 1 if i == 0 else 2; hy0, hy1, hx0, hx1 = Y0 // f, Y1 // f, X0 // f, X1 // f
                    up = acc[i] if (hy1 - hy0, hx1 - hx0) == acc[i].shape else cv2.resize(acc[i], (hx1 - hx0, hy1 - hy0), interpolation=cv2.INTER_CUBIC)
                    planes[i][hy0:hy1, hx0:hx1] = np.clip(planes[i][hy0:hy1, hx0:hx1].astype(np.float32) + up, 0, 255).astype(np.uint8)
                changed += 1
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
    s = {"mode": a.mode, "frames": n, "changed_frames": changed, "hi_fps": float(hi_i["rate"]), "lo_frames": Lz.n_read, "seconds": round(el, 1), "fps": round(n / max(el, 1e-6), 1),
         "flow_s": round(t_flow, 1), "flow_mean_px": round(float(np.mean(flow_mag)), 3) if flow_mag else None}
    print("== fps_native_merge", json.dumps(s), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except SystemExit as e:
        if e.code not in (0, None): print("== fps_native_merge FAILED:", e.code, flush=True); sys.exit(1)
        raise
