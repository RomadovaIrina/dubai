#!/usr/bin/env python3
"""r15 T (edge case res4k_vertical = face_gen_failed): the face stages' LOW-resolution result put back onto the FULL-resolution source.

For a source whose long side is above E2E_FACE_MAXSIDE (default 1920), scripts/r9/e2e.sh runs the face stages (generator + composite) on a
downscaled copy (lo_src). This step then adds only what the composite CHANGED onto the original frames, per Y / U / V plane:
    out = hi + up(M * (lo_comp - lo_src))
    M   = where the composite changed the frame: blurred |delta| above an adaptive threshold (max(--thr, frame median + --k x MAD), so re-encode
          noise and a global colour drift of the composite do not count), closed, dilated, feathered; built at low res, upsampled with the delta
Everything outside the changed (mouth) region keeps the source's full-resolution pixels bit for bit (YUV, no colour conversion); the mouth
gets the generated content (LatentSync works on a 512 px face crop, so face work at <= 1920 px loses no generated detail).
Streaming: three decoders in lockstep, one frame of each in RAM. Output = the hi source's size, pts and time base, colour-tagged like lo_comp.
Exit 0 = OUT written; any error = exit 1 (e2e.sh then falls back to the source picture, PARTIAL_SUCCESS).
    python scripts/r15/hires_merge.py --hi SRC.mp4 --lo-src SRC_LO.mp4 --lo-comp COMP_LO.mp4 --out OUT.mp4 [--json stats.json]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time
import numpy as np, cv2


def yuv_frames(path: str, info: dict | None = None):
    """-> (pts, [Y, U, V]) per frame, 8-bit 4:2:0 planes (10-bit / full-range sources converted like ffmpeg -pix_fmt yuv420p does)"""
    import av
    c = av.open(path); s = c.streams.video[0]; s.thread_type = "AUTO"
    if info is not None: info.update(rate=s.average_rate, tb=s.time_base)
    try:
        for fr in c.decode(s):
            if fr.format.name != "yuv420p": fr = fr.reformat(format="yuv420p")
            P = []
            for i, p in enumerate(fr.planes):
                h_, w_ = (fr.height, fr.width) if i == 0 else ((fr.height + 1) // 2, (fr.width + 1) // 2)
                P.append(np.frombuffer(p, np.uint8).reshape(-1, p.line_size)[:h_, :w_])
            yield fr.pts, P
    finally:
        c.close()


def x264_colour(path: str) -> str:
    """x264-params colour tags = the stream's own tags (ffprobe names = x264 names for the usual values); untagged -> nothing"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_space,color_primaries,color_transfer",
                        "-of", "json", path], capture_output=True, text=True)
    s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]; ok = lambda v: v and v not in ("unknown", "reserved", "unspecified")
    kv = [(k, s.get(f)) for k, f in (("colorprim", "color_primaries"), ("transfer", "color_transfer"), ("colormatrix", "color_space"))]
    return ":".join(f"{k}={v}" for k, v in kv if ok(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hi", required=True); ap.add_argument("--lo-src", required=True); ap.add_argument("--lo-comp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=3.0, help="minimum threshold on the blurred |delta| (8-bit levels; Y, and chroma x1.5)")
    ap.add_argument("--k", type=float, default=6.0, help="adaptive threshold: frame median + k x MAD of the blurred |delta| (noise / global drift)")
    ap.add_argument("--blur", type=float, default=0.004, help="sigma of the |delta| blur, fraction of the low-res long side")
    ap.add_argument("--min-px", type=int, default=64, help="fewer changed low-res pixels than this = an untouched frame (the source frame is written)")
    ap.add_argument("--close", type=float, default=0.015, help="closing kernel, fraction of the low-res long side")
    ap.add_argument("--dilate", type=float, default=0.008, help="mask dilation, fraction of the low-res long side")
    ap.add_argument("--feather", type=float, default=0.006, help="mask feather sigma, fraction of the low-res long side")
    ap.add_argument("--crf", type=int, default=14); ap.add_argument("--preset", default="veryfast", help="x264 preset (4K: veryfast ~3x faster than medium)")
    ap.add_argument("--threads", type=int, default=0, help="x264 threads (0 = auto)"); ap.add_argument("--json", default="")
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    hi_i: dict = {}
    G_hi, G_ls, G_lc = yuv_frames(a.hi, hi_i), yuv_frames(a.lo_src), yuv_frames(a.lo_comp)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".v_" + out.name)
    c = None; st = None; n = changed = tail_src = 0; area = []; thr_used = []
    try:
        for k, (pts, hp) in enumerate(G_hi):
            H, W = hp[0].shape
            if c is None:
                if H % 2 or W % 2: raise SystemExit(f"odd source size {W}x{H}")
                c = av.open(str(tmp), "w"); st = c.add_stream("libx264", rate=hi_i["rate"]); st.width, st.height = W, H
                st.pix_fmt = "yuv420p"; st.time_base = hi_i["tb"]; st.codec_context.time_base = hi_i["tb"]
                st.options = {"crf": str(a.crf), "preset": a.preset, "threads": str(a.threads)}
                col = x264_colour(a.lo_comp)
                if col: st.options = {**st.options, "x264-params": col}
            try:
                _, sp = next(G_ls)
            except StopIteration:
                raise SystemExit(f"lo-src ended at frame {k}: not the same frames as hi")
            try:
                _, cp = next(G_lc); lc_ok = True
            except StopIteration:
                lc_ok = False
            planes = hp
            if lc_ok:
                if k == 0:
                    if [p.shape for p in cp] != [p.shape for p in sp]: raise SystemExit(f"lo-comp {cp[0].shape} != lo-src {sp[0].shape}")
                    h, w = sp[0].shape; L = max(h, w); sy, sx = H / h, W / w
                    kc = max(3, int(round(a.close * L)) | 1); kd = max(1, int(round(a.dilate * L))); sg = max(1.0, a.feather * L); sb = max(1.0, a.blur * L)
                    pad = kd + int(3 * sg) + 2
                    Kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)); Kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1))
                d = [cq.astype(np.int16) - sq.astype(np.int16) for cq, sq in zip(cp, sp)]
                mc = cv2.resize(np.maximum(np.abs(d[1]), np.abs(d[2])).astype(np.float32) * 1.5, (w, h), interpolation=cv2.INTER_NEAREST)
                mag = cv2.GaussianBlur(np.maximum(np.abs(d[0]).astype(np.float32), mc), (0, 0), sb)
                sub = mag[::4, ::4]; med = float(np.median(sub)); mad = float(np.median(np.abs(sub - med))) * 1.4826
                thr = max(a.thr, med + a.k * mad); thr_used.append(thr)
                m0 = (mag > thr).astype(np.uint8)
                if int(m0.sum()) >= a.min_px:
                    m0 = cv2.dilate(cv2.morphologyEx(m0, cv2.MORPH_CLOSE, Kc), Kd)
                    ys, xs = np.nonzero(m0)   # the ROI on even coordinates (4:2:0 chroma aligned), in both resolutions
                    y0, y1 = max(0, ys.min() - pad) & ~1, min(h, ys.max() + pad + 2) & ~1; x0, x1 = max(0, xs.min() - pad) & ~1, min(w, xs.max() + pad + 2) & ~1
                    M = cv2.GaussianBlur(m0[y0:y1, x0:x1].astype(np.float32), (0, 0), sg)
                    Y0, Y1 = int(round(y0 * sy / 2)) * 2, min(H, int(round(y1 * sy / 2)) * 2); X0, X1 = int(round(x0 * sx / 2)) * 2, min(W, int(round(x1 * sx / 2)) * 2)
                    planes = [p.copy() for p in hp]
                    for i in range(3):
                        f = 1 if i == 0 else 2
                        ly0, ly1, lx0, lx1 = y0 // f, y1 // f, x0 // f, x1 // f; hy0, hy1, hx0, hx1 = Y0 // f, Y1 // f, X0 // f, X1 // f
                        Mi = M if f == 1 else cv2.resize(M, (lx1 - lx0, ly1 - ly0), interpolation=cv2.INTER_AREA)
                        up = cv2.resize(d[i][ly0:ly1, lx0:lx1].astype(np.float32) * Mi, (hx1 - hx0, hy1 - hy0), interpolation=cv2.INTER_CUBIC)
                        planes[i][hy0:hy1, hx0:hx1] = np.clip(planes[i][hy0:hy1, hx0:hx1].astype(np.float32) + up, 0, 255).astype(np.uint8)
                    changed += 1; area.append(float(M.sum()) / (h * w))
            else:
                tail_src += 1
            fr = av.VideoFrame.from_ndarray(np.concatenate([p.ravel() for p in planes]).reshape(H * 3 // 2, W), format="yuv420p")
            fr.pts = pts; fr.time_base = hi_i["tb"]
            for p in st.encode(fr): c.mux(p)
            n += 1
        if c is None: raise SystemExit("hi has no frames")
        for p in st.encode(): c.mux(p)
        c.close(); c = None
        if tail_src > 2: raise SystemExit(f"lo-comp is {tail_src} frames shorter than the source")
        tmp.replace(out)
    finally:
        if c is not None: c.close()
        tmp.unlink(missing_ok=True)
    el = time.perf_counter() - t0
    s = {"frames": n, "changed_frames": changed, "tail_source_frames": tail_src, "mask_area_mean": round(float(np.mean(area)), 4) if area else 0.0,
         "mask_area_max": round(float(np.max(area)), 4) if area else 0.0, "thr_median": round(float(np.median(thr_used)), 2) if thr_used else None,
         "hi": [W, H] if n else None, "seconds": round(el, 1), "fps": round(n / max(1e-6, el), 1)}
    print("== hires_merge", json.dumps(s), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if e.code not in (0, None): print("== hires_merge FAILED:", e.code, flush=True); sys.exit(1)
        raise
