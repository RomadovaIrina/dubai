#!/usr/bin/env python3
"""r9 F: temporal stabiliser of the mouth DETAIL in the generated layer (management on m18: "explosions in the mouth"). On m18 the texture
flash frames go raw LatentSync 10 -> + CodeFormer 25 -> + detail transfer + layer SR 120 (30 fps, frame-aligned): CodeFormer's detail is drawn
independently per frame and the SR amplifies it, so the fine texture pops from frame to frame. Per frame k, inside a feathered box around the
mouth: keep the frame's own low frequencies (shape / opening / motion) and replace its high frequencies by the per-pixel MEDIAN of the
high frequencies of frames k-R..k+R, each flow-aligned (DIS) to frame k. A 1-frame detail pop is outvoted; articulation (low frequencies) is untouched.
    python scripts/r9/r9_hf_stabilise.py --ls L.mp4 --lm L.mediapipe.npz --out L_stab.mp4 [--radius 2] [--sigma 0.04]"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r8"))
from r8_mouth_flicker import LIPS_OUTER


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ls", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=int, default=2); ap.add_argument("--sigma", type=float, default=0.04, help="low-pass sigma as a fraction of the inter-ocular distance")
    ap.add_argument("--box", type=float, default=0.9, help="half size of the mouth box in IOD"); ap.add_argument("--crf", type=int, default=10)
    a = ap.parse_args(); t0 = time.perf_counter()
    import av
    c = av.open(a.ls); s = c.streams.video[0]; s.thread_type = "AUTO"; rate = s.average_rate
    F = [f.to_ndarray(format="bgr24") for f in c.decode(s)]; c.close(); n = len(F); H, W = F[0].shape[:2]
    lm = np.load(a.lm)["lm"]; lm = np.concatenate([lm, np.full((max(0, n - len(lm)),) + lm.shape[1:], np.nan, lm.dtype)])[:n]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); out = list(F); done = 0
    for k in range(n):
        p = lm[k]
        if not np.isfinite(p[LIPS_OUTER + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); cx, cy = p[LIPS_OUTER, :2].mean(0); r = int(a.box * iod)
        y0, y1, x0, x1 = max(0, int(cy - r)), min(H, int(cy + r)), max(0, int(cx - r)), min(W, int(cx + r))
        if y1 - y0 < 16 or x1 - x0 < 16: continue
        sg = max(0.8, a.sigma * iod); ck = F[k][y0:y1, x0:x1].astype(np.float32); lf = cv2.GaussianBlur(ck, (0, 0), sg); gk = cv2.cvtColor(F[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        hfs = [ck - lf]; gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        for j in range(k - a.radius, k + a.radius + 1):
            if j == k or j < 0 or j >= n: continue
            cj = F[j][y0:y1, x0:x1]; fl = dis.calc(gk, cv2.cvtColor(cj, cv2.COLOR_BGR2GRAY), None)
            wj = cv2.remap(cj.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            hfs.append(wj - cv2.GaussianBlur(wj, (0, 0), sg))
        if len(hfs) < 3: continue
        hf = np.median(np.stack(hfs), axis=0)
        m = np.zeros((y1 - y0, x1 - x0), np.float32); cv2.ellipse(m, (int(cx - x0), int(cy - y0)), (int(0.75 * r), int(0.6 * r)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.12 * iod))[..., None]
        out[k] = F[k].copy(); out[k][y0:y1, x0:x1] = np.clip(ck * (1 - m) + (lf + hf) * m, 0, 255).astype(np.uint8); done += 1
    o = av.open(a.out, "w"); st = o.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.options = {"crf": str(a.crf), "preset": "medium"}
    for f in out:
        for pk in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close()
    rep = {"frames": n, "stabilised": done, "radius": a.radius, "seconds": round(time.perf_counter() - t0, 1)}
    pathlib.Path(a.out).with_suffix(".stab.json").write_text(json.dumps(rep)); print("== hf_stabilise", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
