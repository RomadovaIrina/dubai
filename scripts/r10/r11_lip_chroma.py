#!/usr/bin/env python3
"""r11 FACE (the root cause of the lip flicker): LatentSync draws PALE, thin lips while the dub mouth is open and full red lips when it closes
(q18: lip redness a* vs the skin ring 7.5 -> 1.6-3.3 on open frames, the source holds 5-7). The red <-> pale alternation is what the eye
reads as 'the lip line vanishes and pops back'. LIP CHROMA LOCK: on the GENERATED lips (the output's own MediaPipe lip polygon, the inner
cavity excluded), each frame's mean Lab a* / b* is pulled to its temporal median over +-W seconds (capped); luminance and shape are untouched.
Input / output: the final video (its own time base / pts kept, audio copied). In memory (test tool).
    python scripts/r10/r11_lip_chroma.py --video FINAL.mp4 --out OUT.mp4 [--win-s 0.5 --max-a 8 --max-b 6 --strength 1.0]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np, cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lm", default=None, help="MediaPipe npz of --video (computed + cached if missing)")
    ap.add_argument("--win-s", type=float, default=0.5); ap.add_argument("--max-a", type=float, default=8.0); ap.add_argument("--max-b", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--crf", type=int, default=13)
    a = ap.parse_args()
    import av, clean_face_stack as cfs
    lmp = pathlib.Path(a.lm) if a.lm else pathlib.Path(a.out).with_suffix(".in_mp.npz")
    lm = np.load(cfs.run_mediapipe(pathlib.Path(a.video), lmp))["lm"].astype(np.float64)
    c = av.open(a.video); s = c.streams.video[0]; tb = s.time_base; rate = s.average_rate
    F, pts = [], []
    for f in c.decode(s): F.append(f.to_ndarray(format="bgr24")); pts.append(f.pts)
    c.close(); n = min(len(F), len(lm)); H, W = F[0].shape[:2]
    geo = [None] * n; A = np.full((n, 2), np.nan)
    for k in range(n):   # lip ring (outer minus the inner cavity) of the OUTPUT's own lips
        p = lm[k]
        if not np.isfinite(p[LIPS_OUTER + LIPS_INNER]).all(): continue
        mw = float(np.linalg.norm(p[61, :2] - p[291, :2])); q = p[LIPS_OUTER, :2]; c_ = q.mean(0); r = int(0.9 * mw)
        y0, y1, x0, x1 = max(0, int(c_[1] - r)), min(H, int(c_[1] + r)), max(0, int(c_[0] - r)), min(W, int(c_[0] + r))
        if y1 - y0 < 12 or x1 - x0 < 12: continue
        mo = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(mo, [np.round(q - [x0, y0]).astype(np.int32)], 1)
        mi = np.zeros_like(mo); cv2.fillPoly(mi, [np.round(p[LIPS_INNER, :2] - [x0, y0]).astype(np.int32)], 1)
        ring = (mo > 0) & (mi == 0)
        if ring.sum() < 20: continue
        lab = cv2.cvtColor(F[k][y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        A[k] = [lab[..., 1][ring].mean(), lab[..., 2][ring].mean()]
        wmask = cv2.GaussianBlur((ring | cv2.dilate(mo, np.ones((3, 3), np.uint8)).astype(bool) & (mi == 0)).astype(np.float32), (0, 0), max(1.0, 0.03 * mw))
        geo[k] = (y0, y1, x0, x1, wmask)
    w_ = max(1, int(round(a.win_s * float(rate)))); tgt = np.full_like(A, np.nan)
    for k in range(n):
        seg = A[max(0, k - w_):k + w_ + 1]; seg = seg[np.isfinite(seg).all(1)]
        if len(seg) >= 3: tgt[k] = np.median(seg, 0)
    out = list(F[:n]); shifts = []
    for k in range(n):
        if geo[k] is None or not np.isfinite(tgt[k]).all() or not np.isfinite(A[k]).all(): continue
        da = float(np.clip(tgt[k, 0] - A[k, 0], -a.max_a, a.max_a)) * a.strength; db = float(np.clip(tgt[k, 1] - A[k, 1], -a.max_b, a.max_b)) * a.strength
        y0, y1, x0, x1, wm = geo[k]; lab = cv2.cvtColor(F[k][y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1] += da * wm; lab[..., 2] += db * wm
        o = F[k].copy(); o[y0:y1, x0:x1] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR); out[k] = o; shifts.append((abs(da), abs(db)))
    outp = pathlib.Path(a.out); vtmp = outp.with_name(".v_" + outp.name)
    o_ = av.open(str(vtmp), "w"); st = o_.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.time_base = tb
    st.options = {"crf": str(a.crf), "preset": "medium"}; st.codec_context.time_base = tb
    for k, f in enumerate(out):
        fr = av.VideoFrame.from_ndarray(f, format="bgr24"); fr.pts = int(pts[k]); fr.time_base = tb
        for pk in st.encode(fr): o_.mux(pk)
    for pk in st.encode(): o_.mux(pk)
    o_.close()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vtmp), "-i", a.video, "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", str(outp)], check=True); vtmp.unlink(missing_ok=True)
    sh = np.array(shifts) if shifts else np.zeros((1, 2))
    rep = {"frames": n, "adjusted": len(shifts), "da_mean": round(float(sh[:, 0].mean()), 2), "da_p95": round(float(np.percentile(sh[:, 0], 95)), 2),
           "db_mean": round(float(sh[:, 1].mean()), 2), "a_std_before": round(float(np.nanstd(A[:, 0])), 2), "args": vars(a)}
    outp.with_suffix(".lipchroma.json").write_text(json.dumps(rep, indent=1)); print("== lip_chroma", json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
