#!/usr/bin/env python3
"""r15 F3: "no Real-ESRGAN on the lips": a layer that is the SR layer outside the mouth and the PRE-SR layer (+ a mild unsharp mask)
on the lips / mouth (each frame's landmarks, feathered). Real-ESRGAN re-invents lip texture per frame (frame-varying detail = shimmer);
the pre-SR mouth is temporally steadier. CPU only, no new model.
    python scripts/r15/lips_nosr.py --pre LAYER.mp4 --sr LAYER_srt1.mp4 --lm LAYER.mediapipe.npz --out LAYER_srnl.mp4 [--amount 0.5 --sigma 1.0]
"""
import argparse, json, subprocess
import numpy as np, cv2
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("pre", "sr", "lm", "out")]
    ap.add_argument("--amount", type=float, default=0.5); ap.add_argument("--sigma", type=float, default=1.0); ap.add_argument("--dilate", type=float, default=0.12)
    ap.add_argument("--crf", type=int, default=10); ap.add_argument("--preset", default="medium")
    a = ap.parse_args(); lm = np.load(a.lm)["lm"][..., :2]; cp, cs = cv2.VideoCapture(a.pre), cv2.VideoCapture(a.sr)
    W, H, fps = int(cs.get(3)), int(cs.get(4)), cs.get(cv2.CAP_PROP_FPS) or 25.0
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-crf", str(a.crf), "-preset", a.preset, "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    k = done = 0
    while True:
        o1, fp = cp.read(); o2, fs = cs.read()
        if not (o1 and o2): break
        out = fs
        if k < len(lm) and np.isfinite(lm[k][OUTER]).all():
            p = lm[k]; mw = float(np.linalg.norm(p[61] - p[291])); c = p[OUTER].mean(0)
            x0, y0 = int(max(0, c[0] - 1.0 * mw)), int(max(0, c[1] - 0.8 * mw)); x1, y1 = int(min(W, c[0] + 1.0 * mw)), int(min(H, c[1] + 0.8 * mw))
            if x1 - x0 > 8 and y1 - y0 > 8:
                m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [np.round(p[OUTER] - [x0, y0]).astype(np.int32)], 1)
                r = max(1, int(a.dilate * mw)); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)).astype(np.float32)
                m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.05 * mw))[..., None]
                pr = fp[y0:y1, x0:x1].astype(np.float32); sh = pr + a.amount * (pr - cv2.GaussianBlur(pr, (0, 0), a.sigma))
                out = fs.copy(); out[y0:y1, x0:x1] = np.clip(fs[y0:y1, x0:x1] * (1 - m) + sh * m + 0.5, 0, 255).astype(np.uint8); done += 1
        enc.stdin.write(out.tobytes()); k += 1
    enc.stdin.close(); enc.wait()
    print("== lips_nosr", json.dumps({"frames": k, "blended": done, "amount": a.amount, "sigma": a.sigma}), flush=True)


if __name__ == "__main__":
    main()
