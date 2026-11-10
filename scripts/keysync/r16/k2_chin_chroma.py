#!/usr/bin/env python3
"""r16d K2: the pink lip-colour smear on the chin (old03 47.7 / 67.9 / 73.2 s, Q + K1: generator). KeySync paints lip colour below the lower
lip, where the source has skin / stubble. Fix on the FINAL: in the chin band (from the lower outer lip down to the chin line, cheek to cheek
at mouth width, from OUR landmarks), where ours is pinker than the source (a* excess > --tol), replace a* / b* with the source's at the same
pixels (feathered); L* (the motion / shading) stays KeySync's. Also measures the band's a* excess (ours - source) per frame.
    python k2_chin_chroma.py --final F.mp4 --master S.mp4 --lm-ours F_mp.npz [--out OUT.mp4 --audio MIX] [--measure-only]"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]   # lower outer lip, corner -> corner
ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--master", required=True); ap.add_argument("--lm-ours", required=True, dest="lm")
ap.add_argument("--out", default=""); ap.add_argument("--audio", default=""); ap.add_argument("--tol", type=float, default=2.0); ap.add_argument("--measure-only", action="store_true")
ap.add_argument("--hf", type=float, default=0.0, help="0..1: the source's luma high-pass (stubble) in the chin band")
a = ap.parse_args()
L = np.load(a.lm)["lm"][..., :2].astype(np.float32)
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); FB = W * H * 3


def reader(p):
    q = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
    while True:
        b = q.stdout.read(FB)
        if len(b) < FB: return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def band(p):
    lo = p[LO]; mw = np.linalg.norm(p[61] - p[291]); chin = p[152]
    # lower lip outer contour moved down 0.06 mouth width (skip the lip itself) ... down to the chin line 0.05 mw above the chin tip
    top = lo + np.array([0, 0.06 * mw], np.float32); bot_y = chin[1] - 0.05 * mw
    bot = np.stack([np.linspace(lo[0, 0], lo[-1, 0], 9), np.full(9, bot_y)], 1).astype(np.float32)
    poly = np.concatenate([top, bot[::-1]]).astype(np.int32)
    m = cv2.fillPoly(np.zeros((H, W), np.uint8), [poly], 255); return m, mw


ex = []; w = None
if a.out and not a.measure_only:
    tmp = a.out + ".v.mp4"
    w = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "16",
                          "-preset", "medium", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", tmp], stdin=subprocess.PIPE)
nfix = 0
for k, (F, S) in enumerate(zip(reader(a.final), reader(a.master))):
    out = F
    p = L[k] if k < len(L) else None
    if p is not None and np.isfinite(p).all():
        m, mw = band(p)
        if m.sum() > 0:
            fl = cv2.cvtColor(F, cv2.COLOR_BGR2LAB).astype(np.float32); sl = cv2.cvtColor(S, cv2.COLOR_BGR2LAB).astype(np.float32)
            d = fl[..., 1] - sl[..., 1]; ex.append(float(d[m > 0].mean()))
            if w is not None:
                pink = np.clip((d - a.tol) / 4.0, 0, 1) * (m > 0)   # only where ours is pinker than the source
                al = cv2.GaussianBlur(pink.astype(np.float32), (0, 0), max(1.0, 0.04 * mw))
                if al.max() > 0.01:
                    fl[..., 1] = fl[..., 1] * (1 - al) + sl[..., 1] * al; fl[..., 2] = fl[..., 2] * (1 - al) + sl[..., 2] * al
                if a.hf > 0:   # the smear is also a smooth, stubble-less patch: the source's fine detail (luma high-pass) in the band, KeySync's shading kept
                    bw = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), max(1.0, 0.05 * mw)) * a.hf
                    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.5)
                    fl[..., 0] = fl[..., 0] + bw * (hp(sl[..., 0]) - hp(fl[..., 0]))
                if al.max() > 0.01 or a.hf > 0:
                    out = cv2.cvtColor(np.clip(fl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR); nfix += 1
        else: ex.append(np.nan)
    else: ex.append(np.nan)
    if w is not None: w.stdin.write(np.ascontiguousarray(out).tobytes())
if w is not None:
    w.stdin.close(); w.wait()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-i", a.audio or a.final, "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", a.out], check=True); os.remove(tmp)
e = np.array(ex); ok = np.isfinite(e)
worst = sorted([(round(float(np.nanmean(e[i:i + 25])), 2), i // 25) for i in range(0, len(e), 25) if np.isfinite(e[i:i + 25]).any()], reverse=True)[:6]
print("K2_CHIN " + json.dumps({"frames": len(e), "a_excess_median": round(float(np.nanmedian(e)), 2), "a_excess_p95": round(float(np.nanpercentile(e[ok], 95)), 2),
                               "share_gt4": round(float((e[ok] > 4).mean()), 3), "worst_seconds": worst, "fixed_frames": nfix}), flush=True)
