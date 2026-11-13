#!/usr/bin/env python3
"""r18 fps: judder numbers for A / B2 / B3 / source on fixed windows, at the SOURCE frame rate (A is upsampled by frame repeat = how a 25 fps
file shows on a 30 Hz timeline, same as fps_ab.py).
  boxes   from the KeySync change mask (|A - master| on Y over the window): lips = the mask bbox core, face = the bbox x1.8 (chin, cheeks, nose)
  cv      fps_ab.py's number: std / mean of the frame-to-frame mean |diff| (repeats -> zeros + jumps -> high; source = reference)
  p5      share of the diff signal's power at 5 Hz +-0.5 (the 1-in-6 cadence of 25-on-30: A freezes everything, B2 freezes the mouth)
  stut    lips-vs-face: share of frames where the lips' change < 25 % of their median while the face's change > 50 % of its median
          (the mouth stops while the head moves = what B2 adds)
    python fps_judder2.py --src SRC --master master_25fps.mp4 --a A.mp4 --v NAME=PATH ... --win 6.04 --win 11.4 [--len 3] [--json out.json]"""
import argparse, json, subprocess
import numpy as np


def frames(path, t0, ln, fps, crop=None, w=None):
    vf = f"fps={fps}" + (f",crop={crop[2] - crop[0]}:{crop[3] - crop[1]}:{crop[0]}:{crop[1]}" if crop else "") + (f",scale={w}:-2" if w else "")
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    W, H = map(int, p.stdout.strip().split(",")[:2])
    cw, ch = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (W, H)
    ow, oh = (w, int(round(ch * w / cw / 2)) * 2) if w else (cw, ch)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t0}", "-t", f"{ln}", "-i", path, "-vf", vf, "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8); n = len(f) // (ow * oh); return f[: n * ow * oh].reshape(n, oh, ow).astype(np.float32)


def stats(fr, fps):
    d = np.abs(np.diff(fr, axis=0)).mean((1, 2)); cv = float(d.std() / (d.mean() + 1e-6))
    x = d - d.mean(); P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; fq = np.fft.rfftfreq(len(x), 1 / fps)
    p5 = float(P[(fq > 4.5) & (fq < 5.5)].sum() / (P[fq > 0.3].sum() + 1e-9)); return d, round(cv, 3), round(p5, 3)


def motion(ff, lips_in_face):
    """DIS flow between consecutive face crops: head = |median flow of the ring (face minus lips)|, art = mean |lips flow - ring median|
    (articulation speed). holds = share of frames where art < 30 % of its median while head > 50 % of its median (mouth stops, head moves)."""
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); x0, y0, x1, y1 = lips_in_face; head, art = [], []
    ring = np.ones(ff.shape[1:], bool); ring[y0:y1, x0:x1] = False
    for a_, b_ in zip(ff[:-1], ff[1:]):
        F = dis.calc(a_.astype(np.uint8), b_.astype(np.uint8), None); m = np.median(F[ring], axis=0)
        head.append(float(np.hypot(*m))); art.append(float(np.hypot(*(F[y0:y1, x0:x1] - m).T).mean()))
    head, art = np.array(head), np.array(art)
    holds = float(np.mean((art < 0.3 * np.median(art)) & (head > 0.5 * np.median(head))))
    jerk = float(np.abs(np.diff(art)).mean() / (art.mean() + 1e-6))   # frame-to-frame change of articulation speed (smooth speech -> low)
    return round(holds, 3), round(jerk, 3), round(float(np.abs(np.diff(head)).mean() / (head.mean() + 1e-6)), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--master", required=True); ap.add_argument("--a", required=True)
    ap.add_argument("--v", action="append", default=[]); ap.add_argument("--win", type=float, action="append", required=True); ap.add_argument("--len", type=float, default=3.0)
    ap.add_argument("--json", default=""); a = ap.parse_args()
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,width,height", "-of", "csv=p=0", a.src], capture_output=True, text=True).stdout.strip().split(",")
    W, H = int(r[0]), int(r[1]); n_, d_ = r[2].split("/"); fps = round(float(n_) / float(d_), 3)
    vids = [("A_25fps", a.a)] + [tuple(s.split("=", 1)) for s in a.v] + [("source", a.src)]; res = {"fps": fps, "windows": {}}
    for t0 in a.win:
        dm = np.abs(frames(a.a, t0, a.len, 25) - frames(a.master, t0, a.len, 25)).mean(0); thr = max(4.0, np.percentile(dm, 99) * 0.25)
        ys, xs = np.nonzero(dm > thr); cy, cx = np.median(ys), np.median(xs); hy, hx = np.percentile(ys, 95) - np.percentile(ys, 5), np.percentile(xs, 95) - np.percentile(xs, 5)
        box = lambda s: (max(0, int(cx - s * hx / 2)) & ~1, max(0, int(cy - s * hy / 2)) & ~1, min(W, int(cx + s * hx / 2)) & ~1, min(H, int(cy + s * hy / 2)) & ~1)
        lips, face = box(0.7), box(1.8); out = {"lips_box": lips, "face_box": face}
        for name, p in vids:
            fl = frames(p, t0, a.len, fps, lips, 96); ff = frames(p, t0, a.len, fps, face, 160)
            dl, cvl, p5l = stats(fl, fps); df, cvf, p5f = stats(ff, fps)
            sc = 320 / (face[2] - face[0]); fm = frames(p, t0, a.len, fps, face, 320)
            lf = (int((lips[0] - face[0]) * sc), int((lips[1] - face[1]) * sc), int((lips[2] - face[0]) * sc), int((lips[3] - face[1]) * sc))
            holds, ajerk, hjerk = motion(fm, lf)
            out[name] = {"face_cv": cvf, "lips_cv": cvl, "lips_p5": p5l, "mouth_holds": holds, "art_jerk": ajerk, "head_jerk": hjerk}
        res["windows"][f"{t0:g}"] = out
    print(json.dumps(res, indent=1))
    if a.json: open(a.json, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
