"""per-class check of the native-fps output B vs A (25 fps final) and the source:
format: fps / frame count / duration of B == the source (hi);  2 windows (30 % and 60 % of the clip, 3 s), boxes from |A - source@25|
numbers: art_jerk / head_jerk on the source-fps timeline (A repeated, as fps_judder2), pulse = lip sharpness pulse % at each file's OWN fps,
mcflick = motion-compensated lip texture flicker on the source-fps timeline.
    python eval_class.py --src HI --a A_FINAL --b B [--name X] [--json out.json]"""
import argparse, json, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fps_judder2 import frames, motion


def probe(p):
    r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                   "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration", "-of", "json", p], capture_output=True, text=True).stdout)
    s = r["streams"][0]; n, d = s["avg_frame_rate"].split("/")
    return {"w": s["width"], "h": s["height"], "fps": round(float(n) / float(d), 3), "frames": int(s["nb_read_packets"]), "dur": round(float(r["format"]["duration"]), 3)}


def pulse(p, t0, ln, fps, box):
    L = frames(p, t0, ln, fps, box); s = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in L])
    med = np.array([np.median(s[max(0, i - 3):i + 4]) for i in range(len(s))]); return round(float((s / med - 1).std() * 100), 2)


def mcflick(p, t0, ln, fps, box):
    L = frames(p, t0, ln, fps, box, 160); dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    hp = [f - cv2.GaussianBlur(f, (0, 0), 1.5) for f in L]; gy, gx = np.mgrid[0:L.shape[1], 0:L.shape[2]].astype(np.float32); fl = []
    for a_, b_, ha, hb in zip(L[:-1], L[1:], hp[:-1], hp[1:]):
        F = dis.calc(b_.astype(np.uint8), a_.astype(np.uint8), None)
        wa = cv2.remap(ha, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); fl.append(float(np.abs(hb - wa)[6:-6, 6:-6].mean()))
    return round(float(np.mean(fl)), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--name", default=""); ap.add_argument("--len", type=float, default=3.0); ap.add_argument("--json", default=""); a = ap.parse_args()
    S, A, B = probe(a.src), probe(a.a), probe(a.b); fps = S["fps"]
    res = {"name": a.name, "src": S, "A": A, "B": B,
           "format_ok": B["w"] == S["w"] and B["h"] == S["h"] and abs(B["fps"] - S["fps"]) < 0.01 and B["frames"] == S["frames"] and abs(B["dur"] - S["dur"]) < 0.1, "windows": {}}
    W, H = S["w"], S["h"]
    for t0 in (round(S["dur"] * 0.3, 2), round(S["dur"] * 0.6, 2)):
        fa, fs_ = frames(a.a, t0, a.len, 25, None, 480), frames(a.src, t0, a.len, 25, None, 480); m_ = min(len(fa), len(fs_))
        dm = np.abs(fa[:m_] - fs_[:m_]).mean(0); sc = W / 480
        thr = max(4.0, np.percentile(dm, 99) * 0.25); ys, xs = np.nonzero(dm > thr)
        if len(ys) < 20: res["windows"][f"{t0:g}"] = "no KeySync change here"; continue
        cy, cx = np.median(ys) * sc, np.median(xs) * sc; hy, hx = (np.percentile(ys, 95) - np.percentile(ys, 5)) * sc, (np.percentile(xs, 95) - np.percentile(xs, 5)) * sc
        hy, hx = max(hy, 24), max(hx, 24)
        box = lambda s: (max(0, int(cx - s * hx / 2)) & ~1, max(0, int(cy - s * hy / 2)) & ~1, min(W, int(cx + s * hx / 2)) & ~1, min(H, int(cy + s * hy / 2)) & ~1)
        lips, face = box(0.7), box(1.8); o = {"lips_box": lips}
        for nm, p, own in (("A_25fps", a.a, 25), ("B", a.b, fps), ("source", a.src, fps)):
            fs = 320 / (face[2] - face[0]); fm = frames(p, t0, a.len, fps, face, 320)
            lf = (int((lips[0] - face[0]) * fs), int((lips[1] - face[1]) * fs), int((lips[2] - face[0]) * fs), int((lips[3] - face[1]) * fs))
            _, aj, hj = motion(fm, lf)
            o[nm] = {"art_jerk": aj, "head_jerk": hj, "pulse_%": pulse(p, t0, a.len, own, lips), "mcflick": mcflick(p, t0, a.len, fps, lips)}
        res["windows"][f"{t0:g}"] = o
    print(json.dumps(res))
    if a.json: open(a.json, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
