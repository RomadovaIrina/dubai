"""all eye-proxy numbers for NAME=PATH videos on the two m18 windows (boxes from judder.json):
art_jerk / head_jerk (fps_judder2.motion), pulse = lip sharpness pulse % (Laplacian var vs 7-frame running median),
mcflick = motion-compensated texture flicker in the lips: mean |cur - warp(prev)| of the high-pass (texture) after DIS alignment."""
import sys, json, numpy as np, cv2
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fps_judder2 import frames, motion

r = json.load(open("judder.json")); vids = [a.split("=", 1) for a in sys.argv[1:]]; out = {}
for w in ["6.04", "11.4"]:
    o = r["windows"][w]; lips, face = o["lips_box"], o["face_box"]; out[w] = {}
    for name, p in vids:
        sc = 320 / (face[2] - face[0]); fm = frames(p, float(w), 3.0, 30.01, face, 320)
        lf = (int((lips[0] - face[0]) * sc), int((lips[1] - face[1]) * sc), int((lips[2] - face[0]) * sc), int((lips[3] - face[1]) * sc))
        holds, aj, hj = motion(fm, lf)
        L = frames(p, float(w), 3.0, 30.01, lips); s = np.array([cv2.Laplacian(f, cv2.CV_32F).var() for f in L])
        med = np.array([np.median(s[max(0, i - 3):i + 4]) for i in range(len(s))]); pulse = float((s / med - 1).std() * 100)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); fl = []
        hp = [f - cv2.GaussianBlur(f, (0, 0), 2.0) for f in L]
        gy, gx = np.mgrid[0:L.shape[1], 0:L.shape[2]].astype(np.float32)
        for a_, b_, ha, hb in zip(L[:-1], L[1:], hp[:-1], hp[1:]):
            F = dis.calc(b_.astype(np.uint8), a_.astype(np.uint8), None)   # b(x) ~ a(x + F)
            wa = cv2.remap(ha, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fl.append(float(np.abs(hb - wa)[8:-8, 8:-8].mean()))
        out[w][name] = {"art_jerk": aj, "head_jerk": hj, "pulse_%": round(pulse, 2), "mcflick": round(float(np.mean(fl)), 3)}
        print(w, f"{name:18s}", "  ".join(f"{k}={v}" for k, v in out[w][name].items()), flush=True)
json.dump(out, open("eval3.json", "w"), indent=1)
