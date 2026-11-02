#!/usr/bin/env python3
"""r11 FACE: KeySync layer repair before our composite (every step its own flag, default off).
  --guard K     explosion guard: KeySync must not touch the upper face, so a frame whose box above the nose differs from the master by more
                than K x its running median (or whose layer eye landmarks leave the source's by > 0.08 IOD) is replaced by the master, +-2 frames,
                with a 2-frame cross-fade (old03 f2195-2198: a melted face)
  --skin-cm S   colour match of the box to the master on the SKIN only (both mouths + lips excluded), Lab mean / std, smoothed +-S frames
  --lip-chroma W  inside the layer's lips (outer minus inner polygon, feathered), Lab a* / b* moved by W x (source lip median - layer lip median),
                the deltas smoothed +-12 frames (KeySync draws the upper lip almost skin-coloured)
  --chin tex|src  below the layer's lower lip down to the jaw line: tex = the layer's low frequencies (shape, shading) + the source's luma detail
                and chroma; src = the source pixels (mask5-like: the source chin under the KeySync mouth)
    ks_fix.py --layer L.mp4 --master master_25fps.mp4 --box box.json --layer-lm L.mediapipe.npz --src-lm source_lm_F.npz --out OUT.mp4 [flags]"""
import argparse, json, subprocess
import numpy as np, cv2

LIPS_O = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LO = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]   # left -> right under the chin
EYES = [33, 133, 159, 145, 362, 263, 386, 374]

ap = argparse.ArgumentParser()
for k in ["layer", "master", "box", "layer_lm", "src_lm", "out"]: ap.add_argument("--" + k.replace("_", "-"), required=True)
ap.add_argument("--guard", type=float, default=0); ap.add_argument("--skin-cm", type=int, default=0); ap.add_argument("--lip-chroma", type=float, default=0)
ap.add_argument("--chin", choices=["off", "tex", "src"], default="off"); ap.add_argument("--chin-sigma", type=float, default=0.05)
ap.add_argument("--guard-report", default=""); a = ap.parse_args()
b = json.load(open(a.box)); x0, y0, s = b["x"], b["y"], b["side"]
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True)
W, H = [int(v) for v in pr.stdout.strip().split(",")]

def frames(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, H, W, 3)

M, Lr = frames(a.master), frames(a.layer); n = min(len(M), len(Lr))
KL = np.load(a.layer_lm)["lm"][..., :2].astype(np.float32); sz = np.load(a.src_lm); SL = sz["lm"][sz["j"]].astype(np.float32)
n = min(n, len(KL), len(SL))

def poly(pts, dil=0):
    m = np.zeros((s, s), np.uint8); cv2.fillPoly(m, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
    if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
    return m.astype(bool)

lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
smooth = lambda v, r: np.array([np.convolve(np.pad(v[:, c], r, mode="edge"), np.ones(2 * r + 1) / (2 * r + 1), mode="valid") for c in range(v.shape[1])]).T
mw = np.linalg.norm(SL[:n, 291] - SL[:n, 61], axis=1); iod = np.linalg.norm(SL[:n, 33] - SL[:n, 263], axis=1)

# ---- explosion guard
bad = np.zeros(n, bool); gw = np.zeros(n, np.float32)
if a.guard > 0:
    d = np.zeros(n); eye = np.linalg.norm(KL[:n, EYES] - SL[:n, EYES], axis=2).mean(1) / iod
    for i in range(n):
        yt = int(SL[i, 1, 1]) - y0                                                    # nose tip row in the box
        if yt > 8: d[i] = np.abs(lab(Lr[i][y0:y0 + yt, x0:x0 + s])[..., 0] - lab(M[i][y0:y0 + yt, x0:x0 + s])[..., 0]).mean()
    med = np.array([np.median(d[max(0, i - 50):i + 51]) for i in range(n)])
    bad = (d > a.guard * np.maximum(med, 0.5)) | (eye > 0.08)
    core = np.where(bad)[0]
    for i in core:
        for k in range(-4, 5):
            j = i + k
            if 0 <= j < n: gw[j] = max(gw[j], 1.0 if abs(k) <= 2 else (1.0 - (abs(k) - 2) / 3))
    if a.guard_report: json.dump({"frames": core.tolist(), "d": d.round(2).tolist(), "eye": eye.round(3).tolist()}, open(a.guard_report, "w"))
    print(f"== guard: {len(core)} frames flagged {core.tolist()[:30]}; median upper-face diff {np.median(d):.2f}", flush=True)

# ---- per-frame stats (skin colour match, lip chroma)
cm = np.zeros((n, 2, 3), np.float32); cm[:, 0] = 1; lipd = np.zeros((n, 2), np.float32); lipk = np.zeros((n, 2), np.float32); lip_ok = np.zeros(n, bool)
if a.skin_cm > 0 or a.lip_chroma > 0:
    st = np.zeros((n, 2, 3), np.float32); st[:, 0] = 1; ok = np.zeros(n, bool)
    for i in range(n):
        Lk, Lm = lab(Lr[i][y0:y0 + s, x0:x0 + s]), lab(M[i][y0:y0 + s, x0:x0 + s]); dil = int(0.12 * mw[i])
        if a.skin_cm > 0:
            sk = ~(poly(KL[i, LIPS_O], dil) | poly(SL[i, LIPS_O], dil))
            km, ksd = Lk[sk].mean(0), Lk[sk].std(0) + 1e-3; mm, msd = Lm[sk].mean(0), Lm[sk].std(0)
            st[i] = np.stack([msd / ksd, mm - msd / ksd * km]); ok[i] = True
        if a.lip_chroma > 0:
            lk = poly(KL[i, LIPS_O]) & ~poly(KL[i, INNER]); ls_ = poly(SL[i, LIPS_O]) & ~poly(SL[i, INNER])
            if lk.sum() > 30 and ls_.sum() > 30:
                lipd[i] = np.median(Lm[ls_][:, 1:], 0); lipk[i] = np.median(Lk[lk][:, 1:], 0); lip_ok[i] = True
    if a.skin_cm > 0: cm = np.stack([smooth(st[:, 0], a.skin_cm), smooth(st[:, 1], a.skin_cm)], 1)
    if a.lip_chroma > 0:
        idx = np.where(lip_ok)[0]
        lipd = lipd - (lipk * cm[:, 0, 1:] + cm[:, 1, 1:])                          # source lip chroma - the layer's AFTER the skin colour match
        for c in range(2): lipd[:, c] = np.interp(np.arange(n), idx, lipd[idx, c])
        lipd = smooth(lipd, 12)
        print(f"== lip chroma: median delta a* {np.median(lipd[:, 0]):+.2f} b* {np.median(lipd[:, 1]):+.2f} ({lip_ok.mean()*100:.0f} % frames measured)", flush=True)

# ---- write
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10",
                      "-preset", "medium", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for i in range(len(Lr)):
    f = Lr[i].copy()
    if i < n:
        k = lab(f[y0:y0 + s, x0:x0 + s]); m_ = lab(M[i][y0:y0 + s, x0:x0 + s]); fe = max(1.0, 0.05 * mw[i])
        if a.skin_cm > 0: k = k * cm[i, 0] + cm[i, 1]
        if a.lip_chroma > 0:
            lm = cv2.GaussianBlur((poly(KL[i, LIPS_O], int(0.02 * mw[i])) & ~poly(KL[i, INNER])).astype(np.float32), (0, 0), fe)
            k[..., 1:] += a.lip_chroma * lipd[i] * lm[..., None]
        if a.chin != "off":
            cp = np.concatenate([KL[i, LO], KL[i, JAW]])                                # lower lip (right -> left) + jaw (left -> right)
            cmk = cv2.erode(poly(cp).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(0.04 * mw[i]))).astype(np.float32)
            cmk = cv2.GaussianBlur(cmk, (0, 0), max(1.0, 0.06 * mw[i]))[..., None]
            if a.chin == "src": k = k * (1 - cmk) + m_ * cmk
            else:
                sg = max(1.0, a.chin_sigma * mw[i])
                lo_k = cv2.GaussianBlur(k[..., 0], (0, 0), sg); hf_m = m_[..., 0] - cv2.GaussianBlur(m_[..., 0], (0, 0), sg)
                t = k.copy(); t[..., 0] = lo_k + hf_m; t[..., 1:] = cv2.GaussianBlur(m_[..., 1:], (0, 0), sg)
                k = k * (1 - cmk) + t * cmk
        f[y0:y0 + s, x0:x0 + s] = cv2.cvtColor(np.clip(k, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        if gw[i] > 0: f = np.clip(f.astype(np.float32) * (1 - gw[i]) + M[i].astype(np.float32) * gw[i], 0, 255).astype(np.uint8)
    w.stdin.write(f.tobytes())
w.stdin.close(); w.wait()
print(f"== ks_fix -> {a.out} (guard {a.guard} -> {int((gw > 0).sum())} frames touched, skin-cm {a.skin_cm}, lip-chroma {a.lip_chroma}, chin {a.chin})")
