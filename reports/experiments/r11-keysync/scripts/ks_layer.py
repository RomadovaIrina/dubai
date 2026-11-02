#!/usr/bin/env python3
"""KeySync bake-off: KeySync's 512^2 / 25 fps crop as a full-frame 25 fps LatentSync-style layer (<work>/<id>_<tag>_25fps_ls_raw.mp4) for our
composite (r7_m7_comp.sh): the crop is scaled back to its box and placed into the master_25fps frames.
  --shift 1 : KeySync frame i is shown at master frame i+1 (its output leads the audio by one 25 fps frame: SyncNet best offset -1 -> 0)
  --cm S    : colour match of the box to the master box, Lab mean / std per channel, the gains smoothed over +-S frames (0 = off)
    ks_layer.py --ks KS.mp4 --master master_25fps.mp4 --box box.json --out LAYER.mp4 [--shift 1 --cm 12]"""
import argparse, json, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--ks", required=True); ap.add_argument("--master", required=True); ap.add_argument("--box", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--shift", type=int, default=1); ap.add_argument("--cm", type=int, default=12); a = ap.parse_args()
b = json.load(open(a.box)); x, y, s = b["x"], b["y"], b["side"]
def frames(p, w, h, vf=None):
    c = ["ffmpeg", "-v", "error", "-i", p] + (["-vf", vf] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(c, capture_output=True, check=True).stdout; return np.frombuffer(r, np.uint8).reshape(-1, h, w, 3)
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1])
M = frames(a.master, W, H); K = frames(a.ks, s, s, f"scale={s}:{s}:flags=lanczos"); n = len(M)
idx = np.clip(np.arange(n) - a.shift, 0, len(K) - 1); valid = (np.arange(n) - a.shift) < len(K)
lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
g = np.zeros((n, 2, 3), np.float32); g[:, 0] = 1
if a.cm > 0:   # per-frame gains: master box stats / KeySync box stats, smoothed in time
    st = []
    for i in range(n):
        km, ks_ = lab(K[idx[i]]).reshape(-1, 3).mean(0), lab(K[idx[i]]).reshape(-1, 3).std(0) + 1e-3
        mm, ms = lab(M[i][y:y + s, x:x + s]).reshape(-1, 3).mean(0), lab(M[i][y:y + s, x:x + s]).reshape(-1, 3).std(0)
        st.append(np.stack([ms / ks_, mm - ms / ks_ * km]))
    st = np.array(st); k = np.ones(2 * a.cm + 1) / (2 * a.cm + 1)
    for j in range(2):
        for c in range(3): g[:, j, c] = np.convolve(np.pad(st[:, j, c], a.cm, mode="edge"), k, mode="valid")
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for i in range(n):
    f = M[i].copy()
    if valid[i]:
        kk = K[idx[i]]
        if a.cm > 0: kk = cv2.cvtColor(np.clip(lab(kk) * g[i, 0] + g[i, 1], 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        f[y:y + s, x:x + s] = kk
    w.stdin.write(f.tobytes())
w.stdin.close(); w.wait()
print(f"== ks_layer {n} frames (KeySync {len(K)}, shift {a.shift}, colour match +-{a.cm}) -> {a.out}; mean L gain {g[:, 0, 0].mean():.3f} offset {g[:, 1, 0].mean():.2f}")
