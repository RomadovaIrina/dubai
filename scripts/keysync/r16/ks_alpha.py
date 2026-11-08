#!/usr/bin/env python3
"""r16 T: K2's ks_dissolve unmix (the same math: a_k = <S_k - T, B - T> / |B - T|^2 on a 1/4 grid, B^_k = (S_k - (1 - a_k) T) / a_k, a_k < amin -> the
nearest valid B^ held) for a face segment with SEVERAL dissolve zones (q03 shot 2: dissolving in at 29.3 s AND out at 37.7 s). Frame indices are
those of the FULL master; the T / B frames may lie outside [k0, k1). Writes the lossless segment master U [k0, k1) + <out>.alpha.json (absolute
frame keys), which ks_dissolve.py assemble reads (--part P.mp4:K0:U.mp4).
    python ks_alpha.py --master M.mp4 --k0 K0 --k1 K1 --zone D0:D1:T:B [--zone ...] --out U.mp4"""
import argparse, json, subprocess
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--k0", type=int, required=True); ap.add_argument("--k1", type=int, required=True)
ap.add_argument("--zone", action="append", default=[]); ap.add_argument("--amin", type=float, default=0.3); ap.add_argument("--out", required=True); a = ap.parse_args()
Z = [tuple(int(x) for x in z.split(":")) for z in a.zone]
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); FB = W * H * 3
need = set(range(a.k0, a.k1)) | {z[2] for z in Z} | {z[3] for z in Z}; last = max(need); F = {}
p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", a.master, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
for k in range(last + 1):
    b = p.stdout.read(FB)
    if len(b) < FB: break
    if k in need: F[k] = np.frombuffer(b, np.uint8).reshape(H, W, 3)
p.stdout.close(); p.wait()
al = {k: 1.0 for k in range(a.k0, a.k1)}; zt = {}
for d0, d1, tf, bf in Z:
    T = F[tf].astype(np.float32); B = F[bf].astype(np.float32); D = (B - T)[::4, ::4].ravel(); DD = float(np.dot(D, D))
    for k in range(max(d0, a.k0), min(d1, a.k1)):
        al[k] = float(np.clip(np.dot((F[k].astype(np.float32) - T)[::4, ::4].ravel(), D) / DD, 0, 1)); zt[k] = T
U = {}
for k in range(a.k0, a.k1):
    if al[k] >= 0.995: U[k] = F[k]
    elif al[k] >= a.amin: U[k] = np.clip((F[k].astype(np.float32) - (1 - al[k]) * zt[k]) / al[k], 0, 255).astype(np.uint8)
ok = sorted(U)
for k in range(a.k0, a.k1):
    if k not in U: U[k] = U[min(ok, key=lambda q: abs(q - k))]
w = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                      "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for k in range(a.k0, a.k1): w.stdin.write(U[k].tobytes())
w.stdin.close(); w.wait()
json.dump({"k0": a.k0, "k1": a.k1, "zones": Z, "alpha": {str(k): round(v, 4) for k, v in al.items()}}, open(a.out + ".alpha.json", "w"))
print(f"KS_ALPHA unmix [{a.k0}, {a.k1}) zones {Z}: {sum(1 for v in al.values() if v < 0.995)} dissolve frames, held {sum(1 for v in al.values() if v < a.amin)}", flush=True)
