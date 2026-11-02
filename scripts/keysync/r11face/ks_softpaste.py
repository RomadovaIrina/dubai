#!/usr/bin/env python3
"""r11 FACE: KeySync soft lower-face paste, outside fps_composite (every composite variant cost KeySync ~1 SyncNet; its own raw square paste
keeps the sync but draws the square's edges / background around the face). The (repaired) 25 fps KeySync layer, motion-interpolated to the
source fps like the raw paste (ffmpeg minterpolate mci), is pasted onto the NATIVE source frames only inside a soft lower-face mask:
the face oval below a line through the cheeks under the eyes and the nostrils, the union of the source's and KeySync's own (time-interpolated)
landmarks, dilated --dil and feathered --feather x the face width. Eyes / nose / forehead / background stay the source's. Audio = --audio.
    ks_softpaste.py --layer L25.mp4 --layer-lm L25.mediapipe.npz --src SRC.mp4 --src-lm source_lm_F.npz --box box.json --audio A --out OUT.mp4"""
import argparse, json, subprocess
import numpy as np, cv2

LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]   # oval, cheek -> chin -> cheek
TOP = [280, 327, 2, 98, 50]                                                                                       # under-eye cheek, nostrils, subnasale
ap = argparse.ArgumentParser()
for k in ["layer", "layer_lm", "src", "src_lm", "box", "audio", "out"]: ap.add_argument("--" + k.replace("_", "-"), required=True)
ap.add_argument("--dil", type=float, default=0.03); ap.add_argument("--feather", type=float, default=0.05); ap.add_argument("--interp", choices=["mci", "nearest", "hybrid"], default="mci")
ap.add_argument("--lip-dil", type=float, default=0.25, help="hybrid: lip mask dilation x mouth width")
ap.add_argument("--mask-out", default=""); a = ap.parse_args()
pr = lambda f, e: subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={e}", "-of", "csv=p=0", f], capture_output=True, text=True).stdout.strip()
W, H = [int(v) for v in pr(a.src, "width,height").split(",")]; fps = pr(a.src, "r_frame_rate"); fr = eval(fps)
b = json.load(open(a.box)); bx, by, bs = b["x"], b["y"], b["side"]
SL = np.load(a.src_lm)["lm"].astype(np.float32); KL = np.load(a.layer_lm)["lm"][..., :2].astype(np.float32)

def rd(cmd, n):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    while True:
        r = p.stdout.read(n)
        if len(r) < n: return
        yield r
vf = f"fps={fps}" if a.interp == "nearest" else f"minterpolate=fps={fps}:mi_mode=mci"
S = rd(["ffmpeg", "-v", "error", "-i", a.src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], W * H * 3)
K = rd(["ffmpeg", "-v", "error", "-i", a.layer, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], W * H * 3)
KN = rd(["ffmpeg", "-v", "error", "-i", a.layer, "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], W * H * 3) if a.interp == "hybrid" else None
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", a.audio, "-map", "0:v", "-map", "1:a",
                      "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", a.out], stdin=subprocess.PIPE)
boxm = np.zeros((H, W), np.float32); boxm[by:by + bs, bx:bx + bs] = 1
boxm = cv2.GaussianBlur(boxm, (0, 0), 6) * (boxm > 0)
masks = []; i = 0; lastk = None
for s_ in S:
    f = np.frombuffer(s_, np.uint8).reshape(H, W, 3); k_ = next(K, None)
    if k_ is not None: lastk = np.frombuffer(k_, np.uint8).reshape(H, W, 3)
    kn_ = next(KN, None) if KN is not None else None
    if lastk is None or i >= len(SL): w.stdin.write(f.tobytes()); i += 1; continue
    t = i / fr * 25.0; j0 = min(int(t), len(KL) - 1); j1 = min(j0 + 1, len(KL) - 1); u = t - int(t)
    kl = KL[j0] * (1 - u) + KL[j1] * u; sl = SL[i]; fw = np.linalg.norm(sl[454] - sl[234])
    if kn_ is not None:   # hybrid: the nearest generated frame inside its lips (+ a margin), the mci blend elsewhere
        jn = min(int(round(t)), len(KL) - 1); mw = np.linalg.norm(KL[jn][291] - KL[jn][61]); lm_ = np.zeros((H, W), np.uint8)
        for L in (KL[jn], kl): cv2.fillPoly(lm_, [np.round(L[LIPS]).astype(np.int32)], 1)
        dd = max(1, int(a.lip_dil * mw)); lm_ = cv2.dilate(lm_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dd + 1, 2 * dd + 1)))
        lm_ = cv2.GaussianBlur(lm_.astype(np.float32), (0, 0), max(1.0, 0.12 * mw))[..., None]
        lastk = (lastk.astype(np.float32) * (1 - lm_) + np.frombuffer(kn_, np.uint8).reshape(H, W, 3).astype(np.float32) * lm_).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    for L in (sl, kl): cv2.fillPoly(m, [np.round(L[LOW + TOP]).astype(np.int32)], 1)
    d = max(1, int(a.dil * fw)); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, a.feather * fw)) * boxm
    if a.mask_out and i % 5 == 0: masks.append((m * 255).astype(np.uint8)[::4, ::4])
    o = f.astype(np.float32) * (1 - m[..., None]) + lastk.astype(np.float32) * m[..., None]
    w.stdin.write(np.clip(o + 0.5, 0, 255).astype(np.uint8).tobytes()); i += 1
w.stdin.close(); w.wait()
if a.mask_out: np.save(a.mask_out, np.array(masks))
print(f"== ks_softpaste {i} frames @ {fps} ({a.interp}, dil {a.dil}, feather {a.feather}) -> {a.out}")
