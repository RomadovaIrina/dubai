#!/usr/bin/env python3
"""KeySync bake-off: raw paste of the KeySync 512^2 / 25 fps face crop back into the source frames (judge the generator itself).
The crop is motion-interpolated to the source fps (ffmpeg minterpolate mci), scaled back to the box, and pasted with only a 24 px feather at
the box edges; then muxed with the given audio.  ks_paste.py <ks.mp4> <src.mp4> <box.json> <audio> <out.mp4>"""
import json, os, subprocess, sys, numpy as np, ks_cms
ks, src, boxf, aud, out = sys.argv[1:6]
b = json.load(open(boxf)); x, y, s, W, H = b["x"], b["y"], b["side"], b["W"], b["H"]
fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip())
rd = lambda cmd, sh: subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=sh * 4)
A = rd(["ffmpeg", "-v", "error", "-i", src] + ks_cms.dec(pix="rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], W * H * 3)
sh = int(os.environ.get("KS_SHIFT", "0"))   # KeySync AV offset fix in 25 fps frames: >0 delay (first frame cloned), <0 advance (frames dropped at the start)
pre = f"tpad=start={sh}:start_mode=clone," if sh > 0 else (f"trim=start_frame={-sh},setpts=PTS-STARTPTS," if sh < 0 else "")
K = rd(["ffmpeg", "-v", "error", "-i", ks] + ks_cms.dec(pre + (f"fps={fps},scale" if os.environ.get("KS_NEAREST") == "1" else f"minterpolate=fps={fps}:mi_mode=mci,scale") + f"={s}:{s}:flags=lanczos", "rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], s * s * 3)
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", aud, "-map", "0:v", "-map", "1:a",
                      "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + ["-c:a", "aac", "-b:a", "192k", "-shortest", out], stdin=subprocess.PIPE)
def ramp(lo_edge, hi_edge):   # 24 px feather, none on a box side that lies on the frame border (nothing to blend with there)
    i = np.arange(s, dtype=np.float32); a = np.ones(s, np.float32)
    if not lo_edge: a = np.minimum(a, i / 24.0)
    if not hi_edge: a = np.minimum(a, (s - 1 - i) / 24.0)
    return np.clip(a, 0, 1)
m = np.minimum.outer(ramp(y == 0, y + s == H), ramp(x == 0, x + s == W))[..., None]
# r13 detail transfer (KS_HF=<sigma px>, KS_HF_LM=<source mediapipe npz at the source fps>): outside the source's lower face (cheeks under the eyes ->
# nostrils -> jaw, dilated KS_HF_DIL x the face width, feathered) the pasted box gets the SOURCE's high frequencies (src - blur) in place of
# KeySync's own (ks - blur): eyes, brows, hair, beard edges and background keep the native resolution; KeySync's mouth / jaw keep their low + high band.
# KS_SHARP=<amount>: an unsharp mask (same sigma) on the remaining KeySync part.
HF = float(os.environ.get("KS_HF", "0")); SHP = float(os.environ.get("KS_SHARP", "0"))
if HF > 0:
    import cv2
    LOWF = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 280, 327, 2, 98, 50]
    SL = np.load(os.environ["KS_HF_LM"])["lm"][..., :2].astype(np.float32); DIL = float(os.environ.get("KS_HF_DIL", "0.12"))
    def hfmask(i):
        p = SL[min(i, len(SL) - 1)]
        if not np.isfinite(p).all() or p.max() <= 0: return None
        mk = np.zeros((s, s), np.uint8); cv2.fillPoly(mk, [np.round(p[LOWF] - [x, y]).astype(np.int32)], 1); fw = float(np.ptp(p[[234, 454], 0]))
        mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(DIL * fw) + 1,) * 2))
        return cv2.GaussianBlur(mk.astype(np.float32), (0, 0), max(1.0, 0.06 * fw))[..., None]   # 1 = KeySync's own detail
KN = float(os.environ.get("KS_KEEP_NOSE", "0"))
if KN > 0:   # the nose polygon (mediapipe): bridge 168 / 6, the alae 129 / 358 / 64 / 294, the nostril bottoms 98 / 327, the columella 2 kept OUT (lip motion lives below it)
    import cv2
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 129, 358, 64, 294, 98, 327, 48, 278, 115, 344]
    NL = np.load(os.environ["KS_HF_LM"])["lm"][..., :2].astype(np.float32)
fk = None; i = 0
while True:
    a = A.stdout.read(W * H * 3)
    if len(a) < W * H * 3: break
    f = np.frombuffer(a, np.uint8).reshape(H, W, 3).copy()
    k = K.stdout.read(s * s * 3)
    if len(k) == s * s * 3: fk = np.frombuffer(k, np.uint8).reshape(s, s, 3)
    if fk is not None:
        roi = f[y:y + s, x:x + s].astype(np.float32); kf = fk.astype(np.float32)
        if HF > 0:
            lm_ = hfmask(i)
            if lm_ is not None:
                hk = kf - cv2.GaussianBlur(kf, (0, 0), HF); hs = roi - cv2.GaussianBlur(roi, (0, 0), HF)
                kf = kf + (1 - lm_) * (hs - hk) + SHP * lm_ * hk
        mm = m
        if KN > 0:   # r15 KS_KEEP_NOSE=<feather sigma, IOD units>: the nose (bridge -> alae -> just above the subnasale) = the SOURCE pixels
            p = NL[min(i, len(NL) - 1)]
            if np.isfinite(p).all() and p.max() > 0:
                iod = float(np.linalg.norm(p[33] - p[263])); nk = np.zeros((s, s), np.uint8)
                cv2.fillPoly(nk, [np.round(cv2.convexHull((p[NOSE] - [x, y]).astype(np.float32))[:, 0]).astype(np.int32)], 1)
                nk = cv2.GaussianBlur(nk.astype(np.float32), (0, 0), max(1.0, KN * iod))[..., None]
                mm = m * (1 - nk)
        f[y:y + s, x:x + s] = np.clip(roi * (1 - mm) + kf * mm + 0.5, 0, 255).astype(np.uint8)
    w.stdin.write(f.tobytes()); i += 1
w.stdin.close(); w.wait(); A.kill(); K.kill()
print(f"pasted {i} frames (source {n}) -> {out}")
