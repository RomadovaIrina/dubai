#!/usr/bin/env python3
"""KeySync bake-off: raw paste of the KeySync 512^2 / 25 fps face crop back into the source frames (judge the generator itself).
The crop is motion-interpolated to the source fps (ffmpeg minterpolate mci), scaled back to the box, and pasted with only a 24 px feather at
the box edges; then muxed with the given audio.  ks_paste.py <ks.mp4> <src.mp4> <box.json> <audio> <out.mp4>"""
import json, os, subprocess, sys, numpy as np
ks, src, boxf, aud, out = sys.argv[1:6]
b = json.load(open(boxf)); x, y, s, W, H = b["x"], b["y"], b["side"], b["W"], b["H"]
fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip())
rd = lambda cmd, sh: subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=sh * 4)
A = rd(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], W * H * 3)
sh = int(os.environ.get("KS_SHIFT", "0"))   # KeySync AV offset fix in 25 fps frames: >0 delay (first frame cloned), <0 advance (frames dropped at the start)
pre = f"tpad=start={sh}:start_mode=clone," if sh > 0 else (f"trim=start_frame={-sh},setpts=PTS-STARTPTS," if sh < 0 else "")
K = rd(["ffmpeg", "-v", "error", "-i", ks, "-vf", pre + (f"fps={fps},scale" if os.environ.get("KS_NEAREST") == "1" else f"minterpolate=fps={fps}:mi_mode=mci,scale") + f"={s}:{s}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], s * s * 3)
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-i", aud, "-map", "0:v", "-map", "1:a",
                      "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], stdin=subprocess.PIPE)
def ramp(lo_edge, hi_edge):   # 24 px feather, none on a box side that lies on the frame border (nothing to blend with there)
    i = np.arange(s, dtype=np.float32); a = np.ones(s, np.float32)
    if not lo_edge: a = np.minimum(a, i / 24.0)
    if not hi_edge: a = np.minimum(a, (s - 1 - i) / 24.0)
    return np.clip(a, 0, 1)
m = np.minimum.outer(ramp(y == 0, y + s == H), ramp(x == 0, x + s == W))[..., None]
fk = None; i = 0
while True:
    a = A.stdout.read(W * H * 3)
    if len(a) < W * H * 3: break
    f = np.frombuffer(a, np.uint8).reshape(H, W, 3).copy()
    k = K.stdout.read(s * s * 3)
    if len(k) == s * s * 3: fk = np.frombuffer(k, np.uint8).reshape(s, s, 3)
    if fk is not None:
        roi = f[y:y + s, x:x + s].astype(np.float32)
        f[y:y + s, x:x + s] = (roi * (1 - m) + fk.astype(np.float32) * m + 0.5).astype(np.uint8)
    w.stdin.write(f.tobytes()); i += 1
w.stdin.close(); w.wait(); A.kill(); K.kill()
print(f"pasted {i} frames (source {n}) -> {out}")
