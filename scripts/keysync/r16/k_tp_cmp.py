import sys, json, subprocess, numpy as np
s_, p_, bx, ms = sys.argv[1:5]; b = json.load(open(bx)); x, y, s = b["x"], b["y"], b["side"]
cnt = lambda f: int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", f], capture_output=True, text=True).stdout or -1)
def fr(f):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f, "-vf", f"crop={s}:{s}:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        r = p.stdout.read(s * s * 3)
        if len(r) < s * s * 3: break
        yield np.frombuffer(r, np.uint8).reshape(s, s, 3).astype(np.float32)
ps = []
for A, B in zip(fr(s_), fr(p_)):
    mse = float(((A - B) ** 2).mean()); ps.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
ps = np.array(ps) if ps else np.array([0.0]); w = int(np.argmin(ps)); bad = np.where(ps < 45)[0]
print(f"{s_.split('/')[-1][:-8]}: frames ser {cnt(s_)} par {cnt(p_)} master {cnt(ms)} | PSNR mean {ps.mean():.2f} min {ps.min():.2f} (f{w} {w/25:.2f}s) <45dB: {len(bad)} {bad[:12].tolist()}")
