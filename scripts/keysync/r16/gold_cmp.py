#!/usr/bin/env python3
"""r16 T golden test: the e2e output vs the ACCEPTED final, inside the KeySync face box (luma PSNR per second; the box of each frame from the e2e
run's box json, scaled from the face-work size to the output size), + the same vs the source (how much each one changed the face). Writes
<out>.json (per second + the worst seconds) and 0.5x clips "accepted | e2e | source" (face-box zoom, time burnt in) of the WORST seconds.
    gold_cmp.py --acc ACC.mp4 --e2e OUT.mp4 --src SRC --box box.json --work-size WxH --out PREFIX [--t1 40.88] [--worst 2]"""
import argparse, json, subprocess, numpy as np
ap = argparse.ArgumentParser()
for k in ("acc", "e2e", "src", "box", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--work-size", required=True); ap.add_argument("--t1", type=float, default=0); ap.add_argument("--worst", type=int, default=2); a = ap.parse_args()
size = lambda v: [int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", v], capture_output=True, text=True).stdout.strip().split(",")[:2]]
W, H = size(a.e2e); ww, wh = [int(x) for x in a.work_size.split("x")]; sc = W / ww
b = json.load(open(a.box)); s = int(round(b["side"] * sc)); XS, YS = b.get("xs"), b.get("ys")
bxy = lambda k: (int(round((XS[min(k, len(XS) - 1)] if XS else b["x"]) * sc)), int(round((YS[min(k, len(YS) - 1)] if YS else b["y"]) * sc)))
def rd(v):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", v, "-vf", f"setpts=PTS-STARTPTS,fps=25,scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        r = p.stdout.read(W * H)
        if len(r) < W * H: break
        yield np.frombuffer(r, np.uint8).reshape(H, W)
psnr = lambda m: 10 * np.log10(255 ** 2 / max(m, 1e-6))
E, S = [], []
for k, (fa, fe, fs) in enumerate(zip(rd(a.acc), rd(a.e2e), rd(a.src))):
    if a.t1 and k >= a.t1 * 25: break
    x, y = bxy(k); x0, y0 = max(0, x), max(0, y); c = lambda f: f[y0:y + s, x0:x + s].astype(np.float32)
    E.append(float(((c(fa) - c(fe)) ** 2).mean())); S.append((float(((c(fe) - c(fs)) ** 2).mean()), float(((c(fa) - c(fs)) ** 2).mean())))
n = len(E); sec = []
for t in range(int(np.ceil(n / 25))):
    e = np.mean(E[t * 25:(t + 1) * 25]); se = np.mean([q[0] for q in S[t * 25:(t + 1) * 25]]); sa = np.mean([q[1] for q in S[t * 25:(t + 1) * 25]])
    sec.append({"t": t, "e2e_vs_acc": round(psnr(e), 2), "e2e_vs_src": round(psnr(se), 2), "acc_vs_src": round(psnr(sa), 2)})
order = sorted(sec, key=lambda r: r["e2e_vs_acc"])
res = {"frames": n, "box_side_px": s, "mean_e2e_vs_acc": round(psnr(np.mean(E)), 2), "mean_e2e_vs_src": round(psnr(np.mean([q[0] for q in S])), 2),
       "mean_acc_vs_src": round(psnr(np.mean([q[1] for q in S])), 2), "worst": order[:a.worst], "per_second": sec}
json.dump(res, open(a.out + ".json", "w"), indent=1)
print("GOLD " + json.dumps({k: v for k, v in res.items() if k != "per_second"}))
F = subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"], capture_output=True, text=True).stdout
for r in order[:a.worst]:   # 0.5x clip of [t - 0.5, t + 1.5] s: accepted | e2e | source, the face box (1.3x) zoomed to 400 px, the source time burnt in
    t0 = max(0.0, r["t"] - 0.5); k = int(r["t"] * 25 + 12); x, y = bxy(k); m = int(0.15 * s); cx, cy, cs = max(0, x - m), max(0, y - m), s + 2 * m
    cs = min(cs, W - cx, H - cy); lab = ["accepted", "e2e", "source"]
    fg = ";".join(f"[{i}:v]setpts=PTS-STARTPTS,fps=25,scale={W}:{H},crop={cs}:{cs}:{cx}:{cy},scale=400:400,drawtext=fontfile={F}:text={lab[i]}:x=8:y=8:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[p{i}]" for i in range(3))
    fg += f";[p0][p1][p2]hstack=3,drawtext=fontfile={F}:text='%{{pts\\:hms\\:{t0}}}':x=(w-tw)/2:y=h-40:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6,setpts=2*PTS[v];[1:a]atempo=0.5[au]"
    o = f"{a.out}_worst_{r['t']}s_half_speed.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + sum([["-ss", f"{t0}", "-t", "2", "-i", v] for v in (a.acc, a.e2e, a.src)], []) +
                   ["-filter_complex", fg, "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", o], check=True)
    print("clip", o)
