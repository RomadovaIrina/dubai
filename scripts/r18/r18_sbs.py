#!/usr/bin/env python3
"""r18: a side-by-side clip for Irina's eye, half speed: "08.11 | restored | +seam a | ... | source" (or blind: the --blind columns shuffled and
named A, B, C...; the key goes to <out>.key.json, never onto the picture).
    r18_sbs.py --in "08.11=ref.mp4" --in "restored=r.mp4" --in "source=src.mp4" --t0 19 --t1 23 --out clip.mp4
               [--crop auto --work /data/jobs/<id>/e2e/work | --crop x,y,w,h (fractions of the frame)] [--height 960] [--speed 0.5]
               [--audio 0 (index of the input whose sound is kept, slowed; -1 = silent)] [--blind restored,a,b --seed 7]
Every input is decoded with its own colour tags to RGB on a 25 fps grid at the FIRST input's size; the clip is written BT.709-tagged, crf 14."""
import argparse, glob, json, os, random, subprocess
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", action="append", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--crop", default=""); ap.add_argument("--work", default=""); ap.add_argument("--height", type=int, default=960)
ap.add_argument("--cols", type=int, default=0, help="columns of the grid (0 = one row)"); ap.add_argument("--speed", type=float, default=0.5); ap.add_argument("--audio", type=int, default=0); ap.add_argument("--blind", default=""); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
IN = [x.split("=", 1) for x in a.inp]
if a.blind:   # shuffle the named columns among their own positions, rename them A, B, ...
    names = a.blind.split(","); pos = [i for i, (l, _) in enumerate(IN) if l in names]; sh = pos[:]; random.Random(a.seed).shuffle(sh)
    new = list(IN); key = {}
    for j, (p_, s_) in enumerate(zip(pos, sh)): new[p_] = [chr(65 + j), IN[s_][1]]; key[chr(65 + j)] = IN[s_][0]
    IN = new; json.dump({"key": key, "t0": a.t0, "t1": a.t1, "seed": a.seed}, open(a.out + ".key.json", "w"), indent=1)
W, H = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", IN[0][1]], capture_output=True, text=True).stdout.strip().split(",")[:2]]
cx, cy, cw, ch = 0.0, 0.0, 1.0, 1.0
if a.crop == "auto" and a.work:   # the paste box (lower face + neck in view): the job's box json of the shot at t0, a little room below
    kw = os.path.join(a.work, "ks"); k = int(a.t0 * 25); bj = None; kb = 0
    segs = json.load(open(os.path.join(kw, "shots.json")))["segments"] if os.path.exists(os.path.join(kw, "shots.json")) else []
    if len(segs) > 1:
        for i, sg in enumerate(segs):
            k0, k1 = (sg[0], sg[1]) if isinstance(sg, (list, tuple)) else (sg.get("k0", sg.get("a")), sg.get("k1", sg.get("b")))
            if k0 <= k < k1: bj = (glob.glob(os.path.join(a.work, "shots", f"s{i}", "ks", "data", "box_*.json")) or [None])[0]; kb = k0
    else: bj = (glob.glob(os.path.join(kw, "data", "box_*.json")) or [None])[0]
    if bj:
        b = json.load(open(bj)); s = b["side"]; x = b["xs"][min(k - kb, len(b["xs"]) - 1)] if b.get("xs") else b["x"]; y = b["ys"][min(k - kb, len(b["ys"]) - 1)] if b.get("ys") else b["y"]
        cx, cy, cw, ch = x / b["W"], (y + 0.05 * s) / b["H"], s / b["W"], 1.1 * s / b["H"]
elif a.crop and a.crop != "auto": cx, cy, cw, ch = [float(v) for v in a.crop.split(",")]
X0, Y0 = max(0, int(cx * W)), max(0, int(cy * H)); X1, Y1 = min(W, int((cx + cw) * W)), min(H, int((cy + ch) * H))
th = a.height; tw = int(round((X1 - X0) * th / (Y1 - Y0) / 2)) * 2; n = int(round((a.t1 - a.t0) * 25)); LB = 44


def rd(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a.t0:.3f}", "-i", p, "-frames:v", str(n), "-vf", f"fps=25,scale={W}:{H}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    while True:
        b = pr.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


NC = a.cols or len(IN); NR = -(-len(IN) // NC)
OW = tw * NC + 4 * (NC - 1); OH = (th + LB) * NR; OH += OH % 2; OW += OW % 2
AU = [] if a.audio < 0 else ["-ss", f"{a.t0:.3f}", "-t", f"{a.t1 - a.t0:.3f}", "-i", IN[a.audio][1]]
AM = ["-an"] if a.audio < 0 else ["-map", "0:v", "-map", "1:a:0", "-af", f"atempo={a.speed}", "-c:a", "aac", "-b:a", "128k"]
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", f"{25 * a.speed}", "-i", "-"] + AU + AM +
                     ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-colorspace", "bt709", "-color_primaries", "bt709",
                      "-color_trc", "bt709", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
k = 0
for fs in zip(*[rd(p) for _, p in IN]):
    can = np.zeros((OH, OW, 3), np.uint8)
    for j, f in enumerate(fs):
        x = (j % NC) * (tw + 4); y = (j // NC) * (th + LB); can[y + LB:y + LB + th, x:x + tw] = cv2.resize(f[Y0:Y1, X0:X1], (tw, th), interpolation=cv2.INTER_LANCZOS4)
        cv2.putText(can, IN[j][0], (x + 8, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(can, f"{a.t0 + k / 25:6.2f}s", (OW - 150, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA)
    w.stdin.write(can.tobytes()); k += 1
w.stdin.close(); w.wait(); print(f"R18_SBS {a.out}: {k} frames, {len(IN)} columns {[l for l, _ in IN]}, crop {X0},{Y0}-{X1},{Y1} of {W}x{H}, tile {tw}x{th}")
