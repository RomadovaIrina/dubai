#!/usr/bin/env python3
"""r14 mouth-flicker metric: the same mouth box (median of the source landmarks, as mouth_2panel.py) in each video, grey 128 x 128 per frame,
4x average pool (shape / brightness, not texture), |second temporal difference| per frame = "jitter". Smooth mouth motion scores low, a
one-frame jump (flicker / pop) scores high. Reports, per video: mean, p95, the count of "pops" (jitter above the SOURCE video's p99: the
real face's own fastest changes), and the mean at the given timestamps (+-0.5 s). The source = the original input (natural motion floor).
    python scripts/r14/flicker_metric.py --lm source_lm_F.npz --video src=in.mp4 k5=k5.mp4 student=st.mp4 [--at 3.5 5.5 21 24 27.5] [--json out.json]"""
import argparse, json, subprocess
import numpy as np

MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]
S = 128


def probe(v):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=p=0", v],
                       capture_output=True, text=True).stdout.strip().split(",")
    n, d = o[2].split("/"); return int(o[0]), int(o[1]), float(n) / float(d)


def box(lm, W, H):
    z = np.load(lm); L = z[next(k for k in ("lm", "lm_all", "landmarks") if k in z.files)]
    if L.ndim == 4: L = L[:, 0]
    P = L[:, MOUTH, :2].reshape(-1, 2); P = P[np.isfinite(P).all(1)]
    if np.nanmax(P) <= 1.5: P = P * np.array([W, H])
    cx, cy = np.median(P, 0); mw = np.percentile(P[:, 0], 95) - np.percentile(P[:, 0], 5); s = int(max(64, 2.0 * mw)); s -= s % 2
    return int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), s


def frames(v, x0, y0, s):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", v, "-vf", f"crop={s}:{s}:{x0}:{y0},scale={S}:{S},format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, S, S).astype(np.float32) / 255.0


def jitter(fr):
    p = fr.reshape(len(fr), S // 4, 4, S // 4, 4).mean((2, 4))
    j = np.abs(p[2:] - 2 * p[1:-1] + p[:-2]).mean((1, 2)); return np.concatenate([[0.0], j, [0.0]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--at", nargs="*", type=float, default=[]); ap.add_argument("--json"); a = ap.parse_args()
    vids = dict(x.split("=", 1) for x in a.video); first = next(iter(vids.values())); W, H, fps = probe(first); x0, y0, s = box(a.lm, W, H)
    J = {k: jitter(frames(v, x0, y0, s)) for k, v in vids.items()}
    thr = float(np.percentile(J["src"][1:-1], 99)) if "src" in J else None
    out = {"box": [x0, y0, s], "fps": fps, "pop_threshold_src_p99": thr, "videos": {}}
    for k, j in J.items():
        r = {"frames": len(j), "mean": float(j.mean()), "p95": float(np.percentile(j, 95))}
        if thr: r["pops"] = int((j > thr).sum()); r["pops_per_10s"] = round(10 * fps * r["pops"] / len(j), 2)
        r["at"] = {str(t): float(j[max(0, int((t - .5) * fps)):int((t + .5) * fps) + 1].mean()) for t in a.at}
        out["videos"][k] = r
        print(f"{k:10s} mean {r['mean']:.4f} p95 {r['p95']:.4f}" + (f" pops {r['pops']} ({r['pops_per_10s']}/10 s)" if thr else "") +
              "".join(f"  @{t} {v:.4f}" for t, v in r["at"].items()))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
