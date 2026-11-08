#!/usr/bin/env python3
"""r16 Q: "the face springs" metric (Irina 08.11: KeySync beards bounce). Method = K1's spring3.py (r16-k1b beard3/k1, definition kept):
per frame pair t -> t+1, DIS flow (medium) in a part box resized to 256 px, RANSAC similarity (thr 0.7 px) on a 6 px grid -> (tx, ty, scale %);
extra = Delta_final - Delta_source for the SAME box (the source's own head motion cancels); mean |extra| per part.
Parts of the paste region: cheekL / cheekR (outer 32 % columns x top 50 % rows), nose (centre 30 % x top 30 %), jaw (bottom 28 %),
upper (the unpasted band above the paste = a control that must stay at the floor). Frame-to-frame (~50 % sign flips): never low-pass it.
Added by Q (validation 08.11, reports/experiments/r16-qa/SPRING.md):
  * a bitrate-MATCHED noise floor: the master re-encoded at the final's own video bitrate, measured the same way. A crf 30 floor is
    too pessimistic (q01 cheek 0.215 % vs matched 0.156 %): compression flicker is the floor, so it must match the final's compression.
  * spring = mean(cheekL, cheekR) |scale| extra; ratio = spring / floor spring; 95 % block-bootstrap CI over 1 s blocks.
  * the region: --region x0,y0,x1,y1, or K1's pass 1 (bbox of mean |final - master| > max(4, 0.25 p99), largest component, +15 % pad).
    Compare two finals of ONE clip only with the SAME region (pass --region from the first).
Validated (q01 / q03 / q04 beards): KeySync r16 ratio 1.75-1.96 (CI clear of 1), LatentSync clean 0.94-1.16 (= the floor; its paste
mask leaves the cheeks mostly source pixels), upper control = the floor everywhere. The three KeySync clips do NOT separate (0.28-0.31 %).
    python r16q_spring.py --final F.mp4 --master SRC_SAME_GRID.mp4 --out S.json [--region x0,y0,x1,y1] [--no-floor]"""
import argparse, json, os, subprocess, tempfile
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True); ap.add_argument("--master", required=True, help="the source on the SAME frame grid as the final")
ap.add_argument("--out", required=True); ap.add_argument("--region", default=None); ap.add_argument("--no-floor", action="store_true")
a = ap.parse_args()


def probe(p, keys="width,height"):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={keys}", "-of", "csv=p=0", p],
                          capture_output=True, text=True).stdout.strip().split(",")


W, H = [int(v) for v in probe(a.final)]
fr = probe(a.master, "r_frame_rate")[0].split("/"); FPS = float(fr[0]) / float(fr[1] if len(fr) > 1 else 1)


def frames(p, vf=None, w=W, h=H):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p] + (["-vf", vf] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        b = pr.stdout.read(w * h)
        if len(b) < w * h: break
        yield np.frombuffer(b, np.uint8).reshape(h, w)


def region():
    q = 4; acc = None; n = 0
    for x, y in zip(frames(a.final, f"scale={W // q}:{H // q}", W // q, H // q), frames(a.master, f"scale={W // q}:{H // q}", W // q, H // q)):
        d = np.abs(x.astype(np.float32) - y); acc = d if acc is None else acc + d; n += 1
    acc /= n; thr = max(4.0, np.percentile(acc, 99) * 0.25); m = (cv2.GaussianBlur(acc, (0, 0), 2) > thr).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(m); i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[i, :4] * q; pad = int(0.15 * max(w, h))
    return [int(max(0, x - pad)), int(max(0, y - pad)), int(min(W, x + w + pad)), int(min(H, y + h + pad))]


x0, y0, x1, y1 = list(map(int, a.region.split(","))) if a.region else region()
hh, ww = y1 - y0, x1 - x0
BOX = {"cheekL": (x0, y0, x0 + int(0.32 * ww), y0 + int(0.5 * hh)), "cheekR": (x1 - int(0.32 * ww), y0, x1, y0 + int(0.5 * hh)),
       "nose": (x0 + int(0.35 * ww), y0, x1 - int(0.35 * ww), y0 + int(0.3 * hh)), "jaw": (x0, y1 - int(0.28 * hh), x1, y1),
       "upper": (x0, max(0, y0 - int(0.55 * hh)), x1, y0 + int(0.05 * hh))}
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def fit(A, B, bx):
    p, q_, r, s = bx; S = 256 / max(r - p, s - q_); w, h = int((r - p) * S), int((s - q_) * S)
    A_ = cv2.resize(A[q_:s, p:r], (w, h), interpolation=cv2.INTER_AREA); B_ = cv2.resize(B[q_:s, p:r], (w, h), interpolation=cv2.INTER_AREA)
    f = DIS.calc(A_, B_, None); gy, gx = np.mgrid[6:h - 6:6, 6:w - 6:6]; P = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    Q = P + f[P[:, 1].astype(int), P[:, 0].astype(int)]
    T, _ = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=0.7, maxIters=500)
    if T is None: return [np.nan] * 3
    return [T[0, 2] / S, T[1, 2] / S, (np.hypot(T[0, 0], T[1, 0]) - 1) * 100]


def extra(final):
    R = {k: [] for k in BOX}; S_ = {k: [] for k in BOX}; pf = ps = None
    for f, s in zip(frames(final), frames(a.master)):
        if pf is not None:
            for k, bx in BOX.items(): R[k].append(fit(pf, f, bx)); S_[k].append(fit(ps, s, bx))
        pf, ps = f, s
    return {k: np.array(R[k]) - np.array(S_[k]) for k in BOX}


rng = np.random.default_rng(0)


def ci(x):
    x = x[np.isfinite(x)]; B = max(1, int(round(FPS))); nb = len(x) // B
    if nb < 3: return [round(float(np.mean(x)), 4)] * 3
    blk = x[:nb * B].reshape(nb, B).mean(1); bs = [blk[rng.integers(0, nb, nb)].mean() for _ in range(2000)]
    return [round(float(blk.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def summary(E):
    sp = (np.abs(E["cheekL"][:, 2]) + np.abs(E["cheekR"][:, 2])) / 2; st = (np.abs(E["cheekL"][:, :2]).sum(1) + np.abs(E["cheekR"][:, :2]).sum(1)) / 2
    B = max(1, int(round(FPS))); sec = [float(np.nanmean(sp[k * B:(k + 1) * B])) for k in range(len(sp) // B)]
    return {"spring_scale_pct": ci(sp), "spring_t_px": ci(st), "parts_scale_pct": {k: round(float(np.nanmean(np.abs(v[:, 2]))), 4) for k, v in E.items()},
            "parts_t_px": {k: round(float(np.nanmean(np.abs(v[:, :2]).sum(1))), 4) for k, v in E.items()},
            "worst_seconds": [[int(k), round(sec[k], 3)] for k in np.argsort(sec)[::-1][:5]]}


res = {"final": a.final, "master": a.master, "fps": FPS, "region": [x0, y0, x1, y1], "final_metric": summary(extra(a.final))}
if not a.no_floor:
    br = probe(a.final, "bit_rate")[0]
    with tempfile.TemporaryDirectory() as td:
        fl = os.path.join(td, "floor.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", a.master, "-an", "-c:v", "libx264", "-preset", "medium", "-b:v", br, "-maxrate", br,
                        "-bufsize", str(2 * int(br)), "-pix_fmt", "yuv420p", fl], check=True)
        res["floor_bitrate"] = int(br); res["floor_metric"] = summary(extra(fl))
    f, g = res["final_metric"]["spring_scale_pct"], res["floor_metric"]["spring_scale_pct"]
    res["spring_ratio"] = round(f[0] / max(g[0], 1e-6), 3); res["spring_above_floor"] = bool(f[1] > g[2])
    res["upper_ratio"] = round(res["final_metric"]["parts_scale_pct"]["upper"] / max(res["floor_metric"]["parts_scale_pct"]["upper"], 1e-6), 3)
json.dump(res, open(a.out, "w"), indent=1)
print("SPRING " + json.dumps({k: res.get(k) for k in ("region", "spring_ratio", "spring_above_floor", "upper_ratio")} | {"spring": res["final_metric"]["spring_scale_pct"],
      "floor": res.get("floor_metric", {}).get("spring_scale_pct"), "worst": res["final_metric"]["worst_seconds"][:3]}))
