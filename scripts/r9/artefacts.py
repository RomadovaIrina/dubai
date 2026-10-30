#!/usr/bin/env python3
"""r9: frame-by-frame MOUTH-RENDERING artefacts (user on m03: "the mouth opens unnaturally, the teeth look drawn, the lips stick together and
unstick", "the mouth slid onto the son's ear"), per layer of the m19b face, so each event is attributed to where it first appears:
  L1 raw LatentSync (_ls_pre) -> L2 + CodeFormer (_ls_raw) -> L3 + detail transfer + SR (_ls_cfd_cfd_srt1, the composite input) -> L4 final output.
All on the 25 fps master timeline, each file with its OWN MediaPipe landmarks; the source = the master's landmarks / pixels. Thresholds are relative
to the SOURCE's own statistics of the same video (generic, no per-video setting). Only frames where LatentSync is used (the fallback npz ls_mask).
  (a) stick / unstick  1-2-frame opening spike or dip: s[t] = o[t] - median(o[t-2..t+2]), o = inner-lip gap / face height;
                       event when |s| > max(0.025, 4 x the source's MAD of s) and the source's own s at t is < half of it (no matching phoneme)
  (b) amplitude        o[t] > 1.15 x the source's 99th-percentile opening (wider than she ever opens)
  (c) teeth            share of inner-mouth pixels with luma > 140 (open frames) exceeds the source's at t by > 0.25 ("drawn" bright teeth)
  (d) slide            lip centre off the source lip centre by > 0.06 IOD, in face coordinates (nose -> chin axis)
Events = runs of flagged frames (gap <= 1 merged). Attribution = the first layer whose run overlaps the final output's run.
    python artefacts.py <id> [--tag-suffix S] --out A.json [--sheet S.jpg]"""
import argparse, json, pathlib, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, "/workspace/dub/dubai/scripts/pilot"); sys.path.insert(0, "/workspace/dub/dubai/scripts/r8")
import clean_face_stack as cfs
from r8_mouth_flicker import LIPS_INNER, LIPS_OUTER
B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag")


def feats(video, lm, n):
    c = cv2.VideoCapture(str(video)); O, T, X, Y = (np.full(n, np.nan) for _ in range(4)); crops = {}
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        p = lm[k]
        if not np.isfinite(p[LIPS_INNER + LIPS_OUTER + [1, 10, 152, 33, 263]]).all(): continue
        fh = np.linalg.norm(p[10, :2] - p[152, :2]); iod = np.linalg.norm(p[33, :2] - p[263, :2]); O[k] = np.linalg.norm(p[13, :2] - p[14, :2]) / fh
        ax = (p[152, :2] - p[1, :2]); ax /= np.linalg.norm(ax); perp = np.array([-ax[1], ax[0]]); mc = p[LIPS_OUTER, :2].mean(0) - p[1, :2]
        X[k], Y[k] = (mc @ perp) / iod, (mc @ ax) / iod
        if O[k] > 0.03:
            m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 1)
            yy = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[m > 0]
            if yy.size >= 12: T[k] = (yy > 140).mean()
        cx, cy = p[LIPS_OUTER, :2].mean(0); r = int(1.0 * iod); crops[k] = (int(cx), int(cy), r)
    return O, T, X, Y


def spike(o):
    s = np.full_like(o, np.nan)
    for t in range(2, len(o) - 2):
        w = o[t - 2:t + 3]
        if np.isfinite(w).sum() >= 4: s[t] = o[t] - np.nanmedian(w)
    return s


def runs(flag):
    ev, k = [], 0
    idx = np.nonzero(flag)[0]
    for t in idx:
        if ev and t - ev[-1][1] <= 2: ev[-1][1] = t
        else: ev.append([t, t])
    return ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--tag-suffix", default=""); ap.add_argument("--out", required=True); ap.add_argument("--sheet", default="")
    ap.add_argument("--gen", default=""); ap.add_argument("--l3", default="")
    a = ap.parse_args(); i, sx = a.id, a.tag_suffix; G, F = (a.gen or f"r9g_{i}{sx}"), f"r9f_{i}{sx}"; W = B / G / f"work_{i}_{G}"; st = f"{i}_{G}_25fps"
    W0 = B / f"r9g_{i}" / f"work_{i}_r9g_{i}"
    srcv = W0 / "video" / "master_25fps.mp4"; srclm = np.load(W0 / "video" / "master_mediapipe.npz")["lm"]; n = len(srclm)
    o25 = D / f"{F}_25.mp4"
    if not o25.exists(): subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(B / F / f"{i}_{F}.mp4"), "-r", "25", "-an", "-c:v", "libx264", "-crf", "12", str(o25)], check=True)
    layers = [("L1 raw LatentSync", W / f"{st}_ls_pre.mp4"), ("L2 +CodeFormer", W / f"{st}_ls_raw.mp4"), ("L3 +detail+SR", pathlib.Path(a.l3) if a.l3 else W / f"{st}_ls_cfd_cfd_srt1.mp4"), ("L4 final", o25)]
    fbp = W / f"{st}_fb_fb7.npz"; lsm = np.load(fbp)["ls_mask"][:n].astype(bool) if fbp.exists() else np.ones(n, bool)
    if len(lsm) < n: lsm = np.pad(lsm, (0, n - len(lsm)))   # the source landmarks can be a frame or two longer than the layer
    So, St, Sx, Sy = feats(srcv, srclm, n); Ss = spike(So)
    mad = np.nanmedian(np.abs(Ss - np.nanmedian(Ss))); thr_s = max(0.025, 4 * 1.4826 * mad); thr_o = 1.15 * np.nanpercentile(So, 99)
    res = {"id": i, "arm": sx or "r9", "ls_frames": int(lsm.sum()), "thr": {"spike": round(float(thr_s), 4), "open_max": round(float(thr_o), 4)}, "layers": {}}
    evs = {}
    for name, p in layers:
        if not p.exists(): res["layers"][name] = "missing"; continue
        try: lm = np.load(cfs.run_mediapipe(p, D / f"{F}_{p.stem}_mp.npz" if name == "L4 final" else p.with_suffix(".art_mp.npz")))["lm"][:n]
        except RuntimeError:   # a layer with a broken fps tag (cf_detail bug, fixed 0fad7e0): video-mode timestamps not increasing -> per-frame image mode
            lm = np.load(cfs.run_mediapipe(p, p.with_suffix(".art_img_mp.npz"), mode="image"))["lm"][:n]
        O, T, X, Y = feats(p, lm, n); s = spike(O)
        fl = {"a_stick_unstick": lsm & (np.abs(s) > thr_s) & ~(np.abs(Ss) > 0.5 * np.abs(s)),
              "b_amplitude": lsm & (O > thr_o),
              "c_teeth": lsm & np.isfinite(T) & np.isfinite(St) & (T - St > 0.25),
              "d_slide": lsm & (np.hypot(X - Sx, Y - Sy) > 0.06)}
        evs[name] = {c: runs(np.nan_to_num(f).astype(bool)) for c, f in fl.items()}
        res["layers"][name] = {c: {"events": len(e), "frames": int(np.nan_to_num(fl[c]).astype(bool).sum())} for c, e in evs[name].items()}
        res["layers"][name]["open_mean_over_src"] = round(float(np.nanmean(O[lsm]) / np.nanmean(So[lsm])), 2)
    fin = evs.get("L4 final", {}); attr = {}; worst = []
    for c, E in fin.items():
        attr[c] = {}
        for e in E:
            origin = "composite / fallback"
            for name in ("L1 raw LatentSync", "L2 +CodeFormer", "L3 +detail+SR"):
                if name in evs and any(not (x[1] < e[0] - 1 or x[0] > e[1] + 1) for x in evs[name][c]): origin = {"L1 raw LatentSync": "raw LatentSync", "L2 +CodeFormer": "CodeFormer", "L3 +detail+SR": "detail transfer / SR"}[name]; break
            attr[c][origin] = attr[c].get(origin, 0) + 1; worst.append({"class": c, "t": [round(e[0] / 25, 2), round(e[1] / 25, 2)], "frames": int(e[1] - e[0] + 1), "origin": origin})
    res["final_events_by_origin"] = attr; res["final_events"] = worst
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    if a.sheet and worst:   # the longest events of each class: source over output, mouth crop, every frame of the event (max 6)
        cs, co = cv2.VideoCapture(str(srcv)), cv2.VideoCapture(str(o25)); tiles = []
        pick = sorted(worst, key=lambda w: -w["frames"])[:10]; pick.sort(key=lambda w: w["t"][0])
        for w in pick:
            k0 = int(w["t"][0] * 25); k = min(n - 1, k0 + (w["frames"] - 1) // 2); p = srclm[k]
            if not np.isfinite(p[[33, 263, 13]]).all(): continue
            iod = np.linalg.norm(p[33, :2] - p[263, :2]); r = int(1.1 * iod); cx, cy = p[LIPS_OUTER, :2].mean(0).astype(int)
            col = []
            for cap in (cs, co):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); h, wd = f.shape[:2]
                cr = f[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(wd, cx + r)]; col.append(cv2.resize(cr, (190, 190)))
            t = np.vstack(col); cv2.putText(t, f"{k / 25:.2f}s {w['class'][:1]} {w['origin'][:9]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1); tiles.append(t)
        if tiles: cv2.imwrite(a.sheet, np.hstack(tiles))
    print(json.dumps({k: v for k, v in res.items() if k != "final_events"}))


if __name__ == "__main__":
    main()
