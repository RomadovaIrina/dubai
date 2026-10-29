#!/usr/bin/env python3
"""r8 F: per-frame ARTEFACT SCORE of the LatentSync layer and the CONFIDENCE FALLBACK to the source mouth (spec 5.5 spirit: "no lip-sync for a
moment" beats a mouth explosion; management on final_r7 01 / 03 starts: "mouth explosions").

All features are measured on the 25 fps master timeline, layer vs the SOURCE master at the same frame, so head / camera motion that the
source also has does not count:
  jerk     |o[t] - (o[t-1] + o[t+1]) / 2| of the layer's inner opening (13-14 / IOD) minus the same of the source (the generated mouth
           snapping open / shut between frames)
  tone     inner-mouth luma HF residual (r8_mouth_flicker model: luma not explained by the opening, 2nd difference) of the layer minus the source
  app      mouth-box appearance jump: mean |crop[t] - (crop[t-1] + crop[t+1]) / 2| (grey, box from frame t's landmarks, same box in the
           neighbours), layer minus 1.5 x source (cavity / teeth blobs, colour pops, lip-shape collapse)
  geo      rigid-landmark disagreement layer vs source (eyes / nose, IOD): the generated face does not sit where the source face is
  joint    the app / jerk excess at a 16-frame window joint counts x --joint-gain (window-joint jumps)
score = max over the features, each scaled so 1 = a clear artefact (--thr-*). fallback weight fb (0..1): score -> clip, max-filter
+-(--dilate) frames, then a smooth ramp (Gaussian --ramp-sigma) = crossfade to the source; a 16-frame window with more than --win-frac of
its frames flagged falls back as a whole. fps_composite --fallback FILE multiplies its master weights by (1 - fb).
    python scripts/r8/r8_fallback.py --work WORK_DIR --ls LAYER.mp4 --lm LAYER.npz --out FB.npz [--wav dubbed_16k.wav]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r8_mouth_flicker import LIPS_INNER, LIPS_OUTER
RIG = [33, 133, 362, 263, 168, 6, 197, 4]


def hf(x):
    o = np.full(len(x), np.nan); o[1:-1] = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:])); return o


def read_feats(video, lm, n):
    """per frame: opening, inner luma, area, grey mouth crops (box from that frame, 48 x 48) of frames t-1, t, t+1 in frame t's box."""
    c = cv2.VideoCapture(str(video)); frames = []
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    c.release(); n = len(frames)
    op = np.full(n, np.nan); yin = np.full(n, np.nan); ar = np.full(n, np.nan); app = np.full(n, np.nan); boxes = [None] * n
    for k in range(n):
        p = lm[k]
        if not np.isfinite(p[LIPS_INNER + LIPS_OUTER + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2]))
        if iod < 5: continue
        op[k] = np.linalg.norm(p[13, :2] - p[14, :2]) / iod; q = p[LIPS_INNER, :2]; ar[k] = cv2.contourArea(q.astype(np.float32)) / iod ** 2
        m = np.zeros(frames[k].shape, np.uint8); cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1); m = cv2.erode(m, np.ones((3, 3), np.uint8))
        if m.sum() >= 6: yin[k] = float(frames[k][m > 0].mean())
        cx, cy = p[LIPS_OUTER, 0].mean(), p[LIPS_OUTER, 1].mean(); r = 0.55 * iod
        boxes[k] = (int(cy - r), int(cy + r), int(cx - r), int(cx + r))
    def crop(k, b):
        y0, y1, x0, x1 = b; H, W = frames[k].shape
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W or y1 - y0 < 8: return None
        return cv2.resize(frames[k][y0:y1, x0:x1], (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    for k in range(1, n - 1):
        if boxes[k] is None: continue
        cs = [crop(j, boxes[k]) for j in (k - 1, k, k + 1)]
        if any(c_ is None for c_ in cs): continue
        app[k] = float(np.abs(cs[1] - 0.5 * (cs[0] + cs[2])).mean())
    return op, yin, ar, app, n


def tone_hf(yin, op, ar, open_min=0.03):
    v = np.isfinite(yin) & np.isfinite(op) & (op >= open_min); r = np.full(len(yin), np.nan)
    if v.sum() > 30:
        X = np.stack([np.ones(v.sum()), op[v], ar[v]], 1); coef, *_ = np.linalg.lstsq(X, yin[v], rcond=None); r[v] = yin[v] - X @ coef
    return hf(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="work dir with video/master_25fps.mp4 + master_mediapipe.npz and the 25fps manifest")
    ap.add_argument("--ls", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr-jerk", type=float, default=0.05); ap.add_argument("--thr-tone", type=float, default=14.0); ap.add_argument("--thr-app", type=float, default=9.0)
    ap.add_argument("--thr-geo", type=float, default=0.12); ap.add_argument("--thr-coll", type=float, default=0.45)
    ap.add_argument("--thr-wash", type=float, default=0.12); ap.add_argument("--thr-dark", type=float, default=0.6); ap.add_argument("--knee", type=float, default=0.5, help="a feature counts from knee x thr (0) to thr (1)")
    ap.add_argument("--joint-gain", type=float, default=1.3); ap.add_argument("--dilate", type=int, default=2); ap.add_argument("--ramp-sigma", type=float, default=2.0); ap.add_argument("--merge-gap", type=int, default=6)
    ap.add_argument("--win-frac", type=float, default=0.4); ap.add_argument("--window", type=int, default=16)
    a = ap.parse_args(); work = pathlib.Path(a.work)
    lml = np.load(a.lm)["lm"]; lms = np.load(work / "video/master_mediapipe.npz")["lm"]; n = min(len(lml), len(lms))
    ol, yl, al_, apl, n1 = read_feats(a.ls, lml, n); os_, ys, as_, aps, n2 = read_feats(work / "video/master_25fps.mp4", lms, n); n = min(n1, n2)
    man = json.loads(next(work.glob("*_25fps.manifest.json")).read_text())
    ls_mask = np.zeros(n, bool); starts = []
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC":
            ls_mask[s["start_frame"]:min(n, s["end_frame"])] = True; starts.append(s["start_frame"])
    jerk = np.nan_to_num(hf(ol[:n]) - hf(os_[:n]), nan=0.0)
    tone = np.nan_to_num(tone_hf(yl[:n], ol[:n], al_[:n]) - np.nan_to_num(tone_hf(ys[:n], os_[:n], as_[:n]), nan=0.0), nan=0.0)
    appx = np.nan_to_num(apl[:n] - 1.5 * np.nan_to_num(aps[:n], nan=0.0), nan=0.0)
    coll = np.zeros(n); wash = np.zeros(n); dark = np.zeros(n)
    cl = cv2.VideoCapture(str(a.ls)); cs = cv2.VideoCapture(str(work / "video/master_25fps.mp4"))
    for k in range(n):
        okl, fl = cl.read(); oks, fs = cs.read()
        if not (okl and oks): break
        q = lms[k]
        if not np.isfinite(q[LIPS_OUTER + LIPS_INNER + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(q[33, :2] - q[263, :2])); m = np.zeros(fs.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cv2.convexHull(q[LIPS_OUTER, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        r_ = max(1, int(round(0.12 * iod))); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1,) * 2)) > 0
        if m.sum() < 20: continue
        gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).astype(np.float32)[m]; gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)[m]
        coll[k] = 1 - gl.std() / max(gs.std(), 1.0)                       # lip / cavity contrast lost vs the source (washed blob)
        wash[k] = (gl.mean() - gs.mean()) / max(gs.mean(), 1.0)           # the mouth region brighter than the source (pale / pink pop)
        dark[k] = (np.percentile(gs, 10) - np.percentile(gl, 10)) / max(np.percentile(gs, 10) + 10, 1.0)   # a black hole darker than the source's cavity
    cl.release(); cs.release()
    geo = np.zeros(n)
    for k in range(n):
        p, q = lml[k], lms[k]
        if np.isfinite(p[RIG]).all() and np.isfinite(q[RIG]).all():
            iod = float(np.linalg.norm(q[33, :2] - q[263, :2])); geo[k] = float(np.median(np.linalg.norm(p[RIG, :2] - q[RIG, :2], axis=1)) / max(iod, 1))
    # window joints: frames within 1 of a 16-frame boundary of a LatentSync segment
    joint = np.zeros(n, bool)
    for s0 in starts:
        for b in range(s0, n, a.window):
            joint[max(0, b - 1):min(n, b + 1)] = True
    sc = lambda x, t: np.clip((x - a.knee * t) / ((1 - a.knee) * t), 0, 1)
    tmed = lambda x: np.array([np.median(x[max(0, k - 6):k + 7]) for k in range(len(x))])   # the transient part (a pop), not a steady offset
    F = {"jerk": sc(jerk, a.thr_jerk), "tone": sc(tone, a.thr_tone), "app": sc(appx, a.thr_app), "geo": sc(geo, a.thr_geo),
         "coll": sc(coll - np.minimum(tmed(coll), 0.15), a.thr_coll), "wash": sc(wash - np.minimum(tmed(wash), 0.05), a.thr_wash), "dark": sc(dark - np.minimum(tmed(dark), 0.2), a.thr_dark)}
    for k_ in ("jerk", "app"): F[k_] = np.where(joint, np.clip(F[k_] * a.joint_gain, 0, 1), F[k_])
    score = np.max(np.stack(list(F.values()), 1), 1) * ls_mask
    fb = score.copy()
    flag = score > 0.5; k = 0   # gaps of <= --merge-gap frames between flagged frames are filled: one held fallback, no blinking in and out
    idx = np.flatnonzero(flag)
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if 1 < i1 - i0 <= a.merge_gap + 1 and ls_mask[i0:i1].all(): fb[i0:i1] = np.maximum(fb[i0:i1], 1.0)
    if a.dilate > 0: fb = np.array([fb[max(0, k - a.dilate):k + a.dilate + 1].max() for k in range(n)])
    # whole-window fallback where a window is mostly bad
    for s0 in starts:
        for b in range(s0, n, a.window):
            seg = slice(b, min(n, b + a.window))
            if ls_mask[seg].any() and (score[seg] > 0.5).mean() > a.win_frac: fb[seg] = 1.0
    if a.ramp_sigma > 0:
        R_ = int(3 * a.ramp_sigma); k_ = np.exp(-0.5 * (np.arange(-R_, R_ + 1) / a.ramp_sigma) ** 2); k_ /= k_.sum(); fb = np.convolve(np.pad(fb, R_, mode="edge"), k_, "same")[R_:-R_]
        fb = np.maximum(fb, np.clip(score, 0, 1))   # the flagged frame itself is never softened
    fb = np.clip(fb, 0, 1) * ls_mask
    nls = max(1, ls_mask.sum()); rep = {"frames": int(n), "ls_frames": int(ls_mask.sum()), "fallback_frames_gt05": int((fb > 0.5).sum()), "fallback_pct_gt05": round(100 * float((fb > 0.5).sum()) / nls, 2),
           "fallback_weight_pct": round(100 * float(fb.sum()) / nls, 2), "by_feature_gt05": {k_: int((v_ > 0.5).sum()) for k_, v_ in F.items()},
           "top_s": [[round(int(k) / 25, 2), round(float(score[k]), 2), max(F, key=lambda f_: F[f_][k])] for k in sorted(np.argsort(-score)[:15])], "args": vars(a)}
    np.savez(a.out, fb=fb.astype(np.float32), score=score.astype(np.float32), ls_mask=ls_mask, **{f"f_{k_}": v_.astype(np.float32) for k_, v_ in F.items()},
             raw_jerk=jerk.astype(np.float32), raw_tone=tone.astype(np.float32), raw_app=appx.astype(np.float32), raw_geo=geo.astype(np.float32), raw_coll=coll.astype(np.float32), raw_wash=wash.astype(np.float32), raw_dark=dark.astype(np.float32))
    pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print("== fallback", json.dumps({k_: v_ for k_, v_ in rep.items() if k_ not in ("args", "top_s")}), flush=True)


if __name__ == "__main__":
    main()
