#!/usr/bin/env python3
"""r20 (mouth): the reference PLAN for ks_lref.py = one identity frame per KeySync keyframe batch (169 frames), picked from the speaker's own frames
NEAR that batch: lit like the batch (the lower-face skin tone), frontal, sharp, eyes open, the face where the batch has it, and with the wanted mouth:
  --mouth closed   a neutral closed mouth (what frame 0 usually is): fixes the tone (04: the band above the lip), draws no teeth
  --mouth teeth    slightly open with CLEAN upper teeth (bright, even, little black): the model gets the speaker's teeth for open mouths;
                   a batch with no clean-teeth frame in reach falls back to closed (never a bad teeth frame)
  --mouth auto     the per-video gate: teeth only if the speaker shows clean teeth in >= --teeth-min-share of the speech frames (m03 0.13: yes;
                   old03 0.001, 04 0: no -> closed; Irina 14.11: the teeth reference wins on m03, "гнилые зубы" on old03)
The pick of the previous batch is kept while it is within --keep of the best (few identity changes; a steady video = one frame for all).
    ks_lrefpick.py --video crop512.mp4 --lm crop_lm.npz [--face-lm lm_f0.npz] [--n N] --out plan.json [--sheet plan.jpg] | --chunks <ks_chunk dir>
prints KS_LREFPICK {...}. Frames >= --n (the mirrored pad) are never picked."""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--face-lm", default="", dest="face_lm")
ap.add_argument("--n", type=int, default=0, help="real frames (without the pad)"); ap.add_argument("--out", default=""); ap.add_argument("--sheet", default="")
ap.add_argument("--mouth", default="auto", choices=["closed", "teeth", "auto"]); ap.add_argument("--batch", type=int, default=169)
ap.add_argument("--reach", type=float, default=3.0, help="s around the batch searched first"); ap.add_argument("--keep", type=float, default=0.5)
ap.add_argument("--lips", type=float, default=1.0, help="weight of the typical-lips term (lip redness and thickness vs the speaker median)"); ap.add_argument("--slot0", type=int, default=0); ap.add_argument("--teeth-min", type=float, default=0.20, dest="tmin", help="upper-teeth share of a teeth frame")
ap.add_argument("--teeth-min-share", type=float, default=0.02, dest="tshare"); ap.add_argument("--feat", default="", help="save / reuse the per-frame features (npz)")
ap.add_argument("--chunks", default="", help="a ks_chunk.py split dir: one plan per chunk (c<j>.plan.json) instead of --out")
ap.add_argument("--when", default="always", choices=["always", "bad"], help="bad = a plan ONLY when the first frame (the default reference) is a bad reference for this clip: its lower-face tone is far from >= --bad-share of the batches (dE > --bad-de: the light changes, 04) or its eye region differs (sunglasses on / off, 05); otherwise keep0 = the caller stays on frame 0, bit-identical to the default")
ap.add_argument("--bad-de", type=float, default=7.0, dest="bad_de"); ap.add_argument("--bad-share", type=float, default=0.25, dest="bad_share"); ap.add_argument("--bad-eye", type=float, default=12.0, dest="bad_eye")
ap.add_argument("--kf-yaw", type=float, default=0.15, dest="kf_yaw", help="per KEYFRAME (every 13th frame): if the batch reference differs from the keyframe in yaw by more than this, that keyframe gets a pose-matched frame from nearby (04 at 28 s: a head turn inside one batch; a frontal reference draws a second jaw line). 0 = off")
ap.add_argument("--global", type=int, default=0, dest="glob", help="1 = ONE frame for the whole clip (the best for the clip median), as KS_REF=<frame>")
a = ap.parse_args()
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
CH = [[116, 123, 147, 187, 205, 36, 100, 117], [345, 352, 376, 411, 425, 266, 329, 346]]; PH = [98, 327, 2, 164, 0, 37, 267, 39, 269, 165, 391]
CHIN = [152, 148, 176, 140, 32, 194, 182, 18, 406, 418, 262, 369, 400, 377]
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); FB = W * H * 3
LM = np.load(a.lm)["lm"][..., :2].astype(np.float32); N = len(LM) if not a.n else min(a.n, len(LM)); ok = np.isfinite(LM.reshape(len(LM), -1)).all(1)
if a.face_lm:   # only frames where THIS face is present in the master track
    fl = np.load(a.face_lm); fm = fl["lm"][..., :2]; J = fl["j"] if "j" in fl.files else np.arange(len(fm))
    ok &= np.array([np.isfinite(fm[min(int(J[min(k, len(J) - 1)]), len(fm) - 1)]).all() for k in range(len(LM))])
hull = lambda p: cv2.convexHull(p.astype(np.float32))[:, 0].astype(np.int32)
def fill(p, idx):
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [hull(p[idx])], 1); return m
KEYS = ["open", "yaw", "roll", "sharp", "L", "A", "B", "pL", "pA", "t_up", "t_L", "t_sd", "t_a", "dark", "cx", "cy", "fh", "lipc", "ear", "lth", "eyeL"]
F = np.full((N, len(KEYS)), np.nan, np.float32)
if a.feat and os.path.exists(a.feat) and np.load(a.feat)["F"].shape == F.shape: F = np.load(a.feat)["F"]
else:
    p_ = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
    for k in range(N):
        b = p_.stdout.read(FB)
        if len(b) < FB: break
        if not ok[k]: continue
        p = LM[k]; rgb = np.frombuffer(b, np.uint8).reshape(H, W, 3); lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        lipm = cv2.dilate(fill(p, LIPS), np.ones((7, 7), np.uint8)); sk = (fill(p, CH[0]) | fill(p, CH[1]) | fill(p, CHIN)) & (1 - lipm); ph = fill(p, PH) & (1 - lipm)
        if sk.sum() < 50 or ph.sum() < 20: continue
        s = np.median(lab[sk > 0], 0); q = np.median(lab[ph > 0], 0); mw = float(np.linalg.norm(p[61] - p[291])); op = float(np.linalg.norm(p[13] - p[14])) / max(mw, 1)
        dl = float(np.linalg.norm(p[1] - p[234])); dr = float(np.linalg.norm(p[454] - p[1])); yaw = (dl - dr) / max(dl + dr, 1.0)
        roll = float(np.degrees(np.arctan2(p[263][1] - p[33][1], p[263][0] - p[33][0])))
        x0, y0 = p[[61, 291, 0, 17]].min(0) - 0.15 * mw; x1, y1 = p[[61, 291, 0, 17]].max(0) + 0.15 * mw
        g = cv2.cvtColor(rgb[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)], cv2.COLOR_RGB2GRAY); sharp = float(cv2.Laplacian(g, cv2.CV_32F).var()) if g.size else 0.0
        inner = cv2.erode(fill(p, INNER), np.ones((3, 3), np.uint8)); up = inner.copy(); up[int(round((p[13][1] + p[14][1]) / 2)):, :] = 0
        L, A = lab[..., 0], lab[..., 1]; teeth = (L > 0.80 * s[0]) & (A < s[1] + 2)
        t_up = float(teeth[up > 0].mean()) if up.sum() > 15 else 0.0; tm = teeth & (up > 0)
        t_L = float(L[tm].mean()) / max(s[0], 1) if tm.sum() > 10 else 0.0; t_sd = float(L[tm].std()) if tm.sum() > 10 else 0.0; t_a = float(A[tm].mean() - s[1]) if tm.sum() > 10 else 0.0
        dark = float((L[inner > 0] < 15).mean()) if inner.sum() > 15 else 0.0
        lh = fill(p, LIPS) & (1 - fill(p, INNER)); lipc = float(np.median(A[lh > 0]) - s[1]) if lh.sum() > 20 else 0.0
        ear = 0.5 * float(np.linalg.norm(p[159] - p[145]) / max(np.linalg.norm(p[33] - p[133]), 1) + np.linalg.norm(p[386] - p[374]) / max(np.linalg.norm(p[362] - p[263]), 1))
        F[k] = [op, yaw, roll, sharp, s[0], s[1], s[2], q[0], q[1], t_up, t_L, t_sd, t_a, dark, p[:, 0].mean(), p[:, 1].mean(), float(np.linalg.norm(p[10] - p[152])), lipc, ear,
                float(np.linalg.norm(p[0] - p[13]) + np.linalg.norm(p[14] - p[17])) / max(mw, 1),
                float(np.median(L[(fill(p, [33, 133, 159, 145, 160, 144, 158, 153]) | fill(p, [362, 263, 386, 374, 385, 380, 387, 373])) > 0]) - s[0])]   # eye region L minus the skin L: sunglasses on / off between the reference and the batch
    p_.kill()
    if a.feat: np.savez_compressed(a.feat, F=F, keys=KEYS)
f = {k: F[:, i] for i, k in enumerate(KEYS)}; V = np.isfinite(F).all(1)
def done(plan, msg):
    if a.out: json.dump(plan, open(a.out, "w"), indent=1)
    print("KS_LREFPICK " + json.dumps({k: v for k, v in plan.items() if k not in ("rows", "kf_refs")}) + " " + msg); sys.exit(0)
if not V.any(): done({"refs": [0], "slot0": a.slot0, "mode": "none"}, "no usable frame")
speech = V & (f["open"] > 0.07)
V = V & (f["ear"] > 0.6 * np.nanmedian(f["ear"]))   # no blink frames
idx = np.flatnonzero(V); ms = np.nanmedian(f["sharp"]); mlc = max(np.nanmedian(f["lipc"]), 1e-3); o10 = np.nanpercentile(f["open"], 10)
closed = V & (f["open"] < max(0.05, o10 + 0.02)) & (f["lipc"] > 0.5 * mlc)
# clean teeth: slightly open, the upper half of the opening is teeth, the teeth bright and even (not a dark / speckled row), little black cavity
tclean = V & (f["open"] > 0.07) & (f["open"] < 0.30) & (f["t_up"] >= a.tmin) & (f["t_L"] > 0.80) & (f["t_sd"] < 9.0) & (f["dark"] < 0.25) & (f["lipc"] > 0.5 * mlc)
tsh = float(tclean.sum()) / max(int(speech.sum()), 1)
# the TYPICAL lips of the speaker (04: a reference with redder / fuller-looking lips than usual made the drawn lips fuller and pinker for the whole video)
mad = lambda v: max(float(np.median(np.abs(v - np.median(v)))), 1e-3)
cl_ = closed if closed.sum() >= 10 else V; lc0, lcs = float(np.median(f["lipc"][cl_])), mad(f["lipc"][cl_]); lt0, lts = float(np.median(f["lth"][cl_])), mad(f["lth"][cl_])
mode = a.mouth if a.mouth != "auto" else ("teeth" if tsh >= a.tshare and tclean.sum() >= 5 else "closed")


def make_plan(origin, lo, hi):
    """batches of a job clip whose frame j is the source frame origin + j; candidates = source frames lo .. hi - 1; refs = job-clip frames"""
    nb = max(1, -(-(hi - origin) // a.batch)); refs, rows = [], []; I = idx[(idx >= lo) & (idx < hi)]
    prev = lo if (V[lo] and (closed[lo] if mode == "closed" else tclean[lo])) else None   # the first frame (the default reference) stays while it is good enough
    for b in range(nb):
        s0, s1 = max(lo, origin + b * a.batch), min(hi, origin + (b + 1) * a.batch)
        if a.glob: s0, s1 = lo, hi
        B = I[(I >= s0) & (I < s1)]
        if not len(B):
            k = prev if prev is not None else (int(I[0]) if len(I) else lo); refs.append(k - origin); rows.append({"b": b, "ref": k - origin, "src": k, "why": "no face in the batch"}); continue
        tone = np.array([np.median(f[c][B]) for c in ("L", "A", "B")]); ton2 = np.array([np.median(f[c][B]) for c in ("pL", "pA")])
        cx, cy, yw, fh = (np.median(f[c][B]) for c in ("cx", "cy", "yaw", "fh"))
        def score(k):   # higher = better: the tone distance in Lab units, the pose in face heights
            dE = float(np.linalg.norm(np.array([f["L"][k], f["A"][k], f["B"][k]]) - tone)); dP = float(np.hypot(f["pL"][k] - ton2[0], f["pA"][k] - ton2[1]))
            pos = float(np.hypot(f["cx"][k] - cx, f["cy"][k] - cy)) / max(fh, 1)
            sc = -dE / 3.0 - dP / 4.0 - 10 * abs(f["yaw"][k] - yw) - 6 * abs(f["yaw"][k]) - 6 * max(0.0, pos - 0.05) - 3 * abs(f["fh"][k] / max(fh, 1) - 1) + 0.5 * min(f["sharp"][k] / max(ms, 1e-6), 1.5)
            sc -= a.lips * (min(abs(f["lipc"][k] - lc0) / (1.4826 * lcs), 3.0) + (min(abs(f["lth"][k] - lt0) / (1.4826 * lts), 3.0) if not tclean[k] else 0.0)) / 2
            if mode == "teeth" and tclean[k]: sc += 1.0 * min(f["t_up"][k], 0.8) - 2.0 * f["dark"][k] - 4 * abs(f["open"][k] - 0.15)
            return float(sc)
        pick, why = None, ""
        for want, name in (((tclean, "teeth"), (closed, "closed")) if mode == "teeth" else ((closed, "closed"),)):
            C = []
            for reach in ((1e9,) if a.glob else (a.reach, 8.0, 1e9)):
                C = I[want[I] & (I >= s0 - reach * 25) & (I < s1 + reach * 25)]
                if len(C): break
            if not len(C): continue
            sc = np.array([score(k) for k in C]); best = int(C[int(np.argmax(sc))]); bs = float(sc.max()); bs0 = bs
            if prev is not None and want[prev] and score(prev) >= bs - a.keep: best, bs = prev, score(prev)
            if name == "teeth" and bs < -3.0: continue   # a far-off / badly lit teeth frame is worse than a good closed one
            pick, why = best, "%s %.2f" % (name, bs); break
        if pick is None: pick, why = (prev if prev is not None else int(B[0])), "fallback"
        k = int(pick); prev = k; refs.append(k - origin)
        rows.append({"b": b, "frames": [int(s0), int(s1 - 1)], "ref": k - origin, "src": k, "t": round(k / 25, 2), "why": why, "open": round(float(f["open"][k]), 3), "t_up": round(float(f["t_up"][k]), 2),
                     "dE": round(float(np.linalg.norm(np.array([f["L"][k], f["A"][k], f["B"][k]]) - tone)), 1), "yaw": round(float(f["yaw"][k]), 3), "lipc": round(float(f["lipc"][k]), 1), "lth": round(float(f["lth"][k]), 3), "s_first": round(score(lo), 2) if V[lo] else None, "s_best": round(bs0, 2)})
        if a.glob: refs = refs * nb; break
    kf = None
    if a.kf_yaw > 0 and not a.glob:   # keyframe j of batch b = job-clip frame 13 (13 b + j - 1), j = 1 .. 13; slot 0 = the anchor
        ysm = np.array([np.nanmedian(f["yaw"][max(0, k - 3):k + 4]) if np.isfinite(f["yaw"][max(0, k - 3):k + 4]).any() else np.nan for k in range(N)]); kf = []; n_over = 0; hold = None
        for b in range(len(refs)):
            row = []
            for j in range(1, 14):
                k = origin + 13 * (13 * b + j - 1); rb = refs[b] + origin; pick = rb
                if lo <= k < hi and np.isfinite(ysm[k]) and V[rb] and abs(f["yaw"][rb] - ysm[k]) > a.kf_yaw:
                    if hold is not None and abs(f["yaw"][hold] - ysm[k]) <= 0.6 * a.kf_yaw and abs(hold - k) < 8 * 25: pick = hold   # the override of the previous keyframe still fits
                    else:
                        C = I[closed[I] & (np.abs(I - k) < a.reach * 25) & (np.abs(f["yaw"][I] - ysm[k]) < 0.6 * a.kf_yaw)]
                        if len(C):
                            tone = np.array([np.nanmedian(f[c][max(lo, k - 25):min(hi, k + 26)]) for c in ("L", "A", "B")])
                            sc = [-float(np.linalg.norm(np.array([f["L"][q], f["A"][q], f["B"][q]]) - tone)) / 3.0 - 10 * abs(f["yaw"][q] - ysm[k]) + 0.5 * min(f["sharp"][q] / max(ms, 1e-6), 1.5)
                                  - a.lips * min(abs(f["lipc"][q] - lc0) / (1.4826 * lcs), 3.0) / 2 for q in C]
                            pick = int(C[int(np.argmax(sc))])
                    hold = pick if pick != rb else None
                else: hold = None
                n_over += int(pick != rb); row.append(int(pick) - origin)
            kf.append([row[0]] + row)
        if n_over == 0: kf = None
        else: rows.append({"kf_overrides": n_over})
    return refs, rows, kf


# the first frame as the reference of every batch: tone distance, eye-region difference (m03 max 6.5 / 3.1, old03 2.6 / 4.5, 01 1.2 / 2.4: good; 04 11.5 / 11.4 with 3 of 8 batches > 7, 05 12.8 / 25.7: bad)
R0 = []
if V0[0] if (V0 := np.isfinite(F).all(1)).any() else False:
    for b in range(max(1, -(-N // a.batch))):
        B = np.flatnonzero(V0[b * a.batch:(b + 1) * a.batch]) + b * a.batch
        if len(B): R0.append((float(np.linalg.norm(np.array([f[c][0] - np.median(f[c][B]) for c in ("L", "A", "B")]))), float(abs(f["eyeL"][0] - np.median(f["eyeL"][B])))))
n_de = sum(1 for d, e in R0 if d > a.bad_de); n_eye = sum(1 for d, e in R0 if e > a.bad_eye)
bad0 = (not R0) or n_de >= max(1.0, a.bad_share * len(R0)) or n_eye >= 2
ref0 = {"batches": len(R0), "de_max": round(max([d for d, e in R0], default=0.0), 1), "de_over": n_de, "eye_max": round(max([e for d, e in R0], default=0.0), 1), "eye_over": n_eye, "bad": bool(bad0)}
if a.when == "bad" and not bad0:
    done({"refs": [0], "slot0": a.slot0, "mode": mode, "keep0": True, "ref0": ref0}, "the first frame is a good reference for the whole clip: kept")
meta = {"ref0": ref0, "slot0": a.slot0, "mode": mode, "batch": a.batch, "teeth_share": round(tsh, 3), "clean_teeth_frames": int(tclean.sum()), "closed_frames": int(closed.sum())}
if a.chunks:   # ks_chunk.py chunks: the job clip = [the reference] + source frames a .. a + in_frames - 2 -> its frame j = source frame a - 1 + j
    CJ = json.load(open(a.chunks + "/chunks.json")); src = []
    for ch in CJ["chunks"]:
        refs, rows, kf = make_plan(ch["a"] - 1, ch["a"], min(N, ch["a"] + ch["in_frames"] - 1))
        json.dump(dict(meta, refs=refs, rows=rows, **({"kf_refs": kf} if kf else {})), open("%s/c%02d.plan.json" % (a.chunks, ch["j"]), "w"), indent=1); src.append([r["src"] for r in rows if "src" in r])
    print("KS_LREFPICK " + json.dumps(dict(meta, chunks=src))); sys.exit(0)
refs, rows, kf = make_plan(0, 0, N); plan = dict(meta, refs=refs, rows=rows, **({"kf_refs": kf} if kf else {})); kfo = rows.pop() if rows and "kf_overrides" in rows[-1] else None; plan["kf_overrides"] = kfo["kf_overrides"] if kfo else 0
if a.sheet:
    c = cv2.VideoCapture(a.video); tiles = []
    for r in rows:
        c.set(cv2.CAP_PROP_POS_FRAMES, r["src"]); im = c.read()[1]; p = LM[r["src"]]; mw = np.linalg.norm(p[61] - p[291]); mx, my = p[[61, 291, 13, 14]].mean(0)
        z = im[max(0, int(my - 0.6 * mw)):int(my + 0.6 * mw), max(0, int(mx - 0.9 * mw)):int(mx + 0.9 * mw)]; t = np.vstack([cv2.resize(im, (240, 240)), cv2.resize(z, (240, 160))])
        for j, s in enumerate(["b%d %s" % (r["b"], r.get("frames")), "ref %d (%.1fs) %s" % (r["src"], r["src"] / 25, r["why"]), "open %s t_up %s dE %s" % (r.get("open"), r.get("t_up"), r.get("dE"))]):
            cv2.putText(t, s, (4, 14 + 15 * j), 0, 0.4, (0, 0, 0), 3); cv2.putText(t, s, (4, 14 + 15 * j), 0, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(a.sheet, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
done(plan, " | ".join("%d:%s" % (r["src"], r["why"]) for r in rows))
