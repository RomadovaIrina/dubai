#!/usr/bin/env python3
"""r16 Q 09.11: Irina's eye defects (ФИДБЕК_0911) as automatic checks. Uses r16q_detect's geometry (same 25 fps grid, the 512 crop around the
target face, MediaPipe landmarks on the source crop lmS and on ours lmO); pass the SAME --npz cache as r16q_detect (made here if missing).

  tail     broken face at the end (m03 / m18 / q01 / q01_subs): per frame the residual |ours - source| (Lab) on the lower face OUTSIDE the
           mouth (+0.3 IOD) and the drift of ours' stable landmarks (nose, eye corners) vs the source, each / its median over the middle
           10-90 % of the video; tail = the 3-frame median max over the last TAIL_S s. A sound tail looks like the middle (~1).
  outside  mask outside the face (q03 collar / neck, old03 right cheek + foliage, long5 neck, q05 window): the frame-to-frame change of the
           residual (ours - source) in bands OUTSIDE the face contour (left / right of the cheeks, and the neck below the jaw), relative
           to the same inside the cheeks. The source pixels there should be untouched -> ~0 change; regenerated pixels wobble every frame.
  chroma   blue in dark areas (long5 1:56): Lab b* of ours' dark pixels (L* < 30) on the lower face vs the source's b* at the same pixels.
           Positive = ours is bluer. Also the share of dark pixels that are clearly blue (b* < -4).
  occlude  occlusion / turn case list (source only, q05 cup + turns, q02 hand, long5 hair): share of lower-face pixels that do not match the
           frame's own cheek-skin colour (Mahalanobis a*b* > 3, the inner mouth excluded), and yaw from the landmarks.
    /venv/mediapipe/bin/python r16q_defects.py --src SRC --ours OURS --out OUT.json [--npz det.npz] [--tail-s 2]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r16q_detect as D

STABLE = [1, 4, 98, 327, 33, 133, 263, 362, 168]
JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]   # same band as K1's seam_jitter.py (one definition)
THR = dict(seam_band=0.08, paint=10.0, paint_rel=4.0, out_ratio_p95=1.2, tail_local=1.5, out_px=6.0, out_share=0.10, occl_rel=0.20, tail=1.5, outside=1.6, outside_abs=1.2, chroma=4.0, chroma_share=0.25, occl=0.25, yaw=0.45)


def lab(img): return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--ours", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None); ap.add_argument("--tail-s", type=float, default=2.0); ap.add_argument("--label", default="")
    ap.add_argument("--from-series", default=None, help="re-summarise a saved .series.npz (no decoding)")
    a = ap.parse_args()
    npz = pathlib.Path(a.npz or (pathlib.Path(a.out).with_suffix("") .as_posix() + ".det.npz"))
    key = f"{D.fingerprint(a.src)}:{D.fingerprint(a.ours)}:0:1e+09"
    ok = False
    if npz.exists():
        z = np.load(npz, allow_pickle=True); ok = "key" in z.files and str(z["key"]) == key
    if not ok:   # the geometry pass of r16q_detect (its cheapest detector) writes the cache
        subprocess.run([sys.executable, str(pathlib.Path(D.__file__)), "--src", a.src, "--ours", a.ours, "--out", str(npz) + ".det.json", "--npz", str(npz), "--only", "black"], check=True)
        z = np.load(npz, allow_pickle=True)
    boxes, lmS, lmO = z["boxes"], z["lmS"], z["lmO"]; N = len(boxes)
    W, H, _ = D.probe(a.src); sc = min(1.0, 1920 / max(W, H)); W, H = int(round(W * sc / 2) * 2), int(round(H * sc / 2) * 2)

    def crop(f, k):
        x0, y0, s = boxes[k]; s_ = D.CROP / s
        M = np.array([[s_, 0, -x0 * s_], [0, s_, -y0 * s_]], np.float32)
        return cv2.warpAffine(f, M, (D.CROP, D.CROP), flags=cv2.INTER_AREA if s_ < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    S = {k: np.full(N, np.nan) for k in ["res_out", "res_low", "sharp", "jaw_dev", "lm_dev", "wob_l", "wob_r", "wob_n", "wob_in", "out_l", "out_r", "out_n", "out_up", "seam_in", "seam_out", "seam_up", "seam_src", "seam_fin", "blue", "blue_share", "dark_n", "occl", "yaw", "edge_box"]}
    prevR = None; prevLS = None; prevLO = None
    if a.from_series:
        zz = np.load(a.from_series); S.update({k: zz[k] for k in zz.files})
    frames = [] if a.from_series else zip(D.reader(a.src, W, H, 0, 1e9), D.reader(a.ours, W, H, 0, 1e9))
    for k, (fS, fO) in enumerate(frames):
        if k >= N: break
        pS, pO = lmS[k], lmO[k]
        if not np.isfinite(pS).all(): prevR = None; prevLS = None; continue
        cS, cO = crop(fS, k), crop(fO, k); LS, LO = lab(cS), lab(cO); sh = cS.shape[:2]
        iod = float(np.linalg.norm(pS[33] - pS[263])); e = lambda r: max(1, int(r * iod))
        mouthX = D.hull_mask(sh, pS[D.LIPS], dil=0.3 * iod)
        if np.isfinite(pO).all(): mouthX |= D.hull_mask(sh, pO[D.LIPS], dil=0.3 * iod)
        low = D.poly_mask(sh, pS[D.LOW]); face = D.hull_mask(sh, pS[D.OVAL])
        R = LO - LS   # residual (Lab)
        m = (low & (1 - mouthX)) > 0
        if m.sum() > 200: S["res_out"][k] = float(np.abs(R[m]).sum(1).mean())
        ml = low > 0
        if ml.sum() > 200:
            S["res_low"][k] = float(np.abs(R[ml]).sum(1).mean())
            gS = cv2.Laplacian(cv2.GaussianBlur(LS[..., 0], (3, 3), 0), cv2.CV_32F); gO = cv2.Laplacian(cv2.GaussianBlur(LO[..., 0], (3, 3), 0), cv2.CV_32F)
            S["sharp"][k] = float(np.abs(gO[ml]).mean() / max(np.abs(gS[ml]).mean(), 1e-3))   # < 1 = ours smeared
        if np.isfinite(pO).all():
            S["lm_dev"][k] = float(np.median(np.linalg.norm(pO[STABLE] - pS[STABLE], axis=1)) / iod * 100)
            S["jaw_dev"][k] = float(np.median(np.linalg.norm(pO[D.LOWOVAL] - pS[D.LOWOVAL], axis=1)) / iod * 100)
        # outside-face bands: ring 0.05..0.45 IOD outside the oval, split left / right of the face centre; neck = below the jaw
        ring = (cv2.dilate(face, np.ones((2 * e(0.45) + 1,) * 2, np.uint8)) & (1 - cv2.dilate(face, np.ones((2 * e(0.05) + 1,) * 2, np.uint8)))) > 0
        cx = float(pS[D.OVAL][:, 0].mean()); yy, xx = np.mgrid[0:sh[0], 0:sh[1]]
        chin = pS[152]; jaw_y = float(np.max(pS[D.LOWOVAL][:, 1]))
        neck = ring & (yy > jaw_y - 0.1 * iod) & (np.abs(xx - chin[0]) < 0.9 * iod)
        side = ring & ~neck & (yy > pS[1][1] - 0.2 * iod)   # cheek level and below (not the hair / forehead)
        bands = {"wob_l": side & (xx < cx), "wob_r": side & (xx >= cx), "wob_n": neck, "wob_in": m}
        eye_y = float(min(pS[33][1], pS[263][1])); up = ring & (yy < eye_y - 0.3 * iod)   # forehead / hair ring: never pasted = codec floor
        aR = np.abs(R).sum(-1)
        for nm, mk in {"out_l": bands["wob_l"], "out_r": bands["wob_r"], "out_n": neck, "out_up": up}.items():
            if mk.sum() > 100: S[nm][k] = float((aR[mk] > THR["out_px"]).mean())
        # jaw seam band: +-SEAM_W x face height around the jaw / chin contour (source landmarks), split inside / outside the face
        band = np.zeros(sh, np.uint8); cv2.polylines(band, [np.round(pS[JAW]).astype(np.int32)], False, 1, thickness=max(2, int(2 * THR["seam_band"] * iod)))
        band = band > 0; fin = face > 0
        if prevR is not None:
            dR = np.abs(R[..., 0] - prevR[..., 0])
            for nm, mk in {"seam_in": band & fin, "seam_out": band & ~fin, "seam_up": up}.items():
                if mk.sum() > 100: S[nm][k] = float(dR[mk].mean())
            if prevLS is not None and band.sum() > 100:
                S["seam_src"][k] = float(np.abs(LS[..., 0] - prevLS[..., 0])[band].mean()); S["seam_fin"][k] = float(np.abs(LO[..., 0] - prevLO[..., 0])[band].mean())
            for nm, mk in bands.items():
                if mk.sum() > 100: S[nm][k] = float(dR[mk].mean())
        prevR = R; prevLS = LS; prevLO = LO
        # chroma: dark pixels on the lower face (mouth included)
        dk = low.astype(bool) & (LO[..., 0] < 30)
        S["dark_n"][k] = int(dk.sum())
        if dk.sum() > 80:
            S["blue"][k] = float(LS[..., 2][dk].mean() - LO[..., 2][dk].mean()); S["blue_share"][k] = float((LO[..., 2][dk] < -4).mean())
        # occlusion (source only): lower-face pixels off the frame's cheek-skin colour
        ck = D.hull_mask(sh, pS[D.REG["lcheek"]]) | D.hull_mask(sh, pS[D.REG["rcheek"]])
        inner = D.hull_mask(sh, pS[D.INNER], dil=0.05 * iod)
        if ck.sum() > 100:
            ab = LS[..., 1:][ck > 0]; mu = ab.mean(0); C = np.cov(ab.T) + np.eye(2) * 1.0; Ci = np.linalg.inv(C)
            q = (low & (1 - inner)) > 0; d = LS[..., 1:][q] - mu
            S["occl"][k] = float((np.einsum("ni,ij,nj->n", d, Ci, d) > 9.0).mean())
        w = np.linalg.norm(pS[234] - pS[454]); S["yaw"][k] = float((np.linalg.norm(pS[1] - pS[234]) - np.linalg.norm(pS[1] - pS[454])) / max(w, 1e-6))
    # ---- summaries
    fps = D.FPS; n = N; mid = slice(int(0.1 * n), int(0.9 * n))
    rel = lambda x: x / max(np.nanmedian(x[mid]), 1e-6)
    from scipy.ndimage import median_filter
    def med3(x): y = x.copy(); y[~np.isfinite(y)] = 0; return median_filter(y, 3, mode="nearest")
    # tail: each feature as a 6-frame (0.24 s) rolling mean; z = (x - mid median) / (mid p99 - mid median); tail = max z over the last TAIL_S s
    T = int(a.tail_s * fps); feats = {"res_out": 1, "res_low": 1, "lm_dev": 1, "jaw_dev": 1, "sharp": -1}
    def roll(x, w=6):
        y = np.where(np.isfinite(x), x, np.nan); out = np.full(len(y), np.nan)
        for i in range(len(y)):
            v = y[max(0, i - w + 1):i + 1]; v = v[np.isfinite(v)]
            if len(v) >= w // 2: out[i] = v.mean()
        return out
    tl = {}; zall = np.full(n, -np.inf)
    for nm, sgn in feats.items():
        x = sgn * roll(S[nm]); md = np.nanmedian(x[mid]); sp = max(np.nanpercentile(x[mid], 99) - md, 1e-6); z = (x - md) / sp
        tw = z[max(0, n - T):]; kk = int(np.nanargmax(tw)) if np.isfinite(tw).any() else 0
        tl[nm] = {"z": round(float(np.nanmax(tw)) if np.isfinite(tw).any() else 0, 2), "at_s": round((n - T + kk) / fps, 2)}
        zall = np.fmax(zall, np.where(np.isfinite(z), z, -np.inf))
    tw = zall[max(0, n - T):]; kt = int(np.argmax(tw))
    res = {"label": a.label, "src": a.src, "ours": a.ours, "frames": n, "thr": THR}
    loc = {}
    for nm in ("res_out", "jaw_dev"):
        x = S[nm]; pre = x[max(0, n - 75):max(0, n - 25)]; last = x[max(0, n - 25):n - 1]
        if np.isfinite(pre).sum() > 10 and np.isfinite(last).sum() >= 4:
            lv = np.convolve(np.where(np.isfinite(last), last, np.nanmedian(pre)), np.ones(4) / 4, "valid")
            loc[nm] = {"ratio": round(float(lv.max() / max(np.nanmedian(pre), 1e-3)), 2), "at_s": round((n - 25 + int(lv.argmax()) + 3) / fps, 2)}
    lr = max([v["ratio"] for v in loc.values()], default=0.0)
    res["tail"] = {"local": loc, "local_max": lr, "flag_local": bool(lr >= THR["tail_local"]), "score": round(float(tw.max()), 2), "at_s": round((n - T + kt) / fps, 2), "flag_global": bool(tw.max() > THR["tail"]), "flag": bool(lr >= THR["tail_local"]), "per_feature": tl,
                   "mid_max": round(float(np.max(zall[mid][np.isfinite(zall[mid])])), 2),
                   "last_frames": {nm: [round(float(v), 2) for v in S[nm][-25:]] for nm in feats}}
    # outside: per second, band wobble / inside-cheek wobble; flag runs where a band is > THR x inside AND > abs floor (L* units)
    out = {}
    for nm in ("wob_l", "wob_r", "wob_n"):
        r = S[nm] / np.fmax(S["wob_in"], 0.3)
        fl = (r > THR["outside"]) & (S[nm] > THR["outside_abs"])
        out[nm] = {"median_ratio": round(float(np.nanmedian(r)), 2), "share": round(float(np.nanmean(fl)), 3), "runs": D.runs(fl, r, 12)[:30]}
    for nm in ("out_l", "out_r", "out_n"):
        ex = S[nm] - np.where(np.isfinite(S["out_up"]), S["out_up"], 0)
        fl = ex > THR["out_share"]
        out[nm] = {"median_excess": round(float(np.nanmedian(ex)), 3), "p90_excess": round(float(np.nanpercentile(ex, 90)), 3) if np.isfinite(ex).any() else None,
                   "share": round(float(np.nanmean(fl)), 3), "runs": D.runs(fl, ex, 12)[:30]}
    # the flag: the outside band wobbles more than the inside cheeks (p95 over frames of band / inside); runs = 1-s rolling median above it
    win = np.fmax(S["wob_in"], 0.3); fl_any = False
    for nm in ("wob_l", "wob_r", "wob_n"):
        r = S[nm] / win; ok = np.isfinite(r)
        p95 = float(np.nanpercentile(r, 95)) if ok.sum() > 25 else 0.0
        rm = np.array([np.nanmedian(r[max(0, i - 12):i + 13]) if ok[max(0, i - 12):i + 13].any() else np.nan for i in range(len(r))])
        out[nm].update({"ratio_p95": round(p95, 2), "flag": bool(p95 >= THR["out_ratio_p95"]),
                        "runs_1s": D.runs(rm >= THR["out_ratio_p95"], rm, 12)[:30]})
        fl_any |= out[nm]["flag"]
    out["flag"] = bool(fl_any)
    res["outside"] = out
    # seam jitter: residual wobble in the jaw band minus the forehead-ring codec floor (L* units / frame); per second + worst seconds
    sx = np.fmax(S["seam_in"], S["seam_out"]) - np.where(np.isfinite(S["seam_up"]), S["seam_up"], 0)
    per_s = [float(np.nanmean(sx[i:i + int(fps)])) if np.isfinite(sx[i:i + int(fps)]).any() else np.nan for i in range(0, n, int(fps))]
    ps = np.array(per_s); order = [int(i) for i in np.argsort(-np.nan_to_num(ps, nan=-9))[:5] if np.isfinite(ps[i])]
    res["seam"] = {"excess_median": round(float(np.nanmedian(sx)), 3) if np.isfinite(sx).any() else None,
                   "excess_p95": round(float(np.nanpercentile(sx, 95)), 3) if np.isfinite(sx).any() else None,
                   "in_median": round(float(np.nanmedian(S["seam_in"])), 3) if np.isfinite(S["seam_in"]).any() else None,
                   "out_median": round(float(np.nanmedian(S["seam_out"])), 3) if np.isfinite(S["seam_out"]).any() else None,
                   "floor_median": round(float(np.nanmedian(S["seam_up"])), 3) if np.isfinite(S["seam_up"]).any() else None,
                   "src_motion_median": round(float(np.nanmedian(S["seam_src"])), 3) if np.isfinite(S["seam_src"]).any() else None,
                   "k1_ratio": round(float(np.nanmean(S["seam_fin"]) / max(np.nanmean(S["seam_src"]), 1e-6)), 3) if np.isfinite(S["seam_fin"]).any() else None,
                   "k1_extra_mean": round(float(np.nanmean(S["seam_fin"] - S["seam_src"])), 3) if np.isfinite(S["seam_fin"]).any() else None,
                   "worst_s": [[i, round(float(ps[i]), 3)] for i in order], "per_s": [None if not np.isfinite(v) else round(v, 3) for v in per_s]}
    bl = (S["blue"] > THR["chroma"]) & (S["blue_share"] > THR["chroma_share"])
    res["chroma"] = {"median_blue": round(float(np.nanmedian(S["blue"])), 2), "share": round(float(np.nanmean(bl)), 3), "runs": D.runs(bl, S["blue"], 6)[:30]}
    oc = (S["occl"] - np.nanmedian(S["occl"])) > THR["occl_rel"]; yw = np.abs(S["yaw"]) > THR["yaw"]
    # painted over an object (q05 cup): ours jaw landmarks leave the sources (1-s rolling median > max(10 % IOD, 4 x the video median))
    j = S["jaw_dev"]; jm = np.nanmedian(j) if np.isfinite(j).any() else 0.0
    jr = np.array([np.nanmedian(j[max(0, i - 6):i + 7]) if np.isfinite(j[max(0, i - 6):i + 7]).any() else np.nan for i in range(n)])
    pt = jr > max(THR["paint"], THR["paint_rel"] * jm)
    res["occlude"] = {"occl_runs": D.runs(oc, S["occl"], 6)[:30], "turn_runs": D.runs(yw, np.abs(S["yaw"]), 12)[:30], "paint_runs": D.runs(pt, jr, 6)[:30],
                      "occl_share": round(float(np.nanmean(oc)), 3), "turn_share": round(float(np.nanmean(yw)), 3)}
    json.dump(res, open(a.out, "w"), indent=1)
    if not a.from_series: np.savez_compressed(pathlib.Path(a.out).with_suffix(".series.npz"), **S)
    print("DEFECTS " + json.dumps({"label": a.label, "tail": {k: res["tail"][k] for k in ("score", "at_s", "flag")},
                                   "outside": {"flag": out["flag"], **{k: v["ratio_p95"] for k, v in out.items() if k.startswith("wob_")}}, "seam": [res["seam"]["excess_median"], res["seam"]["excess_p95"], res["seam"]["k1_ratio"], res["seam"]["worst_s"][:3]], "chroma": (res["chroma"]["share"], len(res["chroma"]["runs"])),
                                   "occl": (res["occlude"]["occl_share"], len(res["occlude"]["occl_runs"])), "turn": (res["occlude"]["turn_share"], len(res["occlude"]["turn_runs"]))}))


if __name__ == "__main__":
    main()
