#!/usr/bin/env python3
"""r15 T: where do the lips vanish? Per-frame lip visibility of a dubbed job, stage by stage, vs the source (CPU only).

For one face (the lip-synced main face of the job, i.e. the face-units face 0), at every 25 fps master frame:
  a_con  = median a* (CIELAB) of the lip ring (outer lip polygon minus the inner one) minus the median a* of a skin ring around it
           (lips are redder than skin; a washed-out / vanished mouth loses it)
  grad   = mean Sobel magnitude of L on a 3 px band along the outer lip contour (a vanished lip has no edge)
both as a ratio to the SOURCE at the same master frame (ra, rg), for every stage the job kept:
  raw (LatentSync output) -> cfd (CodeFormer detail layer, profile current only) -> sr (Real-ESRGAN layer) -> rep (layer repair)
  -> final (the composite = what the user sees: fallback / wash / dark swaps, colour match, sharpening)
The polygon of each stage comes from its own MediaPipe pass (<stage>.mediapipe.npz); the final uses the rep stage's
landmarks at the matching source frame (master_timing_F.json "j").

    lip_vanish.py --job t15_ab_m03_cur --job t15_ab_m03_cln --src <source.mp4> --out-dir <dir> [--sheet <jpg>] [--n 8]
"""
import argparse, json, pathlib, sys
import cv2, numpy as np

A = pathlib.Path("/workspace/dub/artifacts/r4m")
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
cv2.setNumThreads(4)


def measure(img, lm):
    """-> (a_con, grad, dL, box) (dL = median L of the skin ring minus that of the lip ring: lips darker than skin > 0) for one BGR frame and 478 x 2 pixel landmarks; NaN when the face is missing"""
    if lm is None or not np.isfinite(lm[OUTER]).all(): return np.nan, np.nan, np.nan, None
    o = lm[OUTER].astype(np.int32); i = lm[INNER].astype(np.int32)
    x0, y0 = o.min(0); x1, y1 = o.max(0); mw = max(8, x1 - x0); pad = int(0.6 * mw)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad); X1, Y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[Y0:Y1, X0:X1]; off = np.array([X0, Y0])
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    mo = np.zeros(crop.shape[:2], np.uint8); cv2.fillPoly(mo, [o - off], 1)
    mi = np.zeros_like(mo); cv2.fillPoly(mi, [i - off], 1)
    ring = (mo == 1) & (mi == 0)
    k1 = max(3, int(0.08 * mw)) | 1; k2 = max(5, int(0.30 * mw)) | 1
    d1 = cv2.dilate(mo, np.ones((k1, k1), np.uint8)); d2 = cv2.dilate(mo, np.ones((k2, k2), np.uint8))
    skin = (d2 == 1) & (d1 == 0)
    if ring.sum() < 20 or skin.sum() < 20: return np.nan, np.nan, np.nan, None
    a_con = float(np.median(lab[..., 1][ring]) - np.median(lab[..., 1][skin]))
    L = lab[..., 0]; g = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    band = np.zeros_like(mo); cv2.polylines(band, [o - off], True, 1, 3)
    return a_con, float(g[band == 1].mean()), float(np.median(L[skin]) - np.median(L[ring])), (int((x0 + x1) / 2), int((y0 + y1) / 2), int(mw))


def frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok: break
        yield f
    cap.release()


def lm_of(npz, face=None):
    z = np.load(npz); L = z["lm"].astype(np.float32)
    if face is not None and "lm_all" in z.files: L = z["lm_all"][:, face].astype(np.float32)
    if np.nanmax(L) <= 1.5: L = L * z["wh"][None, None, :]
    return L


def stages_of(job):
    W = A / f"r9g_{job}" / f"work_{job}_r9g_{job}"; p = f"{job}_r9g_{job}_25fps"
    st = [("raw", W / f"{p}_ls_raw.mp4")]
    if (W / f"{p}_ls_cfd_ia04.mp4").exists():
        st += [("cfd", W / f"{p}_ls_cfd_ia04.mp4"), ("sr", W / f"{p}_ls_cfd_ia04_srt1.mp4"), ("rep", W / f"{p}_ls_cfd_ia04_srt1_rep2s.mp4")]
        rj = W / f"{p}_ls_cfd_ia04_srt1_rep2s.repair.json"
    else:
        st += [("sr", W / f"{p}_ls_srt1.mp4"), ("rep", W / f"{p}_ls_srt1_rep2s.mp4")]; rj = W / f"{p}_ls_srt1_rep2s.repair.json"
    return W, p, st, rj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--job", action="append", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--sheet"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.45, help="a dip: the final's lip contrast (min of the colour / lightness ratios) below this")
    ap.add_argument("--t", type=float, nargs="*", help="extra instants (s) for the sheet (the eye's complaints)")
    a = ap.parse_args(); out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res, crops = {}, {}
    j0 = a.job[0]; W0, p0, _, _ = stages_of(j0)
    fc = json.load(open(A / f"r9g_{j0}" / "timing" / f"{j0}_master_timing_F.json")); J = np.array(fc["j"])   # master frame k -> source frame J[k]
    Lsrc = lm_of(W0 / "video" / "master_mediapipe_2faces.npz", face=0)
    src = np.array([measure(f, Lsrc[k])[:3] for k, f in enumerate(frames(W0 / "video" / "master_25fps.mp4"))])
    nM = len(src); res["source"] = {"a_con": src[:, 0], "grad": src[:, 1]}
    print(f"master frames {nM}; source a_con median {np.nanmedian(src[:, 0]):.1f}, grad {np.nanmedian(src[:, 1]):.1f}", flush=True)
    for job in a.job:
        W, p, st, rj = stages_of(job); r = {}
        for name, v in st:
            L = lm_of(v.with_suffix(".mediapipe.npz")); m = np.array([measure(f, L[k])[:3] for k, f in enumerate(frames(v))])[:nM]
            r[name] = {"a_con": m[:, 0], "grad": m[:, 1], "dL": m[:, 2]}; print(f"{job} {name}: {len(m)} frames", flush=True)
        Lrep = lm_of(W / f"{st[-1][1].stem}.mediapipe.npz")
        fin = list(frames(pathlib.Path(f"/workspace/dub/r15t/jobs/{job}/{job}.mp4"))); m = []; crops[job] = []
        for k in range(nM):
            f = fin[min(J[k], len(fin) - 1)]; ac, g, dl, box = measure(f, Lrep[k]); m.append((ac, g, dl)); crops[job].append((f, box))
        m = np.array(m); r["final"] = {"a_con": m[:, 0], "grad": m[:, 1], "dL": m[:, 2]}
        fbz = np.load(W / f"{p}_fb_fb7s_r9f_{job}_rep2s.npz"); r["fb_weight"] = fbz["fb"][:nM]
        r["repair"] = json.load(open(rj)); r["fb_json"] = json.load(open(W / f"{p}_fb_fb7s.json"))
        res[job] = r; del fin
    sfr = list(frames(pathlib.Path(a.src)))
    # ---- ratios, dips, attribution
    rows, report = [], {"source": a.src, "jobs": {}, "thr": a.thr}
    for job in a.job:
        r = res[job]; names = [n for n in ("raw", "cfd", "sr", "rep", "final") if n in r]
        ra = {n: r[n]["a_con"] / np.maximum(src[:, 0], 1.0) for n in names}; rg = {n: r[n]["grad"] / np.maximum(src[:, 1], 1e-3) for n in names}
        rd = {n: r[n]["dL"] / np.maximum(src[:, 2], 2.0) for n in names}   # lightness contrast: a pale (washed) lip goes to ~0 or below
        wash = {n: np.minimum(ra[n], rd[n]) for n in names}   # washed = the lips lose their colour OR lightness contrast to the skin
        vis = wash["final"]; dip = (vis < a.thr) & ((src[:, 0] > 4) | (src[:, 2] > 4))   # where the source has visible lips
        runs, k = [], 0
        while k < nM:
            if dip[k]:
                e = k
                while e + 1 < nM and dip[e + 1]: e += 1
                seg = range(k, e + 1); worst = min(seg, key=lambda q: vis[q])
                first = next((n for n in names if min(wash[n][q] for q in seg) < a.thr + 0.15), "final")
                runs.append({"t0": round(k / 25, 2), "t1": round((e + 1) / 25, 2), "frames": e - k + 1, "worst_t": round(worst / 25, 2),
                             "ra_final": round(float(ra["final"][worst]), 2), "rd_final": round(float(rd["final"][worst]), 2), "rg_final": round(float(rg["final"][worst]), 2),
                             "by_stage_wash": {n: round(float(min(wash[n][q] for q in seg)), 2) for n in names},
                             "first_stage_below": first, "fb_weight_max": round(float(np.max(r["fb_weight"][k:e + 1])), 2),
                             "repaired": [x for x in r["repair"].get("repaired", []) if x[1] >= k and x[0] <= e]})
                k = e + 1
            else: k += 1
        report["jobs"][job] = {"dip_frames": int(dip.sum()), "dip_pct": round(100 * dip.mean(), 1), "runs": runs,
                               "median_ra": {n: round(float(np.nanmedian(ra[n])), 2) for n in names},
                               "median_rg": {n: round(float(np.nanmedian(rg[n])), 2) for n in names},
                               "median_rd": {n: round(float(np.nanmedian(rd[n])), 2) for n in names},
                               "frames_washed": {n: int(np.nansum((wash[n] < a.thr) & ((src[:, 0] > 4) | (src[:, 2] > 4)))) for n in names},
                               "fb_top_s": r["fb_json"].get("top_s"), "repair_by_class": r["repair"].get("by_class")}
        np.savez(out / f"{job}_lipvis.npz", src_a=src[:, 0], src_g=src[:, 1], src_d=src[:, 2], **{f"ra_{n}": ra[n] for n in names}, **{f"rg_{n}": rg[n] for n in names},
                 **{f"rd_{n}": rd[n] for n in names}, **{f"w_{n}": wash[n] for n in names}, fb=r["fb_weight"])
        res[job]["vis"] = vis
    (out / "lip_vanish.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)[:6000])
    if a.sheet:   # worst frames of the FIRST job (+ the same instants in the others and the source), at least 0.3 s apart
        v0 = np.nan_to_num(res[j0]["vis"], nan=9); order = np.argsort(v0); pick = []
        for q in order:
            if src[q, 0] <= 4: continue
            if all(abs(q - x) >= 8 for x in pick): pick.append(int(q))
            if len(pick) >= a.n: break
        pick = sorted(set(pick) | {int(round(t * 25)) for t in (a.t or [])}); tiles = []
        for q in pick:
            f0, box = crops[j0][q]
            if box is None: continue
            cx, cy, mw = box; s = int(2.6 * mw); x0 = int(np.clip(cx - s / 2, 0, f0.shape[1] - s)); y0 = int(np.clip(cy - s / 2, 0, f0.shape[0] - s))
            row = []
            for job in a.job:
                c = cv2.resize(crops[job][q][0][y0:y0 + s, x0:x0 + s], (300, 300), interpolation=cv2.INTER_LANCZOS4)
                cv2.putText(c, f"{job.replace('t15_ab_', '')} lip contrast {res[job]['vis'][q]:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2); row.append(c)
            sf = sfr[min(J[q], len(sfr) - 1)]; c = cv2.resize(sf[y0:y0 + s, x0:x0 + s], (300, 300), interpolation=cv2.INTER_LANCZOS4)
            cv2.putText(c, f"source t={q / 25:.2f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2); row.append(c)
            tiles.append(np.hstack(row))
        sheet = np.vstack(tiles); cv2.imwrite(a.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); print("sheet", a.sheet, sheet.shape, "t =", [round(q / 25, 2) for q in pick])


if __name__ == "__main__":
    sys.exit(main())
