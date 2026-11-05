#!/usr/bin/env python3
"""r15 F3: mouth "EXPLOSIONS" (single / double-frame pops) of finals, and the STAGE each one is born in.
pop[k] = mean |f_k - (w_{k-1} + w_{k+1}) / 2| over the mouth (lip hull + 0.3 IOD), neighbours flow-warped onto k (as r12 mouth_pop):
a smooth motion leaves ~0, a frame that differs from BOTH neighbours (a pop) leaves a residual.
Finals and the source on the source timeline (source landmarks); events = the primary final's pop minus the source's > --thr, peaks
>= 3 frames apart, the top --top. For each event (source frame k, master frame m = round(mu[k]), retime phase frac(mu)) the pop of every
--stage layer (25 fps master timeline, its own landmarks) at m-1..m+1 minus the master's own pop there; origin = the first stage whose
excess reaches 60 % of the final's (>= 2 grey levels); "composite / retime" when no layer stage has it.
    explode.py --work W --timing T.json --src SRC --src-lm source_lm_F.npz --final recipe=F1.mp4 --final prod=F2.mp4
               --stage pre=L0.mp4 --stage sr=L1.mp4 ... [--top 20] [--sheet out.jpg --sheet-n 8] [--json out.json]"""
import argparse, json, pathlib
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def read_all(p, gray=True):
    c = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f)
    return out


def pop_at(F, lm, k, dil=0.3):
    if k < 1 or k + 1 >= len(F) or k >= len(lm): return np.nan
    p = lm[k]
    if not np.isfinite(p[LIPS + [33, 263], :2]).all(): return np.nan
    iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); q = p[LIPS, :2]; r = int(0.6 * iod)
    x0, y0 = np.maximum(q.min(0).astype(int) - r, 0); x1, y1 = q.max(0).astype(int) + r
    A, B, C = [np.ascontiguousarray(F[j][y0:y1, x0:x1]) for j in (k - 1, k, k + 1)]
    if B.size < 400: return np.nan
    gy, gx = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(np.float32)
    f = dis.calc(B, A, None); g = dis.calc(B, C, None)
    wa = cv2.remap(A, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    wc = cv2.remap(C, gx + g[..., 0], gy + g[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    m = np.zeros(B.shape, np.uint8); cv2.fillPoly(m, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dil * iod) + 1,) * 2)) > 0
    return float(np.abs(B.astype(np.float32) - 0.5 * (wa + wc))[m].mean())


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("work", "timing", "src", "src-lm")]
    ap.add_argument("--final", action="append", required=True); ap.add_argument("--stage", action="append", default=[])
    ap.add_argument("--events", help="the gate's flash.json of the PRIMARY final: its flash_list frames are the events"); ap.add_argument("--ratio", type=float, default=1.6, help="a stage has the pop when its pop at the event is >= this x its own median")
    ap.add_argument("--thr", type=float, default=5.0); ap.add_argument("--top", type=int, default=20); ap.add_argument("--json"); ap.add_argument("--sheet"); ap.add_argument("--sheet-n", type=int, default=8)
    a = ap.parse_args(); W = pathlib.Path(a.work)
    z = np.load(a.src_lm); slm = z["lm"].astype(np.float64); mu = z["mu"]
    S = read_all(a.src); n = min(len(S), len(slm))
    ps = np.array([pop_at(S, slm, k) for k in range(n)]); print(f"source pop mean {np.nanmean(ps):.2f}", flush=True)
    fin = {}
    for spec in a.final:
        lab, v = spec.split("=", 1); Fv = read_all(v)
        if Fv[0].shape != S[0].shape: Fv = [cv2.resize(x, (S[0].shape[1], S[0].shape[0])) for x in Fv]
        pv = np.array([pop_at(Fv, slm, k) for k in range(min(n, len(Fv)))]); fin[lab] = (Fv, pv)
        x = pv - ps[:len(pv)]; print(f"{lab}: pop mean {np.nanmean(pv):.2f}, excess mean {np.nanmean(x):.2f}, frames > thr {int(np.nansum(x > a.thr))}", flush=True)
    prim = a.final[0].split("=", 1)[0]; pv = fin[prim][1]; x = np.nan_to_num(pv - ps[:len(pv)], nan=0)
    if a.events:   # the gate's own strong-flash list (r4 flash.json: the frames the eye calls explosions), strongest first
        fl = sorted(json.load(open(a.events))["flash_list"], key=lambda e: -e.get("mc_vs_src", 0)); ev = []
        for e in fl:
            if len(ev) >= a.top: break
            if all(abs(e["frame"] - q) >= 3 for q in ev): ev.append(int(e["frame"]))
    else:
        ev = []
        for k in np.argsort(-x):
            if x[k] <= a.thr or len(ev) >= a.top: break
            if all(abs(k - e) >= 3 for e in ev): ev.append(int(k))
    ev.sort()
    M = read_all(W / "video/master_25fps.mp4"); mlm = np.load(W / "video/master_mediapipe.npz")["lm"].astype(np.float64)
    nm_ = len(M); mpop = np.array([pop_at(M, mlm, j) for j in range(nm_)]); mmed = float(np.nanmedian(mpop))
    st = []
    for spec in a.stage:   # each stage's FULL pop series on the master timeline: an event is judged by its ratio to the stage's own median
        lab, v = spec.split("=", 1); v = pathlib.Path(v); F = read_all(v); lm = np.load(v.with_suffix(".mediapipe.npz"))["lm"].astype(np.float64)
        sp = np.array([pop_at(F, lm, j) for j in range(min(len(F), nm_))]); st.append((lab, sp, float(np.nanmedian(sp)))); del F
        print(f"stage {lab}: pop median {np.nanmedian(sp):.2f}", flush=True)
    rows = []; origins = {}
    for k in ev:
        m = int(round(mu[k])); ph = float(mu[k] - np.floor(mu[k])); fx = float(x[k]) if k < len(x) else np.nan
        win = [j for j in (m - 1, m, m + 1) if 0 <= j < nm_]
        base = float(np.nanmax(mpop[win])) / max(mmed, 1e-6)
        sv = {lab: round(float(np.nanmax(sp[[j for j in win if j < len(sp)]])) / max(med, 1e-6), 2) for lab, sp, med in st}
        org = next((lab for lab, _, _ in st if sv[lab] >= a.ratio and sv[lab] >= base + 0.4), "composite/retime")
        origins[org] = origins.get(org, 0) + 1
        row = {"t": round(k / 29.97, 2), "src_frame": k, "master": m, "phase": round(ph, 2), "final_excess": {lab: round(float(fin[lab][1][k] - ps[k]), 2) for lab in fin},
               "master_ratio": round(base, 2), "stage_ratio": sv, "origin": org}
        rows.append(row); print(json.dumps(row), flush=True)
    rep = {"events": rows, "origin_counts": origins, "thr": a.thr}
    print("== explode", json.dumps({"n_events": len(rows), "origin_counts": origins}), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(rep, indent=1))
    if a.sheet and rows:   # the worst --sheet-n events: each final at k-1, k, k+1 + the source at k
        top = sorted(rows, key=lambda r: -r["final_excess"][prim])[:a.sheet_n]; Sc = read_all(a.src, gray=False); tiles = []
        Fc = {lab: read_all(spec.split("=", 1)[1], gray=False) for lab, spec in zip(fin, a.final)}
        for r in top:
            k = r["src_frame"]; p = slm[k]; c = p[LIPS, :2].mean(0); mw = np.linalg.norm(p[61, :2] - p[291, :2]); h = int(0.9 * mw)
            y0, x0 = int(max(0, c[1] - h)), int(max(0, c[0] - 1.2 * h)); y1, x1 = y0 + 2 * h, x0 + int(2.4 * h); row = []
            for lab in fin:
                for j in (k - 1, k, k + 1):
                    im = Fc[lab][j]; im = cv2.resize(im, (Sc[0].shape[1], Sc[0].shape[0])) if im.shape != Sc[0].shape else im
                    t_ = cv2.resize(im[y0:y1, x0:x1], (240, 200)); cv2.putText(t_, f"{lab} {j - k:+d}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); row.append(t_)
            t_ = cv2.resize(Sc[k][y0:y1, x0:x1], (240, 200)); cv2.putText(t_, f"src t={r['t']} {r['origin']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); row.append(t_)
            tiles.append(np.hstack(row))
        cv2.imwrite(a.sheet, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("sheet", a.sheet, flush=True)


if __name__ == "__main__":
    main()
