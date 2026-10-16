#!/usr/bin/env python3
"""Mouth / face jitter of a dubbed output vs its 25 fps master, split by SITUATION (lip-sync track, 2026-10-16).

Question: is the visible twitching of the mouth / face made by the lip-sync generator itself, or by what is around it (crop / affine,
mask, paste-back, gating, LS <-> original switches)? face_temporal.py gives whole-video ratios; this tool splits every metric into the
situations the human sees, lists their frame ranges, and adds two measurements that separate generator-internal jitter:

Situations (frames of the 25 fps master, from the run manifest + the master's own landmarks):
  S1 speech      LS frames inside continuous dubbed speech: > --win frames from an LS<->original switch, inside a dub VAD interval >= 1 s,
                 inside an LS segment >= 2 s (the steady state of the generator)
  S2 switch      +-win frames around an LS <-> original switch (manifest segments)
  S3 head_turn   LS frames where the master head moves: smoothed per-frame |d yaw| > --turn-yaw or |d pitch| > --turn-pitch (nose tip vs
                 eye midpoint, inter-ocular units) or |d roll| > --turn-roll deg or eye-midpoint translation > --turn-shift inter-ocular units
  S4 closed      LS frames where the generated (output) mouth is closed / almost closed (inner-lip gap 60-62 < --closed-thr
                 inter-ocular units); `closed_master` = same on the master (the source speaker's closed mouth)
  (orig_core = pass-through frames > win from a switch: the control, output == master there)

Per frame, in a HEAD-STABILISED face frame (one similarity transform per frame from the MASTER's eye centroids + nose points, temporally
smoothed, applied identically to master and output so head motion cancels and both are measured on the same pixels):
  mouth_diff     mean |f_t - f_t-1| (grey, mouth ROI)                     plain frame-to-frame change
  mouth_flicker  mean |f_t+1 - 2 f_t + f_t-1| (mouth ROI)                 temporal 2nd difference (face_temporal's flicker)
  mouth_warp     warping error: f_t vs f_t-1 warped by DIS optical flow (fwd/bwd-consistent pixels only), mouth ROI (Lai et al. 2018,
                 "Learning Blind Video Temporal Consistency"): change that smooth motion does NOT explain = texture popping / twitch
  upper_*        the same on the upper face (eyes / brows): must stay = master with the mouth-only paste-back
  lip_jitter     2nd difference of the output's own mouth landmarks (52-71) in the stabilised frame (inter-ocular units)
  mouth_sharp    Laplacian variance of the mouth ROI
All are reported as output / master ratios per situation (1.0 = as calm as the source at the same frames).

Window test (generator-internal): LatentSync denoises every LS segment in independent 16-frame windows starting at the segment start.
For transitions INTO a window start (frame s0 + 16k, k >= 1, crossfade frames excluded) vs all other LS transitions:
mean mouth_diff / mouth_warp, output and master; boundary/interior ratio of the output divided by that of the master (control) = the
step that the window cut adds. Also the 16-bin profile of mouth_warp over (t - s0) mod 16.

Landmarks: LatentSync's own FaceDetector (insightface buffalo_l: det_10g + 2d106det). NOTE: InsightFace model packs are licensed for
non-commercial research only; used here as an INTERNAL measurement (same detector the shipped LatentSync crop already uses, see
reports/experiments/lipsync-temporal/LICENSES.md), never shipped by this tool.

    python scripts/quality/lipsync_jitter.py --manifest RUN/final/03_final.manifest.json --out DIR --tag baseline [--video X.mp4] [--t0 40 --t1 60]
    (a window manifest of ls_window_ab.py is measured on its own window against the full master: no --t0/--t1 needed)
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parents[1] / "third_party" / "latentsync"))
FPS = 25; NWIN = 16
EYE_L, EYE_R, NOSE = list(range(33, 43)), list(range(87, 97)), list(range(72, 87))
MOUTH = list(range(52, 72)); INNER_UP, INNER_LO = 60, 62
CAN = 192                                   # canonical face crop (px)


def read_range(path, f0, f1):
    import cv2
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); out = []
    for _ in range(f1 - f0):
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def detect(frames):
    import cv2
    import face_aware_latentsync as fa
    det = fa.face_detector(); L = np.full((len(frames), 106, 2), np.nan, np.float32)
    for i, f in enumerate(frames):
        bb, lm = det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if bb is not None: L[i] = np.asarray(lm, np.float32)
    return L


def interp_nan(a):
    a = a.copy(); idx = np.arange(len(a)); flat = a.reshape(len(a), -1)
    for c in range(flat.shape[1]):
        m = np.isfinite(flat[:, c])
        if 2 <= m.sum() < len(m): flat[~m, c] = np.interp(idx[~m], idx[m], flat[m, c])
    return flat.reshape(a.shape)


def gsmooth(x, sigma):
    if sigma <= 0: return x
    r = int(3 * sigma + 0.5); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, xp)


def stable_points(L):
    return np.concatenate([L[:, EYE_L].mean(1, keepdims=True), L[:, EYE_R].mean(1, keepdims=True), L[:, NOSE]], axis=1)   # (T, 17, 2)


def similarity(src, dst):
    """least-squares similarity (4 DOF) mapping src (N,2) -> dst (N,2): returns 2x3."""
    ms, md = src.mean(0), dst.mean(0); s, d = src - ms, dst - md
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / (s ** 2).sum(); b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / (s ** 2).sum()
    R = np.array([[a, -b], [b, a]]); t = md - R @ ms
    return np.hstack([R, t[:, None]])


def runs(mask, min_len=3, merge=2):
    """contiguous True runs as [start, end) frame pairs, gaps <= merge merged, runs < min_len dropped."""
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            if out and i - out[-1][1] <= merge: out[-1][1] = j
            else: out.append([i, j])
            i = j
        else: i += 1
    return [r for r in out if r[1] - r[0] >= min_len]


def main() -> int:
    import cv2
    cv2.setNumThreads(8)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--video", default=None); ap.add_argument("--master", default=None)
    ap.add_argument("--dub16k", default=None, help="dubbed voice 16 kHz (default: manifest dubbed_track.wav16k) for the S1 speech mask")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--win", type=int, default=5); ap.add_argument("--stab-sigma", type=float, default=1.5)
    ap.add_argument("--turn-yaw", type=float, default=0.005); ap.add_argument("--turn-pitch", type=float, default=0.007)
    ap.add_argument("--turn-roll", type=float, default=0.4, help="deg per frame"); ap.add_argument("--turn-shift", type=float, default=0.015, help="inter-ocular units per frame")
    ap.add_argument("--closed-thr", type=float, default=0.05, help="closed mouth: inner-lip gap (lmk 60-62) / inter-ocular distance (03 master, silent frames: p25 0.043, median 0.054)")
    ap.add_argument("--master-cache", default=None, help="npz with master landmarks (reused across candidates; default <out>/<id>_master_lm.npz)")
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); vid = pathlib.Path(a.video or man["output"])
    master_p = pathlib.Path(a.master or (pathlib.Path(man["work_dir"]) / "master_25fps.mp4"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); vid_id = pathlib.Path(man["input"]).stem; stem = f"{vid_id}_{a.tag}"
    n_all = int(man["lipsync"]["master_frames"])
    win = man.get("window") or {}; voff = int(win.get("f0", 0))                 # ls_window_ab.py clips start at master frame f0
    f0 = int(round(a.t0 * FPS)) if (a.t0 or not win) else int(win["f0"])
    f1 = min(n_all, int(round(a.t1 * FPS)) if a.t1 else int(win.get("f1", n_all)))
    M = read_range(master_p, f0, f1); O = read_range(vid, f0 - voff, f1 - voff); n = min(len(M), len(O)); M, O = M[:n], O[:n]; f1 = f0 + n
    # ---- landmarks (master cached: identical for every candidate of this video)
    cache = pathlib.Path(a.master_cache or (out / f"{vid_id}_master_lm.npz"))
    LMall = None
    if cache.exists():
        z = np.load(cache); LMall = z["lm"] if str(z["master"]) == str(master_p) and len(z["lm"]) == n_all else None
    if LMall is None:
        LMall = detect(read_range(master_p, 0, n_all)); np.savez_compressed(cache, lm=LMall, master=str(master_p))
    LM = LMall[f0:f1]
    oc = out / f"{stem}_out_lm.npz"; st = vid.stat(); key = f"{vid}|{st.st_size}|{int(st.st_mtime)}|{f0}|{f1}"
    LO = None
    if oc.exists():
        z = np.load(oc); LO = z["lm"] if str(z["key"]) == key else None
    if LO is None:
        LO = detect(O); np.savez_compressed(oc, lm=LO, key=key)
    det_ok_m = np.isfinite(LM[:, 0, 0]); det_ok_o = np.isfinite(LO[:, 0, 0])
    LMi = interp_nan(LM); LOi = interp_nan(LO)
    # ---- stabilisation: similarity from the master's stable points to a per-video template (median shape), smoothed over time
    P = stable_points(LMi); iod = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    norm = (P - P.mean(1, keepdims=True)) / iod[:, None, None]; tmpl = np.nanmedian(norm, axis=0)
    tmpl = tmpl * (0.30 * CAN) + np.array([CAN / 2, 0.36 * CAN])            # eyes ~0.3 CAN apart, eye/nose centroid at 36 % height
    A = np.stack([similarity(P[i], tmpl) for i in range(n)]); A = gsmooth(A.reshape(n, 6), a.stab_sigma).reshape(n, 2, 3)
    def warp_pts(L, Ai): return L @ Ai[:, :2].T + Ai[:, 2]
    LMc = np.stack([warp_pts(LMi[i], A[i]) for i in range(n)]); LOc = np.stack([warp_pts(LOi[i], A[i]) for i in range(n)])
    # ROIs in the canonical frame (fixed): mouth = box around the median master mouth points; upper = eyes + brows
    mm = np.nanmedian(LMc[:, MOUTH], axis=0); mx0, my0 = mm.min(0); mx1, my1 = mm.max(0); w = mx1 - mx0
    mouth_roi = (int(max(0, mx0 - 0.25 * w)), int(max(0, my0 - 0.35 * w)), int(min(CAN, mx1 + 0.25 * w)), int(min(CAN, my1 + 0.45 * w)))
    up = np.nanmedian(LMc[:, EYE_L + EYE_R + list(range(43, 52)) + list(range(97, 106))], axis=0)
    upper_roi = (int(max(0, up[:, 0].min() - 8)), int(max(0, up[:, 1].min() - 8)), int(min(CAN, up[:, 0].max() + 8)), int(min(CAN, up[:, 1].max() + 8)))
    GM = np.stack([cv2.cvtColor(cv2.warpAffine(M[i], A[i], (CAN, CAN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), cv2.COLOR_BGR2GRAY) for i in range(n)]).astype(np.float32)
    GO = np.stack([cv2.cvtColor(cv2.warpAffine(O[i], A[i], (CAN, CAN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), cv2.COLOR_BGR2GRAY) for i in range(n)]).astype(np.float32)
    del M, O
    def roi(G, r): x0, y0, x1, y1 = r; return G[:, y0:y1, x0:x1]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    def warp_err(G, r):
        """per-transition warping error in ROI r: |f_t - warp(f_t-1 -> t)| over fwd/bwd-consistent pixels (0 at t = 0)."""
        x0, y0, x1, y1 = r; e = np.zeros(len(G)); H, W = G.shape[1:]
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        for t in range(1, len(G)):
            a8, b8 = G[t - 1].astype(np.uint8), G[t].astype(np.uint8)
            fb = dis.calc(b8, a8, None); ff = dis.calc(a8, b8, None)          # fb: t -> t-1 ; ff: t-1 -> t
            mx, my = gx + fb[..., 0], gy + fb[..., 1]
            warped = cv2.remap(G[t - 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            back = cv2.remap(ff, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cons = np.linalg.norm(fb + back, axis=2) < 0.5 + 0.05 * (np.linalg.norm(fb, axis=2) + np.linalg.norm(back, axis=2))
            d = np.abs(G[t] - warped)[y0:y1, x0:x1]; c = cons[y0:y1, x0:x1]
            e[t] = float(d[c].mean()) if c.any() else float(d.mean())
        return e
    def diff1(X): d = np.zeros(len(X)); d[1:] = np.abs(X[1:] - X[:-1]).mean(axis=(1, 2)); return d
    def diff2(X): d = np.zeros(len(X)); d[1:-1] = np.abs(X[2:] - 2 * X[1:-1] + X[:-2]).mean(axis=(1, 2)); return d
    def sharp(X): return np.array([cv2.Laplacian(x, cv2.CV_32F).var() for x in X])
    sig = {}
    for key, r in (("mouth", mouth_roi), ("upper", upper_roi)):
        Rm, Ro = roi(GM, r), roi(GO, r)
        sig[f"{key}_diff"] = (diff1(Ro), diff1(Rm)); sig[f"{key}_flicker"] = (diff2(Ro), diff2(Rm))
        sig[f"{key}_warp"] = (warp_err(GO, r), warp_err(GM, r))
    sig["mouth_sharp"] = (sharp(roi(GO, mouth_roi)), sharp(roi(GM, mouth_roi)))
    s_io = np.linalg.norm(LMc[:, EYE_R].mean(1) - LMc[:, EYE_L].mean(1), axis=1)   # ~constant after stabilisation
    def acc2(X):
        d = np.zeros(len(X)); d[1:-1] = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1).mean(-1); return d
    sig["lip_jitter"] = (acc2(LOc[:, MOUTH]) / s_io, acc2(LMc[:, MOUTH]) / s_io)
    open_o = np.linalg.norm(LOc[:, INNER_UP] - LOc[:, INNER_LO], axis=1) / s_io; open_m = np.linalg.norm(LMc[:, INNER_UP] - LMc[:, INNER_LO], axis=1) / s_io
    sig["open_jitter"] = (np.abs(np.convolve(open_o, [1, -2, 1], "same")), np.abs(np.convolve(open_m, [1, -2, 1], "same")))
    # ---- situations
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    ls_all = np.zeros(n_all, bool); seg_len = np.zeros(n_all); seg_start = np.full(n_all, -1); xf_mask = np.zeros(n_all, bool)
    for s in segs:
        if s["action"] == "LATENT_SYNC":
            ls_all[s["start_frame"]:s["end_frame"]] = True; seg_len[s["start_frame"]:s["end_frame"]] = s["frames"]; seg_start[s["start_frame"]:s["end_frame"]] = s["start_frame"]
            xf = s.get("crossfade") or {}
            xf_mask[s["start_frame"]:s["start_frame"] + (xf.get("in") or 0)] = True; xf_mask[s["end_frame"] - (xf.get("out") or 0):s["end_frame"]] = True
    sw = [b["start_frame"] for a_, b in zip(segs, segs[1:]) if (a_["action"] == "LATENT_SYNC") != (b["action"] == "LATENT_SYNC")]
    near_all = np.zeros(n_all, bool)
    for f in sw: near_all[max(0, f - a.win):min(n_all, f + a.win)] = True
    dub = a.dub16k or man["dubbed_track"]["wav16k"]
    import run_clean_pipeline_05 as rp
    dub_iv = [iv for iv in rp.run_vad(pathlib.Path(dub)) if iv["end"] - iv["start"] >= 1.0]
    speech_all = np.zeros(n_all, bool)
    for iv in dub_iv: speech_all[int(np.ceil(iv["start"] * FPS)):int(iv["end"] * FPS)] = True
    sl = slice(f0, f1); ls, near, speech, slen = ls_all[sl], near_all[sl], speech_all[sl], seg_len[sl]
    eye_mid = 0.5 * (LMi[:, EYE_L].mean(1) + LMi[:, EYE_R].mean(1)); iod_raw = np.linalg.norm(LMi[:, EYE_R].mean(1) - LMi[:, EYE_L].mean(1), axis=1)
    nose_tip = LMi[:, 86]                                                       # lowest nose-bridge point (tip area)
    yaw = (nose_tip[:, 0] - eye_mid[:, 0]) / iod_raw; pitch = (nose_tip[:, 1] - eye_mid[:, 1]) / iod_raw
    roll = np.degrees(np.unwrap(np.arctan2(LMi[:, EYE_R].mean(1)[:, 1] - LMi[:, EYE_L].mean(1)[:, 1], LMi[:, EYE_R].mean(1)[:, 0] - LMi[:, EYE_L].mean(1)[:, 0])))
    dyaw = np.abs(np.gradient(gsmooth(yaw, 2.0))); dpitch = np.abs(np.gradient(gsmooth(pitch, 2.0))); droll = np.abs(np.gradient(gsmooth(roll, 2.0)))
    shift = np.linalg.norm(np.gradient(gsmooth(eye_mid, 2.0), axis=0), axis=1) / gsmooth(iod_raw, 12.0)
    turn = (dyaw > a.turn_yaw) | (dpitch > a.turn_pitch) | (droll > a.turn_roll) | (shift > a.turn_shift)
    closed_o = gsmooth(open_o, 1.0) < a.closed_thr; closed_m = gsmooth(open_m, 1.0) < a.closed_thr
    sit = {"S1_speech": ls & ~near & speech & (slen >= 2 * FPS), "S2_switch": near.copy(), "S3_head_turn": ls & ~near & turn,
           "S4_closed_out": ls & ~near & closed_o, "S4_closed_master": ls & ~near & closed_m, "ls_core": ls & ~near, "orig_core": ~ls & ~near}
    valid = det_ok_m.copy(); valid[0] = False                                   # transition metrics need t-1
    def agg(key, mask):
        so, sm = sig[key]; m = mask & valid & np.isfinite(so) & np.isfinite(sm)
        if m.sum() < 5: return None
        mo, mm_ = float(np.mean(so[m])), float(np.mean(sm[m]))
        return {"n": int(m.sum()), "out": round(mo, 4), "master": round(mm_, 4), "ratio": round(mo / max(mm_, 1e-9), 3),
                "p95_ratio": round(float(np.percentile(so[m], 95) / max(np.percentile(sm[m], 95), 1e-9)), 3)}
    table = {s: {k: agg(k, m) for k in sig} for s, m in sit.items()}
    # ---- window-boundary test (inside LS, crossfade frames and switch windows excluded)
    gidx = np.arange(f0, f1); phase = np.where(seg_start[sl] >= 0, (gidx - seg_start[sl]) % NWIN, -1)
    inner = ls & ~near & ~xf_mask[sl] & valid & (gidx - seg_start[sl] >= 1)
    bnd = inner & (phase == 0); intr = inner & (phase != 0)
    wtest = {}
    for key in ("mouth_diff", "mouth_warp", "mouth_flicker", "lip_jitter", "upper_diff"):
        so, sm = sig[key]
        if bnd.sum() >= 3 and intr.sum() >= 10:
            ro = float(so[bnd].mean() / max(so[intr].mean(), 1e-9)); rm = float(sm[bnd].mean() / max(sm[intr].mean(), 1e-9))
            wtest[key] = {"n_boundary": int(bnd.sum()), "out_boundary_vs_interior": round(ro, 3), "master_boundary_vs_interior": round(rm, 3), "window_step_factor": round(ro / max(rm, 1e-9), 3)}
    prof = {k: [round(float(sig[k][0][inner & (phase == p)].mean()), 4) if (inner & (phase == p)).any() else None for p in range(NWIN)] for k in ("mouth_warp", "mouth_diff")}
    prof_m = {k: [round(float(sig[k][1][inner & (phase == p)].mean()), 4) if (inner & (phase == p)).any() else None for p in range(NWIN)] for k in ("mouth_warp", "mouth_diff")}
    # ---- situation frame ranges (seconds, absolute)
    ranges = {s: [[round((f0 + r[0]) / FPS, 2), round((f0 + r[1]) / FPS, 2)] for r in runs(m)] for s, m in sit.items() if s not in ("ls_core", "orig_core")}
    # ---- worst 1 s windows per situation by mouth_warp ratio (for side-by-side clips)
    def worst(mask, k=3):
        so, sm = sig["mouth_warp"]; base = max(float(np.median(sm[valid & ls])) if (valid & ls).any() else 1.0, 1e-6); cand = []
        for i0 in range(0, n - FPS, 5):
            mm_ = mask[i0:i0 + FPS] & valid[i0:i0 + FPS]
            if mm_.sum() >= 10: cand.append((float(so[i0:i0 + FPS][mm_].mean() / base), i0))
        cand.sort(reverse=True); pick = []
        for v, i0 in cand:
            if all(abs(i0 - p) > 2 * FPS for _, p in pick): pick.append((v, i0))
            if len(pick) >= k: break
        return [{"t": round((f0 + i0) / FPS, 2), "mouth_warp_vs_master_ls_median": round(v, 2)} for v, i0 in pick]
    worst_by = {s: worst(m) for s, m in sit.items() if s.startswith("S")}
    tl = [{"t": round((f0 + t) / FPS, 2), "ls": bool(ls[t]), "phase": int(phase[t]), "mouth_warp_out": round(float(sig["mouth_warp"][0][t]), 3), "mouth_warp_master": round(float(sig["mouth_warp"][1][t]), 3),
           "mouth_diff_out": round(float(sig["mouth_diff"][0][t]), 3), "mouth_diff_master": round(float(sig["mouth_diff"][1][t]), 3), "lip_jitter_out": round(float(sig["lip_jitter"][0][t]), 5),
           "lip_jitter_master": round(float(sig["lip_jitter"][1][t]), 5), "open_out": round(float(open_o[t]), 4), "open_master": round(float(open_m[t]), 4), "dyaw": round(float(dyaw[t]), 4),
           "dpitch": round(float(dpitch[t]), 4), "droll": round(float(droll[t]), 3), "shift": round(float(shift[t]), 4), "turn": bool(turn[t])} for t in range(n)]
    rep = {"video": str(vid), "master": str(master_p), "tag": a.tag, "frames": [f0, f1], "params": {k: v for k, v in vars(a).items() if k not in ("manifest", "out")},
           "landmarker": "insightface buffalo_l 2d106det via LatentSync FaceDetector (non-commercial weights; internal measurement only)",
           "det_rate": {"master": round(float(det_ok_m.mean()), 4), "output": round(float(det_ok_o.mean()), 4)}, "rois_canonical": {"mouth": mouth_roi, "upper": upper_roi, "size": CAN},
           "situation_frames": {s: int(m.sum()) for s, m in sit.items()},
           "open_percentiles": {"out_ls": np.round(np.percentile(open_o[ls], [5, 25, 50, 75, 95]), 4).tolist() if ls.any() else None,
                                "master_ls": np.round(np.percentile(open_m[ls], [5, 25, 50, 75, 95]), 4).tolist() if ls.any() else None,
                                "master_orig": np.round(np.percentile(open_m[~ls], [5, 25, 50, 75, 95]), 4).tolist() if (~ls).any() else None}, "table": table, "window_test": wtest, "window_profile_out": prof, "window_profile_master": prof_m,
           "situation_ranges_s": ranges, "worst_1s_by_situation": worst_by, "switch_frames": [f for f in sw if f0 <= f < f1], "timeline": tl}
    (out / f"{stem}.lipsync_jitter.json").write_text(json.dumps(rep, indent=1))
    keys = ("mouth_warp", "mouth_diff", "mouth_flicker", "lip_jitter", "open_jitter", "upper_warp", "upper_diff", "mouth_sharp")
    lines = [f"# lipsync_jitter {stem} frames {f0}-{f1} (output / master ratios; 1.0 = as calm as the source)", "",
             "| situation | frames | " + " | ".join(keys) + " |", "|---|---:|" + "---:|" * len(keys)]
    for s, row in table.items():
        lines.append(f"| {s} | {rep['situation_frames'][s]} | " + " | ".join("–" if row[k] is None else f"{row[k]['ratio']}" for k in keys) + " |")
    lines += ["", "window test (LatentSync 16-frame windows): " + ", ".join(f"{k} step x{v['window_step_factor']} (out {v['out_boundary_vs_interior']}, master {v['master_boundary_vs_interior']}, n={v['n_boundary']})" for k, v in wtest.items())]
    (out / f"{stem}.lipsync_jitter.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
