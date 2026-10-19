#!/usr/bin/env python3
"""Clean temporal stabilisation / composite of the lip-synced lower face (track B), applied to the stage frames of a decomposition window
(`decompose_video_stage.py --face-stack mediapipe`): master.mp4 + S0_ls_raw.mp4 (+ window_manifest.json -> the run manifest / work dir).
CLEAN-ONLY: every landmark comes from MediaPipe Face Landmarker (scripts/pilot/clean_face_stack.py, Apache-2.0); guard_nc() blocks the
non-commercial loaders in this process. No restorer.

  composite                   exactly clean0's mouth-only composite: feathered lower-face ellipse from the replayed clean detections
                              (MediaPipe -> LatentSync bbox formula, landmark smoothing +-2, mask smoothing +-3) + the 4-frame LS <-> original
                              crossfade. `--boundary off --skin-texture off --silence none` reproduces clean0 (control).
  C2 boundary interpolation   LatentSync 1.6 denoises independent 16-frame windows: the two frames around every window boundary are replaced by
                              flow-interpolated frames at t = 1/3, 2/3 (DIS flow both ways, occlusion-aware), only inside the paste mask.
  skin texture                outside a lip mask (MediaPipe outer lip contour of the LatentSync frame and of the source, dilated) the high
  (--skin-texture source)     frequencies of the lower face come from the SOURCE frame, flow-aligned to the LatentSync geometry: chin / stubble /
                              skin keep the source's sharp, steady texture. `--lip-texture on`: the exclusion shrinks to the inner mouth
                              (MediaPipe inner lip contour), so the lip surface also takes the source texture where the alignment is confident.
  dub-silent stretches        frames of an LS piece where the source speaks (manifest VAD) and the dub is silent (no placed TTS part) for at
  (--silence ...)             least --silence-min-s: the LatentSync mouth is shut and still while the source face around it articulates.
                              still = OPT1: the paste mask grows to the whole lower face (MediaPipe face oval below the nose) there, eased in /
                                      out over --silence-ramp frames, so jaw / cheeks are as still as the mouth (texture still from the source).
                              orig  = OPT2: the original frames are passed through there with a cosine crossfade of --silence-ramp frames.

    python scripts/quality/temporal_mouth_stabilize.py --dir DECOMP_DIR --tag C2T --boundary on --skin-texture source [--silence still]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
P = 16; FPS = 25
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_BASE = [2, 98, 327]   # subnasale + alar bases


def read_video(p) -> list:
    import cv2
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


class _Shim:   # MouthOnlyRunner needs an inner runner only for attributes
    config, calls, steps, guidance, seed, load_s = {}, [], None, None, None, None


class Flow:
    def __init__(self):
        import cv2
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); self.grid = None

    def g(self, h, w):
        if self.grid is None or self.grid[0].shape != (h, w):
            self.grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return self.grid

    def calc(self, a_gray, b_gray):
        return self.dis.calc(a_gray, b_gray, None)

    def warp(self, img, flow):
        """backward warp: out(x) = img(x + flow(x))"""
        import cv2
        gx, gy = self.g(*flow.shape[:2])
        return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def consistent(self, f_ab, f_ba, tol=0.75):
        back = self.warp(f_ba, f_ab)
        return np.hypot(f_ab[..., 0] + back[..., 0], f_ab[..., 1] + back[..., 1]) < tol + 0.05 * np.hypot(f_ab[..., 0], f_ab[..., 1])


def gray(x):
    import cv2
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def interp(fl: Flow, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
    """Frame at time t between y (t=0) and z (t=1): Super-SloMo approximation of the intermediate flows, visibility from the
    forward-backward check, plain cross-dissolve where neither side is consistent."""
    gy_, gz_ = gray(y), gray(z); F01, F10 = fl.calc(gy_, gz_), fl.calc(gz_, gy_)
    Ft0 = -(1 - t) * t * F01 + t * t * F10; Ft1 = (1 - t) ** 2 * F01 - t * (1 - t) * F10
    w0 = fl.warp(y, Ft0); w1 = fl.warp(z, Ft1)
    v0 = fl.warp(fl.consistent(F01, F10).astype(np.float32), Ft0); v1 = fl.warp(fl.consistent(F10, F01).astype(np.float32), Ft1)
    a0, a1 = (1 - t) * v0, t * v1; s = a0 + a1
    return np.where((s > 1e-3)[..., None], (a0[..., None] * w0 + a1[..., None] * w1) / np.maximum(s, 1e-3)[..., None], (1 - t) * y + t * z)


def boundary_interp(fl: Flow, frames: list, starts: list) -> tuple[list, int]:
    out = [f.copy() for f in frames]; n = 0
    for b in starts:
        if b - 1 < 0 or b >= len(frames): continue
        y, z = frames[b - 1], frames[b]
        out[b - 1] = interp(fl, y, z, 1 / 3); out[b] = interp(fl, y, z, 2 / 3); n += 1
    return out, n


def poly_mask(pts, roi, dilate_px: float, blur_px: float) -> np.ndarray:
    import cv2
    ya, yb, xa, xb = roi; m = np.zeros((yb - ya, xb - xa), np.uint8)
    if pts is None or not np.isfinite(pts).all(): return m.astype(np.float32)
    cv2.fillPoly(m, [np.round(np.asarray(pts)[:, :2] - [xa, ya]).astype(np.int32)], 255)
    if dilate_px >= 1:
        k = int(2 * round(dilate_px) + 1); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = m.astype(np.float32) / 255.0
    if blur_px > 0:
        k = int(2 * round(blur_px) + 1); m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def mouth_width(lm) -> float:
    return float(np.linalg.norm(lm[291, :2] - lm[61, :2])) if lm is not None and np.isfinite(lm[[61, 291]]).all() else 40.0


def skin_texture_transfer(fl: Flow, base: list, master: list, excl: list, sigma: float, tau: float) -> list:
    """Y = base + (1 - excl) * conf * (HP(warp(master)) - HP(base)), HP(x) = x - Gauss_sigma(x); master flow-aligned to the base."""
    import cv2
    out = []
    for k in range(len(base)):
        b, m = base[k], master[k]
        F = fl.calc(gray(b), gray(m)); mw = fl.warp(m, F)
        lb, lm = cv2.GaussianBlur(b, (0, 0), sigma), cv2.GaussianBlur(mw, (0, 0), sigma)
        conf = np.exp(-np.abs(cv2.GaussianBlur(lm - lb, (0, 0), 2.0)).mean(-1) / tau)
        wgt = (1.0 - excl[k]) * conf
        out.append(b + wgt[..., None] * ((mw - lm) - (b - lb)))
    return out


def load_gaps(path) -> list:
    """Residual-gap list (track C / the coordinator): a JSON list of [start_s, end_s], or {"gaps": [...]} whose items are [s, e] or
    {"start": s, "end": e} (also "start_s"/"end_s"). Seconds on the source timeline."""
    d = json.loads(pathlib.Path(path).read_text()); items = d.get("gaps", d.get("residual_gaps", d)) if isinstance(d, dict) else d; out = []
    for it in items:
        if isinstance(it, dict): out.append((float(it.get("start", it.get("start_s"))), float(it.get("end", it.get("end_s")))))
        else: out.append((float(it[0]), float(it[1])))
    return sorted(out)


def silent_stretches(man: dict, n_frames: int, A: int, min_s: float, merge_frames: int = 4, gaps: list | None = None) -> np.ndarray:
    """Per window frame: True where the SOURCE speaks (manifest speech_intervals) and NO placed dub part sounds, for >= min_s.
    With `gaps` (load_gaps) the given stretches are used instead (still subject to min_s)."""
    t = (np.arange(n_frames) + A) / FPS
    if gaps is not None:
        m = np.zeros(n_frames, bool)
        for g0, g1 in gaps:
            if g1 - g0 >= min_s: m |= (t >= g0) & (t < g1)
        return m
    src = np.zeros(n_frames, bool); dub = np.zeros(n_frames, bool)
    for iv in man.get("speech_intervals", []): src |= (t >= iv["start"]) & (t < iv["end"])
    for u in man.get("units", []):
        for p in u.get("alignment", {}).get("placements", []):
            if p.get("placed"): dub |= (t >= p["placed"][0]) & (t < p["placed"][1])
    m = src & ~dub; runs = []; i = 0
    while i < n_frames:
        if m[i]:
            j = i
            while j < n_frames and m[j]: j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_frames: merged[-1][1] = r[1]
        else: merged.append(r)
    out = np.zeros(n_frames, bool)
    for a_, b_ in merged:
        if (b_ - a_) / FPS >= min_s: out[a_:b_] = True
    return out


def ramp(mask: np.ndarray, k: int) -> np.ndarray:
    """0/1 per frame -> eased weight: cosine ramps of k frames inside each run (full weight in the middle)."""
    w = np.zeros(len(mask), np.float32); i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        L = j - i; kk = min(k, L // 2)
        for q in range(L):
            d = min(q + 1, L - q); w[i + q] = 1.0 if kk == 0 or d > kk else 0.5 - 0.5 * math.cos(math.pi * d / (kk + 1))
        i = j
    return w


def crossfade(rep, master, k_in, k_out):
    import cv2
    n = len(rep); res = []
    for k in range(n):
        a = 1.0
        if k_in and k < k_in: a = (k + 1) / (k_in + 1)
        if k_out and k >= n - k_out: a = min(a, (n - k) / (k_out + 1))
        res.append(cv2.addWeighted(rep[k], a, master[k], 1.0 - a, 0.0) if a < 1.0 else rep[k])
    return res


def main() -> int:
    import cv2
    import clean_face_stack as cfs, face_aware_latentsync_accel as faa
    from candidate_video_stage import MouthOnlyRunner
    from decompose_video_stage import write_video
    cfs.guard_nc()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--base", default="S0_ls_raw")
    ap.add_argument("--boundary", choices=["on", "off"], default="on")
    ap.add_argument("--skin-texture", choices=["off", "source"], default="off"); ap.add_argument("--lip-texture", choices=["off", "on"], default="off")
    ap.add_argument("--tex-sigma", type=float, default=1.5); ap.add_argument("--tex-tau", type=float, default=12.0); ap.add_argument("--lip-dilate", type=float, default=0.12)
    ap.add_argument("--silence", choices=["none", "still", "orig"], default="none"); ap.add_argument("--silence-min-s", type=float, default=0.3)
    ap.add_argument("--silence-ramp", type=int, default=6); ap.add_argument("--gaps", default=None, help="residual-gap JSON (load_gaps); default: computed from the manifest")
    ap.add_argument("--lm-smooth", type=int, default=2); ap.add_argument("--mask-smooth", type=int, default=3); ap.add_argument("--margin", type=int, default=16)
    a = ap.parse_args(); d = pathlib.Path(a.dir); wm = json.loads((d / "window_manifest.json").read_text()); A = wm["window"][0]
    man = json.loads(pathlib.Path(wm["source_manifest"]).read_text()); work = pathlib.Path(man["work_dir"])
    t0 = time.perf_counter()
    master = read_video(d / "master.mp4"); base_all = read_video(d / f"{a.base}.mp4"); H, W = master[0].shape[:2]; n_all = len(master)
    mp_full = np.load(cfs.run_mediapipe(work / "master_25fps.mp4", work / "master_mediapipe.npz"))["lm"]; mp_src = mp_full[A:A + n_all]
    need_ls_lm = a.skin_texture == "source"
    mp_ls = np.load(cfs.run_mediapipe(d / f"{a.base}.mp4", d / f"{a.base}.mediapipe.npz"))["lm"] if need_ls_lm else None
    dets_all = cfs.ls_detections(mp_full, W, H, calib=True)
    silent = silent_stretches(man, n_all, A, a.silence_min_s, gaps=load_gaps(a.gaps) if a.gaps else None) if a.silence != "none" else np.zeros(n_all, bool)
    fl = Flow(); out = [f.copy() for f in master]; info = []
    runner = MouthOnlyRunner(_Shim(), 0.52, 0.08, a.mask_smooth, defer=True)
    for pc in wm["pieces"]:
        s, e = pc["start"] - A, pc["end"] - A; n = e - s
        dets = list(dets_all[pc["start"]:pc["end"]])
        if a.lm_smooth > 0: dets = faa.smooth_detections(dets, a.lm_smooth)
        boxes = runner.boxes(dets, n)
        rois = [runner.mask(b, H, W)[1] for b in boxes if b is not None]
        grow_w = ramp(silent[s:e], a.silence_ramp) if a.silence == "still" else np.zeros(n, np.float32)
        if a.silence == "still":   # the grown region must fit in the ROI
            for j in range(n):
                lm = mp_src[s + j]
                if grow_w[j] > 0 and np.isfinite(lm).all():
                    x0, y0 = lm[FACE_OVAL, :2].min(0); x1, y1 = lm[FACE_OVAL, :2].max(0); rois.append((int(y0), int(y1) + 1, int(x0), int(x1) + 1))
        ya, yb = max(0, min(r[0] for r in rois) - a.margin), min(H, max(r[1] for r in rois) + a.margin)
        xa, xb = max(0, min(r[2] for r in rois) - a.margin), min(W, max(r[3] for r in rois) + a.margin); roi = (ya, yb, xa, xb)
        base = [base_all[i][ya:yb, xa:xb].astype(np.float32) for i in range(s, e)]
        starts = list(range(P, n, P)); nb = 0
        if a.boundary == "on": base, nb = boundary_interp(fl, base, starts)
        y = base
        if a.skin_texture == "source":
            excl = []
            for i in range(s, e):
                mw_ = mouth_width(mp_src[i]); idx = LIPS_INNER if a.lip_texture == "on" else LIPS_OUTER
                m1 = poly_mask(mp_ls[i][idx] if mp_ls is not None else None, roi, a.lip_dilate * mw_, 0.08 * mw_)
                m2 = poly_mask(mp_src[i][idx], roi, a.lip_dilate * mw_, 0.08 * mw_) if a.lip_texture == "off" else np.zeros_like(m1)
                excl.append(np.maximum(m1, m2))
            mcrop = [master[i][ya:yb, xa:xb].astype(np.float32) for i in range(s, e)]
            y = skin_texture_transfer(fl, y, mcrop, excl, a.tex_sigma, a.tex_tau)
        full = []
        for j, i in enumerate(range(s, e)):
            f = base_all[i].copy(); f[ya:yb, xa:xb] = np.clip(y[j], 0, 255).round().astype(np.uint8); full.append(f)
        mst = [master[i] for i in range(s, e)]
        comp = runner.composite(full, mst, boxes)
        if a.silence == "still":   # OPT1: grown lower-face mask (max with the ellipse) where the dub is silent and the source speaks
            for j in range(n):
                if grow_w[j] <= 0: continue
                lm = mp_src[s + j]
                if not np.isfinite(lm).all(): continue
                oval = lm[FACE_OVAL, :2]; ny = float(lm[NOSE_BASE, 1].mean()); low = oval[oval[:, 1] >= ny - 1]
                hull = cv2.convexHull(np.concatenate([low, [[oval[:, 0].min(), ny], [oval[:, 0].max(), ny]]]).astype(np.float32))[:, 0]
                g = poly_mask(hull, roi, 0, 0.15 * mouth_width(lm)) * grow_w[j]
                c = comp[j][ya:yb, xa:xb].astype(np.float32); f = full[j][ya:yb, xa:xb].astype(np.float32)
                c = np.maximum(0, c); comp[j] = comp[j].copy()
                comp[j][ya:yb, xa:xb] = np.clip(c + g[..., None] * (f - c), 0, 255).round().astype(np.uint8)
        fin = crossfade(comp, mst, pc["xf_in"], pc["xf_out"])
        if a.silence == "orig":      # OPT2: original frames in dub-silent stretches, cosine crossfade
            wv = ramp(silent[s:e], a.silence_ramp)
            fin = [cv2.addWeighted(mst[j], float(wv[j]), fin[j], 1.0 - float(wv[j]), 0.0) if wv[j] > 0 else fin[j] for j in range(n)]
        for j, i in enumerate(range(s, e)): out[i] = fin[j]
        info.append({"piece": [pc["start"], pc["end"]], "roi": [ya, yb, xa, xb], "boundaries_interpolated": nb, "silent_frames": int(silent[s:e].sum())})
        print(f"  piece {pc['start']}:{pc['end']} roi {yb - ya}x{xb - xa} boundaries {nb} silent {int(silent[s:e].sum())}", flush=True)
    write_video(d / f"{a.tag}.mp4", out)
    meta = {"tag": a.tag, "args": vars(a), "pieces": info, "silent_frames_total": int(silent.sum()), "seconds": round(time.perf_counter() - t0, 1),
            "face_stack": "mediapipe (clean)", "restorer": None}
    (d / f"{a.tag}.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ("tag", "silent_frames_total", "seconds")})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
