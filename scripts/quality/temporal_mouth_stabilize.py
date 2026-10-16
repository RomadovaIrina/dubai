#!/usr/bin/env python3
"""Temporal stabilisation of the lip-synced mouth (track B candidates C1 / C2), applied to the stage frames of a decomposition window
(`decompose_video_stage.py`): the LatentSync raw frames (base) and, optionally, a restorer's output on the same frames (detail source).

  C2 boundary interpolation   LatentSync 1.6 denoises independent 16-frame windows: the mouth jumps between the last frame of a window and
                              the first of the next. The two frames around every boundary are replaced by flow-interpolated frames at
                              t = 1/3 and 2/3 between them (DIS flow both ways, Super-SloMo-style intermediate flows, occlusion-aware), so
                              the jump is spread over three frame steps instead of one. Only inside the paste-back mask.
  C1 detail stabilisation     detail = restorer output - LatentSync raw (what the restorer adds: lip / teeth / stubble texture). It is
                              filtered by a motion-compensated recursive filter: D'(t) = a D(t) + (1 - a) warp(D'(t-1)) with the backward
                              DIS flow of the BASE (LatentSync raw, steady texture -> reliable flow); a = 1 where the flow is
                              inconsistent or the base changes after warping (new content: mouth opens, teeth appear), --alpha elsewhere.
                              Output = base + D'. The mouth shape / timing stays exactly LatentSync's; only the added detail is smoothed.

Then the same mouth-only composite (smoothed LatentSync detector boxes, ellipse top 0.52, feather 0.08) and the same 4-frame crossfade
as cand5. Writes <dir>/<tag>.mp4 (lossless) for mouth_jitter_hr.py / window_syncnet.py.

    python scripts/quality/temporal_mouth_stabilize.py --dir DECOMP_DIR --tag C1C2_cf --detail S1_ls_cf --boundary on --alpha 0.35
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
P = 16


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


def mask_boxes(master_rgb: np.ndarray, lm_smooth: int, mask_smooth: int):
    """Exactly the cand5 mask boxes: LatentSync FaceDetector on the master frames -> smooth_detections(lm_smooth) -> MouthOnlyRunner.boxes(mask_smooth)."""
    import face_aware_latentsync as fa, face_aware_latentsync_accel as faa
    from candidate_video_stage import MouthOnlyRunner
    dets = fa.detect_frames(master_rgb)
    if lm_smooth > 0: dets = faa.smooth_detections(dets, lm_smooth)
    r = MouthOnlyRunner(_Shim(), 0.52, 0.08, mask_smooth, defer=True)
    return r, r.boxes(dets, len(master_rgb))


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
        """on grid a: f_ab(x) + f_ba(x + f_ab(x)) ~ 0"""
        back = self.warp(f_ba, f_ab)
        return np.hypot(f_ab[..., 0] + back[..., 0], f_ab[..., 1] + back[..., 1]) < tol + 0.05 * np.hypot(f_ab[..., 0], f_ab[..., 1])


def gray(x):
    import cv2
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def interp(fl: Flow, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
    """Frame at time t between y (t=0) and z (t=1): Super-SloMo approximation of the intermediate flows, visibility from the
    forward-backward check, plain cross-dissolve where neither side is consistent."""
    import cv2
    gy_, gz_ = gray(y), gray(z); F01, F10 = fl.calc(gy_, gz_), fl.calc(gz_, gy_)
    Ft0 = -(1 - t) * t * F01 + t * t * F10; Ft1 = (1 - t) ** 2 * F01 - t * (1 - t) * F10
    w0 = fl.warp(y, Ft0); w1 = fl.warp(z, Ft1)
    v0 = fl.warp(fl.consistent(F01, F10).astype(np.float32), Ft0); v1 = fl.warp(fl.consistent(F10, F01).astype(np.float32), Ft1)
    a0, a1 = (1 - t) * v0, t * v1; s = a0 + a1
    out = np.where((s > 1e-3)[..., None], (a0[..., None] * w0 + a1[..., None] * w1) / np.maximum(s, 1e-3)[..., None], (1 - t) * y + t * z)
    return out


def boundary_interp(fl: Flow, frames: list, starts: list, strength: int = 2) -> tuple[list, int]:
    """frames: float32 crops of one LS piece; starts: in-piece indices of window starts (16, 32, ...). Replace b-1 / b by t = 1/3, 2/3."""
    out = [f.copy() for f in frames]; n = 0
    for b in starts:
        if b - 1 < 0 or b >= len(frames): continue
        y, z = frames[b - 1], frames[b]
        out[b - 1] = interp(fl, y, z, 1 / 3); out[b] = interp(fl, y, z, 2 / 3); n += 1
    return out, n


def detail_filter(fl: Flow, base: list, rest: list, alpha: float, change_thr: float, gain: float = 1.0) -> list:
    """C1: motion-compensated recursive filter of the detail layer (rest - base), flow from the base."""
    import cv2
    out = []; Dp = None
    for k in range(len(base)):
        D = rest[k] - base[k]
        if Dp is None: Dp = D; out.append(base[k] + gain * D); continue
        gb, gp = gray(base[k]), gray(base[k - 1]); Fb = fl.calc(gb, gp); Ff = fl.calc(gp, gb)   # Fb: grid k -> k-1
        ok = fl.consistent(Fb, Ff)
        err = np.abs(gb.astype(np.float32) - fl.warp(gp.astype(np.float32), Fb))
        a = np.where(ok & (err < change_thr), alpha, 1.0).astype(np.float32)
        a = cv2.GaussianBlur(a, (7, 7), 0)[..., None]
        Dp = a * D + (1 - a) * fl.warp(Dp, Fb); out.append(base[k] + gain * Dp)
    return out


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
    from decompose_video_stage import write_video
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="S0_ls_raw"); ap.add_argument("--detail", default=None, help="restored stage (e.g. S1_ls_cf); none = no restorer")
    ap.add_argument("--boundary", choices=["on", "off"], default="on"); ap.add_argument("--detail-filter", choices=["on", "off"], default="on")
    ap.add_argument("--alpha", type=float, default=0.35); ap.add_argument("--change-thr", type=float, default=10.0)
    ap.add_argument("--gain", type=float, default=1.0, help="gain on the filtered detail (averaging misaligned texture lowers its amplitude)")
    ap.add_argument("--lm-smooth", type=int, default=2); ap.add_argument("--mask-smooth", type=int, default=3); ap.add_argument("--margin", type=int, default=12)
    a = ap.parse_args(); d = pathlib.Path(a.dir); wm = json.loads((d / "window_manifest.json").read_text()); A = wm["window"][0]
    t0 = time.perf_counter()
    master = read_video(d / "master.mp4"); base_all = read_video(d / f"{a.base}.mp4"); rest_all = read_video(d / f"{a.detail}.mp4") if a.detail else None
    H, W = master[0].shape[:2]; fl = Flow(); out = [f.copy() for f in master]; info = []
    for pc in wm["pieces"]:
        s, e = pc["start"] - A, pc["end"] - A; n = e - s
        mrgb = np.stack([cv2.cvtColor(master[i], cv2.COLOR_BGR2RGB) for i in range(s, e)])
        runner, boxes = mask_boxes(mrgb, a.lm_smooth, a.mask_smooth)
        rois = [runner.mask(b, H, W)[1] for b in boxes if b is not None]
        ya, yb = max(0, min(r[0] for r in rois) - a.margin), min(H, max(r[1] for r in rois) + a.margin)
        xa, xb = max(0, min(r[2] for r in rois) - a.margin), min(W, max(r[3] for r in rois) + a.margin)
        base = [base_all[i][ya:yb, xa:xb].astype(np.float32) for i in range(s, e)]
        starts = list(range(P, n, P))   # window starts inside the piece (pieces start on the run's 16-frame grid)
        nb = 0
        if a.boundary == "on":
            base, nb = boundary_interp(fl, base, starts)
        if rest_all is not None:
            rest = [rest_all[i][ya:yb, xa:xb].astype(np.float32) for i in range(s, e)]
            if a.boundary == "on":   # the restorer's detail at the replaced frames: interpolate it the same way
                rest, _ = boundary_interp(fl, rest, starts)
            y = detail_filter(fl, base, rest, a.alpha, a.change_thr, a.gain) if a.detail_filter == "on" else rest
        else:
            y = base
        full = []
        for j, i in enumerate(range(s, e)):
            f = base_all[i].copy(); f[ya:yb, xa:xb] = np.clip(y[j], 0, 255).round().astype(np.uint8); full.append(f)
        comp = runner.composite(full, [master[i] for i in range(s, e)], boxes)
        fin = crossfade(comp, [master[i] for i in range(s, e)], pc["xf_in"], pc["xf_out"])
        for j, i in enumerate(range(s, e)): out[i] = fin[j]
        info.append({"piece": [pc["start"], pc["end"]], "roi": [ya, yb, xa, xb], "boundaries_interpolated": nb})
        print(f"  piece {pc['start']}:{pc['end']} roi {yb - ya}x{xb - xa} boundaries {nb}", flush=True)
    write_video(d / f"{a.tag}.mp4", out)
    meta = {"tag": a.tag, "args": vars(a), "pieces": info, "seconds": round(time.perf_counter() - t0, 1)}
    (d / f"{a.tag}.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
