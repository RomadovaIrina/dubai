#!/usr/bin/env python3
"""Source-frame-rate composite (track B, P0a): the output keeps the SOURCE frames, frame count, timestamps (CFR or VFR) and resolution;
only the lip-synced lower face comes from LatentSync, retimed from the 25 fps master to every source timestamp.

Why: LatentSync runs on a 25 fps master (ffmpeg fps=25: one source frame in six dropped for a 30 fps source) and the pipeline muxed that
master -> 25 fps output, uneven head motion (judder) and spec 9.4 (source fps) violated.

Per source frame k at time t_k (master position m = 25 t_k, between master frames m0 = floor(m) and m1 = m0 + 1, alpha = m - m0):
  LatentSync layer   L(t_k) = flow interpolation between the LatentSync raw frames L[m0], L[m1] at alpha (DIS both ways, Super-SloMo
                     intermediate flows, occlusion-aware; flows cached per master pair); --boundary on first replaces the two LatentSync
                     frames around every 16-frame window boundary by t = 1/3, 2/3 interpolations (C2).
  LS weight          w(t_k) = the clean0 per-master-frame weight (1 inside LATENT_SYNC segments, the 4-frame crossfade ramps at switches,
                     0 elsewhere) linearly interpolated; w = 0 -> the output frame IS the source frame (pixels untouched before encoding).
  mask               clean0's lower-face ellipse (LatentSync bbox formula on the clean MediaPipe landmarks, landmarks interpolated to t_k,
                     smoothing +-2 / +-3 master frames) with an explicit edge: alpha goes 0.1 -> 0.9 over --edge-px output pixels (10-15).
  sharpen            (--ls-sharpen A) unsharp mask of the LatentSync layer: L += A (L - Gauss_sigma(L)), deterministic, no model.
  skin texture       (--skin-texture source) outside a lip mask, the high frequencies come from the SOURCE frame k itself, flow-aligned
                     to the LatentSync geometry (no extra frame, no extra model).
  dub-silent         (--silence still|orig) as temporal_mouth_stabilize.py: OPT1 grows the mask to the lower face, OPT2 passes the source
                     through, both eased over --silence-ramp master frames.
  robustness         any per-frame exception -> the previous valid output frame (or the source frame); counted; the run reports
                     fallback_fraction and exits 3 when it exceeds 2 %.
Output: H.264 crf 13 yuv420p with the source time base / pts (PyAV), then the audio stream of the run output (--manifest "output", the
final mix) COPIED in (md5 identity with the baseline).

    python scripts/quality/fps_composite.py --manifest CLEAN0.manifest.json --ls-raw S0_ls_raw.mp4 --out out.mp4 [--boundary on]
        [--skin-texture source] [--silence none|still|orig] [--edge-px 12] [--threads 12]
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, math, pathlib, subprocess, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
import temporal_mouth_stabilize as tms   # Flow, interp, poly_mask, lip / oval indices, silent_stretches, ramp   # noqa: E402
FPS_M = 25.0; P = 16


def decode_source(path: str):
    import av
    c = av.open(path); s = c.streams.video[0]; frames, pts = [], []
    for fr in c.decode(s):
        frames.append(fr.to_ndarray(format="bgr24")); pts.append(fr.pts)
    tb = s.time_base; rate = s.average_rate; c.close()
    return frames, np.array(pts, np.int64), tb, rate


def master_weights(man: dict, n: int, xf_frames: int = 0, shape: str = "linear") -> np.ndarray:
    """clean0's per-master-frame LS weight: 1 in LATENT_SYNC segments, the assembly crossfade ramps at LS <-> original switches.
    r5 F: xf_frames > 0 overrides the ramp length (capped at a third of the segment), shape smooth = smoothstep."""
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"]); w = np.zeros(n, np.float32)
    ease = (lambda x: x * x * (3 - 2 * x)) if shape == "smooth" else (lambda x: x)
    for s in segs:
        if s["action"] != "LATENT_SYNC": continue
        xf = dict(s.get("crossfade") or {}); L = s["end_frame"] - s["start_frame"]
        if xf_frames > 0:
            for k_ in ("in", "out"):
                if xf.get(k_): xf[k_] = min(xf_frames, L // 3)
        for k in range(L):
            a = 1.0
            if xf.get("in") and k < xf["in"]: a = ease((k + 1) / (xf["in"] + 1))
            if xf.get("out") and k >= L - xf["out"]: a = min(a, ease((L - k) / (xf["out"] + 1)))
            if s["start_frame"] + k < n: w[s["start_frame"] + k] = a
    return w


def seg_joints(man: dict) -> list:
    """r5 F: master frames b where one LATENT_SYNC generation ends and the next begins with no source frame between (a hard switch)."""
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    return [b["start_frame"] for a_, b in zip(segs, segs[1:]) if a_["action"] == b["action"] == "LATENT_SYNC" and a_["end_frame"] == b["start_frame"]]


def ellipse_alpha(bbox, H, W, edge_px: float, top: float = 0.52):
    """clean0's lower-face ellipse (MouthOnlyRunner.mask geometry) with an explicit edge width: Gaussian sigma = edge_px / 2.563 gives a
    0.1 -> 0.9 transition of edge_px pixels across a straight edge. Returns (alpha crop, roi)."""
    import cv2
    x1, y1, x2, y2 = [float(v) for v in bbox]; w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cy = y1 + top * h; cx = (x1 + x2) / 2; axes = (w * 0.55, (y2 + 0.06 * h - cy) * 0.5 + 1); centre = (cx, (cy + y2 + 0.06 * h) / 2)
    sig = edge_px / 2.563; pad = int(math.ceil(3 * sig)) + 2
    xa, xb = int(max(0, centre[0] - axes[0] - pad)), int(min(W, centre[0] + axes[0] + pad + 1)); ya, yb = int(max(0, centre[1] - axes[1] - pad)), int(min(H, centre[1] + axes[1] + pad + 1))
    if xb <= xa or yb <= ya: return None, None
    m = np.zeros((yb - ya, xb - xa), np.float32)
    cv2.ellipse(m, (int(round(centre[0] - xa)), int(round(centre[1] - ya))), (int(round(axes[0])), int(round(axes[1]))), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), sig, borderType=cv2.BORDER_CONSTANT), (ya, yb, xa, xb)


def edge_width_px(alpha: np.ndarray) -> float | None:
    """Measured 10-90 % transition width of a soft mask: area(0.1 < a < 0.9) / perimeter of the a = 0.5 contour."""
    import cv2
    band = float(((alpha > 0.1) & (alpha < 0.9)).sum()); cs, _ = cv2.findContours((alpha >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return round(band / per, 2) if per > 0 else None


def interp_safe(fl, y, z, t, d0=3.5, c0=0.95, sc=2.0):
    """tms.interp + a confidence: on the pixels that change between the two LatentSync frames (|grey diff| > 10), D = mean |w0 - w1| (the two
    warped candidates disagree -> averaging them gives the double / torn lip contour the user saw) and the forward-backward flow consistency.
    conf = clip(1 - max((D - d0) / sc, (c0 - cons) / 0.1), 0, 1); conf >= 0.6: the interpolation; below: the NEARER frame alone, warped to t
    (single source = no double contour; a soft blend with the nearest frame was tried and itself ghosts).
    Calibrated on 04 r3cand (flash frames: D median 7.8, cons 0.82; others D median 2.3, p90 4.7, cons 0.98); d0 3.5 / sc 2 / c0 0.95:
    mean confidence 0.08 on the flash frames, 0.82 elsewhere (frame 670: D 4.4, cons 0.946 -> 0.55)."""
    gy_, gz_ = tms.gray(y), tms.gray(z); F01, F10 = fl.calc(gy_, gz_), fl.calc(gz_, gy_)
    Ft0 = -(1 - t) * t * F01 + t * t * F10; Ft1 = (1 - t) ** 2 * F01 - t * (1 - t) * F10
    w0 = fl.warp(y, Ft0); w1 = fl.warp(z, Ft1)
    c01 = fl.consistent(F01, F10); v0 = fl.warp(c01.astype(np.float32), Ft0); v1 = fl.warp(fl.consistent(F10, F01).astype(np.float32), Ft1)
    a0, a1 = (1 - t) * v0, t * v1; s_ = a0 + a1
    it = np.where((s_ > 1e-3)[..., None], (a0[..., None] * w0 + a1[..., None] * w1) / np.maximum(s_, 1e-3)[..., None], (1 - t) * y + t * z)
    mov = np.abs(gy_.astype(np.float32) - gz_.astype(np.float32)) > 10
    if mov.sum() < 20: return it, 1.0
    D = float(np.abs(w0 - w1).mean(2)[mov].mean()); cons = float(c01[mov].mean())
    conf = float(np.clip(1 - max((D - d0) / sc, (c0 - cons) / 0.1), 0, 1))
    if conf >= 0.6: return it, conf
    # low confidence: NO averaging of two disagreeing mouths (that is what draws the double contour); the nearer frame alone, warped to t
    return (w0 if t < 0.5 else w1), conf


def main() -> int:
    import cv2
    import clean_face_stack as cfs, face_aware_latentsync_accel as faa
    cfs.guard_nc()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="clean0 run manifest (segments, crossfades, work dir with master_mediapipe.npz, audio)")
    ap.add_argument("--ls-raw", required=True, help="LatentSync raw frames on the 25 fps master (decompose_video_stage.py S0_ls_raw.mp4, full video)")
    ap.add_argument("--out", required=True); ap.add_argument("--boundary", choices=["on", "off"], default="on")
    ap.add_argument("--skin-texture", choices=["off", "source"], default="off"); ap.add_argument("--lip-texture", choices=["off", "on"], default="off")
    ap.add_argument("--tex-sigma", type=float, default=1.5); ap.add_argument("--tex-tau", type=float, default=12.0); ap.add_argument("--lip-dilate", type=float, default=0.12)
    ap.add_argument("--silence", choices=["none", "still", "orig"], default="none"); ap.add_argument("--silence-min-s", type=float, default=0.3); ap.add_argument("--silence-ramp", type=int, default=6)
    ap.add_argument("--gaps", default=None, help="residual-gap JSON (temporal_mouth_stabilize.load_gaps; e.g. track C's list); default: computed from the manifest")
    ap.add_argument("--sharpen-lips-max", type=float, default=0.0, help="r3 D: cap of the EXTRA lip unsharp amount after --sharpen-auto (the closed loop "
                    "chased the source's open-mouth texture up to 1.6 on 04 and drew a hard pink edge); 0 = no cap")
    ap.add_argument("--sanitize-layer", type=float, default=0.0, help="r3 D (user: 'washed-out pink lips / torn lips'): raw LatentSync frames whose lip-box "
                    "texture collapses (min(lip Laplacian, lip contrast) / median of the frames 2..6 away < this, e.g. 0.83) are replaced, in runs of <= 4, by a "
                    "flow interpolation between the nearest good frames; 0 = off")
    ap.add_argument("--sanitize-mode", choices=["replace", "texture"], default="replace", help="r3 D: replace = the flagged frame becomes the interpolation of "
                    "its good neighbours; texture = keep the flagged frame's own low frequencies (mouth opening / shape = the lip-sync) and take only the "
                    "high-frequency detail (lip line, texture) from that interpolation (Gaussian split at --sanitize-sigma px)")
    ap.add_argument("--sanitize-sigma", type=float, default=2.0)
    ap.add_argument("--sanitize-guard", choices=["on", "off"], default="on", help="r6 F: a flagged frame is replaced only if the interpolated neighbours "
                    "are aligned with it on the skin ring around the lip box (grey NCC >= --sanitize-align); otherwise the original stays "
                    "(off = r3 D behaviour: replace every flagged frame)")
    ap.add_argument("--sanitize-align", type=float, default=0.8, help="r6 F: ring NCC threshold of --sanitize-guard (03 real flashes 0.95-0.98, "
                    "05 smear frames < 0.15, 05 p10 of all flagged 0.83)")
    ap.add_argument("--sanitize-min-conf", type=float, default=0.3, help="r7 F: a flagged frame is fixed only when the neighbours' flow interpolation has "
                    "at least this confidence (interp_safe); at 0 the fallback warp tears the lip line into 'cracks' (04 m8 frame 956); 0 = off")
    ap.add_argument("--sanitize-tear", type=float, default=0.0, help="r7 F: with the rigid fallback, pixels where the dense warp and the rigid warp of the "
                    "nearer frame differ by more than this (grey, blurred 2 px) take the rigid one; 0 (default) = rigid everywhere: a partial blend doubled the lip line on 04 948 / 954")
    ap.add_argument("--sanitize-mild", type=float, default=0.0, help="r7 F (01 m8 7.6 s: a pursed bilabial flagged as 'collapsed' got frame 188's flat lip "
                    "line, then frame 190 showed the bulge = a one-frame pop): a MILDLY flagged frame (collapse ratio >= this, i.e. its own lip line exists) "
                    "is fixed only when the donated high frequencies correlate with its own on the lip box (grey NCC >= --sanitize-hf-ncc); 0 = off")
    ap.add_argument("--sanitize-hf-ncc", type=float, default=0.3); ap.add_argument("--sanitize-replace-below", type=float, default=0.0, help="r7 F: texture mode, but a frame whose collapse ratio is below this is replaced whole (washed-pink blob); 0 = off")
    ap.add_argument("--sanitize-colour", choices=["on", "off"], default="off", help="r3 D, texture mode: inside the flagged frame's lip box also match the "
                    "LUMINANCE mean / contrast (gain 0.7-1.6) of its low frequencies to the neighbours' interpolation (pale lips); per-channel matching over-saturated")
    ap.add_argument("--chin-clip", type=float, default=0.0, help="r4 V (user: 'second / third chin moves like a wave'): with --face-clip, the clip polygon's "
                    "jaw edge moves up toward the lower lip by this fraction (0 = the jawline, 1 = the lip), so the regenerated chin returns to the source "
                    "pixels above the jawline; the edge_px feather stays inside the polygon")
    ap.add_argument("--mouth-region", type=float, default=0.0, help="r4 V task 5 (chin / jaw 'wave': the regenerated chin stubble / skin swims "
                    "frame to frame, 03 68 s): the paste weight is capped to the source lips + the layer's own lips (its jaw drop) dilated by this "
                    "many inter-ocular distances; the edge_px feather stays inside, everything farther (chin, jaw, neck) is the source")
    ap.add_argument("--mouth-region-closed", type=float, default=0.0, help="r7 F: --mouth-region used while the layer's mouth is closed (opening < --jaw-closed IOD) "
                    "and the source's is open (> --jaw-open): the generator's closed jaw is pasted with its lips (05 24 s pressed lips); 0 = off")
    ap.add_argument("--jaw-open", type=float, default=0.05); ap.add_argument("--jaw-closed", type=float, default=0.03); ap.add_argument("--mr-ramp", type=float, default=4.0)
    ap.add_argument("--chin-warp", choices=["on", "off"], default="off", help="r4 V task 5 (with --mouth-region): outside the generated mouth region the "
                    "SOURCE chin is shifted down by the layer's jaw drop (layer lower lip y - source lower lip y, clipped to -0.2..0.5 IOD), full from the lips "
                    "to the chin tip, fading to 0 over 0.25 IOD below the jawline and toward the jaw corners: the dub's jaw motion with the source's skin "
                    "/ folds (no regenerated chin texture to 'wave')")
    ap.add_argument("--face-clip", choices=["on", "off"], default="off", help="r3 D (user: 'the background beside the cheek wobbles'): the paste weight "
                    "is capped by the distance INSIDE the source frame's MediaPipe face contour (0 on / outside it, 1 at --edge-px inside): nothing outside "
                    "the face is pasted / warped / colour-matched")
    ap.add_argument("--only-frames", default=None, help="debug a:b: composite only these OUTPUT frames, the rest is the source frame (fast A/B of a window)")
    ap.add_argument("--safe-scale", type=float, default=2.0, help="r3 D flow_safe: grey levels over --safe-d at which the confidence reaches 0")
    ap.add_argument("--safe-d", type=float, default=3.5, help="r3 D flow_safe: warp disagreement (grey levels, moving pixels) where the confidence starts to drop (-> 0 at +3)")
    ap.add_argument("--safe-cons", type=float, default=0.95, help="r3 D flow_safe: forward-backward consistency of the moving pixels where the confidence starts to drop (-> 0 at -0.1)")
    ap.add_argument("--retime", choices=["flow", "flow_safe", "nearest", "nearest_rigid"], default="flow", help="LatentSync layer at a source timestamp: flow interpolation between the two master "
                    "frames, or the nearest master frame (exact LatentSync shapes, the mouth repeats one frame in six for a 30 fps source); nearest_rigid = the "
                    "nearest frame moved by the head's similarity motion (MediaPipe eyes / nose bridge, master -> source time): exact shapes at the right position")
    ap.add_argument("--ls-sharpen", type=float, default=0.0, help="unsharp-mask amount on the LatentSync layer (0 = off), radius --ls-sharpen-sigma")
    ap.add_argument("--ls-sharpen-sigma", type=float, default=1.2)
    ap.add_argument("--ls-sharpen-scope", choices=["all", "lips"], default="all", help="lips = only inside the lip mask (the skin gets the source texture instead)")
    ap.add_argument("--ls-sharpen-lips", type=float, default=0.0, help="EXTRA unsharp amount inside the outer lip contour of the LatentSync frame (on top of --ls-sharpen)")
    ap.add_argument("--edge-px", type=float, default=12.0); ap.add_argument("--threads", type=int, default=12); ap.add_argument("--crf", type=int, default=13)
    ap.add_argument("--max-fallback", type=float, default=0.02)
    ap.add_argument("--paste-guard", type=float, default=0.0, help="r7 F (user: 05 25-26 s 'a rectangle with the face plus a background from another frame "
                    "is pasted' while the camera shows trees): the layer is pasted only when the CURRENT source frame really holds the face at the paste "
                    "position: grey NCC between the source frame and the (registered, colour-matched) layer on the paste region outside the lips "
                    ">= this; below it the frame passes through untouched (counted: paste_guard_skipped). 0 = off; 0.5 = the r7 default")
    ap.add_argument("--guard-min-run", type=int, default=3, help="r7 F: --paste-guard passes a frame through only inside a run of >= this many "
                    "flagged frames (a lost face is lost for a while; a single flagged frame is motion blur on a small face and keeps its paste)")
    ap.add_argument("--max-hold", type=int, default=2, help="r7 F: a source frame whose landmarks are NaN in the raw MediaPipe pass but finite after "
                    "any hold / interpolation is pasted only if the run of such frames is <= this (never carry the face position further); "
                    "the source-lm npz has no hold today, this is the guard for any future one")
    ap.add_argument("--lip-guard", choices=["on", "off"], default="off", help="r3: mask = max(ellipse, hull of the source + layer lips dilated 2.6 sigma and "
                    "feathered with the same edge): the lips always sit at alpha >= 0.98 (small faces: the 12 px ramp otherwise reaches them)")
    ap.add_argument("--master-fps", type=float, default=25.0, help="r3 experiment: frame rate of the master / LatentSync layer (30 = LatentSync run on the source frames)")
    ap.add_argument("--master-timing", default=None, help="r2: <id>_master_timing.json (master frame m shows source time tau[m], NOT m/25: ffmpeg -r 25 lags ~60 ms on 30 fps "
                    "sources). Given: mask / landmarks / registration target follow the TRUE time of each source frame (mu = tau^-1(t_k)); the mouth stays audio-synced (25 t_k)")
    ap.add_argument("--register", choices=["off", "rigid", "sim", "sim_ring"], default="off", help="r2: move the LatentSync layer from its own head pose to the shown source frame's: "
                    "rigid = similarity on MediaPipe eyes / nose bridge / nose tip (layer pose -> true-time pose); sim = rigid + DIS-flow RANSAC similarity on the stable lower-face "
                    "parts vs the source frame (composite_register.py); sim_ring = sim + smooth flow refinement on the paste ring")
    ap.add_argument("--register-prior", choices=["landmarks", "image"], default="landmarks", help="r2: first (rigid) step of --register: landmark similarity, or "
                    "image = DIS + RANSAC similarity of the upper face between the SOURCE frame the LatentSync layer was made from (j[m]) and the shown frame "
                    "(same camera image both sides: no detector lag on fast motion)")
    ap.add_argument("--ring-region", choices=["all", "upper"], default="upper", help="r2: where the ring refinement may move the layer: all = the whole paste ring "
                    "(chin / jaw included: pulls the generated jaw towards the SOURCE jaw, which articulates the original speech); upper = only above the "
                    "mouth-corner line (cheeks, nasolabial, upper-lip skin), faded out over 0.1 IOD below it")
    ap.add_argument("--small-face-iod", type=float, default=80.0, help="r2: below this inter-ocular distance the registration flow runs at full resolution")
    ap.add_argument("--source-lm", default=None, help="r3: MediaPipe landmarks of every SOURCE frame (r3_source_lm.py npz). Mask, registration target and the layer's "
                    "pose come from the source frames (the master is a subset of them: master m = source j[m]) instead of master landmarks interpolated "
                    "at mu (wrong on the source frames the master dropped: head shake is not linear). Needs --master-timing (j)")
    ap.add_argument("--seam-lf", type=int, default=0, help="r3: LatentSync 16-frame window joints: the EXCESS low-frequency appearance step at every window start "
                    "(motion-compensated residual, Gaussian --seam-lf-sigma x box width, minus the mean of the neighbouring frames' residuals) is spread "
                    "linearly over +-N master frames, moved with the head (landmark similarity); 0 = off. Runs before --boundary")
    ap.add_argument("--seam-lf-sigma", type=float, default=0.06)
    ap.add_argument("--colour-match", choices=["off", "band", "upper"], default="off", help="r3: per output frame, low-frequency colour / brightness correction of the "
                    "LatentSync layer towards the source frame, estimated in the mask border band above the mouth-corner line (lips excluded; normalised "
                    "Gaussian, sigma --colour-sigma x mouth width) and extended over the whole pasted region; upper = estimated on ALL mask skin above the "
                    "mouth-corner line (alpha > 0.05: nose base, cheeks, upper-lip skin), not only the band")
    ap.add_argument("--colour-sigma", type=float, default=0.6); ap.add_argument("--colour-max", type=float, default=20.0)
    ap.add_argument("--sharpen-auto", type=float, default=0.0, help="r3: per-VIDEO unsharp amounts (constant -> no flicker) so that the LatentSync layer's sharpness "
                    "(mean |Laplacian|) reaches TARGET x the source's, calibrated on 48 LatentSync frames against the source frame each was generated from "
                    "(same pose): lips (outer contour, dilated) -> --ls-sharpen-lips, skin in the mask -> --ls-sharpen (0 when --skin-texture source); 0 = off")
    ap.add_argument("--src-anchor", default="flow:4", help="r3, with --source-lm: landmark filter around the source-rate rigid head motion (anchor_filters spec)")
    ap.add_argument("--mask-track", choices=["dets", "head"], default="dets", help="r3, with --source-lm: dets = LatentSync box of the filtered landmarks, +-3 frame "
                    "mean (clean0 recipe at the source rate); head = box locked to the rigid head motion (anchor_filters.head_locked_boxes: sides / top "
                    "smoothed in head-stabilised coordinates, chin edge sigma 1)")
    ap.add_argument("--seg-joint", type=int, default=0, help="r5 F (face twitch): LS|LS joints between two SEPARATE generations (dub-silence splits, "
                    "--max-ls-frames caps; crossfade 0 there) -> the N master frames each side are a confidence-checked flow interpolation between the "
                    "last frame before and the first after, eased (smoothstep); 0 = off (hard switch)")
    ap.add_argument("--xfade-frames", type=int, default=0, help="r5 F: LS <-> source crossfade length in master frames (0 = the manifest's, 4)")
    ap.add_argument("--xfade-shape", choices=["linear", "smooth"], default="linear", help="r5 F: crossfade ramp shape (smooth = smoothstep: no velocity step)")
    ap.add_argument("--xfade-morph", choices=["on", "off"], default="off", help="r5 F: inside LS <-> source crossfades, a flow morph instead of an alpha "
                    "blend: both the layer and the source are warped part of the way toward each other (no double jaw / lip contour)")
    a = ap.parse_args()
    global FPS_M; FPS_M = float(a.master_fps)
    t_all = time.perf_counter(); times = {}
    man = json.loads(pathlib.Path(a.manifest).read_text()); work = pathlib.Path(man["work_dir"]); src = man["input"]
    t0 = time.perf_counter(); S, pts, tb, rate = decode_source(src); times["decode_source"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter(); L = tms.read_video(a.ls_raw); times["decode_ls"] = round(time.perf_counter() - t0, 2)
    H, W = S[0].shape[:2]; nm = len(L)
    if L[0].shape[:2] != (H, W): raise SystemExit(f"LatentSync frames {L[0].shape[:2]} != source {(H, W)}")
    mp = np.load(cfs.run_mediapipe(work / "master_25fps.mp4", work / "master_mediapipe.npz"))["lm"][:nm]
    wm = master_weights(man, nm, a.xfade_frames, a.xfade_shape)
    dets = cfs.ls_detections(mp, W, H, calib=True); dets = faa.smooth_detections(dets, 2)
    bb = np.array([d[0] if d[0] is not None else (np.nan,) * 4 for d in dets], np.float64)
    for k in range(4):   # mask smoothing +-3 (MouthOnlyRunner.boxes) on the master grid
        col = bb[:, k]; ok = np.isfinite(col); sm = col.copy()
        for i in np.flatnonzero(ok): nb = col[max(0, i - 3):i + 4]; sm[i] = np.nanmean(nb)
        bb[:, k] = sm
    silent = tms.silent_stretches(man, nm, 0, a.silence_min_s, gaps=tms.load_gaps(a.gaps) if a.gaps else None) if a.silence != "none" else np.zeros(nm, bool)
    sil_w = tms.ramp(silent, a.silence_ramp)
    mp_ls = np.load(cfs.run_mediapipe(pathlib.Path(a.ls_raw), pathlib.Path(a.ls_raw).with_suffix(".mediapipe.npz")))["lm"][:nm] if (a.skin_texture == "source" or a.ls_sharpen_scope == "lips" or a.ls_sharpen_lips > 0 or a.sharpen_auto > 0 or a.lip_guard == "on") else None
    # C2 on the LatentSync layer (master grid): two frames around every 16-frame window boundary of every LATENT_SYNC segment
    fl0 = tms.Flow(); nbnd = 0; nsan = 0; san_frames = []
    def lip_stats(img, lm_):   # (lip-box high-frequency energy, contrast) of a layer frame, or None
        if not np.isfinite(lm_[tms.LIPS_OUTER]).all(): return None
        q = lm_[tms.LIPS_OUTER, :2]; w_ = q[:, 0].max() - q[:, 0].min(); x0, y0 = (q.min(0) - 0.1 * w_).astype(int); x1, y1 = (q.max(0) + 0.1 * w_).astype(int)
        g_ = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8)[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if g_.size < 50: return None
        return float(np.abs(cv2.Laplacian(g_, cv2.CV_32F, ksize=3)).mean()), float(g_.std())
    def ring_align(own_, cand_, lm_):   # grey NCC of two layer frames on the skin ring around the lip box (box +-0.6 w minus box +-0.1 w), or None
        if not np.isfinite(lm_[tms.LIPS_OUTER]).all(): return None
        q = lm_[tms.LIPS_OUTER, :2]; w_ = q[:, 0].max() - q[:, 0].min(); x0, y0 = (q.min(0) - 0.1 * w_).astype(int); x1, y1 = (q.max(0) + 0.1 * w_).astype(int)
        X0, Y0 = np.maximum((q.min(0) - 0.6 * w_).astype(int), 0); X1, Y1 = (q.max(0) + 0.6 * w_).astype(int)
        ga = cv2.cvtColor(np.clip(own_, 0, 255).astype(np.uint8)[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(np.clip(cand_, 0, 255).astype(np.uint8)[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if ga.size < 100: return None
        msk = np.ones(ga.shape, bool); msk[max(0, y0 - Y0):max(0, y1 - Y0), max(0, x0 - X0):max(0, x1 - X0)] = False
        if msk.sum() < 50 or ga[msk].std() < 1e-3 or gb[msk].std() < 1e-3: return None
        return float(np.corrcoef(ga[msk], gb[msk])[0, 1])
    nguard = 0; nlowconf = 0; nrigid = 0; mr_state = [a.mouth_region]; mr_frames = [0]   # r7 F --mouth-region-closed: current radius, frames widened
    if a.sanitize_layer > 0 and mp_ls is not None:   # r3 D: generator outliers (pink blob: the lip line vanishes for 1-3 frames) out of the layer first
        t0 = time.perf_counter(); hf = np.full(nm, np.nan); ct = np.full(nm, np.nan)
        for m_ in range(nm):
            if wm[m_] <= 0: continue
            st_ = lip_stats(L[m_], mp_ls[m_])
            if st_ is not None: hf[m_], ct[m_] = st_
        bad = np.zeros(nm, bool); ratio_ = np.full(nm, np.nan)
        for m_ in range(nm):
            if not np.isfinite(hf[m_]): continue
            nb = [i for i in range(max(0, m_ - 6), min(nm, m_ + 7)) if abs(i - m_) >= 2 and np.isfinite(hf[i])]
            if len(nb) >= 4:
                ratio_[m_] = min(hf[m_] / np.median(hf[nb]), ct[m_] / np.median(ct[nb]))
                if ratio_[m_] < a.sanitize_layer: bad[m_] = True
        nmild = 0; mild_frames = []
        m_ = 0
        while m_ < nm:
            if not bad[m_]: m_ += 1; continue
            e_ = m_
            while e_ < nm and bad[e_]: e_ += 1
            lo_, hi_ = m_ - 1, e_
            if e_ - m_ <= 4 and lo_ >= 0 and hi_ < nm and wm[lo_] > 0 and wm[hi_] > 0:
                y_, z_ = L[lo_].astype(np.float32), L[hi_].astype(np.float32)
                for i_ in range(m_, e_):
                    it_, conf_ = interp_safe(fl0, y_, z_, (i_ - lo_) / (hi_ - lo_), a.safe_d, a.safe_cons, a.safe_scale)
                    if conf_ < a.sanitize_min_conf:   # r7 F (04 m8 frame 956: a 'cracked' lower lip): at confidence ~0 the dense flow is wrong and the
                        # fallback (the nearer frame warped by that flow) tears the lip line into cracks -> the nearer good frame moved by a SIMILARITY
                        # fitted on the mouth + rigid landmarks instead (a similarity cannot tear); no landmarks -> the frame is kept as it is
                        near_ = lo_ if (i_ - lo_) <= (hi_ - i_) else hi_; idx_ = tms.LIPS_OUTER + [33, 133, 362, 263, 168, 6, 197, 4]   # lips + eyes / nose (RIG_POSE is defined later)
                        P_, Q_ = mp_ls[near_][idx_, :2], mp_ls[i_][idx_, :2]; Mr_ = None
                        if np.isfinite(P_).all() and np.isfinite(Q_).all(): Mr_, _ = cv2.estimateAffinePartial2D(P_.astype(np.float32), Q_.astype(np.float32), method=cv2.LMEDS)
                        if Mr_ is None: nlowconf += 1; continue
                        rig_ = cv2.warpAffine(L[near_].astype(np.float32), Mr_, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE); nrigid += 1
                        # per pixel: the dense warp where it agrees with the rigid one (it aligns the lip line better), the rigid one where they
                        # disagree by more than --sanitize-tear grey levels (the torn region); soft switch over 4 levels
                        dis_ = cv2.GaussianBlur(np.abs(it_ - rig_).mean(2), (0, 0), 2.0); wr_ = np.clip((dis_ - a.sanitize_tear) / 4.0 + 0.5, 0, 1)[..., None]
                        it_ = it_ + wr_ * (rig_ - it_)
                    # r7 F (03 37.4 / 37.8 s: a washed-pink lip blob keeps its colour in texture mode because the LF is its own): a STRONGLY
                    # collapsed frame (ratio < --sanitize-replace-below) takes the whole interpolation; mild ones keep texture mode
                    if a.sanitize_mode == "texture" and not (a.sanitize_replace_below > 0 and ratio_[i_] < a.sanitize_replace_below):   # own low frequencies (shape, opening) + the neighbours' high frequencies (lip line)
                        own = L[i_].astype(np.float32); sg = a.sanitize_sigma; lo_own = cv2.GaussianBlur(own, (0, 0), sg); lo_it = cv2.GaussianBlur(it_, (0, 0), sg)
                        if a.sanitize_colour == "on" and np.isfinite(mp_ls[i_][tms.LIPS_OUTER]).all():
                            q = mp_ls[i_][tms.LIPS_OUTER, :2]; w_ = q[:, 0].max() - q[:, 0].min()
                            bx0, by0 = np.maximum((q.min(0) - 0.15 * w_).astype(int), 0); bx1, by1 = (q.max(0) + 0.15 * w_).astype(int)
                            ro, ri = lo_own[by0:by1, bx0:bx1], lo_it[by0:by1, bx0:bx1]
                            if ro.size > 30:
                                yo = ro @ np.array([0.114, 0.587, 0.299], np.float32); yi = ri @ np.array([0.114, 0.587, 0.299], np.float32)   # BGR -> luma
                                gain = float(np.clip((yi.std() + 1e-3) / (yo.std() + 1e-3), 0.7, 1.6))
                                fix = ro + ((yo - yo.mean()) * gain + yi.mean() - yo)[..., None]   # luminance only: darkness / lip-line contrast, no chroma push
                                fw = tms.poly_mask(q, (by0, by1, bx0, bx1), 0.15 * w_, 0.1 * w_)[..., None] if hasattr(tms, "poly_mask") else 1.0
                                lo_own[by0:by1, bx0:bx1] = ro + fw * (fix - ro)
                        if a.sanitize_mild > 0 and ratio_[i_] >= a.sanitize_mild and np.isfinite(mp_ls[i_][tms.LIPS_OUTER]).all():
                            q = mp_ls[i_][tms.LIPS_OUTER, :2]; w_ = q[:, 0].max() - q[:, 0].min()
                            bx0, by0 = np.maximum((q.min(0) - 0.15 * w_).astype(int), 0); bx1, by1 = (q.max(0) + 0.15 * w_).astype(int)
                            ho = (own - lo_own)[by0:by1, bx0:bx1].mean(2).ravel(); hd_ = (it_ - lo_it)[by0:by1, bx0:bx1].mean(2).ravel()
                            ncc_ = float(np.corrcoef(ho, hd_)[0, 1]) if ho.size > 30 and ho.std() > 1e-3 and hd_.std() > 1e-3 else 1.0
                            print(f"  sanitize-mild frame {i_}: ratio {ratio_[i_]:.3f} hf-ncc {ncc_:.3f} -> {'kept' if ncc_ < a.sanitize_hf_ncc else 'fixed'}", flush=True)
                            if ncc_ < a.sanitize_hf_ncc: nmild += 1; mild_frames.append(int(i_)); continue
                        it_ = lo_own + (it_ - lo_it)
                    if a.sanitize_guard == "on":   # r6 F (05 23.7 s, the user's face twitch): the fix is taken only if the interpolated neighbours are
                        # ALIGNED with the frame on the skin ring around the lip box (grey NCC >= --sanitize-align). A real flash changes the lips only
                        # (03: ring NCC 0.95-0.98 on every real flash); on a small / moving face the flow fails (interp confidence 0) and the warped
                        # neighbour lands elsewhere (05 smear frames: ring NCC -0.13..0.14) -> the "fix" was a mouth-less smear for 2-4 frames.
                        al_ = ring_align(L[i_], it_, mp_ls[i_])
                        if al_ is None or al_ < a.sanitize_align: nguard += 1; continue
                    L[i_] = np.clip(it_, 0, 255).astype(np.uint8); nsan += 1; san_frames.append(int(i_))
            m_ = e_
        times["sanitize_layer"] = {"seconds": round(time.perf_counter() - t0, 2), "flagged": int(bad.sum()), "replaced": nsan, "guard_kept": nguard, "lowconf_kept": nlowconf, "rigid_fallback": nrigid, "mild_kept": nmild, "mild_frames": mild_frames[:200], "frames": san_frames[:200]}
        print(f"  sanitize-layer: {int(bad.sum())} flagged, {nsan} replaced, {nguard} kept by the guard, {nrigid} by the rigid fallback (interpolation confidence < {a.sanitize_min_conf}), {nlowconf} kept, {nmild} mild kept (own lip line, donor misaligned)", flush=True)
    nseam = 0
    if a.seam_lf > 0:   # r3: window-joint appearance steps -> ramps
        t0 = time.perf_counter(); N = a.seam_lf; dis_s = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        STAB = [33, 133, 362, 263, 168, 6, 197, 4, 98, 327, 2, 50, 280]
        def lf_resid(m, roi, sig):
            ya, yb, xa, xb = roi; a_ = L[m - 1][ya:yb, xa:xb].astype(np.float32); b_ = L[m][ya:yb, xa:xb].astype(np.float32)
            F = dis_s.calc(cv2.cvtColor(L[m][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY), cv2.cvtColor(L[m - 1][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY), None)
            yy, xx = np.mgrid[0:yb - ya, 0:xb - xa].astype(np.float32)
            wa = cv2.remap(a_, xx + F[..., 0], yy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return cv2.GaussianBlur(b_ - wa, (0, 0), sig)
        corr = {}
        for s_ in man["lipsync"]["segments"]:
            if s_["action"] != "LATENT_SYNC": continue
            lo, hi = s_["start_frame"], min(s_["end_frame"], nm)
            for b in range(lo + P, hi, P):
                if b + 1 >= hi or not np.isfinite(bb[b]).all() or not np.isfinite(mp[b][STAB]).all(): continue
                x1, y1, x2, y2 = bb[b]; w_ = x2 - x1; pad = 0.3 * w_
                roi = (int(max(0, y1 - pad)), int(min(H, y2 + pad)), int(max(0, x1 - pad)), int(min(W, x2 + pad))); sig = a.seam_lf_sigma * w_
                D = lf_resid(b, roi, sig) - 0.5 * (lf_resid(b - 1, roi, sig) + lf_resid(b + 1, roi, sig)); off = np.array([roi[2], roi[0]], np.float64)
                for m_ in range(max(lo + 1, b - N), min(hi, b + N)):
                    r_ = (m_ - (b - N) + 0.5) / (2 * N); c_ = r_ if m_ < b else -(1 - r_)
                    Dm = D
                    if m_ != b and np.isfinite(mp[m_][STAB]).all():
                        Mr, _ = cv2.estimateAffinePartial2D(mp[b][STAB].astype(np.float32), mp[m_][STAB].astype(np.float32))
                        if Mr is not None:
                            Mr = np.concatenate([Mr[:, :2], (Mr[:, :2] @ off + Mr[:, 2] - off)[:, None]], 1)
                            Dm = cv2.warpAffine(D, Mr, (roi[3] - roi[2], roi[1] - roi[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    corr.setdefault(m_, []).append((roi, c_ * Dm))
                nseam += 1
        for m_, lst in corr.items():
            f_ = L[m_].astype(np.float32)
            for (ya, yb, xa, xb), C in lst: f_[ya:yb, xa:xb] += C
            L[m_] = np.clip(f_ + 0.5, 0, 255).astype(np.uint8)
        times["seam_lf"] = round(time.perf_counter() - t0, 2)
    if a.boundary == "on":
        t0 = time.perf_counter()
        for s in man["lipsync"]["segments"]:
            if s["action"] != "LATENT_SYNC": continue
            for b in range(s["start_frame"] + P, s["end_frame"], P):
                if b >= nm: continue
                y, z = L[b - 1].astype(np.float32), L[b].astype(np.float32)
                L[b - 1] = np.clip(tms.interp(fl0, y, z, 1 / 3), 0, 255).astype(np.uint8); L[b] = np.clip(tms.interp(fl0, y, z, 2 / 3), 0, 255).astype(np.uint8); nbnd += 1
        times["c2_boundaries"] = round(time.perf_counter() - t0, 2)
    njoint = 0
    if a.seg_joint > 0:   # r5 F: two generations meet with no crossfade -> eased interpolation across the joint
        t0 = time.perf_counter(); N = a.seg_joint
        for b in seg_joints(man):
            lo_, hi_ = b - N - 1, b + N
            if lo_ < 0 or hi_ >= nm or wm[lo_] <= 0 or wm[hi_] <= 0: continue
            y_, z_ = L[lo_].astype(np.float32), L[hi_].astype(np.float32)
            for i_ in range(lo_ + 1, hi_):
                x_ = (i_ - lo_) / (hi_ - lo_); x_ = x_ * x_ * (3 - 2 * x_)
                L[i_] = np.clip(interp_safe(fl0, y_, z_, x_, a.safe_d, a.safe_cons, a.safe_scale)[0], 0, 255).astype(np.uint8)
            njoint += 1
        times["seg_joints"] = {"seconds": round(time.perf_counter() - t0, 2), "joints": njoint}
    tk = pts.astype(np.float64) * float(tb); mpos = tk * FPS_M
    mu_arr = mpos.copy()   # geometry index: nominal (= the old behaviour) unless the true master timing is given
    if a.master_timing:
        tau = np.asarray(json.loads(pathlib.Path(a.master_timing).read_text())["tau_s"], np.float64)[:nm]
        mu_arr = np.interp(tk, tau, np.arange(len(tau), dtype=np.float64))
        jsrc = np.asarray(json.loads(pathlib.Path(a.master_timing).read_text())["j"], np.int64)[:nm]   # master frame m = source frame jsrc[m]
        times["geometry_minus_audio_index_mean"] = round(float(np.mean(mu_arr - mpos)), 3)
    import composite_register as creg
    RIG_POSE = [33, 133, 362, 263, 168, 6, 197, 4]   # eyes / nose bridge / nose tip: the layer's head pose
    import anchor_filters as af
    def dis_for(iod):
        key = "dis_full" if iod < a.small_face_iod else "dis"
        if not hasattr(local, key): setattr(local, key, creg.Dis(0, 5) if key == "dis_full" else creg.Dis(1, 5))
        return getattr(local, key)
    def src_rigid(js, k, lm_src):
        """similarity mapping source frame js -> source frame k (full-frame coords), from DIS on the upper face (anchor_filters' region)."""
        if js == k: return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        if not np.isfinite(lm_src).all(): return None
        iod = float(np.linalg.norm(lm_src[33] - lm_src[263])); msk, (x0, y0, x1, y1) = af._region_mask(lm_src, H, W, pad=int(0.6 * iod))
        X0, Y0, X1, Y1 = max(0, x0 - 16), max(0, y0 - 16), min(W, x1 + 16), min(H, y1 + 16)
        Fr = dis_for(iod)(creg.gray(S[js][Y0:Y1, X0:X1]), creg.gray(S[k][Y0:Y1, X0:X1]))
        ys, xs = np.nonzero(msk[::2, ::2]); ys = ys * 2 + (y0 - Y0); xs = xs * 2 + (x0 - X0)
        if len(xs) < 12: return None
        p0 = np.stack([xs + X0, ys + Y0], 1).astype(np.float32); p1 = p0 + Fr[ys, xs]
        Mj, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.5, maxIters=2000, confidence=0.995)
        return Mj if Mj is not None and float(inl.mean()) > 0.3 else None
    def lerp(arr, m):
        m0 = int(np.clip(math.floor(m), 0, nm - 1)); m1 = min(m0 + 1, nm - 1); al = float(m - math.floor(m))
        if not (np.isfinite(arr[m0]).all() and np.isfinite(arr[m1]).all()): return arr[m0] if np.isfinite(arr[m0]).all() else arr[m1]
        return (1 - al) * arr[m0] + al * arr[m1]
    import threading
    local = threading.local()
    def flow():
        if not hasattr(local, "f"): local.f = tms.Flow()
        return local.f
    if a.sharpen_auto > 0:   # r3: calibrate the unsharp amounts once per video
        t0 = time.perf_counter(); sig_ = a.ls_sharpen_sigma; js_ = jsrc if a.master_timing else np.round(np.arange(nm) * float(rate) / FPS_M).astype(int)
        cand = [m for m in range(nm) if wm[m] >= 1 and np.isfinite(mp[m]).all() and np.isfinite(bb[m]).all() and 0 <= js_[m] < len(S)]
        pick = cand[:: max(1, len(cand) // 48)][:48]; amts = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32); res = {"lips": [], "skin": []}
        for m in pick:
            x1, y1, x2, y2 = bb[m]; al_, roi_ = ellipse_alpha(bb[m], H, W, a.edge_px)
            if al_ is None: continue
            ya, yb, xa, xb = roi_; lw = tms.mouth_width(mp[m]); lip_ = tms.poly_mask(mp[m][tms.LIPS_OUTER], roi_, 0.08 * lw, 0) > 0.5
            skin_ = (al_ > 0.9) & ~(tms.poly_mask(mp[m][tms.LIPS_OUTER], roi_, 0.25 * lw, 0) > 0.5)
            gl = cv2.cvtColor(L[m][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32); gs = cv2.cvtColor(S[js_[m]][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
            ls_ = np.abs(cv2.Laplacian(gs, cv2.CV_32F)); det = gl - cv2.GaussianBlur(gl, (0, 0), sig_)
            for nm_, rg in (("lips", lip_), ("skin", skin_)):
                if rg.sum() < 30: continue
                res[nm_].append([float(np.abs(cv2.Laplacian(gl + A_ * det, cv2.CV_32F))[rg].mean() / max(ls_[rg].mean(), 1e-3)) for A_ in amts])
        auto = {}
        for nm_ in ("lips", "skin"):
            if not res[nm_]: auto[nm_] = 0.0; continue
            r_ = np.median(np.array(res[nm_]), 0); auto[nm_] = float(np.interp(a.sharpen_auto, r_, amts)) if r_[0] < a.sharpen_auto else 0.0
            auto[nm_ + "_slope"] = float(max(np.interp(auto[nm_] + 0.5, amts, r_) - np.interp(auto[nm_], amts, r_), 0.01) / 0.5)
            auto[nm_ + "_ratio_before"] = round(float(r_[0]), 3)
        a.ls_sharpen = round(auto["skin"], 3); a.ls_sharpen_lips = round(max(0.0, auto["lips"] - a.ls_sharpen), 3)
        times["sharpen_auto"] = round(time.perf_counter() - t0, 2); times["sharpen_auto_amounts"] = {**auto, "ls_sharpen": a.ls_sharpen, "ls_sharpen_lips": a.ls_sharpen_lips}
        print("  sharpen-auto:", times["sharpen_auto_amounts"], flush=True)
    lm_f = None; jj = None
    if a.source_lm:   # r3: geometry from the source frames themselves
        if not a.master_timing: raise SystemExit("--source-lm needs --master-timing")
        t0 = time.perf_counter(); lm_s = np.load(a.source_lm)["lm"][:len(S)].astype(np.float64)
        if len(lm_s) != len(S): raise SystemExit(f"--source-lm has {len(lm_s)} frames, source {len(S)}")
        rawp = pathlib.Path(a.source_lm).with_name(pathlib.Path(a.source_lm).stem + "_mp.npz"); nheld_drop = 0   # r7 F --max-hold: never carry a face position
        if rawp.exists():
            raw_ok = np.isfinite(np.load(rawp)["lm"][:len(S), 0, 0]); held = np.isfinite(lm_s[:, 0, 0]) & ~raw_ok
            for s_, e_ in af.runs(held):
                if e_ - s_ > a.max_hold: lm_s[s_:e_] = np.nan; nheld_drop += e_ - s_
        times["held_frames_dropped"] = nheld_drop
        rp = pathlib.Path(a.source_lm).with_name(pathlib.Path(a.source_lm).stem + "_rigid.npz")
        if rp.exists() and len(np.load(rp)["H"]) == len(S): Hs = np.load(rp)["H"]
        else: Hs = af.rigid_motion(S, lm_s); np.savez_compressed(rp, H=Hs)
        times["source_rigid_motion"] = round(time.perf_counter() - t0, 2)
        lm_f = af.filter_landmarks(lm_s, Hs, a.src_anchor) if a.src_anchor != "raw" else lm_s
        jj = np.clip(jsrc, 0, len(S) - 1)
        dets_s = cfs.ls_detections(lm_f.astype(np.float32), W, H, calib=True)
        if a.mask_track == "head":
            l106 = np.stack([d[1] if d[1] is not None else np.full((106, 2), np.nan, np.float32) for d in dets_s]).astype(np.float64)
            hb = af.head_locked_boxes(l106, Hs); bb_s = np.array([hb[i] if d[0] is not None else (np.nan,) * 4 for i, d in enumerate(dets_s)], np.float64)
        else:
            bb_s = np.array([af.ls_bbox_float(d[1].astype(np.float64)) if d[0] is not None else (np.nan,) * 4 for d in dets_s], np.float64)
            for c_ in range(4):
                col = bb_s[:, c_].copy()
                for i in np.flatnonzero(np.isfinite(col)): bb_s[i, c_] = np.nanmean(col[max(0, i - 3):i + 4])
        times["source_geometry"] = round(time.perf_counter() - t0, 2)
    def lm_at(k, mu):   # landmarks of the SHOWN frame
        return lm_f[k] if lm_f is not None else lerp(mp, mu)
    def lm_layer(m):    # landmarks of the head pose the LatentSync layer at master position m was generated from
        if lm_f is None: return lerp(mp, m)
        m0 = int(np.clip(math.floor(m), 0, nm - 1)); m1 = min(m0 + 1, nm - 1); al = float(m - math.floor(m))
        return (1 - al) * lm_f[jj[m0]] + al * lm_f[jj[m1]]
    edge_meas = []; lip_cov = []; guard_vals = []; guard_skip = []
    only = tuple(int(x) for x in a.only_frames.split(":")) if a.only_frames else None
    def one(k):
        if only and not (only[0] <= k < only[1]): return S[k], 0, None
        m = mpos[k]; m0 = int(np.clip(math.floor(m), 0, nm - 1)); m1 = min(m0 + 1, nm - 1); al = float(m - math.floor(m)); mu = float(mu_arr[k])
        w = float((1 - al) * wm[m0] + al * wm[m1])
        if w <= 0: return S[k], 0, None
        box = lerp(bb, mu) if lm_f is None else bb_s[k]
        if not np.isfinite(box).all(): return S[k], 0, None
        alpha, roi = ellipse_alpha(box, H, W, a.edge_px)
        if alpha is None: return S[k], 0, None
        if a.lip_guard == "on":   # spec 8.4 on small faces: keep the 10-15 px ramp OUTSIDE the lips (source lips + the layer's own lips)
            pts_ = [q[tms.LIPS_OUTER] for q in (lm_at(k, mu), lerp(mp_ls, m) if mp_ls is not None else None) if q is not None and np.isfinite(q[tms.LIPS_OUTER]).all()]
            if pts_:
                hull = cv2.convexHull(np.concatenate(pts_).astype(np.float32))[:, 0]; sg_ = a.edge_px / 2.563
                gm_ = cv2.GaussianBlur(tms.poly_mask(hull, roi, 2.6 * sg_, 0), (0, 0), sg_)
                alpha = np.maximum(alpha, gm_)
        ya, yb, xa, xb = roi; f = flow(); roi_r = roi
        conf_k = 1.0
        if a.register != "off":   # build the LatentSync layer on a larger box: it moves by up to head speed x 60 ms before the roi is cut
            pad = int(0.25 * (box[2] - box[0])); ya, yb, xa, xb = max(0, ya - pad), min(H, yb + pad), max(0, xa - pad), min(W, xb + pad)
        if a.retime in ("nearest", "nearest_rigid"):
            mn = m0 if (al < 0.5 or wm[m1] <= 0) else m1
            if a.retime == "nearest_rigid" and abs(m - mn) > 1e-3:
                tgt = lerp(mp, m); src_ = mp[mn]; RIG = [33, 133, 362, 263, 168, 6, 197, 4]
                Mr, _ = cv2.estimateAffinePartial2D(src_[RIG, :2].astype(np.float32), tgt[RIG, :2].astype(np.float32)) if np.isfinite(src_[RIG]).all() and np.isfinite(tgt[RIG]).all() else (None, None)
                if Mr is not None:
                    Lk = cv2.warpAffine(L[mn], Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[ya:yb, xa:xb].astype(np.float32)
                else:
                    Lk = L[mn][ya:yb, xa:xb].astype(np.float32)
            else:
                Lk = L[mn][ya:yb, xa:xb].astype(np.float32)
        elif al < 0.04 or m1 == m0 or wm[m1] <= 0: Lk = L[m0][ya:yb, xa:xb].astype(np.float32)
        elif al > 0.96 or wm[m0] <= 0: Lk = L[m1][ya:yb, xa:xb].astype(np.float32)
        elif a.retime == "flow_safe":   # r3 D (user defect "torn / whitish lips on single 30 fps frames"): confidence-checked interpolation
            Lk, conf_k = interp_safe(f, L[m0][ya:yb, xa:xb].astype(np.float32), L[m1][ya:yb, xa:xb].astype(np.float32), al, a.safe_d, a.safe_cons, a.safe_scale)
        else: Lk = tms.interp(f, L[m0][ya:yb, xa:xb].astype(np.float32), L[m1][ya:yb, xa:xb].astype(np.float32), al)
        if a.register != "off":
            mn_ = m0 if (al < 0.5 or wm[m1] <= 0) else m1
            pose_l = (mp[mn_] if lm_f is None else lm_f[jj[mn_]]) if a.retime == "nearest" else lm_layer(m); pose_t = lm_at(k, mu)
            if np.isfinite(pose_l[RIG_POSE]).all() and np.isfinite(pose_t[RIG_POSE]).all():
                off = np.array([xa, ya], np.float64); Mr = None
                if a.register_prior == "image":   # layer pose = source frame(s) jsrc[m0] / jsrc[m1]; blend the two similarities by the retime weight
                    if a.retime == "nearest": Mr = src_rigid(int(jsrc[mn_]), k, pose_l)
                    else:
                        M0 = src_rigid(int(jsrc[m0]), k, mp[m0] if lm_f is None else lm_f[jj[m0]]); M1 = (src_rigid(int(jsrc[m1]), k, mp[m1] if lm_f is None else lm_f[jj[m1]]) if al >= 0.04 else M0)
                        Mr = None if (M0 is None or M1 is None) else (1 - al) * M0 + al * M1
                    if Mr is not None: Mr = np.concatenate([Mr[:, :2], (Mr[:, :2] @ off + Mr[:, 2] - off)[:, None]], 1)   # -> crop coords
                if Mr is None: Mr, _ = cv2.estimateAffinePartial2D((pose_l[RIG_POSE] - off).astype(np.float32), (pose_t[RIG_POSE] - off).astype(np.float32), method=cv2.LMEDS)
                if Mr is not None: Lk = cv2.warpAffine(Lk, Mr, (xb - xa, yb - ya), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                if a.register in ("sim", "sim_ring"):
                    local.dis = dis_for(creg.face_axes(pose_t)[2])
                    Lt = pose_t - off; iod = creg.face_axes(pose_t)[2]; gs = creg.gray(S[k][ya:yb, xa:xb]); gl = cv2.cvtColor(np.clip(Lk, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    stab, lip = creg.stable_mask(Lt, (yb - ya, xb - xa), iod, 0.12)
                    F = local.dis(gs, gl); ys, xs = np.nonzero(stab)
                    if len(xs) >= 30:
                        P_ = np.stack([xs[::2], ys[::2]], 1).astype(np.float32); Q_ = P_ + F[ys[::2], xs[::2]]
                        Ms, inl = cv2.estimateAffinePartial2D(P_, Q_, method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=2000, confidence=0.995, refineIters=10)
                        if Ms is not None and float(inl.mean()) >= 0.4 and abs(math.hypot(Ms[0, 0], Ms[1, 0]) - 1) < 0.04 and np.linalg.norm(Ms[:, 2]) < 0.15 * iod + 5:
                            Lk = cv2.warpAffine(Lk, Ms, (xb - xa, yb - ya), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                    if a.register == "sim_ring":
                        al_c = np.zeros((yb - ya, xb - xa), np.float32); al_c[roi_r[0] - ya:roi_r[1] - ya, roi_r[2] - xa:roi_r[3] - xa] = alpha
                        F2 = local.dis(gs, cv2.cvtColor(np.clip(Lk, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)); ringw = int(max(3, 0.35 * iod))
                        band = cv2.dilate(((al_c > 0.02) & (al_c < 0.98)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ringw + 1, 2 * ringw + 1))) > 0
                        wgt = ((band | stab) & ~lip).astype(np.float32); sg = 0.12 * iod
                        if a.ring_region == "upper":   # never chin / jaw: their position is the dub's articulation, not the head's
                            u_, v_, _ = creg.face_axes(pose_t); s_mc = 0.5 * float((Lt[61] + Lt[291]) @ v_)
                            yy_, xx_ = np.mgrid[0:yb - ya, 0:xb - xa].astype(np.float32); sv = xx_ * v_[0] + yy_ * v_[1]
                            wgt *= np.clip(1 - (sv - s_mc) / max(0.1 * iod, 1.0), 0, 1)
                        den = cv2.GaussianBlur(wgt, (0, 0), sg) + 1e-4
                        Fs = np.stack([cv2.GaussianBlur(F2[..., c_] * wgt, (0, 0), sg) / den for c_ in range(2)], -1)
                        fade = np.clip(creg.lip_distance(lip, lip.shape) / max(0.25 * iod, 1.0), 0, 1).astype(np.float32); Fs *= fade[..., None] * (den > 0.02)[..., None]
                        yy, xx = np.mgrid[0:yb - ya, 0:xb - xa].astype(np.float32)
                        Lk = cv2.remap(Lk, xx + Fs[..., 0], yy + Fs[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Lk = Lk[roi_r[0] - ya:roi_r[1] - ya, roi_r[2] - xa:roi_r[3] - xa].astype(np.float32); ya, yb, xa, xb = roi_r
        sk = S[k][ya:yb, xa:xb].astype(np.float32)
        if a.colour_match != "off":   # r3: low-frequency colour match to the source, estimated in the upper border band
            lm_c = lm_at(k, mu)
            if np.isfinite(lm_c[[61, 291, 13]]).all():
                mwc = tms.mouth_width(lm_c); ymc = float((lm_c[61, 1] + lm_c[291, 1]) / 2) - ya
                lipm = tms.poly_mask(lm_c[tms.LIPS_OUTER], (ya, yb, xa, xb), 0.25 * mwc, 0)
                yy_ = np.arange(yb - ya, dtype=np.float32)[:, None]
                wgt = ((alpha > 0.05) & ((alpha < 0.95) | (a.colour_match == "upper"))).astype(np.float32) * (yy_ < ymc + 0.1 * mwc) * (1 - lipm)
                if wgt.sum() > 50:
                    sg_ = a.colour_sigma * mwc; d_ = sk - Lk
                    num = cv2.GaussianBlur(d_ * wgt[..., None], (0, 0), sg_); den = cv2.GaussianBlur(wgt, (0, 0), sg_)
                    gm = (d_ * wgt[..., None]).sum((0, 1)) / wgt.sum(); eps = 0.05
                    Lk = Lk + np.clip((num + eps * gm) / (den[..., None] + eps), -a.colour_max, a.colour_max)
        if a.paste_guard > 0:   # r7 F: paste only if the CURRENT source frame holds the face here (layer vs source agree outside the lips)
            lm_g = lm_at(k, mu); ring = alpha > 0.5
            if np.isfinite(lm_g[tms.LIPS_OUTER]).all(): ring &= ~(tms.poly_mask(lm_g[tms.LIPS_OUTER], roi, 0.3 * tms.mouth_width(lm_g), 0) > 0.5)
            if ring.sum() >= 50:
                gs_ = cv2.cvtColor(np.clip(sk, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[ring]
                gl_ = cv2.cvtColor(np.clip(Lk, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[ring]
                ncc = float(np.corrcoef(gs_, gl_)[0, 1]) if gs_.std() > 1e-3 and gl_.std() > 1e-3 else 0.0
                guard_vals.append((k, ncc))
                if ncc < a.paste_guard: guard_skip.append(k)   # applied after the pass: only runs of >= --guard-min-run frames pass through
        ex = None
        if a.skin_texture == "source" or (a.ls_sharpen > 0 and a.ls_sharpen_scope == "lips"):
            lm_s = lm_at(k, mu); lm_l = lerp(mp_ls, m) if mp_ls is not None else None; mw_ = tms.mouth_width(lm_s)
            idx = tms.LIPS_INNER if a.lip_texture == "on" else tms.LIPS_OUTER
            ex = tms.poly_mask(lm_l[idx] if lm_l is not None else None, roi, a.lip_dilate * mw_, 0.08 * mw_)
            if a.lip_texture == "off": ex = np.maximum(ex, tms.poly_mask(lm_s[idx], roi, a.lip_dilate * mw_, 0.08 * mw_))
        if a.ls_sharpen > 0 or a.ls_sharpen_lips > 0:   # deterministic spatial unsharp mask of the (soft) LatentSync layer
            det = Lk - cv2.GaussianBlur(Lk, (0, 0), a.ls_sharpen_sigma); amt = np.full(Lk.shape[:2], a.ls_sharpen, np.float32)
            if a.ls_sharpen_scope == "lips": amt = a.ls_sharpen * ex
            if a.ls_sharpen_lips > 0:
                lm_l2 = lerp(mp_ls, m) if mp_ls is not None else lerp(mp, m); mw2 = tms.mouth_width(lm_l2)   # LatentSync layer's own lips (its pose)
                amt = amt + a.ls_sharpen_lips * tms.poly_mask(lm_l2[tms.LIPS_OUTER], roi, 0.08 * mw2, 0.08 * mw2)
            Lk = Lk + (amt * conf_k)[..., None] * det   # r3 D: a low-confidence (partly nearest-frame) layer is never sharpened more than a real one
        if a.skin_texture == "source":
            Lk = tms.skin_texture_transfer(f, [Lk], [sk], [ex], a.tex_sigma, a.tex_tau)[0]
        aw = alpha * w
        if a.silence == "still" and (sil_w[m0] > 0 or sil_w[m1] > 0):
            g = (1 - al) * sil_w[m0] + al * sil_w[m1]; lm_s = lm_at(k, mu)
            if np.isfinite(lm_s).all():
                oval = lm_s[tms.FACE_OVAL, :2]; ny = float(lm_s[tms.NOSE_BASE, 1].mean()); low = oval[oval[:, 1] >= ny - 1]
                hull = cv2.convexHull(np.concatenate([low, [[oval[:, 0].min(), ny], [oval[:, 0].max(), ny]]]).astype(np.float32))[:, 0]
                gm = tms.poly_mask(hull, roi, 0, 0) ; gm = cv2.GaussianBlur(gm, (0, 0), a.edge_px / 2.563) * g * w
                aw = np.maximum(aw, gm)
        if a.silence == "orig" and (sil_w[m0] > 0 or sil_w[m1] > 0):
            aw = aw * (1.0 - ((1 - al) * sil_w[m0] + al * sil_w[m1]))
        if a.face_clip == "on":
            lm_fc = lm_at(k, mu)
            if lm_fc is not None and np.isfinite(lm_fc[tms.FACE_OVAL]).all():
                ov = lm_fc[tms.FACE_OVAL, :2].copy()
                if a.chin_clip > 0 and np.isfinite(lm_fc[17]).all():   # jaw points (below the mouth corners) pulled toward the lower lip
                    lip_y = lm_fc[17, 1]; low = ov[:, 1] > lip_y
                    ov[low] = ov[low] + a.chin_clip * (np.stack([ov[low, 0], np.full(low.sum(), lip_y)], 1) - ov[low]) * [0.35, 1.0]
                fm = np.zeros(aw.shape[:2], np.uint8); cv2.fillPoly(fm, [np.round(ov - [xa, ya]).astype(np.int32)], 1)
                aw = np.minimum(aw, np.clip(cv2.distanceTransform(fm, cv2.DIST_L2, 3) / max(a.edge_px, 1.0), 0, 1).astype(aw.dtype))
        if a.mouth_region > 0:
            lm_mr = lm_at(k, mu); pts_ = [q[tms.LIPS_OUTER, :2] for q in (lm_mr, lerp(mp_ls, m) if mp_ls is not None else None)
                                         if q is not None and np.isfinite(q[tms.LIPS_OUTER]).all()]
            if pts_ and np.isfinite(lm_mr[[33, 263]]).all():
                iod = float(np.linalg.norm(lm_mr[33, :2] - lm_mr[263, :2])); hull = cv2.convexHull(np.concatenate(pts_).astype(np.float32))[:, 0]
                hm = np.ones(aw.shape[:2], np.uint8); cv2.fillPoly(hm, [np.round(hull - [xa, ya]).astype(np.int32)], 0)
                dist = cv2.distanceTransform(hm, cv2.DIST_L2, 3)   # px outside the lips hull (0 inside)
                lq = lerp(mp_ls, m) if mp_ls is not None else None
                mr_ = a.mouth_region
                if a.mouth_region_closed > 0 and lq is not None and np.isfinite(lq[[13, 14]]).all() and np.isfinite(lm_mr[[13, 14]]).all():
                    # r7 F (05 24-25 s 'glued, pressed lips': the dub is silent, the generator closes the mouth, but the 0.38 IOD region keeps the
                    # SOURCE's talking, dropped jaw under the closed lips -> lips stretched over an open jaw). When the layer is closed while the
                    # source holds the mouth open, the paste region widens to --mouth-region-closed so the generator's closed jaw comes with its
                    # lips; the radius moves by at most 1/--mr-ramp of the difference per frame (no region pop).
                    os_ = float(np.linalg.norm(lm_mr[13, :2] - lm_mr[14, :2])) / iod; ol_ = float(np.linalg.norm(lq[13, :2] - lq[14, :2])) / iod
                    tgt_ = a.mouth_region_closed if (os_ > a.jaw_open and ol_ < a.jaw_closed) else a.mouth_region
                    mr_state[0] += np.clip(tgt_ - mr_state[0], -abs(a.mouth_region_closed - a.mouth_region) / a.mr_ramp, abs(a.mouth_region_closed - a.mouth_region) / a.mr_ramp)
                    mr_ = float(mr_state[0]); mr_frames[0] += int(mr_ > a.mouth_region + 1e-6)
                aw = np.minimum(aw, np.clip((mr_ * iod - dist) / max(a.edge_px, 1.0), 0, 1).astype(aw.dtype))
                if a.chin_warp == "on" and lq is not None and np.isfinite(lq[17]).all() and np.isfinite(lm_mr[[17, 152, 172, 397]]).all():
                    dy = float(np.clip(lq[17, 1] - lm_mr[17, 1], -0.2 * iod, 0.5 * iod))
                    if abs(dy) > 0.5:
                        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32); ly, cy = lm_mr[17, 1], lm_mr[152, 1]; cx = lm_mr[152, 0]
                        hw = max(0.5 * abs(lm_mr[397, 0] - lm_mr[172, 0]), 1.0)
                        wv = np.where(yy < ly, np.clip(1 - (ly - yy) / (0.15 * iod), 0, 1), np.where(yy <= cy, 1.0, np.clip(1 - (yy - cy) / (0.25 * iod), 0, 1)))
                        wh = np.clip(1 - ((xx - cx) / hw) ** 2, 0, 1)
                        sk = cv2.remap(sk, xx - xa, yy - ya - dy * wv * wh, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if a.xfade_morph == "on" and 0 < w < 1:   # r5 F: morph, not a double exposure: the source and the layer meet half way in geometry
            Fsl = f.calc(tms.gray(sk), tms.gray(Lk)); gx, gy = f.g(*sk.shape[:2])   # sk(x) ~ Lk(x + Fsl)
            Sw = cv2.remap(sk, gx - w * Fsl[..., 0], gy - w * Fsl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Lw = cv2.remap(Lk, gx + (1 - w) * Fsl[..., 0], gy + (1 - w) * Fsl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Lk = (1 - w) * Sw + w * Lw; aw = np.minimum(aw / w, 1.0)   # the blend weight is now inside Lk; the spatial mask stays
        o = S[k].copy(); o[ya:yb, xa:xb] = np.clip(sk + aw[..., None] * (Lk - sk), 0, 255).round().astype(np.uint8)
        if w >= 1:   # spec 8.4 on small faces: the feather ramp must not reach the lips (source lips and the layer's own lips)
            lm_q = lm_at(k, mu); lips_ = [lm_q] + ([lerp(mp_ls, m)] if mp_ls is not None else [])
            cov = []
            for lq in lips_:
                if np.isfinite(lq[tms.LIPS_OUTER]).all():
                    lpm = tms.poly_mask(lq[tms.LIPS_OUTER], (ya, yb, xa, xb), 0, 0) > 0.5
                    if lpm.any():
                        inner = cv2.distanceTransform((alpha >= 0.9).astype(np.uint8), cv2.DIST_L2, 3)   # px from the 0.9 contour, inside
                        cov.append((float(alpha[lpm].min()), float(inner[lpm].min())))
            if cov: lip_cov.append((min(c[0] for c in cov), min(c[1] for c in cov), float(np.linalg.norm(lm_q[33] - lm_q[263])) if np.isfinite(lm_q[[33, 263]]).all() else np.nan))
        return o, 1, (alpha if k % 50 == 0 else None)
    if a.sharpen_auto > 0:   # closed loop: composite sample frames, measure OUTPUT vs source sharpness (lips / lower-face skin), correct twice
        t0 = time.perf_counter(); ks_ = [k for k in range(len(S)) if wm[int(np.clip(round(mpos[k]), 0, nm - 1))] >= 1 and np.isfinite(lm_at(k, mu_arr[k])).all()]
        ks_ = ks_[:: max(1, len(ks_) // 40)][:40]; hist = []
        def meas():
            rl, rs = [], []
            for k in ks_:
                o, t_, _ = one(k)
                if not t_: continue
                lk = lm_at(k, mu_arr[k]); box = bb_s[k] if lm_f is not None else lerp(bb, mu_arr[k]); al_, roi_ = ellipse_alpha(box, H, W, a.edge_px)
                if al_ is None: continue
                ya, yb, xa, xb = roi_; lw = tms.mouth_width(lk)
                lip_ = tms.poly_mask(lk[tms.LIPS_OUTER], roi_, 0.08 * lw, 0) > 0.5; skin_ = (al_ > 0.9) & ~(tms.poly_mask(lk[tms.LIPS_OUTER], roi_, 0.25 * lw, 0) > 0.5)
                go = np.abs(cv2.Laplacian(cv2.cvtColor(o[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
                gs = np.abs(cv2.Laplacian(cv2.cvtColor(S[k][ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
                if lip_.sum() > 30: rl.append(go[lip_].mean() / max(gs[lip_].mean(), 1e-3))
                if skin_.sum() > 30: rs.append(go[skin_].mean() / max(gs[skin_].mean(), 1e-3))
            return float(np.median(rl)) if rl else None, float(np.median(rs)) if rs else None
        for it in range(3):
            rl, rs = meas(); hist.append({"ls_sharpen": a.ls_sharpen, "ls_sharpen_lips": a.ls_sharpen_lips, "out_ratio_lips": rl, "out_ratio_skin": rs})
            if it == 2: break
            if rs is not None: a.ls_sharpen = round(float(np.clip(a.ls_sharpen + (a.sharpen_auto - rs) / auto["skin_slope"], 0, 8)), 3)
            if rl is not None:
                tot = float(np.clip(a.ls_sharpen + a.ls_sharpen_lips + (a.sharpen_auto - rl) / auto["lips_slope"] if it == 0 else a.ls_sharpen + a.ls_sharpen_lips + (a.sharpen_auto - rl) / auto["lips_slope"], 0, 10))
                a.ls_sharpen_lips = round(max(0.0, tot - a.ls_sharpen), 3)
        best = min(hist, key=lambda h: abs((h["out_ratio_lips"] or a.sharpen_auto) - a.sharpen_auto) + abs((h["out_ratio_skin"] or a.sharpen_auto) - a.sharpen_auto))
        a.ls_sharpen, a.ls_sharpen_lips = best["ls_sharpen"], best["ls_sharpen_lips"]
        if a.sharpen_lips_max > 0: a.ls_sharpen_lips = min(a.ls_sharpen_lips, a.sharpen_lips_max)
        times["sharpen_closed_loop"] = {"iterations": hist, "chosen": best, "seconds": round(time.perf_counter() - t0, 2)}; print("  sharpen closed loop:", best, flush=True)
    lip_cov.clear(); edge_meas.clear(); guard_vals.clear(); guard_skip.clear()   # calibration calls of one() do not count
    t0 = time.perf_counter(); out = [None] * len(S); fails = 0; touched = 0
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = {ex.submit(one, k): k for k in range(len(S))}
        for fu in cf.as_completed(futs):
            k = futs[fu]
            try:
                o, t_, al_ = fu.result(); out[k] = o; touched += t_
                if al_ is not None: edge_meas.append(edge_width_px(al_))
            except Exception as e:   # never crash the worker: previous valid frame, counted
                out[k] = None; fails += 1
                if fails <= 3: print(f"  frame {k} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
    for k in range(len(out)):
        if out[k] is None: out[k] = out[k - 1] if k > 0 and out[k - 1] is not None else S[k]
    guard_applied = []
    if guard_skip:   # r7 F: a face that is really gone is gone for many frames (a pan); a single low-NCC frame is motion blur on a small face (05 frame 705)
        fl_ = np.zeros(len(out), bool); fl_[guard_skip] = True
        for s_, e_ in af.runs(fl_):
            if e_ - s_ >= a.guard_min_run:
                for k in range(s_, e_): out[k] = S[k]; guard_applied.append(k)
        print(f"  paste-guard: {len(guard_skip)} frames below NCC {a.paste_guard}, {len(guard_applied)} passed through (runs >= {a.guard_min_run})", flush=True)
    times["composite"] = round(time.perf_counter() - t0, 2)
    # encode with the source time base / pts, then copy the run's own final audio stream in
    import av
    t0 = time.perf_counter(); outp = pathlib.Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True); vtmp = outp.with_name(".v_" + outp.name)
    c = av.open(str(vtmp), "w"); st = c.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.time_base = tb
    st.options = {"crf": str(a.crf), "preset": "medium"}; st.codec_context.time_base = tb
    for k, o in enumerate(out):
        fr = av.VideoFrame.from_ndarray(o, format="bgr24"); fr.pts = int(pts[k]); fr.time_base = tb
        for p in st.encode(fr): c.mux(p)
    for p in st.encode(): c.mux(p)
    c.close()
    # audio: the run output's own (final-mix) audio stream, copied -> bit-identical to the baseline / clean0 output
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vtmp), "-i", str(man["output"]), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(outp)], check=True)
    vtmp.unlink(missing_ok=True); times["encode_mux"] = round(time.perf_counter() - t0, 2); times["total"] = round(time.perf_counter() - t_all, 2)
    dur = float(tk[-1] - tk[0] + (tk[-1] - tk[-2] if len(tk) > 1 else 0))
    em = [e for e in edge_meas if e]
    rep = {"input": src, "output": str(outp), "source_frames": len(S), "source_fps_avg": float(rate), "master_frames": nm, "ls_touched_frames": touched,
           "fallback_frames": fails, "fallback_fraction": round(fails / max(len(S), 1), 5), "paste_guard": {"threshold": a.paste_guard, "min_run": a.guard_min_run, "flagged": len(guard_skip), "flagged_frames": sorted(guard_skip)[:300], "skipped": len(guard_applied), "skipped_frames": guard_applied[:300], "ncc_p1": round(float(np.percentile([v for _, v in guard_vals], 1)), 3) if guard_vals else None, "ncc_p5": round(float(np.percentile([v for _, v in guard_vals], 5)), 3) if guard_vals else None, "ncc_min": round(float(min(v for _, v in guard_vals)), 3) if guard_vals else None, "ncc_median": round(float(np.median([v for _, v in guard_vals])), 3) if guard_vals else None}, "boundaries_interpolated": nbnd, "seam_lf_boundaries": nseam, "edge_px_target": a.edge_px,
           "edge_px_measured": {"median": float(np.median(em)) if em else None, "min": min(em) if em else None, "max": max(em) if em else None, "n": len(em)},
           "silent_master_frames": int(silent.sum()), "mouth_region_closed": {"radius": a.mouth_region_closed, "widened_frames": mr_frames[0]},
           "lip_coverage": (lambda c: {"frames": len(c), "alpha_min_p1": round(float(np.percentile(c[:, 0], 1)), 4), "alpha_min_min": round(float(c[:, 0].min()), 4),
                                       "frames_alpha_below_0_98": int((c[:, 0] < 0.98).sum()), "lips_to_ramp_px_p1": round(float(np.percentile(c[:, 1], 1)), 2),
                                       "lips_to_ramp_px_min": round(float(c[:, 1].min()), 2), "iod_median": round(float(np.nanmedian(c[:, 2])), 1)} if len(c) else None)(np.array(lip_cov, np.float64).reshape(-1, 3)), "args": vars(a), "seconds": times, "s_per_video_min": round(times["total"] / (dur / 60), 1)}
    outp.with_suffix(".fps_composite.json").write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("source_frames", "ls_touched_frames", "fallback_frames", "paste_guard", "edge_px_measured", "lip_coverage", "s_per_video_min")}))
    return 3 if rep["fallback_fraction"] > a.max_fallback else 0


if __name__ == "__main__":
    raise SystemExit(main())
