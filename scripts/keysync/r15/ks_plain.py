#!/usr/bin/env python3
"""r15 K: PLAIN paste of the raw KeySync 512 crop back into the 25 fps master (the eye: the raw crop is good, the full-frame cfhf chain breaks it).
Exactly the inverse of ks_prep.py's STATIC crop (box x / y / side; the crop was scaled side -> 512 with lanczos): the crop is scaled 512 -> side
and placed at (x, y). No per-frame registration, no flow, no retime: the output is the 25 fps master grid, KeySync frame i at master frame
i + shift (= the SyncNet AV fix, as ks_layer --shift). Mask = the speaker's face oval (source landmarks via the master timing map j), dilated
--dil IOD (+ --down IOD of room below the chin for an opened jaw), minus the other faces (k_faces2.py, dilated 0.08 of their IOD), feathered,
x the box-edge feather (24 px, none on a frame border). ONE global colour match: Lab mean / std of the KeySync skin vs the master skin inside
the mask (lips excluded) over every 5th frame -> a constant gain / offset per channel (no per-frame colour match). After KeySync's output
ends, a fade to the master over 8 frames. Colour-managed I/O (ks_cms, BT.709).
    KS_CMS=1 python ks_plain.py --ks KS.mp4 --master master_25fps.mp4 --box box.json --lm source_lm_F.npz [--others faces2.npz] --shift 1 --audio MIX --out OUT.mp4"""
import argparse, json, math, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ks_cms

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAWC = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]   # r16c K1: cheek + jaw contour (ear to ear via the chin)
JAWLOW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]   # r16c K1: the jaw below the mouth corners
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser()
for k in ("ks", "master", "box", "lm", "audio", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--cm-lonly", type=int, default=0, help="r16c K1 colour in dark / warm scenes: 1 = the colour match (global gain / offset + --lfm maps) corrects LIGHTNESS only; KeySync's chroma is not shifted (the a / b offset turned dark areas blue)")
ap.add_argument("--cm-shadow", default="", help="r16c K1 + K2 (long5 1:56 blue mouth): L0:L1 = fade the whole colour correction out in the darks by the RAW KeySync lightness (0 below L0, full above L1): the global offset no longer lifts the black floor or neutralises the warmth of a dark mouth / hair; e.g. 10:30")
ap.add_argument("--mask", default="oval", choices=["oval", "tight"], help="r16c K1 rule 5: oval = the r15/r16 mask (oval + 0.15 IOD down + 0.10 IOD dilation: reaches neck / collar / background); tight = strictly INSIDE the face: the cheek / jaw contour (union of the source jaw and KeySync's generated jaw), inset, feathered inwards")
ap.add_argument("--mask-chin", type=int, default=0, help="tight: 1 = do NOT inset the chin: the jaw below the mouth corners (source and generated) + the lips are added back un-eroded, so the generated chin motion is kept (the inset there cost SyncNet); the cheeks keep the inset")
ap.add_argument("--mask-tsmooth", type=float, default=0.0, help="r16c K1 (Irina 09.11: the neck-face seam swims, in ALL videos): the mask polygon (oval or tight) from landmark tracks gaussian-smoothed over time (sigma in frames; source and generated); symmetric + global, so ks_plain_par == serial")
ap.add_argument("--mask-chin-feather", type=float, default=0.0, help="tight + chin: an extra VERTICAL-only feather of the chin part (sigma x IOD), so the generated-jaw / source-neck step fades over a taller band")
ap.add_argument("--neck-follow", type=float, default=0.0, help="r16c K1 (Irina 10.11 seam option a): warp the SOURCE neck / chin strip below the jaw so it follows the GENERATED jaw: per column dy = generated - source jaw y (clamped +-0.12 IOD, tapered at the jaw ends), weight 1 at the jaw line -> 0 at this many IOD below; temporal smoothing via --mask-tsmooth")
ap.add_argument("--mask-chin-down", type=float, default=0.0, help="r16d K1 (ghost second chin): extend the chin part this many IOD BELOW the lower of the source / generated jaw, so the feather lies on plain neck, never on a jaw contour (no double chin); use with --neck-follow")
ap.add_argument("--mask-inset", type=float, default=0.04, help="tight: inset from the face contour, x IOD")
ap.add_argument("--mask-feather", type=float, default=0.04, help="tight: feather sigma, x IOD (the whole transition lies inside the inset contour)")
ap.add_argument("--parse", default="", help="r16c K1: face_parse.py npz for the master (per-frame face-skin probability in box coords): inside the mask, non-face pixels (hair / hands / clothes / background) take the source; the generated lips stay")
ap.add_argument("--others", default=""); ap.add_argument("--shift", type=int, default=0); ap.add_argument("--dil", type=float, default=0.10); ap.add_argument("--down", type=float, default=0.15)
ap.add_argument("--no-cm", action="store_true")
ap.add_argument("--align-ecc", default="", help="r16 K1: k_ecc.py npz (per-frame KS -> source affine in 512 px, image-based) instead of the landmark fit for --align")
ap.add_argument("--align-scale", type=float, default=-1, help="r16 K1: scale component of --align: -1 = smoothed like the rest (r16 smart), 0 = LOCKED (no per-frame scale), >0 = its own smoothing sigma in frames")
ap.add_argument("--scale-fix", default="", help="r16 K1 m18 'breathing': k_breath.py npz measured on a no-align paste of the SAME render (master grid); "
                "the high-frequency part of KeySync's lower-face scale vs the source (x / its 1 s moving mean) is divided out per frame, about the lower-face centroid")
ap.add_argument("--others-grow", type=float, default=0.0, help="r16 K1 m03: protect the OTHER faces' whole head: their oval scaled by this factor about its centre "
                "and stretched up by 0.35 of their IOD (hair), feathered on the speaker side; 0 = the r15 clamp (oval + 0.08 IOD)")
ap.add_argument("--lfm", type=float, default=0.0, help="r16 K1 #15/#16: per-frame LOW-FREQUENCY Lab match (KeySync -> master skin, normalized blur at 0.35 IOD, mouth zone excluded), "
                "gaussian-smoothed over time with this sigma in frames, on top of the global gain; 0 = off (one global offset)")
ap.add_argument("--sharp", default="0", help="r16 K1 #17: unsharp amount on the KeySync luma (sigma 1 px at 512 scale), or 'auto' = match the source's skin high-frequency energy")
ap.add_argument("--keep-nose", type=float, default=0.0, help="r16 K1 #8: the source nose pixels (bridge -> alae, columella out), feather sigma in IOD units (e.g. 0.04); 0 = off")
ap.add_argument("--upper-src", default="", help="r16 K1 #17: 'T0:T1' = the UPPER face from the full-res source (KeySync does not change it; the 512 crop softens eyes / glasses / brows): "
                "the paste weight ramps 0 -> 1 between T0 and T1 along the eye-midpoint -> chin axis (fractions of that length, e.g. 0.12:0.32); empty = off")
ap.add_argument("--diag", default="", help="r16 K1: write per-video diagnostics json (edge-band colour error vs the source, its drift over time, skin HF ratio)")
ap.add_argument("--align", type=float, default=0.0, help="r16 K1 smart paste: per-frame similarity of KeySync's stable landmarks (eye corners, nose bridge; --ks-lm) onto the source's at the shown master frame, "
                "gaussian-smoothed over time (sigma = this many frames), clamped to 0.06 IOD / 3 %% scale / 2 deg; 0 = off (the static inverse crop)")
ap.add_argument("--ks-lm", default="", help="landmarks of the raw KeySync output (512 grid, its frame index): the GENERATED lips for --tex / --cavity")
ap.add_argument("--tex", type=float, default=2.0, help="r15 stubble / pores: the source's luma high frequencies (gaussian sigma px), DIS-flow-warped onto KeySync's "
                "geometry, replace KeySync's inside the face mask outside the generated lips (dilated 0.15 IOD); 0 = off")
ap.add_argument("--tex-rgb", type=int, default=0, help="r16b K1 texture-keep: 1 = the source high frequencies on all 3 channels (beard / skin colour boil), 0 = luma only")
ap.add_argument("--tex-flow", type=int, default=1, help="r16b K1: 1 = DIS-warp the source texture onto KeySync's geometry (per frame), 0 = no warp (box / ECC aligned only)")
ap.add_argument("--tex-dil", type=float, default=0.15, help="r16b K1: the generated-lip keep zone = the outer-lip hull dilated by this x IOD (soft edge 0.05 IOD)")
ap.add_argument("--cavity", type=float, default=0.0, help="r15 black-hole mouth: lift strength (0..1) of the near-black inner-mouth pixels toward the source's open-mouth "
                "interior tone (L 30th percentile + mean a / b of the source's open frames); 0 = off")
ap.add_argument("--cm-file", default="", help="r16 T (parallel paste, ks_plain_par.py): the global colour match: loaded (gain / off) if the file exists, else computed and saved with its sufficient statistics (Lab sums of the sampled skin pixels, so chunks combine exactly)")
ap.add_argument("--cm-only", action="store_true", help="r16 T: compute --cm-file over --write-range and exit")
ap.add_argument("--write-range", default="", help="r16 T: 'A:B' = process and write only master frames A <= k < B (a chunk; the frames around it are context)")
ap.add_argument("--absent", default="hold", choices=["hold", "skip"], help="r16 T: a master frame with no source landmarks (NaN): hold = paste with the last mask (r15), skip = leave the source frame (e2e: the face is absent there - a cutaway / outside its presence; ks_e2e_prep bridges short gaps)")
ap.add_argument("--k-base", type=int, default=0, help="r16 K1 (set by ks_plain_par): the GLOBAL master index of this run's frame 0 (the k %% 3 / 5 / 10 sampling grids follow the global index)")
ap.add_argument("--ks-base", type=int, default=0, help="r16 K1 (set by ks_plain_par): the GLOBAL KeySync index of this run's KeySync frame 0")
ap.add_argument("--ks-total", type=int, default=-1, help="r16 K1 (set by ks_plain_par): the GLOBAL KeySync frame count (the end fade fires only when KeySync is really shorter than the master)")
ap.add_argument("--n-total", type=int, default=-1, help="r16 K1 (set by ks_plain_par): the GLOBAL master frame count")
ap.add_argument("--video-only", action="store_true", help="r16 K1 (set by ks_plain_par): write no audio (the chunks are joined and muxed with the mix once)")
ap.add_argument("--lip-keep", default="", help="r16 K2: per-master-frame weights (.npy, k2_pause.py): the SOURCE lips (hull + 0.10 IOD) show through the paste mask")
a = ap.parse_args()
b = json.load(open(a.box)); x0, y0, s, W, H = b["x"], b["y"], b["side"], b["W"], b["H"]
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM))
OT = np.load(a.others)["others"] if a.others else None
SHW = tuple(float(x) for x in a.cm_shadow.split(':')) if a.cm_shadow else None
PARSE = None
if a.parse:   # r16c K1: ONE parser module (also T's occlusion guard): the flat kit dir or ../r16
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r16')]
    import face_parse; PARSE = face_parse.load(a.parse)
LK = np.load(a.lip_keep) if a.lip_keep else None
XS, YS = b.get("xs"), b.get("ys")   # r16 K2: a TRACKED box (k2_prep_track.py): constant side, per-master-frame corner
def bxy(k): return (int(XS[min(max(k, 0), len(XS) - 1)]), int(YS[min(max(k, 0), len(YS) - 1)])) if XS else (x0, y0)
def BXY(k): return np.array(bxy(k), np.float32)
def crop(f, k):
    bx, by = bxy(k); return f[by:by + s, bx:bx + s]
IODM = float(np.nanmedian(np.linalg.norm(LM[:, 33] - LM[:, 263], axis=1)))
KL = np.load(a.ks_lm)["lm"][..., :2].astype(np.float32) * (s / 512.0) if a.ks_lm else None
LIPS_O = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if a.tex > 0 else None
GX, GY = np.meshgrid(np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32))


def frames(path, w, h, vf=None):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path] + ks_cms.dec(vf, "rgb24") + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    while True:
        r = p.stdout.read(w * h * 3)
        if len(r) < w * h * 3: break
        yield np.frombuffer(r, np.uint8).reshape(h, w, 3)


K = np.stack(list(frames(a.ks, s, s, f"scale={s}:{s}:flags=lanczos")))   # the raw crop, scaled back to the box (the inverse of ks_prep's scale)
n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip())
print(f"plain paste: KeySync {len(K)} frames, master {n}, shift {a.shift}, box {x0},{y0} side {s}", flush=True)
el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(r)) + 1,) * 2)
hull = lambda pts: np.round(cv2.convexHull(pts.astype(np.float32))[:, 0]).astype(np.int32)
def ramp(lo_edge, hi_edge):   # 24 px feather at the box edges, none on a side that lies on the frame border
    i_ = np.arange(s, dtype=np.float32); r = np.ones(s, np.float32)
    if not lo_edge: r = np.minimum(r, i_ / 24.0)
    if not hi_edge: r = np.minimum(r, (s - 1 - i_) / 24.0)
    return np.clip(r, 0, 1)


_BF = {}
def boxf(k):
    bx, by = bxy(k); key = (by == 0, by + s == H, bx == 0, bx + s == W)
    if key not in _BF: _BF[key] = np.minimum.outer(ramp(key[0], key[1]), ramp(key[2], key[3])).astype(np.float32)
    return _BF[key]
last = [None]


def _tsm(A, sig):   # gaussian smoothing along time of an (n, P, 2) landmark track, NaN-aware
    if A is None or sig <= 0: return A
    A = A.astype(np.float32).copy(); n_ = len(A); ok = np.isfinite(A).all(axis=(1, 2))
    if ok.sum() < 2: return A
    idx = np.arange(n_); B = A.copy()
    for pi in range(A.shape[1]):
        for c in range(2): B[~ok, pi, c] = np.interp(idx[~ok], idx[ok], A[ok, pi, c])
    r_ = int(3 * sig); g_ = np.exp(-0.5 * (np.arange(-r_, r_ + 1) / sig) ** 2); g_ /= g_.sum()
    Bp = np.concatenate([np.repeat(B[:1], r_, 0), B, np.repeat(B[-1:], r_, 0)], 0)
    S_ = sum(g_[j] * Bp[j:j + n_] for j in range(2 * r_ + 1)); S_[~ok] = np.nan; return S_
LMS = _tsm(LM[..., :2], a.mask_tsmooth) if a.mask_tsmooth > 0 else None
KLS = _tsm(KL, a.mask_tsmooth) if (a.mask_tsmooth > 0 and KL is not None) else KL



def neck_warp(roi, k, i):   # r16c K1 option (a): the source neck follows the generated jaw (no step at a jaw-line seam)
    src = LMS if LMS is not None else LM[..., :2]
    ps = src[min(int(J[min(k, len(J) - 1)]), len(src) - 1)]; KK = KLS if KLS is not None else KL
    if KK is None or not (0 <= i < len(KK)): return roi
    kp = KK[i]
    if not (np.isfinite(ps[JAWLOW]).all() and np.isfinite(kp[JAWLOW]).all() and np.isfinite(ps[[33, 263]]).all()): return roi
    iod = float(np.linalg.norm(ps[33] - ps[263])); q_ = ps - BXY(k); xs_ = np.arange(s, dtype=np.float32)
    def env(pts):   # the jaw polyline as y(x) on the columns it spans (nan outside)
        o_ = np.argsort(pts[:, 0]); x_, y_ = pts[o_, 0], pts[o_, 1]; y = np.interp(xs_, x_, y_); y[(xs_ < x_[0]) | (xs_ > x_[-1])] = np.nan; return y, x_[0], x_[-1]
    ys_, x0_, x1_ = env(q_[JAWLOW]); yg_, _, _ = env(kp[JAWLOW])
    dy = np.clip(yg_ - ys_, -0.12 * iod, 0.12 * iod); dy[~np.isfinite(dy)] = 0
    wdt = max(x1_ - x0_, 1.0); tp = np.clip(np.minimum(xs_ - x0_, x1_ - xs_) / (0.15 * wdt), 0, 1); dy = dy * (0.5 - 0.5 * np.cos(np.pi * tp))
    ysf = np.where(np.isfinite(ys_), ys_, 1e9).astype(np.float32)[None, :]
    below = np.clip(1 - (GY - ysf) / max(a.neck_follow * iod, 1.0), 0, 1) * (GY >= ysf)
    above = np.clip(1 - (ysf - GY) / max(0.05 * iod, 1.0), 0, 1) * (GY < ysf)
    wgt = (below + above).astype(np.float32)
    if float(np.abs(dy).max()) < 0.25: return roi
    return cv2.remap(roi, GX, (GY - dy[None, :] * wgt).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

CZ = [None]
def mask(k):
    i = int(J[min(k, len(J) - 1)]); p = LM[min(i, len(LM) - 1)]
    if not np.isfinite(p).all(): return (None, None) if a.absent == "skip" else (last[0], None)
    iod = float(np.linalg.norm(p[33] - p[263])); q = p - BXY(k); mk = np.zeros((s, s), np.uint8)
    qs = q; CZ[0] = None
    if LMS is not None:   # r16c K1: the mask polygon from the time-smoothed track (oval AND tight): the seam does not swim
        ps_ = LMS[min(i, len(LMS) - 1)]
        if np.isfinite(ps_).all(): qs = ps_ - BXY(k)
    if a.mask == "tight":   # r16c K1 rule 5: never neck / collar / hair / background
        pts = [qs[OVAL]]
        ii = kidx(k)
        if KLS is not None and 0 <= ii < len(KLS) and np.isfinite(KLS[ii][JAWC]).all(): pts.append(KLS[ii][JAWC])   # the generated jaw may open lower than the source's
        cv2.fillPoly(mk, [hull(np.concatenate(pts))], 1); own = mk.copy()
        mk = cv2.erode(mk, el((a.mask_inset + 2 * a.mask_feather) * iod))
        own = mk.copy()
        if a.mask_chin:   # the chin / lips area (below the mouth corners) un-eroded: source jaw U generated jaw U lips
            yc = float(qs[[61, 291], 1].mean()); low = [qs[JAWLOW]]
            if KLS is not None and 0 <= ii < len(KLS) and np.isfinite(KLS[ii][JAWLOW]).all(): low.append(KLS[ii][JAWLOW])
            if a.mask_chin_down > 0: low = low + [x_ + np.array([0, a.mask_chin_down * iod], np.float32) for x_ in low]   # r16d K1: the band below both jaws
            cz = np.zeros_like(mk); cv2.fillPoly(cz, [hull(np.concatenate(low + [qs[LIPS]]))], 1); cz[:max(0, int(yc)), :] = 0
            mk = np.maximum(mk, cz); CZ[0] = cz
        own = mk.copy()   # r16c K1: m03 bug: the 2-face 'own face' re-fill used the full contour and undid the tight mask
    else:
        hp = hull(qs[OVAL]); cv2.fillPoly(mk, [hp], 1); cv2.fillPoly(mk, [hp + np.array([0, int(round(a.down * iod))], np.int32)], 1)
        own = mk.copy()   # r16 K1: the speaker's own face (oval + the room below the chin): the other-face protection must never cover it
        mk = cv2.dilate(mk, el(a.dil * iod))
    if OT is not None:
        for f2 in OT[min(i, len(OT) - 1)]:
            if np.isfinite(f2).all():
                om = np.zeros((s, s), np.uint8); oi = float(np.linalg.norm(f2[33] - f2[263]))
                if a.others_grow > 0:
                    ov_ = f2[OVAL] - BXY(k); cc = ov_.mean(0); g_ = cc + (ov_ - cc) * a.others_grow
                    cv2.fillPoly(om, [hull(np.concatenate([g_, g_ - [0, 0.35 * oi]]))], 1)
                else:
                    cv2.fillPoly(om, [hull(f2[OVAL] - BXY(k))], 1); om = cv2.dilate(om, el(0.08 * oi))
                if a.others_grow > 0: om[cv2.dilate(own, el(0.03 * iod)) > 0] = 0
                mk[om > 0] = 0
    m = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), max(1.0, (a.mask_feather if a.mask == "tight" else 0.04) * iod)) * boxf(k)
    if a.mask == "tight" and a.mask_chin and a.mask_chin_feather > 0 and CZ[0] is not None:   # a taller, vertical-only fade of the chin edge
        mc = cv2.GaussianBlur(CZ[0].astype(np.float32), (0, 0), sigmaX=max(1.0, a.mask_feather * iod), sigmaY=max(1.0, a.mask_chin_feather * iod)) * boxf(k)
        yc_ = float(qs[152, 1]) - 0.15 * iod   # crossfade from the normal mask to the vertically feathered chin mask around the chin
        wv = np.clip((GY - (yc_ - 0.1 * iod)) / max(0.2 * iod, 1.0), 0, 1).astype(np.float32)
        m = m * (1 - wv) + mc * wv
    if PARSE is not None:   # r16c K1: hair / hands / clothes / background inside the mask -> the source (the generated lips stay)
        fs = face_parse.face_prob(PARSE, k + a.k_base, s)   # global master frame (beards are face skin to the model)
        if fs is not None:
            lz = np.zeros((s, s), np.uint8); cv2.fillPoly(lz, [hull(q[LIPS])], 1)
            ii = kidx(k)
            if KL is not None and 0 <= ii < len(KL) and np.isfinite(KL[ii][LIPS]).all(): cv2.fillPoly(lz, [hull(KL[ii][LIPS])], 1)
            lz = cv2.GaussianBlur(cv2.dilate(lz, el(0.12 * iod)).astype(np.float32), (0, 0), max(1.0, 0.03 * iod))
            m = m * np.maximum(fs, lz)
    if OT is not None and a.others_grow > 0:   # the feather of the cut lies OUTSIDE the speaker's face: fully KeySync inside her oval
        m = np.maximum(m, np.clip(2 * cv2.GaussianBlur(own.astype(np.float32), (0, 0), max(1.0, 0.04 * iod)), 0, 1) * boxf(k))
    if LK is not None and k < len(LK) and LK[k] > 0.01:   # r16 K2 pause closure: the source lips through the mask
        lk = np.zeros((s, s), np.uint8); cv2.fillPoly(lk, [hull(q[LIPS])], 1); lk = cv2.dilate(lk, el(0.10 * iod))
        m = m * (1 - LK[k] * cv2.GaussianBlur(lk.astype(np.float32), (0, 0), max(1.0, 0.04 * iod)))
    lip = np.zeros((s, s), np.uint8); cv2.fillPoly(lip, [hull(q[LIPS])], 1); lip = cv2.dilate(lip, el(0.15 * iod))
    last[0] = m; return m, (mk > 0) & (lip == 0)


NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 129, 358, 64, 294, 98, 327, 48, 278, 115, 344]
def mouth_zone(k, i, iod):   # source lips at master k  U  KeySync lips at its frame i, dilated 0.3 IOD
    z_ = np.zeros((s, s), np.uint8); p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)] - BXY(k)
    if np.isfinite(p[LIPS]).all(): cv2.fillPoly(z_, [hull(p[LIPS])], 1)
    if KL is not None and 0 <= i < len(KL) and np.isfinite(KL[i][LIPS]).all(): cv2.fillPoly(z_, [hull(KL[i][LIPS])], 1)
    return cv2.dilate(z_, el(0.3 * iod)) > 0
def lab_of(img): return cv2.cvtColor(np.clip(img, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
def hf_energy(y, wm): return float(np.sqrt((cv2.Laplacian(y, cv2.CV_32F)[wm] ** 2).mean())) if wm.sum() > 200 else np.nan
IODM = float(np.nanmedian(np.linalg.norm(LM[:, 33] - LM[:, 263], axis=1)))
SHARP = 0.0 if a.sharp == "auto" else float(a.sharp)
UP = tuple(float(x) for x in a.upper_src.split(":")) if a.upper_src else None
kidx = lambda k: k - a.shift   # KeySync frame shown at master frame k
AT = None
ECM = np.load(a.align_ecc)["M"] if a.align_ecc else None
if a.align > 0 and KL is not None:   # r16 K1: per-frame similarity KeySync -> source (box coords), smoothed params
    STB = [33, 263, 133, 362, 168, 6]; nkl = len(KL); prm = np.full((n, 4), np.nan)
    for k in range(n):
        i = kidx(k)
        if not (0 <= i < nkl): continue
        p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)] - BXY(k); q = KL[i]
        if not (np.isfinite(p[STB]).all() and np.isfinite(q[STB]).all()): continue
        if ECM is not None:
            if i >= len(ECM) or not np.isfinite(ECM[i]).all(): continue
            M = ECM[i].astype(np.float64).copy(); M[:, 2] *= s / 512.0   # 512 grid -> box grid (linear part unchanged)
            A_ = M[:, :2]; sc_ = float(np.sqrt(abs(np.linalg.det(A_)))); th = float(np.arctan2(A_[1, 0] - A_[0, 1], A_[0, 0] + A_[1, 1]))
            c0 = q[STB].mean(0); t_ = M @ np.r_[c0, 1] - c0; prm[k] = [t_[0], t_[1], sc_ - 1, th]; continue
        M, _ = cv2.estimateAffinePartial2D(q[STB].astype(np.float32), p[STB].astype(np.float32), method=cv2.LMEDS)
        if M is None: continue
        sc_ = float(np.hypot(M[0, 0], M[1, 0])); th = float(np.arctan2(M[1, 0], M[0, 0])); c0 = q[STB].mean(0)   # params about the anchor centroid
        t_ = M @ np.r_[c0, 1] - c0; prm[k] = [t_[0], t_[1], sc_ - 1, th]
    ok = np.isfinite(prm).all(1)
    if ok.sum() > 5:
        for d in range(4): prm[~ok, d] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), prm[ok, d])
        g = np.exp(-0.5 * (np.arange(-int(3 * a.align), int(3 * a.align) + 1) / a.align) ** 2); g /= g.sum(); pd_ = len(g) // 2
        raw_ds = prm[:, 2].copy()
        prm = np.stack([np.convolve(np.pad(prm[:, d], (pd_, pd_), mode="edge"), g, "valid") for d in range(4)], 1)
        if a.align_scale == 0: prm[:, 2] = 0.0
        elif a.align_scale > 0:
            g2 = np.exp(-0.5 * (np.arange(-int(3 * a.align_scale), int(3 * a.align_scale) + 1) / a.align_scale) ** 2); g2 /= g2.sum(); p2 = len(g2) // 2
            prm[:, 2] = np.convolve(np.pad(raw_ds, (p2, p2), mode="edge"), g2, "valid")
        iodm = float(np.nanmedian(np.linalg.norm(LM[:, 33] - LM[:, 263], axis=1))); lim = 0.06 * iodm
        prm[:, :2] = np.clip(prm[:, :2], -lim, lim); prm[:, 2] = np.clip(prm[:, 2], -0.03, 0.03); prm[:, 3] = np.clip(prm[:, 3], -np.radians(2), np.radians(2))
        AT = prm; print(f"align: |t| mean {np.linalg.norm(prm[:, :2], axis=1).mean():.2f} px, scale {np.abs(prm[:, 2]).mean() * 100:.2f} %, rot {np.degrees(np.abs(prm[:, 3])).mean():.2f} deg", flush=True)
KC = None
if KL is not None and len(KL):
    KC = KL[:, [33, 263, 133, 362, 168, 6]].mean(1); ok_ = np.isfinite(KC).all(1)
    if ok_.any() and not ok_.all():
        for d in range(2): KC[~ok_, d] = np.interp(np.flatnonzero(~ok_), np.flatnonzero(ok_), KC[ok_, d])
    elif not ok_.any(): KC = None
def warp_ks(kf, k, i):
    if AT is None or not (0 <= i < len(KL)): return kf
    tx, ty, ds, th = AT[k]; c0 = KC[i] if KC is not None else np.array([s / 2, s / 2])
    M = cv2.getRotationMatrix2D((float(c0[0]), float(c0[1])), float(np.degrees(-th)), float(1 + ds)); M[:, 2] += [tx, ty]
    return cv2.warpAffine(kf, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
gain = np.ones(3, np.float32); off = np.zeros(3, np.float32); CAV = None
is_open = lambda p: np.isfinite(p).all() and np.linalg.norm(p[61] - p[291]) > 5 and np.linalg.norm(p[13] - p[14]) / np.linalg.norm(p[61] - p[291]) > 0.15
CMF = bool(a.cm_file) and os.path.exists(a.cm_file) and not a.cm_only
CMZ = np.load(a.cm_file) if CMF else None
WR = tuple(int(x) for x in a.write_range.split(":")) if a.write_range else (0, 1 << 30)
need_cm = not a.no_cm and (CMZ is None or "gain" not in CMZ.files)
need_cav = a.cavity > 0 and (CMZ is None or "cav_l" not in CMZ.files)
need_sh = a.sharp == "auto" and (CMZ is None or "sharp" not in CMZ.files)
if CMZ is not None and (need_cav or need_sh): print(f"WARNING: {a.cm_file} has no {'cavity tone ' if need_cav else ''}{'sharp auto' if need_sh else ''}: computed from THIS run's frames only", flush=True)
if need_cm or need_cav or need_sh:   # pass 1: the global colour match on the skin; the source's open-mouth interior tone (for --cavity)
    SK, SM, SC, HFR = [], [], [], []
    for k, f in enumerate(frames(a.master, W, H)):
        if need_cav and (k + a.k_base) % 3 == 0 and WR[0] <= k < WR[1]:
            p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]
            if is_open(p):
                mk = np.zeros((H, W), np.uint8); cv2.fillPoly(mk, [hull(p[INNER])], 1); mk = cv2.erode(mk, np.ones((3, 3), np.uint8))
                if mk.sum() > 30: SC.append(cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[mk > 0])
        if not (need_cm or need_sh) or (k + a.k_base) % 5 or not (0 <= kidx(k) < len(K)) or not (WR[0] <= k < WR[1]): continue
        _, sk = mask(k)
        if sk is None or sk.sum() < 500: continue
        lk = cv2.cvtColor(K[kidx(k)].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[sk]; lm_ = cv2.cvtColor(crop(f, k).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[sk]
        if need_cm: SK.append(lk[:: 7]); SM.append(lm_[:: 7])
        if need_sh and (k + a.k_base) % 10 == 0:
            core = sk & (cv2.erode(sk.astype(np.uint8), el(0.1 * IODM)) > 0)
            HFR.append((hf_energy(cv2.cvtColor(crop(f, k), cv2.COLOR_RGB2GRAY).astype(np.float32), core), hf_energy(cv2.cvtColor(K[kidx(k)], cv2.COLOR_RGB2GRAY).astype(np.float32), core)))
    if a.cm_only:   # r16 K1: the sufficient statistics of EVERY global estimate (colour match sums, sharp-auto pairs, cavity samples): the parent combines them exactly
        ex_ = {"hfr": np.array(HFR, np.float64).reshape(-1, 2), "sc": (np.concatenate(SC) if SC else np.zeros((0, 3), np.float32))}
        if SK:
            SK, SM = np.concatenate(SK), np.concatenate(SM)
            np.savez(a.cm_file, n=len(SK), s1k=SK.sum(0, dtype=np.float64), s2k=(SK.astype(np.float64) ** 2).sum(0), s1m=SM.sum(0, dtype=np.float64), s2m=(SM.astype(np.float64) ** 2).sum(0), **ex_)
        else: np.savez(a.cm_file, n=0, **ex_)
        sys.exit(0)
    if need_cm and SK:
        SK, SM = np.concatenate(SK), np.concatenate(SM)
        gain = np.clip(SM.std(0) / np.maximum(SK.std(0), 1e-3), 0.9, 1.1).astype(np.float32); off = (SM.mean(0) - SK.mean(0) * gain).astype(np.float32)
        print(f"global colour match (Lab): gain {gain.round(3).tolist()} offset {off.round(2).tolist()}", flush=True)
    if need_sh and HFR:
        h_ = np.array([x for x in HFR if np.isfinite(x).all() and x[1] > 0]); r_ = float(np.median(h_[:, 0] / h_[:, 1])) if len(h_) else 1.0
        SHARP = float(np.clip((r_ - 1) * 1.5, 0, 1.5)); print(f"sharp auto: skin HF source / KeySync {r_:.3f} -> unsharp {SHARP:.2f}", flush=True)
    if need_cav and SC:
        SC = np.concatenate(SC); CAV = (float(np.percentile(SC[:, 0], 30)), SC[:, 1:].mean(0).astype(np.float32))
        print(f"source open-mouth interior: L p30 {CAV[0]:.1f}, a/b {CAV[1].round(1).tolist()} ({len(SC)} px)", flush=True)
if CMZ is not None:
    if "gain" in CMZ.files and not a.no_cm:
        gain, off = CMZ["gain"].astype(np.float32), CMZ["off"].astype(np.float32); print(f"colour match from {a.cm_file}: gain {gain.round(3).tolist()} offset {off.round(2).tolist()}", flush=True)
    if a.sharp == "auto" and "sharp" in CMZ.files: SHARP = float(CMZ["sharp"]); print(f"sharp auto from {a.cm_file}: unsharp {SHARP:.2f}", flush=True)
    if a.cavity > 0 and "cav_l" in CMZ.files and float(CMZ["cav_l"]) >= 0: CAV = (float(CMZ["cav_l"]), CMZ["cav_ab"].astype(np.float32)); print(f"cavity tone from {a.cm_file}: L p30 {CAV[0]:.1f}", flush=True)
if a.cm_only: sys.exit(0)
NT = [0, 0]


def fix_ks(kf, i, m, src_crop):   # r15 --cavity / --tex on one KeySync crop (float RGB 0..255) -> float RGB
    kp = KL[int(np.clip(i, 0, len(KL) - 1))] if KL is not None else None
    if a.cavity > 0 and CAV is not None and kp is not None and is_open(kp):
        mc = np.zeros((s, s), np.uint8); cv2.fillPoly(mc, [hull(kp[INNER])], 1); mc = cv2.erode(mc, np.ones((3, 3), np.uint8))
        if mc.sum() > 30:
            dn = cv2.distanceTransform(mc, cv2.DIST_L2, 3); dn = dn / max(float(dn.max()), 1.0)   # 0 at the lip rim, 1 at the cavity centre: a lighter rim, a darker throat
            mc = cv2.GaussianBlur(mc.astype(np.float32), (0, 0), 1.5); lab = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB); Lt, abt = CAV
            w_ = np.clip((Lt - lab[..., 0]) / max(Lt, 1.0), 0, 1) * mc * a.cavity * (1 - 0.65 * dn)
            lab[..., 0] += w_ * (Lt - lab[..., 0]); lab[..., 1:] += w_[..., None] * (abt - lab[..., 1:]); NT[0] += 1
            kf = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)
    if a.tex > 0 and m is not None:
        sc = src_crop.astype(np.float32); h2 = s // 2
        gk = cv2.cvtColor(kf.astype(np.uint8), cv2.COLOR_RGB2GRAY); gs = cv2.cvtColor(src_crop, cv2.COLOR_RGB2GRAY)
        if a.tex_flow:
            fl = DIS.calc(cv2.resize(gk, (h2, h2), interpolation=cv2.INTER_AREA), cv2.resize(gs, (h2, h2), interpolation=cv2.INTER_AREA), None)
            fl = cv2.resize(fl, (s, s)) * (s / h2)
        else: fl = np.zeros((s, s, 2), np.float32)
        yk = cv2.cvtColor(kf.astype(np.float32), cv2.COLOR_RGB2YCrCb); ys = cv2.cvtColor(sc, cv2.COLOR_RGB2YCrCb)[..., 0]
        ws = cv2.remap(ys, GX + fl[..., 0], GY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lx = np.zeros((s, s), np.uint8)
        lp = kp if kp is not None and np.isfinite(kp).all() else LM[min(int(J[min(i + a.shift, len(J) - 1)]), len(LM) - 1)] - BXY(i + a.shift)
        if np.isfinite(lp).all():
            iod = float(np.linalg.norm(lp[33] - lp[263])); cv2.fillPoly(lx, [hull(lp[LIPS_O])], 1); lx = cv2.dilate(lx, el(a.tex_dil * iod))
            lx = cv2.GaussianBlur(lx.astype(np.float32), (0, 0), max(1.0, 0.05 * iod))
        T = m * (1 - lx)
        hp = lambda img: img - cv2.GaussianBlur(img, (0, 0), a.tex)
        if a.tex_rgb:   # r16b K1 texture-keep: all channels; the source HF (warped) replaces KeySync's outside the lip zone
            wr = cv2.remap(sc, GX + fl[..., 0], GY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            kf = np.clip(kf + T[..., None] * (hp(wr) - hp(kf.astype(np.float32))), 0, 255); NT[1] += 1
        else:
            yk[..., 0] += T * (hp(ws) - hp(yk[..., 0])); NT[1] += 1
            kf = np.clip(cv2.cvtColor(yk, cv2.COLOR_YCrCb2RGB), 0, 255)
    return kf
LG = 64; LFM = None
if a.lfm > 0:   # r16 K1: per-frame low-frequency Lab difference maps (LG x LG), temporally smoothed
    D = np.zeros((n, LG, LG, 3), np.float32); V = np.zeros((n, LG, LG, 1), np.float32)
    for k, f in enumerate(frames(a.master, W, H)):
        i = kidx(k)
        if not (0 <= i < len(K)): continue
        m, sk = mask(k)
        if sk is None: continue
        p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]; iod = float(np.linalg.norm(p[33] - p[263])) if np.isfinite(p).all() else IODM
        wm = ((m > 0.05) & ~mouth_zone(k, i, iod)).astype(np.float32)
        kf = warp_ks(K[i].astype(np.float32), k, i); lk = lab_of(kf) * gain + off; lm_ = lab_of(crop(f, k))
        sg = 0.35 * iod * LG / s; ds = lambda x: cv2.resize(x, (LG, LG), interpolation=cv2.INTER_AREA)
        wd = cv2.GaussianBlur(ds(wm), (0, 0), sg); ok_ = wd > 0.02
        dd = (cv2.GaussianBlur(ds(lm_ * wm[..., None]), (0, 0), sg) - cv2.GaussianBlur(ds(lk * wm[..., None]), (0, 0), sg)) / np.maximum(wd, 1e-3)[..., None]
        D[k] = dd * ok_[..., None]; V[k, ..., 0] = ok_
    r_ = int(3 * a.lfm); g = np.exp(-0.5 * (np.arange(-r_, r_ + 1) / a.lfm) ** 2).astype(np.float32); g /= g.sum()
    def sm(x):
        xp = np.concatenate([np.repeat(x[:1], r_, 0), x, np.repeat(x[-1:], r_, 0)], 0); o_ = np.zeros_like(x)
        for j_, gj in enumerate(g): o_ += gj * xp[j_: j_ + len(x)]
        return o_
    LFM = sm(D) / np.maximum(sm(V), 1e-3); LFM = np.clip(LFM, -12, 12).astype(np.float32)
    print(f"lfm: |D| mean L {np.abs(LFM[..., 0]).mean():.2f} a {np.abs(LFM[..., 1]).mean():.2f} b {np.abs(LFM[..., 2]).mean():.2f} (sigma_t {a.lfm})", flush=True)
SFX = None
if a.scale_fix:
    xs_ = np.load(a.scale_fix)["lower"].astype(np.float64); tr_ = np.convolve(np.pad(xs_, (12, 12), mode="edge"), np.ones(25) / 25, "valid")
    SFX = np.clip(tr_ / xs_, 0.97, 1.03); print(f"scale-fix: correction std {(SFX - 1).std() * 100:.3f} %, max {np.abs(SFX - 1).max() * 100:.2f} %", flush=True)
LOWC = [234, 454, 132, 361, 58, 288, 172, 397, 150, 379, 152, 1]
def scale_fix(kf, k):
    if SFX is None or k >= len(SFX): return kf
    p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]
    if not np.isfinite(p[LOWC]).all(): return kf
    c_ = p[LOWC].mean(0) - BXY(k); M = cv2.getRotationMatrix2D((float(c_[0]), float(c_[1])), 0.0, float(SFX[k]))
    return cv2.warpAffine(kf, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
DG = []
fps_m = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip()
def audio_probe(audio):
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name:format=duration", "-of", "json", audio], capture_output=True, text=True).stdout
    js = json.loads(pr or "{}"); return (js.get("streams") or [{}])[0].get("codec_name", ""), float(js.get("format", {}).get("duration", 0) or 0)
def audio_args(audio, nv, fps):   # r16 K1: the accepted mix is stream-copied (no AAC re-encode); the length is set explicitly (-frames:v / -t), never by -shortest
    codec, dur = audio_probe(audio); fr = eval(fps) if isinstance(fps, str) else float(fps)
    if abs(dur - nv / fr) > 1.0 / fr: print(f"WARNING: audio {dur:.3f} s vs video {nv / fr:.3f} s ({nv} frames): the output is cut to {nv} frames", flush=True)
    ac = ["-c:a", "copy"] if codec in ("aac", "mp3", "ac3", "eac3", "alac") else ["-c:a", "aac", "-b:a", "192k"]
    if ac[1] != "copy": print(f"WARNING: audio codec '{codec}' cannot be stream-copied into mp4: re-encoded to AAC 192k", flush=True)
    return ac + ["-t", f"{nv / fr:.6f}"]
FR_M = eval(fps_m)   # frames out = min(master, floor(audio * fps)), the same rule as ks_plain_par (it was -shortest, which dropped 1 extra frame)
NOUT = n if a.video_only else (min(n, int(math.floor(audio_probe(a.audio)[1] * FR_M + 1e-3))) or n)
nw_ = (min(WR[1], n) - WR[0]) if a.write_range else NOUT
AU = ["-an"] if a.video_only else ["-i", a.audio, "-map", "0:v", "-map", "1:a"]
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps_m, "-i", "-"] + AU +
                     ["-c:v", "libx264", "-crf", os.environ.get("KS_PLAIN_CRF", "16"), "-preset", "medium"] + ks_cms.enc() + ["-frames:v", str(nw_)] + ([] if a.video_only else audio_args(a.audio, nw_, fps_m)) + [a.out], stdin=subprocess.PIPE)
last[0] = None; nk = len(K)
NKT = a.ks_total if a.ks_total >= 0 else nk; NTT = a.n_total if a.n_total >= 0 else n; SHG = a.k_base - a.ks_base + a.shift   # global KeySync length / master length / shift
KS_SHORT = NKT - SHG < NTT   # the end fade exists ONLY for a KeySync output that is really shorter than the master (never because a chunk was cut)
if KS_SHORT: print(f"WARNING: KeySync has {NKT} frames < master {NTT} (+ shift {SHG}): the last {NTT - NKT + SHG} frames fall back to the source with a 1 s fade; pad the KeySync inputs", flush=True)
NFADE = 0
for k, f in enumerate(frames(a.master, W, H)):
    if k < WR[0]: continue
    if k >= WR[1] or (not a.write_range and k >= NOUT): break
    f = f.copy(); i = kidx(k); m, _ = mask(k)
    if m is not None:
        ig = i + a.ks_base; fade = 1.0 if (not KS_SHORT or ig < NKT - 25) else max(0.0, (NKT - ig) / 25.0)   # r16: only if KeySync is SHORTER than the master (pad the inputs); 1 s fade as a last resort
        NFADE += fade < 1.0
        kf = scale_fix(warp_ks(K[int(np.clip(i, 0, nk - 1))].astype(np.float32), k, i), k)
        if not a.no_cm:
            lab = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB); lab0 = lab.copy() if SHW else None
            if a.cm_lonly: lab[..., 0] = lab[..., 0] * gain[0] + off[0]
            else: lab = lab * gain + off
            if LFM is not None:
                d_ = cv2.resize(LFM[min(k, n - 1)], (s, s), interpolation=cv2.INTER_CUBIC)
                if a.cm_lonly: lab[..., 0] += d_[..., 0]
                else: lab += d_
            if SHW:   # fade the correction out in the darks (by the raw lightness)
                w_ = np.clip((lab0[..., 0] - SHW[0]) / max(SHW[1] - SHW[0], 1e-3), 0, 1)[..., None]; lab = lab0 + w_ * (lab - lab0)
            kf = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)
        if SHARP > 0:
            yk = cv2.cvtColor(kf.astype(np.float32), cv2.COLOR_RGB2YCrCb); yk[..., 0] += SHARP * (yk[..., 0] - cv2.GaussianBlur(yk[..., 0], (0, 0), max(0.6, s / 512.0)))
            kf = np.clip(cv2.cvtColor(yk, cv2.COLOR_YCrCb2RGB), 0, 255)
        if a.tex > 0 or a.cavity > 0: kf = fix_ks(kf, i, m, crop(f, k))
        roi = crop(f, k).astype(np.float32); mm2 = m * fade
        if a.neck_follow > 0: roi = neck_warp(roi, k, i)
        p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]; iod = float(np.linalg.norm(p[33] - p[263])) if np.isfinite(p).all() else IODM
        if UP is not None and np.isfinite(p[[33, 263, 133, 362, 152]]).all():
            em = p[[33, 263, 133, 362]].mean(0) - BXY(k); ax = (p[152] - BXY(k)) - em; L2 = float(ax @ ax)
            if L2 > 1:
                t_ = ((GX - em[0]) * ax[0] + (GY - em[1]) * ax[1]) / L2
                mm2 = mm2 * np.clip((t_ - UP[0]) / (UP[1] - UP[0]), 0, 1).astype(np.float32)
        if a.keep_nose > 0 and np.isfinite(p[NOSE]).all():
            nmk = np.zeros((s, s), np.uint8); cv2.fillPoly(nmk, [hull(p[NOSE] - BXY(k))], 1)
            mm2 = mm2 * (1 - cv2.GaussianBlur(nmk.astype(np.float32), (0, 0), max(1.0, a.keep_nose * iod)))
        mm = mm2[..., None]; comp = np.clip(roi * (1 - mm) + kf * mm + 0.5, 0, 255)
        if a.diag and k % 2 == 0:
            mz = mouth_zone(k, i, iod); band = (mm2 > 0.5) & (mm2 < 0.92) & ~mz; core = (mm2 > 0.97) & ~mz
            if band.sum() > 200:
                lc, ls_ = lab_of(comp), lab_of(roi); dl = (lc[band] - ls_[band]).mean(0)
                DG.append([k, *dl.tolist(), hf_energy(cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), core) / max(hf_energy(cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), core), 1e-6)])
        crop(f, k)[:] = comp.astype(np.uint8)
    w.stdin.write(f.tobytes())
if a.diag and DG:
    dg = np.array(DG, np.float64); de = np.linalg.norm(dg[:, 1:4], axis=1)
    slow = np.stack([np.convolve(dg[:, c], np.ones(13) / 13, "same") for c in (1, 2, 3)], 1)
    out_ = {"edge_dE_mean": round(float(de.mean()), 3), "edge_dL_mean": round(float(dg[:, 1].mean()), 3), "edge_da_mean": round(float(dg[:, 2].mean()), 3), "edge_db_mean": round(float(dg[:, 3].mean()), 3),
            "drift_std_L": round(float(slow[6:-6, 0].std()), 3), "drift_std_ab": round(float(np.linalg.norm(slow[6:-6, 1:3].std(0))), 3), "hf_ratio_core": round(float(np.nanmedian(dg[:, 4])), 3)}
    json.dump(out_, open(a.diag, "w")); print("diag", json.dumps(out_), flush=True)
if NFADE: print(f"WARNING: end fade KeySync -> source on {NFADE} frames (global KeySync {NKT} frames, master {NTT}, shift {SHG})", flush=True)
w.stdin.close(); w.wait(); print(f"plain paste -> {a.out} ({n} frames; cavity lifted in {NT[0]}, texture on {NT[1]})", flush=True)
