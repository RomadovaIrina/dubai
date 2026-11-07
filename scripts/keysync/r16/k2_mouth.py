#!/usr/bin/env python3
"""r16 K2: the black-hole mouth on the RAW KeySync 512 crop (before ks_plain.py). Inside the GENERATED inner-lip hull (KeySync's own landmarks),
on open frames only (opening ramp gap/width 0.10 -> 0.18), one of:
  bank   a SOURCE interior bank: open, sharp source frames of the same speaker (the static ks_prep crop of the master, 512 grid); per generated
         frame the nearest source mouth by inner-lip shape (20 points, normalised by the corner axis / width) + opening, with hysteresis (keep the
         previous pick unless a new one is > --hyst better) so the teeth do not jump; the source interior is warped onto the generated inner lip by
         an affine fit of the 20 inner-lip points, colour = the source's own (it is the same face), mixed --strength over KeySync's interior.
  synth  a synthetic interior from the generated landmarks + the source mouth tone: KeySync's interior lifted toward the source's open-interior tone
         (lighter rim, darker throat, as ks_plain --cavity) + an upper-teeth band hanging from the upper inner lip (height min(0.38 gap, 0.10 width),
         source teeth tone = L p92 of the source open interior, fading at the corners, faint vertical tooth gaps), KeySync's high frequencies kept.
  lift   = ks_plain --cavity (for reference).
Source pixels at the master frame of the SAME time are never needed (the bank is built from the whole video); frames where the source mouth is
occluded / blurred (Laplacian var < 0.35 x the median, or a finger: --others hull overlap) are not banked.
    python k2_mouth.py --mode bank --ks KS.mp4 --ks-lm KS.mediapipe.npz --master master_25fps.mp4 --box box.json --lm source_lm_F.npz [--others faces2.npz]
           --out KS_fixed.mp4 [--strength 0.85] [--json stats.json]"""
import argparse, json, subprocess, sys, os
import numpy as np, cv2

INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]   # upper inner lip, left corner -> right corner
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
ap = argparse.ArgumentParser()
for k in ("mode", "ks", "ks_lm", "master", "box", "lm", "out"): ap.add_argument("--" + k.replace("_", "-"), required=True, dest=k)
ap.add_argument("--others", default=""); ap.add_argument("--strength", type=float, default=0.85); ap.add_argument("--hyst", type=float, default=1.0)
ap.add_argument("--json", default=""); ap.add_argument("--detail", type=float, default=0.0, help="bank: source interior luma HF on every open frame (#6 flat teeth)"); ap.add_argument("--teeth", type=float, default=0.8, help="synth: upper-teeth band opacity")
a = ap.parse_args()
b = json.load(open(a.box)); x0, y0, s, W, H = b["x"], b["y"], b["side"], b["W"], b["H"]; S = 512; sc = S / s
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM))
KL = np.load(a.ks_lm)["lm"][..., :2].astype(np.float32)
OT = np.load(a.others)["others"] if a.others else None
hull = lambda pts: np.round(cv2.convexHull(pts.astype(np.float32))[:, 0]).astype(np.int32)


def frames(path, w, h, vf=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-vf", vf] if vf else []) + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    while True:
        r = p.stdout.read(w * h * 3)
        if len(r) < w * h * 3: break
        yield np.frombuffer(r, np.uint8).reshape(h, w, 3)


def opening(p):
    if not np.isfinite(p).all(): return 0.0
    mw = float(np.linalg.norm(p[61] - p[291])); return float(np.linalg.norm(p[13] - p[14])) / mw if mw > 5 else 0.0


def desc(p):   # inner-lip shape in the mouth frame: centred, rotated to the corner axis, / mouth width; + 2 x opening
    c = p[INNER].mean(0); ax = p[291] - p[61]; mw = float(np.linalg.norm(ax)); ax = ax / mw; R = np.array([[ax[0], ax[1]], [-ax[1], ax[0]]])
    q = (p[INNER] - c) @ R.T / mw; return np.concatenate([q.ravel(), [2 * opening(p)]]).astype(np.float32)


def inner_mask(p, er=1):
    m = np.zeros((S, S), np.uint8); cv2.fillPoly(m, [hull(p[INNER])], 1)
    return cv2.erode(m, np.ones((3, 3), np.uint8), iterations=er) if er else m


# ---- pass 1: the source bank (512 crop grid), the source interior / teeth tone
BANK_P, BANK_D, BANK_I, BANK_B, SC = [], [], [], [], []; lapv = []
for k, f in enumerate(frames(a.master, W, H)):
    i = int(J[min(k, len(J) - 1)]); p = LM[min(i, len(LM) - 1)]
    o = opening(p)
    if o < 0.12: continue
    q = (p - [x0, y0]) * sc
    if (q < 0).any() or (q >= S).any(): continue
    if OT is not None:   # a finger / the child's face over the mouth: skip
        bad = False; mh = inner_mask(q, 0)
        for f2 in OT[min(i, len(OT) - 1)]:
            if np.isfinite(f2).all():
                om = np.zeros((S, S), np.uint8); cv2.fillPoly(om, [hull((f2[OVAL] - [x0, y0]) * sc)], 1)
                if (om & mh).sum() > 0.05 * mh.sum(): bad = True
        if bad: continue
    crop = cv2.resize(f[y0:y0 + s, x0:x0 + s], (S, S), interpolation=cv2.INTER_LANCZOS4)
    xs, ys = q[INNER, 0], q[INNER, 1]; mw = float(np.linalg.norm(q[61] - q[291])); pad = int(0.35 * mw)
    bx0, by0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad); bx1, by1 = min(S, int(xs.max()) + pad), min(S, int(ys.max()) + pad)
    g = cv2.cvtColor(crop[by0:by1, bx0:bx1], cv2.COLOR_RGB2GRAY); lv = float(cv2.Laplacian(g, cv2.CV_32F).var())
    mk = inner_mask(q); Lm = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[mk > 0] if mk.sum() > 30 else None
    if Lm is not None and k % 2 == 0: SC.append(Lm)
    BANK_B.append(float(np.percentile(Lm[:, 0], 90)) if Lm is not None else 0.0)
    BANK_P.append((crop[by0:by1, bx0:bx1].copy(), (bx0, by0), q)); BANK_D.append(desc(q)); BANK_I.append(k); lapv.append(lv)
lapv = np.array(lapv); keep = lapv >= 0.35 * np.median(lapv) if len(lapv) else np.zeros(0, bool)
BB = np.array(BANK_B)
if keep.sum() > 20: keep &= BB >= np.percentile(BB[keep], 50)   # the brighter half: frames where the source shows teeth / a lit interior
BANK_P = [x for x, kk in zip(BANK_P, keep) if kk]; BANK_D = np.array([x for x, kk in zip(BANK_D, keep) if kk]); BANK_I = [x for x, kk in zip(BANK_I, keep) if kk]
SC = np.concatenate(SC) if SC else np.zeros((0, 3), np.float32)
CAV_L = float(np.percentile(SC[:, 0], 30)) if len(SC) else 20.0; CAV_AB = SC[:, 1:].mean(0) if len(SC) else np.array([12, 6], np.float32)
TT = SC[SC[:, 0] > np.percentile(SC[:, 0], 85)] if len(SC) else np.zeros((0, 3))
TEETH = (float(np.percentile(SC[:, 0], 92)), TT[:, 1:].mean(0) if len(TT) else np.array([2, 8], np.float32)) if len(SC) else (70.0, np.array([2, 8], np.float32))
print(f"bank {len(BANK_I)} source open frames (of {len(keep)} open, sharp kept); interior L p30 {CAV_L:.1f} a/b {np.round(CAV_AB, 1).tolist()}; "
      f"teeth L {TEETH[0]:.1f} a/b {np.round(TEETH[1], 1).tolist()}", flush=True)


def lift(kf, p, w):   # ks_plain --cavity, on one frame (float RGB 0..255), weight w
    mc = inner_mask(p)
    if mc.sum() <= 30: return kf
    dn = cv2.distanceTransform(mc, cv2.DIST_L2, 3); dn = dn / max(float(dn.max()), 1.0)
    mc = cv2.GaussianBlur(mc.astype(np.float32), (0, 0), 1.5); lab = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB)
    ww = np.clip((CAV_L - lab[..., 0]) / max(CAV_L, 1.0), 0, 1) * mc * w * (1 - 0.65 * dn)
    lab[..., 0] += ww * (CAV_L - lab[..., 0]); lab[..., 1:] += ww[..., None] * (CAV_AB - lab[..., 1:])
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)


GX, GY = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))


def synth(kf, p, w):
    kf = lift(kf, p, 0.5 * w)   # a lighter rim only; the throat stays dark under the teeth
    o = opening(p); mw = float(np.linalg.norm(p[61] - p[291])); gap = float(np.linalg.norm(p[13] - p[14]))
    th = min(0.28 * gap, 0.075 * mw)
    if th < 1.5: return kf
    mc = inner_mask(p, 1).astype(np.float32)
    up = p[UPPER]; ax = (p[291] - p[61]) / mw; nrm = np.array([-ax[1], ax[0]], np.float32)
    if nrm[1] < 0: nrm = -nrm   # pointing down (into the mouth)
    # distance below the upper inner lip along the normal, and the position along the mouth (0 = corner, 1 = the other corner)
    ts = ((np.stack([GX, GY], -1) - p[61]) @ ax) / mw
    ux, uy = ((up - p[61]) @ ax) / mw, (up - p[61]) @ nrm; o_ = np.argsort(ux)
    xs = np.clip(ts, 0, 1); ul = np.interp(xs.ravel(), ux[o_], uy[o_]).reshape(S, S)
    dv = (np.stack([GX, GY], -1) - p[61]) @ nrm - ul
    band = np.clip((th + 0.8 - dv) / 1.6, 0, 1) * np.clip((dv + 1.0) / 1.5, 0, 1)   # a crisp-edged strip right under the upper lip
    cen = np.clip(1 - np.abs(ts - 0.5) / 0.36, 0, 1) ** 0.7   # fade toward the corners (teeth recede into shadow)
    gaps = 1 - 0.10 * (np.cos(2 * np.pi * (ts - 0.5) / 0.115) > 0.93)   # faint vertical gaps (incisors ~ 0.115 mouth width)
    shade = 1 - 0.25 * np.clip(dv / max(th, 1), 0, 1)   # brighter right under the lip, darker at the edge
    al = band * cen * mc * a.teeth * w * np.clip((o - 0.10) / 0.10, 0, 1)
    lab = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB); hfL = lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 2.0)
    tL = TEETH[0] * shade * gaps + 0.5 * hfL
    al = al * np.clip((tL - lab[..., 0]) / 6, 0, 1)   # brighten only
    lab[..., 0] += al * (tL - lab[..., 0]); lab[..., 1:] += al[..., None] * (TEETH[1] - lab[..., 1:])
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)


def dark_gate(kf, p):   # 1 when KeySync drew a black hole, 0 when it already shows teeth / a lit interior
    mc = inner_mask(p)
    if mc.sum() <= 30: return 0.0
    Li = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB)[..., 0][mc > 0]
    return float(np.clip((26 - Li.mean()) / 8, 0, 1) * np.clip((0.06 - (Li > 55).mean()) / 0.04, 0, 1))


def lighten(kf, tgt, al):   # move toward tgt only where tgt is brighter (keeps any teeth KeySync drew)
    L = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB)[..., 0]; Lt = cv2.cvtColor(np.clip(tgt, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    g = np.clip((Lt - L) / 6, 0, 1)[..., None] * al
    return kf * (1 - g) + tgt * g


last = [None]; ST = {"bank_switches": 0, "fixed": 0, "picks": []}


def bank(kf, p, w, i, wd=0.0):
    if not len(BANK_I): return kf
    d = desc(p); dist = np.linalg.norm(BANK_D - d, axis=1); j = int(np.argmin(dist))
    if last[0] is not None and dist[last[0]] <= dist[j] * (1 + a.hyst) + 1e-6: j = last[0]
    elif last[0] is not None: ST["bank_switches"] += 1
    last[0] = j; ST["picks"].append(BANK_I[j])
    patch, (bx0, by0), q = BANK_P[j]
    M, _ = cv2.estimateAffine2D((q[INNER] - [bx0, by0]).astype(np.float32), p[INNER].astype(np.float32), method=cv2.LMEDS)
    if M is None: return kf
    wp = cv2.warpAffine(patch, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    mc = cv2.GaussianBlur(inner_mask(p, 1).astype(np.float32), (0, 0), 1.2)
    al = (mc * a.strength * w)[..., None]
    out = lighten(kf, wp, al) if w > 0 else kf
    if a.detail > 0:   # #6 teeth detail: the source interior's luma high frequencies (tooth gaps / edges) on KeySync's flat teeth bar
        y = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2YCrCb); ys = cv2.cvtColor(wp, cv2.COLOR_RGB2YCrCb)[..., 0]
        hp = lambda im: im - cv2.GaussianBlur(im, (0, 0), 1.6)
        y[..., 0] += mc * a.detail * wd * hp(ys); out = np.clip(cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB), 0, 255)
    return out


n = 0; KF = list(frames(a.ks, S, S))
TP = []   # KeySync's OWN teeth tone: bright interior pixels of the frames where it does draw teeth (consistent with the rest of the video)
for i in range(0, len(KF), 2):
    p = KL[min(i, len(KL) - 1)]
    if opening(p) < 0.12: continue
    mc = inner_mask(p)
    if mc.sum() <= 30: continue
    lab = cv2.cvtColor(KF[i].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[mc > 0]
    if (lab[:, 0] > 55).mean() > 0.08: TP.append(lab[lab[:, 0] > 55])
if TP:
    TP = np.concatenate(TP); TEETH = (float(np.percentile(TP[:, 0], 60)), np.median(TP[:, 1:], 0).astype(np.float32))
else: TEETH = (float(np.clip(TEETH[0] + 20, 58, 72)), np.array([4.0, 11.0], np.float32))
print(f"teeth tone {'from KeySync' if len(TP) else 'fallback'}: L {TEETH[0]:.1f} a/b {np.round(TEETH[1], 1).tolist()}", flush=True)
fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", a.ks], capture_output=True, text=True).stdout.strip()
wr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "12",
                       "-preset", "medium", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
for i, f in enumerate(KF):
    p = KL[min(i, len(KL) - 1)]; o = opening(p); w = float(np.clip((o - 0.10) / 0.08, 0, 1))
    kf = f.astype(np.float32)
    wd = w
    if w > 0 and a.mode != "lift": w *= dark_gate(kf, p)
    if a.mode == "bank" and a.detail > 0 and wd > 0:
        ST["fixed"] += 1; kf = bank(kf, p, w, i, wd)
    elif w > 0:
        ST["fixed"] += 1
        kf = lift(kf, p, 0.8 * w) if a.mode == "lift" else synth(kf, p, w) if a.mode == "synth" else bank(kf, p, w, i)
    else: last[0] = None if a.mode == "bank" and o < 0.06 else last[0]
    wr.stdin.write(np.clip(kf + 0.5, 0, 255).astype(np.uint8).tobytes())
wr.stdin.close(); wr.wait()
ST["frames"] = len(KF); ST["bank_size"] = len(BANK_I); ST["picks"] = ST["picks"][:: max(1, len(ST["picks"]) // 40)]
print(f"k2_mouth {a.mode} -> {a.out}: {ST['fixed']} of {len(KF)} frames touched, bank switches {ST['bank_switches']}", flush=True)
if a.json: json.dump(ST, open(a.json, "w"))
