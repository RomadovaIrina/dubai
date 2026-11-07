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
import argparse, json, os, subprocess, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ks_cms

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser()
for k in ("ks", "master", "box", "lm", "audio", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--others", default=""); ap.add_argument("--shift", type=int, default=0); ap.add_argument("--dil", type=float, default=0.10); ap.add_argument("--down", type=float, default=0.15)
ap.add_argument("--no-cm", action="store_true")
ap.add_argument("--ks-lm", default="", help="landmarks of the raw KeySync output (512 grid, its frame index): the GENERATED lips for --tex / --cavity")
ap.add_argument("--tex", type=float, default=2.0, help="r15 stubble / pores: the source's luma high frequencies (gaussian sigma px), DIS-flow-warped onto KeySync's "
                "geometry, replace KeySync's inside the face mask outside the generated lips (dilated 0.15 IOD); 0 = off; r16: default 2 (Irina 07.11: old03 better)")
ap.add_argument("--cavity", type=float, default=0.0, help="r15 black-hole mouth: lift strength (0..1) of the near-black inner-mouth pixels toward the source's open-mouth "
                "interior tone (L 30th percentile + mean a / b of the source's open frames); 0 = off")
a = ap.parse_args()
b = json.load(open(a.box)); x0, y0, s, W, H = b["x"], b["y"], b["side"], b["W"], b["H"]
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM))
OT = np.load(a.others)["others"] if a.others else None
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


BOXF = np.minimum.outer(ramp(y0 == 0, y0 + s == H), ramp(x0 == 0, x0 + s == W)).astype(np.float32)
last = [None]


def mask(k):
    i = int(J[min(k, len(J) - 1)]); p = LM[min(i, len(LM) - 1)]
    if not np.isfinite(p).all(): return last[0], None
    iod = float(np.linalg.norm(p[33] - p[263])); q = p - [x0, y0]; mk = np.zeros((s, s), np.uint8)
    hp = hull(q[OVAL]); cv2.fillPoly(mk, [hp], 1); cv2.fillPoly(mk, [hp + np.array([0, int(round(a.down * iod))], np.int32)], 1)
    mk = cv2.dilate(mk, el(a.dil * iod))
    if OT is not None:
        for f2 in OT[min(i, len(OT) - 1)]:
            if np.isfinite(f2).all():
                om = np.zeros((s, s), np.uint8); cv2.fillPoly(om, [hull(f2[OVAL] - [x0, y0])], 1)
                mk[cv2.dilate(om, el(0.08 * float(np.linalg.norm(f2[33] - f2[263])))) > 0] = 0
    m = cv2.GaussianBlur(mk.astype(np.float32), (0, 0), max(1.0, 0.04 * iod)) * BOXF
    lip = np.zeros((s, s), np.uint8); cv2.fillPoly(lip, [hull(q[LIPS])], 1); lip = cv2.dilate(lip, el(0.15 * iod))
    last[0] = m; return m, (mk > 0) & (lip == 0)


kidx = lambda k: k - a.shift   # KeySync frame shown at master frame k
gain = np.ones(3, np.float32); off = np.zeros(3, np.float32); CAV = None
is_open = lambda p: np.isfinite(p).all() and np.linalg.norm(p[61] - p[291]) > 5 and np.linalg.norm(p[13] - p[14]) / np.linalg.norm(p[61] - p[291]) > 0.15
if not a.no_cm or a.cavity > 0:   # pass 1: the global colour match on the skin; the source's open-mouth interior tone (for --cavity)
    SK, SM, SC = [], [], []
    for k, f in enumerate(frames(a.master, W, H)):
        if a.cavity > 0 and k % 3 == 0:
            p = LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]
            if is_open(p):
                mk = np.zeros((H, W), np.uint8); cv2.fillPoly(mk, [hull(p[INNER])], 1); mk = cv2.erode(mk, np.ones((3, 3), np.uint8))
                if mk.sum() > 30: SC.append(cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[mk > 0])
        if a.no_cm or k % 5 or not (0 <= kidx(k) < len(K)): continue
        _, sk = mask(k)
        if sk is None or sk.sum() < 500: continue
        lk = cv2.cvtColor(K[kidx(k)].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[sk]; lm_ = cv2.cvtColor(f[y0:y0 + s, x0:x0 + s].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[sk]
        SK.append(lk[:: 7]); SM.append(lm_[:: 7])
    if not a.no_cm:
        SK, SM = np.concatenate(SK), np.concatenate(SM)
        gain = np.clip(SM.std(0) / np.maximum(SK.std(0), 1e-3), 0.9, 1.1).astype(np.float32); off = (SM.mean(0) - SK.mean(0) * gain).astype(np.float32)
        print(f"global colour match (Lab): gain {gain.round(3).tolist()} offset {off.round(2).tolist()}", flush=True)
    if a.cavity > 0 and SC:
        SC = np.concatenate(SC); CAV = (float(np.percentile(SC[:, 0], 30)), SC[:, 1:].mean(0).astype(np.float32))
        print(f"source open-mouth interior: L p30 {CAV[0]:.1f}, a/b {CAV[1].round(1).tolist()} ({len(SC)} px)", flush=True)
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
        fl = DIS.calc(cv2.resize(gk, (h2, h2), interpolation=cv2.INTER_AREA), cv2.resize(gs, (h2, h2), interpolation=cv2.INTER_AREA), None)
        fl = cv2.resize(fl, (s, s)) * (s / h2)
        yk = cv2.cvtColor(kf.astype(np.float32), cv2.COLOR_RGB2YCrCb); ys = cv2.cvtColor(sc, cv2.COLOR_RGB2YCrCb)[..., 0]
        ws = cv2.remap(ys, GX + fl[..., 0], GY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lx = np.zeros((s, s), np.uint8)
        lp = kp if kp is not None and np.isfinite(kp).all() else LM[min(int(J[min(i + a.shift, len(J) - 1)]), len(LM) - 1)] - [x0, y0]
        if np.isfinite(lp).all():
            iod = float(np.linalg.norm(lp[33] - lp[263])); cv2.fillPoly(lx, [hull(lp[LIPS_O])], 1); lx = cv2.dilate(lx, el(0.15 * iod))
            lx = cv2.GaussianBlur(lx.astype(np.float32), (0, 0), max(1.0, 0.05 * iod))
        T = m * (1 - lx)
        hp = lambda img: img - cv2.GaussianBlur(img, (0, 0), a.tex)
        yk[..., 0] += T * (hp(ws) - hp(yk[..., 0])); NT[1] += 1
        kf = np.clip(cv2.cvtColor(yk, cv2.COLOR_YCrCb2RGB), 0, 255)
    return kf
fps_m = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip()
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", fps_m, "-i", "-", "-i", a.audio, "-map", "0:v", "-map", "1:a",
                      "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + ["-c:a", "aac", "-b:a", "192k", "-shortest", a.out], stdin=subprocess.PIPE)
last[0] = None; nk = len(K)
for k, f in enumerate(frames(a.master, W, H)):
    f = f.copy(); i = kidx(k); m, _ = mask(k)
    if m is not None:
        fade = 1.0 if i < nk - 8 else max(0.0, (nk - i) / 8.0)   # KeySync's output ends a few frames early: fade to the master
        kf = K[int(np.clip(i, 0, nk - 1))].astype(np.float32)
        if not a.no_cm:
            lab = cv2.cvtColor(kf / 255, cv2.COLOR_RGB2LAB) * gain + off; kf = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255, 0, 255)
        if a.tex > 0 or a.cavity > 0: kf = fix_ks(kf, i, m, f[y0:y0 + s, x0:x0 + s])
        roi = f[y0:y0 + s, x0:x0 + s].astype(np.float32); mm = (m * fade)[..., None]
        f[y0:y0 + s, x0:x0 + s] = np.clip(roi * (1 - mm) + kf * mm + 0.5, 0, 255).astype(np.uint8)
    w.stdin.write(f.tobytes())
w.stdin.close(); w.wait(); print(f"plain paste -> {a.out} ({n} frames; cavity lifted in {NT[0]}, texture on {NT[1]})", flush=True)
