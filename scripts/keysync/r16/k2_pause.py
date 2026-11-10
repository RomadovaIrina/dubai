#!/usr/bin/env python3
"""r16 K2: lips that rest parted in dub pauses (side effect of the ref_teeth identity frame). Pause = a run of >= --min-run master frames where
the dub voice is > --db below its peak. Two arms (Irina's rule: no KeySync <-> source fade over the visible face):
  weights  -> <out>.npy: per MASTER frame keep-weight for ks_plain --lip-keep (arm a: the SOURCE lips show through the paste mask, lips region
              only, where the pause frame's source mouth is closed too; eased --ease frames in / out)
  warp     -> <out>.mp4: arm b, on the RAW KeySync 512 crop: KeySync's OWN lips closed geometrically. Per column across the mouth, a piecewise-linear
              vertical warp: the band [upper outer lip - margin, upper inner lip] is stretched down to the closing line and [lower inner lip,
              lower outer lip + margin] up to it (closing line = the gap's middle; margin 0.35 mouth width keeps the skin moving with the lip);
              closure fraction eased --ease frames in / out over each pause, no pixel ever comes from the source.
    python k2_pause.py --mode warp --ks KS.mp4 --ks-lm KS.npz --wav dub16k.wav --shift S --out KSclosed.mp4
    python k2_pause.py --mode weights --src-lm source_crop.npz(or source_lm_F.npz) --wav dub16k.wav --n N --out keep.npy"""
import argparse, json, subprocess, wave
import numpy as np, cv2

UO = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]   # upper outer lip, corner -> corner
UI = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]  # upper inner
LI = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # lower inner
LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291] # lower outer
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["warp", "weights"]); ap.add_argument("--wav", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ks", default=""); ap.add_argument("--ks-lm", default="", dest="ks_lm"); ap.add_argument("--shift", type=int, default=0)
ap.add_argument("--src-lm", default="", dest="src_lm"); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--db", type=float, default=35.0); ap.add_argument("--min-run", type=int, default=6, dest="min_run"); ap.add_argument("--ease", type=int, default=4)
ap.add_argument("--closed", type=float, default=0.05, help="weights: source gap/width below this = closed")
ap.add_argument("--pause", type=int, default=1, help="warp: close the lips in dub pauses (arm b); 0 = only the cap")
ap.add_argument("--cap", type=float, default=0.0, help="warp: cap the generated lip gap at CAP x the source speaker's --cap-q gap (0 = off; r16d black holes)")
ap.add_argument("--cap-q", type=float, default=0.9, dest="cap_q"); ap.add_argument("--cap-max", type=float, default=0.6, dest="cap_max")
ap.add_argument("--cap-src-lm", default="", dest="cap_src_lm", help="the SOURCE crop landmarks (crop_lm_<tag>.npz) for the speaker's opening range"); a = ap.parse_args()


def pause_weight(n):
    w = wave.open(a.wav); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; hop = w.getframerate() // 25
    db = np.array([20 * np.log10(np.sqrt((x[i * hop:(i + 1) * hop] ** 2).mean()) + 1e-9) if i * hop < len(x) else -120 for i in range(n)])
    sil = db < db.max() - a.db; out = np.zeros(n, np.float32); i = 0
    while i < n:
        if not sil[i]: i += 1; continue
        j = i
        while j < n and sil[j]: j += 1
        if j - i >= a.min_run:
            for k in range(i, j): out[k] = min(1.0, (k - i + 1) / a.ease, (j - k) / a.ease)
        i = j
    return out


op = lambda p: float(np.linalg.norm(p[13] - p[14]) / max(np.linalg.norm(p[61] - p[291]), 1)) if np.isfinite(p).all() else 1.0
if a.mode == "weights":
    z = np.load(a.src_lm); L = z["lm"][..., :2]; J = z["j"] if "j" in z.files else np.arange(len(L)); n = a.n or len(J)
    pw = pause_weight(n); cl = np.array([op(L[min(int(J[min(k, len(J) - 1)]), len(L) - 1)]) < a.closed for k in range(n)], np.float32)
    # the closed-source condition eased too (no hard switch inside a pause)
    ce = np.convolve(cl, np.ones(2 * a.ease + 1) / (2 * a.ease + 1), mode="same"); w = pw * np.clip((ce - 0.5) * 2, 0, 1)
    np.save(a.out, w.astype(np.float32)); print("K2_PAUSE weights", json.dumps({"frames": n, "pause_frames": int((pw > 0).sum()), "keep_frames": int((w > 0.01).sum())}))
    raise SystemExit
KL = np.load(a.ks_lm)["lm"][..., :2].astype(np.float32)
fr = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", a.ks], capture_output=True, text=True).stdout.strip()
nk = int(fr); pw = pause_weight(nk + max(a.shift, 0) + 2)
wk = np.array([pw[i + a.shift] if 0 <= i + a.shift < len(pw) else 0 for i in range(nk)], np.float32)   # KeySync frame i is shown at master frame i + shift
if not a.pause: wk[:] = 0
NCAP = 0
if a.cap > 0 and a.cap_src_lm:   # r16d: the black hole = a generated opening far wider than the speaker ever opens -> close it to CAP x her q-gap
    SL = np.load(a.cap_src_lm)["lm"][..., :2].astype(np.float32); sg = np.array([op(q) for q in SL]); sg = sg[np.isfinite(sg) & (sg > 0.03) & (sg < 1)]
    capv = a.cap * float(np.percentile(sg, a.cap_q * 100)) if len(sg) > 20 else 1.0
    kg = np.array([op(KL[min(i, len(KL) - 1)]) for i in range(nk)]); kg[~np.isfinite(kg)] = 0
    wc = np.where(kg > capv, np.clip(1 - capv / np.maximum(kg, 1e-3), 0, a.cap_max), 0).astype(np.float32)
    wc = np.convolve(wc, np.array([1, 2, 3, 2, 1], np.float32) / 9, "same"); NCAP = int((wc > 0.02).sum())
    print("K2_PAUSE cap " + json.dumps({"cap_gap": round(capv, 3), "src_q": a.cap_q, "frames_capped": NCAP, "mean_close": round(float(wc[wc > 0.02].mean()) if NCAP else 0, 3)}), flush=True)
    wk = np.maximum(wk, wc)
S = 512; GY = np.arange(S, dtype=np.float32)
r = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.ks, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{S}x{S}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
nt = 0
for i in range(nk):
    f = np.frombuffer(r.stdout.read(S * S * 3), np.uint8).reshape(S, S, 3); p = KL[min(i, len(KL) - 1)]
    if wk[i] > 0 and np.isfinite(p).all() and op(p) > 0.02:
        mw = float(np.linalg.norm(p[61] - p[291])); mg = 0.35 * mw
        xs = np.arange(int(min(p[61][0], p[291][0])) + 1, int(max(p[61][0], p[291][0])))
        it = lambda idx: np.interp(xs, *zip(*sorted(zip(p[idx, 0], p[idx, 1]))))
        yuo, yui, yli, ylo = it(UO), it(UI), it(LI), it(LO)
        gap = np.clip(yli - yui, 0, None); ym = yui + gap / 2
        tu = yui + wk[i] * (ym - yui); tl = yli - wk[i] * (yli - ym)   # the inner contours' target rows
        mapy = np.tile(GY[:, None], (1, S)).copy(); mapx = np.tile(np.arange(S, dtype=np.float32)[None, :], (S, 1))
        for c, x in enumerate(xs):
            top, bot = yuo[c] - mg, ylo[c] + mg
            # target rows -> source rows (inverse map), piecewise linear: [top, tu] <- [top, yui]; [tl, bot] <- [yli, bot]; between tu and tl (closed: none)
            yy = GY
            m1 = (yy >= top) & (yy <= tu[c]); mapy[m1, x] = top + (yy[m1] - top) * (yui[c] - top) / max(tu[c] - top, 1e-3)
            m2 = (yy >= tl[c]) & (yy <= bot); mapy[m2, x] = yli[c] + (yy[m2] - tl[c]) * (bot - yli[c]) / max(bot - tl[c], 1e-3)
            m3 = (yy > tu[c]) & (yy < tl[c]); mapy[m3, x] = np.where(yy[m3] < (tu[c] + tl[c]) / 2, yui[c], yli[c])   # a remaining gap: lip edges
        mapy = cv2.GaussianBlur(mapy, (0, 0), 2.0)   # no column seams
        f = cv2.remap(f, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); nt += 1
    w.stdin.write(np.ascontiguousarray(f).tobytes())
w.stdin.close(); w.wait(); print("K2_PAUSE warp", json.dumps({"frames": nk, "pause_frames": int((wk > 0).sum()), "warped": nt}))
