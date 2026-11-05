#!/usr/bin/env python3
"""r15 F3: mouth-motion JUDDER of a final ("worms"): lip-contour acceleration relative to the head, at the video's own fps (30), and the
5 Hz periodicity a 25 -> 30 fps retime can leave (one duplicated / nearest-picked step every 6 output frames).
Input: MediaPipe landmarks of the FINAL (scripts/quality/mp_landmarks.py on the final video) and of the SOURCE (same tool or source_lm_F).
Per frame: a similarity transform (eyes corners, nose bridge, nose tip) to the video's median face removes the head motion; the 20 outer +
20 inner lip points in that frame, / IOD. acc[t] = |x[t+1] - 2 x[t] + x[t-1]| averaged over the points; jerk = |acc[t+1] - acc[t]|.
Printed: mean acc / jerk (x 1000 IOD) for the final and the source, their ratio, and the share of the 1-14 Hz power of acc(t) in 4.5-5.5 Hz
(a pure 25 -> 30 judder) for both.
    python scripts/r15/lip_jitter.py --lm FINAL_lm.npz --src-lm SOURCE_lm.npz --fps 29.97 --label x [--json out.jsonl]"""
import argparse, json
import numpy as np, cv2
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
RIG = [33, 133, 362, 263, 168, 6, 197, 4]


def lips_stab(lm):
    lm = lm[..., :2].astype(np.float64); ok = np.isfinite(lm[:, RIG + OUTER + INNER]).all((1, 2)); ref = np.nanmedian(lm[ok][:, RIG], 0)
    out = np.full((len(lm), 40, 2), np.nan)
    for k in np.flatnonzero(ok):
        M, _ = cv2.estimateAffinePartial2D(lm[k, RIG].astype(np.float32), ref.astype(np.float32), method=cv2.LMEDS)
        if M is None: continue
        q = lm[k, OUTER + INNER] @ M[:, :2].T + M[:, 2]; out[k] = q
    iod = float(np.linalg.norm(ref[0] - ref[3])); return out / max(iod, 1.0)


def stats(x, fps):
    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=2).mean(1); jerk = np.abs(np.diff(acc))
    v = np.nan_to_num(acc - np.nanmean(acc)); f = np.fft.rfftfreq(len(v), 1 / fps); P = np.abs(np.fft.rfft(v)) ** 2
    band = (f >= 1) & (f <= 14); pk = (f >= 4.5) & (f <= 5.5)
    return {"acc_x1000": round(1000 * float(np.nanmean(acc)), 3), "jerk_x1000": round(1000 * float(np.nanmean(jerk)), 3),
            "share_5Hz": round(float(P[pk].sum() / max(P[band].sum(), 1e-12)), 4), "flat_share_5Hz": round(float(pk.sum() / max(band.sum(), 1)), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--src-lm", required=True); ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--label", default=""); ap.add_argument("--json")
    a = ap.parse_args(); F = lips_stab(np.load(a.lm)["lm"]); S = lips_stab(np.load(a.src_lm)["lm"]); n = min(len(F), len(S))
    rf, rs = stats(F[:n], a.fps), stats(S[:n], a.fps)
    r = {"label": a.label, "final": rf, "source": rs, "acc_ratio": round(rf["acc_x1000"] / max(rs["acc_x1000"], 1e-6), 3), "jerk_ratio": round(rf["jerk_x1000"] / max(rs["jerk_x1000"], 1e-6), 3)}
    print("== lip_jitter", json.dumps(r), flush=True)
    if a.json: open(a.json, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
