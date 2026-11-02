#!/usr/bin/env python3
"""r12 KS 'лаги': frame-to-frame mouth motion spikes of a final (source fps) and where they sit vs KeySync's 13-frame keyframe windows.
Per frame: mouth opening (lm 13-14 / IOD) and the lip-centre position (/ IOD); jerk = |2nd difference|; a spike = jerk > K x the median
(same metric on the source = the natural level). Each output frame t -> KeySync 25 fps frame round(t * 25) - shift -> phase mod 13.
    ks_lag.py --lm FINAL.mediapipe.npz --src-lm SRC.mediapipe.npz --fps 30 --shift S --out PREFIX"""
import argparse, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--lm"); ap.add_argument("--src-lm"); ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--shift", type=int, default=0); ap.add_argument("--k", type=float, default=4.0); ap.add_argument("--out"); a = ap.parse_args()
def feats(p):
    d = np.load(p); lm = d["lm"][..., :2].astype(np.float64); iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1)
    op = np.linalg.norm(lm[:, 13] - lm[:, 14], axis=1) / iod; c = lm[:, [61, 291, 13, 14]].mean(1) / iod[:, None]
    nose = lm[:, 1] / iod[:, None]; c = c - nose   # lip centre relative to the nose: head motion removed
    return op, c
def jerk(op, c):
    jo = np.abs(np.diff(op, 2)); jc = np.linalg.norm(np.diff(c, 2, axis=0), axis=1); return np.pad(jo, (1, 1)), np.pad(jc, (1, 1))
op, c = feats(a.lm); sop, sc = feats(a.src_lm); n = min(len(op), len(sop))
jo, jc = jerk(op[:n], c[:n]); sjo, sjc = jerk(sop[:n], sc[:n])
thr_o, thr_c = a.k * np.median(sjo[sjo > 0]), a.k * np.median(sjc[sjc > 0])
sp = (jo > thr_o) | (jc > thr_c); ssp = (sjo > thr_o) | (sjc > thr_c)
t = np.arange(n) / a.fps; kf = np.round(t * 25).astype(int) - a.shift; ph = np.mod(kf, 13)
h = np.bincount(ph[sp], minlength=13); hn = np.bincount(ph, minlength=13)
r = {"frames": int(n), "spikes": int(sp.sum()), "spikes_source": int(ssp.sum()), "spike_rate": round(float(sp.mean()), 4), "source_rate": round(float(ssp.mean()), 4),
     "phase_hist": h.tolist(), "phase_rate": [round(float(x), 4) for x in h / np.maximum(hn, 1)],
     "worst_s": [round(float(x), 2) for x in t[np.argsort(-(jo / thr_o + jc / thr_c))[:12]]]}
json.dump(r, open(a.out + ".json", "w"), indent=1); print(json.dumps({k: v for k, v in r.items()}))
f, ax = plt.subplots(2, 1, figsize=(14, 6))
ax[0].plot(t, jc[:n] / thr_c, lw=0.6, label="KeySync final"); ax[0].plot(t, sjc / thr_c, lw=0.6, alpha=0.6, label="source")
for b in np.where(np.mod(kf, 13) == 0)[0]: ax[0].axvline(t[b], color="r", lw=0.3, alpha=0.4)
ax[0].axhline(1, color="k", lw=0.5); ax[0].set_ylabel("lip-centre jerk / thr"); ax[0].legend(); ax[0].set_title("red = KeySync window boundaries (every 13 frames at 25 fps)")
ax[1].bar(range(13), r["phase_rate"]); ax[1].set_xlabel("phase in the 13-frame window"); ax[1].set_ylabel("spike rate")
plt.tight_layout(); plt.savefig(a.out + ".png", dpi=90)
