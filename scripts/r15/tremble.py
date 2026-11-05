#!/usr/bin/env python3
"""r15 W: TREMBLE metric (why LS_RATE was rejected: "the face twitches / trembles"). MediaPipe 478 landmarks of each video (the cached
scripts/r14/mouth_open.py pass, <video>.mp_lm.npz), taken RELATIVE to a head anchor (eye corners 33 / 263 + nose bridge 168 / 6, so head
motion cancels) and normalised by the median inter-ocular distance. Per frame, |second temporal difference| of
  mouth = the lip ring (61 291 0 17 37 267 84 314 40 270 91 321 13 14)      chin = chin / jaw (152 148 377 176 400 149 378)
= jitter (smooth motion ~0, a back-and-forth one-frame wobble high). Reports the mean x 1000 and the share of frames above the SOURCE's
p95 ("shake share"); the first video named is the reference for the ratios (q2).
    python scripts/r15/tremble.py --src in.mp4 --video q2=a.mp4 tkf2=b.mp4 ... [--json out.json]"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r14"))
from mouth_open import landmarks

MOUTH = [61, 291, 0, 17, 37, 267, 84, 314, 40, 270, 91, 321, 13, 14]; CHIN = [152, 148, 377, 176, 400, 149, 378]; ANCHOR = [33, 263, 168, 6]


def jit(v):
    lm, ok = landmarks(v); iod = np.median(np.linalg.norm(lm[ok][:, 33] - lm[ok][:, 263], axis=1))
    rel = lm - lm[:, ANCHOR].mean(1, keepdims=True); good = ok[2:] & ok[1:-1] & ok[:-2]; out = {}
    for name, idx in (("mouth", MOUTH), ("chin", CHIN)):
        p = rel[:, idx]; d2 = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=2).mean(1) / iod
        out[name] = d2[good]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--video", nargs="+", required=True); ap.add_argument("--json")
    a = ap.parse_args(); vids = dict(x.split("=", 1) for x in a.video)
    S = jit(a.src); thr = {k: float(np.percentile(v, 95)) for k, v in S.items()}; res = {"src": {}}; ref = None
    for k in ("mouth", "chin"): res["src"][k] = {"jitter_x1000": round(1000 * float(S[k].mean()), 3), "shake_share": 0.05}
    for name, v in vids.items():
        J = jit(v); r = {}
        for k in ("mouth", "chin"):
            r[k] = {"jitter_x1000": round(1000 * float(J[k].mean()), 3), "shake_share": round(float((J[k] > thr[k]).mean()), 4)}
        res[name] = r; ref = ref or name
    for name in vids:
        for k in ("mouth", "chin"):
            res[name][k]["vs_" + ref] = round(res[name][k]["jitter_x1000"] / max(res[ref][k]["jitter_x1000"], 1e-9), 3)
        print(f"{name:8s} mouth jitter {res[name]['mouth']['jitter_x1000']:.2f} (x{res[name]['mouth']['vs_' + ref]:.2f} vs {ref}, shake {100 * res[name]['mouth']['shake_share']:.1f} %)  "
              f"chin jitter {res[name]['chin']['jitter_x1000']:.2f} (x{res[name]['chin']['vs_' + ref]:.2f}, shake {100 * res[name]['chin']['shake_share']:.1f} %)")
    print(f"src      mouth jitter {res['src']['mouth']['jitter_x1000']:.2f}  chin jitter {res['src']['chin']['jitter_x1000']:.2f}  (shake = share of frames above the source p95)")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
