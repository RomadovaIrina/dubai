#!/usr/bin/env python3
"""r16c K1: neck <-> face SEAM jitter (Irina 09.11: "дрожание перехода от шеи к лицу").
Band = the source jaw line (cheek corner -> chin -> cheek corner, MediaPipe 58..288 via 152) drawn +-BAND x IOD thick, per frame.
jitter(video) = mean over the band of |V_t - V_t-1| (luma). Reported: final, source, ratio final / source (1.0 = moves exactly like the
source), extra = final - source, and the worst seconds.
    python seam_jitter.py --final F.mp4 --master master_25fps.mp4 --lm lm_f0.npz [--band 0.08] [--out J.json]"""
import argparse, json, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--master", required=True); ap.add_argument("--lm", required=True)
ap.add_argument("--band", type=float, default=0.08); ap.add_argument("--out", default=""); a = ap.parse_args()
JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
W, H = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.final], capture_output=True, text=True).stdout.strip().split(",")]
def frames(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        b = pr.stdout.read(W * H)
        if len(b) < W * H: break
        yield np.frombuffer(b, np.uint8).astype(np.float32).reshape(H, W)
L = np.load(a.lm)["lm"][..., :2]; jf, js, ks = [], [], []; pf = ps = None
for k, (F, S) in enumerate(zip(frames(a.final), frames(a.master))):
    if pf is not None and k < len(L) and np.isfinite(L[k][JAW]).all() and np.isfinite(L[k][[33, 263]]).all():
        iod = float(np.linalg.norm(L[k][33] - L[k][263])); m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [np.round(L[k][JAW]).astype(np.int32)], False, 1, thickness=max(2, int(2 * a.band * iod)))
        if m.sum() > 50:
            mb = m > 0; jf.append(float(np.abs(F - pf)[mb].mean())); js.append(float(np.abs(S - ps)[mb].mean())); ks.append(k)
    pf, ps = F, S
jf, js, ks = np.array(jf), np.array(js), np.array(ks); ex = jf - js
sec = {}
for k, e in zip(ks, ex): sec.setdefault(int(k // 25), []).append(e)
worst = sorted(((float(np.mean(v)), s_) for s_, v in sec.items()), reverse=True)[:5]
r = {"final": a.final, "frames": int(len(jf)), "jitter_final": round(float(jf.mean()), 3), "jitter_source": round(float(js.mean()), 3),
     "ratio": round(float(jf.mean() / max(js.mean(), 1e-6)), 3), "extra_mean": round(float(ex.mean()), 3), "extra_p95": round(float(np.percentile(ex, 95)), 3),
     "worst_seconds": [[s_, round(v, 3)] for v, s_ in worst]}
print("SEAM_JITTER " + json.dumps(r))
if a.out: json.dump(r, open(a.out, "w"))
