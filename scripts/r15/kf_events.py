#!/usr/bin/env python3
"""r15 W: the worst mouth events of a keyframe arm vs the reference, and whether they sit on INTERPOLATED frames.
Per output frame: the mouth-box jitter (scripts/r14/flicker_metric.py: |2nd temporal difference| of the 4x pooled grey box) and the
mouth-box luma step |dY| (a flash). Events = the arm's top-N frames by jitter (and by luma step) with the reference's value at the same
frame; master frame m = round(t * 25); interpolated = inside a LATENT_SYNC segment of the arm's generator manifest at an odd offset from
the segment start and not its last frame (scripts/r15/kf.py: keyframes 0, k, 2k, ... + last, here k = 2).
    python scripts/r15/kf_events.py --lm source_lm_F.npz --ref q2=a.mp4 --arm tkf2=b.mp4 --manifest gen_25fps.manifest.json [--k 2] [--top 8]"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r14"))
import flicker_metric as fm

ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--manifest", required=True); ap.add_argument("--k", type=int, default=2); ap.add_argument("--top", type=int, default=8); a = ap.parse_args()
rn, rv = a.ref.split("=", 1); an, av = a.arm.split("=", 1)
W, H, fps = fm.probe(av); x0, y0, s = fm.box(a.lm, W, H)
FA, FR = fm.frames(av, x0, y0, s), fm.frames(rv, x0, y0, s); n = min(len(FA), len(FR)); FA, FR = FA[:n], FR[:n]
JA, JR = fm.jitter(FA), fm.jitter(FR); LA = np.abs(np.diff(FA.mean((1, 2)), prepend=FA[0].mean())); LR = np.abs(np.diff(FR.mean((1, 2)), prepend=FR[0].mean()))
m = json.load(open(a.manifest)); segs = [(sg["start_frame"], sg["end_frame"]) for sg in ((m.get("lipsync") or {}).get("segments") or m.get("segments") or []) if sg.get("action") == "LATENT_SYNC"]


def interp(mf):
    for s0, e0 in segs:
        if s0 <= mf < e0: return "interp" if ((mf - s0) % a.k and mf != e0 - 1) else "key"
    return "pass"


for name, X, R in (("jitter", JA, JR), ("luma step", LA, LR)):
    print(f"== top {a.top} {an} events by {name} (ref {rn}: same frame); output {fps:.2f} fps")
    for f in np.argsort(-X)[:a.top]:
        t = f / fps; mf = int(round(t * 25)); near = sorted({interp(mf + d) for d in (-1, 0, 1)})
        print(f"  t={t:6.2f} s  frame {f:4d}  {an} {X[f]:.4f}  {rn} {R[f]:.4f}  (x{X[f] / max(R[f], 1e-6):.1f})  master {mf} {interp(mf)}  (+-1: {'/'.join(near)})")
it = np.array([interp(int(round(f / fps * 25))) == "interp" for f in range(n)]); thr = np.percentile(JR, 99)
print(f"== frames above the ref's p99 jitter: {an} {int((JA > thr).sum())} ({int((JA[it] > thr).sum())} on interpolated master frames, {100 * it.mean():.0f} % of frames are interpolated), {rn} {int((JR > thr).sum())}")
