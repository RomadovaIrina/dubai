#!/usr/bin/env python3
"""r15 Z: WHERE the lip colour / brightness jumps of a 25 fps generator layer sit relative to LatentSync's 16-frame windows. Per master
frame: mean L and a* (Lab) inside the outer lip polygon (landmarks: the cached <video>.mp_lm.npz or .mp_lm_img.npz), |dL| / |da*| between
consecutive frames; jumps = above the SOURCE (master) p99. For the jumps inside LATENT_SYNC segments (the generator manifest): the histogram
of the offset from the segment start modulo 16 (window seams at offset 0; q2 runs 16-frame windows from the segment start + 8-frame
bridges centred on the seams) and the share within +-1 frame of a seam vs the share of frames that are.
Also the slide split: the 1 s low-pass of the mouth-corner midpoint relative to the head anchor, excess over the master, x and y apart.
    python scripts/r15/colour_seams.py --master master_25fps.mp4 --layer ls_pre.mp4 --manifest gen_25fps.manifest.json"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_jitter as SJ


def series(v):
    lm, ok = SJ.landmarks(v); dL, dA = SJ.colour(v, lm, ok); return lm, ok, dL, dA


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--layer", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--lm", choices=["img", "video"], default="video"); a = ap.parse_args(); SJ.MODE["lm"] = a.lm
    lmM, okM, dLM, dAM = series(a.master); lmL, okL, dLL, dAL = series(a.layer)
    m = json.load(open(a.manifest)); segs = [(s["start_frame"], s["end_frame"]) for s in ((m.get("lipsync") or {}).get("segments") or m.get("segments") or []) if s.get("action") == "LATENT_SYNC"]
    tL, tA = np.percentile(dLM, 99), np.percentile(dAM, 99); n = min(len(dLL), len(dAL))
    jump = (dLL[:n] > tL) | (dAL[:n] > tA)   # dL[i] = change between frame i and i+1 (both lip-visible); index ~ frame i+1
    off = np.full(n, -1)
    for s0, e0 in segs:
        for f in range(max(s0, 1), min(e0, n + 1)): off[f - 1] = (f - s0) % 16
    ins = off >= 0; J = jump & ins
    hist = np.bincount(off[J], minlength=16); near = np.isin(off, [15, 0, 1])
    print(f"jumps {int(jump.sum())} of {n} frame pairs ({int(J.sum())} inside generator segments); master p99 thresholds dL {tL:.2f} da {tA:.2f}")
    print(f"offset mod 16 histogram (0 = window seam): {hist.tolist()}")
    print(f"within +-1 of a seam: {100 * (J & near).sum() / max(1, J.sum()):.0f} % of the jumps vs {100 * (ins & near).sum() / max(1, ins.sum()):.0f} % of the frames")
    for name, lm, ok in (("master", lmM, okM), ("layer", lmL, okL)):
        g = SJ.geom(lm, ok, 25.0)
        if name == "master": GM = g
        else:
            d = g["lp"] - GM["lp"][:len(g["lp"])] if len(GM["lp"]) >= len(g["lp"]) else g["lp"][:len(GM["lp"])] - GM["lp"]
            k = min(len(d), len(g["ok"]))
            ex = np.std(d[:k][g["ok"][:k]], 0) * 1000
            print(f"slide (1 s low-pass of the corner midpoint vs the master), IOD x 1000: x {ex[0]:.2f}  y {ex[1]:.2f}")


if __name__ == "__main__":
    main()
