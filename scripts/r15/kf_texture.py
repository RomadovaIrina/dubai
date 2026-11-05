#!/usr/bin/env python3
"""r15 Z: do the keyframe arms alternate texture (a generated frame, then a DIS-interpolated one = a 12.5 Hz shimmer)? On a 25 fps generator
layer: per master frame, the mouth box (scripts/r14/flicker_metric.py box) lip sharpness (Laplacian variance of grey) and mean luma;
frames split into KEY (generated) and INTERP (odd offset inside a LATENT_SYNC segment, not its last frame: scripts/r15/kf.py, k = 2);
reports interp / key sharpness ratio, the mean |frame - average of its two neighbours| for both classes (the alternating component), and
the share of the frame-to-frame sharpness power at the Nyquist (12.5 Hz) bin.
    python scripts/r15/kf_texture.py --lm source_lm_F.npz --layer name=ls_pre.mp4 --manifest gen_25fps.manifest.json [...]"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r14"))
import flicker_metric as fm


def frames(v, x0, y0, s):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", v, "-vf", f"crop={s}:{s}:{x0}:{y0},scale=128:128,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 128, 128).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--layer", nargs="+", required=True); ap.add_argument("--manifest", nargs="+", required=True)
    a = ap.parse_args()
    for spec, man in zip(a.layer, a.manifest):
        name, v = spec.split("=", 1); W, H, fps = fm.probe(v); x0, y0, s = fm.box(a.lm, W, H); F = frames(v, x0, y0, s)
        lap = np.array([np.var(f[1:-1, 1:-1] * 4 - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]) for f in F]); lum = F.mean((1, 2))
        m = json.load(open(man)); segs = [(x["start_frame"], x["end_frame"]) for x in ((m.get("lipsync") or {}).get("segments") or m.get("segments") or []) if x.get("action") == "LATENT_SYNC"]
        cls = np.full(len(F), "", object)
        for s0, e0 in segs:
            for f in range(s0, min(e0, len(F))): cls[f] = "interp" if ((f - s0) % 2 and f != e0 - 1) else "key"
        mid = np.arange(1, len(F) - 1)
        def alt(x): return np.abs(x[mid] - (x[mid - 1] + x[mid + 1]) / 2)
        out = {}
        for c in ("key", "interp"):
            sel = cls[mid] == c
            if sel.any(): out[c] = (float(np.log(lap[mid][sel]).mean()), float(alt(np.log(lap))[sel].mean()), float(alt(lum)[sel].mean()))
        d = np.diff(np.log(lap)); P = np.abs(np.fft.rfft(d - d.mean())) ** 2; nyq = float(P[-3:].sum() / P.sum())
        r = np.exp(out["interp"][0] - out["key"][0]) if "interp" in out and "key" in out else float("nan")
        print(f"{name:10s} sharpness interp / key x{r:.3f}" + "".join(f"  {c}: log-sharp {v[0]:.3f} alt-sharp {v[1]:.3f} alt-luma {v[2]:.3f}" for c, v in out.items()) + f"  Nyquist share of d(sharpness) {100 * nyq:.1f} %")


if __name__ == "__main__":
    main()
