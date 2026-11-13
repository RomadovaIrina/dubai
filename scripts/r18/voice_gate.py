#!/usr/bin/env python3
"""r18: the voice KeySync hears, when only the accepted MIX of a take survives: Spleeter vocals of the mix are the dub's voice (m03: corr 0.998
with the true stem) but in the dub's pauses they carry the bed's leak (m03: -52..-63 dBFS where the true stem is digital silence). This gate
returns the pauses to silence: 20 ms frames, open when the level is above (speech median - --below dB), held --hold ms on both sides, 10 ms fades.
    voice_gate.py --in vocals_16k.wav --out dubbed_16k.wav [--below 28 --hold 120] [--truth true_16k.wav  (report only)]"""
import argparse, numpy as np, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--below", type=float, default=28.0); ap.add_argument("--hold", type=float, default=120.0); ap.add_argument("--truth", default=""); a = ap.parse_args()
x, sr = sf.read(a.inp, dtype="float32"); w = sr // 50; k = len(x) // w
db = lambda v: 20 * np.log10(np.sqrt((v[:k * w].reshape(k, w) ** 2).mean(1)) + 1e-9)
L = db(x); sp = float(np.median(L[L > L.max() - 30])); op = L > sp - a.below; h = max(1, int(round(a.hold / 20)))
op = np.convolve(op.astype(np.float32), np.ones(2 * h + 1), "same") > 0   # hold: h frames before and after
g = np.repeat(op.astype(np.float32), w); g = np.concatenate([g, np.full(len(x) - len(g), g[-1] if len(g) else 0, np.float32)])
f = sr // 100; g = np.convolve(g, np.ones(f, np.float32) / f, "same")   # 10 ms fades
y = (x * g).astype(np.float32); sf.write(a.out, y, sr, subtype="PCM_16")
print(f"VOICE_GATE speech median {sp:.1f} dBFS, threshold {sp - a.below:.1f}, open {100 * op.mean():.1f} % of the frames")
if a.truth:
    t, _ = sf.read(a.truth, dtype="float32"); n = min(len(t), len(y)) // w; T = db(t)[:n]; Y = db(y)[:n]; X = L[:n]
    sil = T < -60; spk = T > T.max() - 35
    print(f"  vs the true stem: its silent frames {int(sil.sum())}: gated to silence {100 * (Y[sil] < -80).mean():.1f} % (before the gate {100 * (X[sil] < -80).mean():.1f} %); "
          f"its speech frames {int(spk.sum())}: lost {100 * (Y[spk] < -80).mean():.2f} %; corr {np.corrcoef(t[:n * w], y[:n * w])[0, 1]:.4f}")
