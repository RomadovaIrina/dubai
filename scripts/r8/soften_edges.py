#!/usr/bin/env python3
"""r8 v3: abrupt edges in the dub voice softened (user on r8v1: words cut off on a hard edge; scripts/r8/cuts.py counts them). On the
pre-placement dub (last step before lip_place), 5-ms frames vs the voice's p95 (ref):
  * an abrupt END (the level falls from >= ref - 25 dB to <= ref - 55 dB within 15 ms): the --decay-ms before the edge get a dB-linear decay to
    -60 dB, the way a word dies out naturally (a 10-ms ramp into digital silence is a cut);
  * an abrupt START out of silence (<= ref - 55 dB for 15 ms, then >= ref - 25 dB within 15 ms): a --rise-ms dB-linear rise from -40 dB.
Timing never changes; samples outside the softened edges are bit-identical.
    python scripts/r8/soften_edges.py --work W --out W2"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np, soundfile as sf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--decay-ms", type=float, default=60.0); ap.add_argument("--rise-ms", type=float, default=10.0); a = ap.parse_args()
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*_r4a.manifest.json")); man = json.loads(mp.read_text())
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); z = y.copy()
    h = int(0.005 * sr); L = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); ref = float(np.percentile(L[L > -90], 95))
    hi, lo, k = ref - 25, ref - 55, 3; ends, starts = [], []; i = k
    while i < len(L) - k:
        if L[i - 1] >= hi and (L[i:i + k] <= lo).any():
            j = i + int(np.argmax(L[i:i + k] <= lo)); e = j * h; n = int(a.decay_ms / 1000 * sr); s0 = max(0, e - n)
            g = 10 ** (np.linspace(0, -60, e - s0) / 20); z[s0:e] *= g.astype(np.float32); ends.append(round(e / sr, 3)); i = j + k; continue
        if L[i + k - 1] >= hi and (L[i - k:i] <= lo).all():
            st = i * h; n = int(a.rise_ms / 1000 * sr); g = 10 ** (np.linspace(-40, 0, n) / 20); z[st:st + n] *= g.astype(np.float32)[:len(z[st:st + n])]
            starts.append(round(st / sr, 3)); i += k; continue
        i += 1
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    man["r8_soften_edges"] = {"ends": ends, "starts": starts, "decay_ms": a.decay_ms, "rise_ms": a.rise_ms}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); print(json.dumps({"ends": len(ends), "starts": len(starts)}))


if __name__ == "__main__":
    main()
