#!/usr/bin/env python3
"""Round 3 D: background-bed "breathing" (user: the bed pumps while the speaker talks). The bed as heard = final mix - g * dub voice
(g = least-squares gain of the dub in the mix: loudness normalisation scales both), compared window by window (200 ms) with the
separated, UNDUCKED bed (bg_44k.wav): gain_dB(w) = 20 log10(rms(bed heard) / rms(bg)). Windows split by the dub voice (active when its
RMS is within 35 dB of its p95). Reports the median gain in dub-speech and in dub-pause windows, their difference (= effective duck depth
incl. residue duck and any dynamic loudness processing), and the window-to-window gain jitter (p95 |delta|) inside each class.
    python scripts/quality/r3_bedswing.py --mix final_mix_44k.wav --dub dubbed_24k.wav --bg bg_44k.wav [--json out.json]"""
import argparse, json, pathlib
import numpy as np, soundfile as sf
from scipy.signal import resample_poly


def mono(p):
    y, sr = sf.read(str(p), dtype="float32"); return (y.mean(1) if y.ndim > 1 else y), sr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mix", required=True); ap.add_argument("--dub", required=True); ap.add_argument("--bg", required=True)
    ap.add_argument("--win", type=float, default=0.2); ap.add_argument("--json", default=None); a = ap.parse_args()
    mix, sr = mono(a.mix); dub, sd = mono(a.dub); bg, sb = mono(a.bg)
    if sd != sr: dub = resample_poly(dub, sr, sd).astype(np.float32)
    if sb != sr: bg = resample_poly(bg, sr, sb).astype(np.float32)
    n = min(len(mix), len(dub), len(bg)); mix, dub, bg = mix[:n], dub[:n], bg[:n]
    g = float(np.dot(mix, dub) / max(np.dot(dub, dub), 1e-9)); bed = mix - g * dub
    w = int(a.win * sr); k = n // w
    rms = lambda x: np.sqrt((x[:k * w].reshape(k, w) ** 2).mean(1)) + 1e-9
    rb, rg, rd = rms(bed), rms(bg), rms(dub)
    db_d = 20 * np.log10(rd); speech = db_d > np.percentile(db_d, 95) - 35; ok = 20 * np.log10(rg) > 20 * np.log10(np.percentile(rg, 95)) - 40
    gain = 20 * np.log10(rb / rg)
    sp, pa = gain[speech & ok], gain[~speech & ok]
    jit = lambda m: float(np.percentile(np.abs(np.diff(gain[m])), 95)) if m.sum() > 3 else None
    r = {"mix": a.mix, "dub_gain_in_mix": round(g, 3), "windows_speech": int((speech & ok).sum()), "windows_pause": int((~speech & ok).sum()),
         "bed_gain_db_speech_median": round(float(np.median(sp)), 2) if len(sp) else None, "bed_gain_db_pause_median": round(float(np.median(pa)), 2) if len(pa) else None,
         "swing_db": round(float(np.median(pa) - np.median(sp)), 2) if len(sp) and len(pa) else None,
         "gain_jitter_p95_db_all": round(float(np.percentile(np.abs(np.diff(gain[ok])), 95)), 2)}
    print(json.dumps(r))
    if a.json: pathlib.Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
