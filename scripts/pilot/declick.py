"""Targeted declick of the assembled dub voice (Agent C r2, 2026-10-17, "посторонние звуки").

audio_stages.py found isolated impulsive clicks in the dub voice of the r2 / cand5 tracks: most are already in the raw Chatterbox take (vocoder
click, 01 2.7 / 14.3 / 33.7 / 71.1 s, 05 42.0 s), two appear after ffmpeg atempo (01 42.3 / 59.3 s). They are found with the SAME detector
(1 ms peak of the > 5 kHz band > 10 x its +-25 ms median, isolated: plosives / fricatives last > 5 ms and are not touched) and repaired locally:
in +-1.5 ms around the peak the signal is replaced by its zero-phase 4 kHz low-pass, with 0.5 ms raised-cosine crossfades. Nothing else changes.
"""
from __future__ import annotations
import numpy as np


def find_clicks(y: np.ndarray, sr: int, k: float = 10.0, floor: float = 0.01, min_sep_s: float = 0.02) -> list:
    from scipy.signal import butter, sosfilt
    from scipy.ndimage import median_filter
    if len(y) < sr // 10: return []
    hp = np.abs(sosfilt(butter(4, 5000, "highpass", fs=sr, output="sos"), y)); f = max(1, sr // 1000); n = len(hp) // f
    pk = hp[:n * f].reshape(n, f).max(1); med = np.maximum(median_filter(pk, size=51, mode="nearest"), 1e-5)
    out = []; last = -1e9
    for i in np.where((pk > k * med) & (pk > floor))[0]:
        lo, hi = max(0, i - 4), min(n, i + 5); nb = np.concatenate([pk[lo:max(lo, i - 1)], pk[min(hi, i + 2):hi]])
        if len(nb) and nb.max() > (k / 3) * med[i]: continue
        t = i / 1000.0
        if t - last >= min_sep_s: out.append({"t": round(t, 3), "peak": round(float(pk[i]), 4), "x_local": round(float(pk[i] / med[i]), 1)}); last = t
    return out


def repair(y: np.ndarray, sr: int, clicks: list, half_ms: float = 1.5, xf_ms: float = 0.5, lp_hz: float = 4000.0) -> tuple[np.ndarray, list]:
    from scipy.signal import butter, sosfiltfilt, sosfilt
    y = y.copy(); log = []; sos_lp = butter(4, lp_hz, "lowpass", fs=sr, output="sos"); hp = np.abs(sosfilt(butter(4, 5000, "highpass", fs=sr, output="sos"), y))
    h, x, ctx = int(half_ms * sr / 1000), max(1, int(xf_ms * sr / 1000)), int(0.02 * sr)
    for c in clicks:
        a0, b0 = max(0, int((c["t"] - 0.002) * sr)), min(len(y), int((c["t"] + 0.003) * sr))
        if b0 <= a0: continue
        i = a0 + int(np.argmax(hp[a0:b0])); a, b = max(0, i - h), min(len(y), i + h)
        s0, s1 = max(0, a - ctx), min(len(y), b + ctx); lp = sosfiltfilt(sos_lp, y[s0:s1])[a - s0:b - s0]
        w = np.ones(b - a, np.float32); r = min(x, (b - a) // 2)
        if r > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r, dtype=np.float32)); w[:r] = ramp; w[-r:] = ramp[::-1]
        before = float(np.abs(y[a:b]).max())
        y[a:b] = np.clip((1 - w) * y[a:b] + w * lp.astype(np.float32), -before, before)   # r3: never louder than before (04 22.98 s: 0.99 -> 1.03 = clipping)
        log.append({"t": c["t"], "sample": int(i), "x_local": c["x_local"], "peak_before": round(before, 4), "peak_after": round(float(np.abs(y[a:b]).max()), 4)})
    return y, log


def declick_file(wav: str) -> dict:
    """In place: dubbed_24k.wav -> repaired; returns the log (re-detection after the repair included)."""
    import soundfile as sf
    y, sr = sf.read(wav, dtype="float32"); c = find_clicks(y, sr); found = len(c); y2, log, passes = y, [], 0
    while c and passes < 3:   # r3: re-detect after each repair (02 96.17 s / 04 22.99 s: a second click 6-8 ms after the repaired one was left)
        y2, lg = repair(y2, sr, c); log += lg; passes += 1; c = find_clicks(y2, sr)
    if log: sf.write(wav, y2, sr, subtype="PCM_16")
    return {"found": found, "passes": passes, "repaired": log, "left_after": find_clicks(y2, sr) if log else c}
