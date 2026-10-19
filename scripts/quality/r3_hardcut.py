#!/usr/bin/env python3
"""Round 3 D: hard phrase cuts in a dub voice track (coordinator's definition): 5 ms RMS frames; OFF = a loud frame (> --loud dBFS) followed
within one frame (5 ms) by a silent one (< --silent dBFS); ON = the reverse. Unit / part joins concatenated without fades show up here.
    python scripts/quality/r3_hardcut.py dubbed_24k.wav [more.wav ...] [--json out.json]"""
import argparse, json, pathlib
import numpy as np, soundfile as sf


def events(path, loud=-35.0, silent=-55.0, win_ms=5.0, hop_ms=1.0):
    """5 ms RMS windows on a 1 ms hop; an event = window at t loud and window at t + 5 ms silent (OFF) or the reverse (ON); events within
    50 ms of each other are merged."""
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    w, h = int(sr * win_ms / 1000), int(sr * hop_ms / 1000); c = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    starts = np.arange(0, len(y) - w, h); db = 10 * np.log10((c[starts + w] - c[starts]) / w + 1e-18); lag = int(win_ms / hop_ms); ev = []
    for i in range(len(db) - lag):
        kind = "OFF" if db[i] > loud and db[i + lag] < silent else "ON" if db[i] < silent and db[i + lag] > loud else None
        if kind:
            t = round((starts[i] + w) / sr, 3)
            if not ev or t - ev[-1][1] > 0.05: ev.append((kind, t, round(float(max(db[i], db[i + lag])), 1)))
    return ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("wav", nargs="+"); ap.add_argument("--json", default=None); a = ap.parse_args(); out = {}
    for w in a.wav:
        e = events(w); out[w] = {"count": len(e), "events": e}; print(f"{w}: {len(e)} hard cuts " + " ".join(f"{k} {t}" for k, t, _ in e))
    if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
