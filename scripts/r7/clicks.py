#!/usr/bin/env python3
"""r7 click / crackle detector: 2 kHz high-pass, 1-ms frames, peak vs the local median of the +-25 ms neighbourhood (a click is a short broadband
transient far above its surroundings) plus a sample-step score (|x[n] - x[n-1]| vs the local median step). Events closer than 10 ms merge.
    python scripts/r7/clicks.py <wav> [--json out.json]                         list of clicks (t, score dB, level dBFS)
    python scripts/r7/clicks.py --tag r7v1b --prev r6v1rx [ids]                  per video: counts in final mix / dub / bed, the new ones (not in PREV)"""
import argparse, json, pathlib
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

A = pathlib.Path("/workspace/dub/artifacts/r4a")


def detect(x: np.ndarray, sr: int, thr_db: float = 18.0, min_dbfs: float = -50.0) -> list[dict]:
    if x.ndim > 1: x = x.mean(1)
    hp = sosfiltfilt(butter(4, 2000, "hp", fs=sr, output="sos"), x)
    F = sr // 1000; n = len(hp) // F
    p = np.abs(hp[:n * F]).reshape(n, F).max(1)
    ref = median_filter(p, size=51, mode="nearest") + 1e-7
    st = np.abs(np.diff(x, prepend=x[:1]))[:n * F].reshape(n, F).max(1); sref = median_filter(st, size=51, mode="nearest") + 1e-7
    p = p + 1e-9; st = st + 1e-9
    sc = 20 * np.log10(p / ref); ss = 20 * np.log10(st / sref)
    hit = np.where(((sc > thr_db) | (ss > thr_db + 4)) & (20 * np.log10(p + 1e-9) > min_dbfs))[0]
    ev = []
    for i in hit:
        if ev and i - ev[-1]["_i"] <= 10:
            e = ev[-1]; e["_i"] = i
            if sc[i] > e["score_db"]: e.update(t=round(i / 1000, 3), score_db=round(float(sc[i]), 1), level_dbfs=round(float(20 * np.log10(p[i] + 1e-9)), 1))
            continue
        ev.append({"_i": i, "t": round(i / 1000, 3), "score_db": round(float(sc[i]), 1), "step_db": round(float(ss[i]), 1), "level_dbfs": round(float(20 * np.log10(p[i] + 1e-9)), 1)})
    for e in ev: e.pop("_i")
    return ev


def load(p):
    x, sr = sf.read(str(p), always_2d=False); return x, sr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("wav", nargs="?"); ap.add_argument("--json"); ap.add_argument("--tag"); ap.add_argument("--prev"); ap.add_argument("--thr", type=float, default=18.0)
    ap.add_argument("ids", nargs="*"); a = ap.parse_args()
    if a.wav and not a.tag:
        ev = detect(*load(a.wav), thr_db=a.thr); print(json.dumps(ev)) if not a.json else pathlib.Path(a.json).write_text(json.dumps(ev)); return
    ids = ([a.wav] if a.wav else []) + a.ids or ["01", "02", "03", "04", "05"]; R = {}   # with --tag the first positional is an id
    for id_ in ids:
        r = {}
        for tag in filter(None, [a.prev, a.tag]):
            d = A / tag / id_; r[tag] = {f: detect(*load(d / f), thr_db=a.thr) for f in ("final_mix_44k.wav", "dubbed_24k.wav", "bg_clean_44k.wav", "bg_capped_44k.wav") if (d / f).exists()}
        if a.prev:
            pt = [e["t"] for e in r[a.prev]["final_mix_44k.wav"]]
            r["new_in_final"] = [e for e in r[a.tag]["final_mix_44k.wav"] if not any(abs(e["t"] - t) < 0.02 for t in pt)]
        R[id_] = r
        print(id_, " | ".join(f"{t}: " + ", ".join(f"{f.split('_')[0]}{'_' + f.split('_')[1] if f.startswith('bg') else ''} {len(v)}" for f, v in r[t].items()) for t in filter(None, [a.prev, a.tag])),
              f"| new in final {len(r.get('new_in_final', []))}: " + " ".join(f"{e['t']}({e['score_db']}/{e['level_dbfs']})" for e in r.get("new_in_final", [])[:40]), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
