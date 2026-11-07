#!/usr/bin/env python3
"""r14: is the student's mouth damped? MediaPipe landmarks on each output (scripts/quality/mp_landmarks.py, cached next to the video), inner-lip
gap |13-14| / mouth width |61-291| per frame, on the VOICED frames of the dubbed track (16 kHz RMS per video frame above max - 30 dB).
Reports mean / p90 / p99 opening (how wide), std and mean |d/dt| (how much it moves), the share of near-closed voiced frames, and the
correlation of the opening with the dub's loudness envelope; ratios vs the first video (the production baseline).
    python scripts/r14/mouth_open.py --wav dubbed_16k.wav --video q2=q2.mp4 s4=s4.mp4 [--json out.json]"""
import argparse, json, pathlib, subprocess, wave
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def landmarks(v):
    out = pathlib.Path(v).with_suffix(".mp_lm.npz")
    if not out.exists():
        subprocess.run(["/venv/mediapipe/bin/python", str(REPO / "scripts/quality/mp_landmarks.py"), "--video", v, "--out", str(out)], check=True,
                       stdout=subprocess.DEVNULL)
    z = np.load(out); return z["lm"][..., :2].astype(np.float64), z["ok"].astype(bool)


def envelope(wav, fps, n):
    w = wave.open(wav); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels()).mean(1)
    hop = sr / fps; r = np.array([np.sqrt(np.mean(x[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) if int(i * hop) < len(x) else 1e-6 for i in range(n)])
    return 20 * np.log10(r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--wav", required=True); ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--json"); a = ap.parse_args()
    vids = dict(x.split("=", 1) for x in a.video); res = {}; base = None
    for k, v in vids.items():
        L, ok = landmarks(v); gap = np.linalg.norm(L[:, 13] - L[:, 14], axis=1); wid = np.linalg.norm(L[:, 61] - L[:, 291], axis=1)
        o = np.where(ok & (wid > 1), gap / np.maximum(wid, 1), np.nan); db = envelope(a.wav, a.fps, len(o)); voiced = (db > db.max() - 30) & np.isfinite(o)
        ov = o[voiced]; d = np.abs(np.diff(o)); dv = d[voiced[1:] & voiced[:-1] & np.isfinite(d)]
        r = {"frames": int(len(o)), "voiced": int(voiced.sum()), "mean": float(ov.mean()), "p90": float(np.percentile(ov, 90)), "p99": float(np.percentile(ov, 99)),
             "std": float(ov.std()), "speed": float(dv.mean()), "closed_share": float((ov < 0.05).mean()),
             "corr_loudness": float(np.corrcoef(ov, db[voiced])[0, 1]), "silent_mean": float(np.nanmean(o[~voiced & np.isfinite(o)]))}
        if base is None: base = r
        else: r["vs_" + next(iter(vids))] = {m: round(r[m] / base[m], 3) for m in ("mean", "p90", "p99", "std", "speed")}
        res[k] = r; print(k, json.dumps({m: (round(x, 4) if isinstance(x, float) else x) for m, x in r.items()}), flush=True)
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
