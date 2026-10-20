#!/usr/bin/env python3
"""r4 merged gate metric mouth_audio_mismatch_s (user on 05 m2 WATCH: "11-12 s the mouth talks with no dub, 13 s the mouth is closed while the dub
speaks"): the visible mouth must follow the DUB audio. MediaPipe on the OUTPUT (face visible = landmarks found); inner-lip gap / inter-ocular
distance per frame; motion = its std over 0.3 s. Calibrated on the sources 03 / 04: motion median 0.02 in speech, 0.001-0.007 in silence.
  (a) dub silent (runs >= 0.3 s, 0.15 s margins at dub edges) and the mouth talks (motion > 0.015), runs >= 0.3 s
  (b) dub speaks and the mouth is closed and still (gap < 0.04 and motion < 0.008), runs >= 0.3 s
    python scripts/r4/r4_mouth_audio.py --video OUT.mp4 --dub dubbed_16k.wav --out x.json"""
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))


def runs(m, min_n):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_n]


def analyse(video, dub):
    import av, clean_face_stack as cfs
    v = pathlib.Path(video); lm = np.load(cfs.run_mediapipe(v, v.with_name(v.stem + "_out_mp.npz")))["lm"]
    c = av.open(str(v)); fps = float(c.streams.video[0].average_rate); c.close(); n = len(lm)
    iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1); gap = np.linalg.norm(lm[:, 13] - lm[:, 14], axis=1) / iod; vis = np.isfinite(gap)
    w = max(1, int(0.15 * fps)); mv = np.array([np.nanstd(gap[max(0, i - w):i + w + 1]) if vis[i] else np.nan for i in range(n)])
    y, sr = sf.read(str(dub), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); h = int(sr / fps)
    e = 20 * np.log10(np.sqrt((y[: len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); e = np.pad(e, (0, max(0, n - len(e))), constant_values=-120)[:n]
    act = e > np.percentile(e[e > -100], 90) - 30 if (e > -100).any() else np.zeros(n, bool)
    sil = np.zeros(n, bool)
    for a, b in runs(~act, int(0.3 * fps)): sil[min(n, a + w):max(a, b - w)] = True
    spk = np.convolve(act.astype(float), np.ones(2 * w + 1), "same") > 0
    A = vis & sil & (np.nan_to_num(mv) > 0.015); Bm = vis & spk & ~sil & (np.nan_to_num(gap, nan=1) < 0.04) & (np.nan_to_num(mv, nan=1) < 0.008)
    ra, rb = runs(A, int(0.3 * fps)), runs(Bm, int(0.3 * fps))
    sa, sb = sum(b - a for a, b in ra) / fps, sum(b - a for a, b in rb) / fps
    worst = sorted([("talking mouth, dub silent", a / fps, (b - a) / fps) for a, b in ra] + [("closed mouth, dub speaks", a / fps, (b - a) / fps) for a, b in rb], key=lambda x: -x[2])
    return {"mouth_audio_mismatch_s": round(sa + sb, 2), "talking_mouth_dub_silent_s": round(sa, 2), "closed_mouth_dub_speaks_s": round(sb, 2),
            "face_visible_s": round(float(vis.sum() / fps), 1), "worst": [[k, round(t, 2), round(d, 2)] for k, t, d in worst[:8]]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--dub", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    r = analyse(a.video, a.dub); pathlib.Path(a.out).write_text(json.dumps(r, indent=1)); print(json.dumps(r))


if __name__ == "__main__":
    main()
