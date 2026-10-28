"""Creak / "fart" detector on a dub track: voiced runs >= min_s whose F0 sits far below the voice's own median (vocal fry, a Chatterbox
hesitation trail) and that are loud enough to hear (> voice p95 - db_below). pyin 40-400 Hz on 10-ms hops.
    python scripts/r7/creak.py --tag r7v3a5 [--prev r7v2] [ids]"""
import argparse, numpy as np, soundfile as sf, librosa
A = "/workspace/dub/artifacts/r4a"

def detect(y, sr, ratio=0.65, min_s=0.15, db_below=32.0, hop_s=0.01):
    if y.ndim > 1: y = y.mean(1)
    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32); sr = 16000
    hop = int(hop_s * sr)
    f0, vf, vp = librosa.pyin(y16, fmin=40, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    db = 20 * np.log10(librosa.feature.rms(y=y16, frame_length=640, hop_length=hop)[0] + 1e-9)[:len(f0)]
    ref = np.percentile(db[db > -80], 95); med = np.nanmedian(f0[vf & (vp > 0.5)])
    low = ~np.isnan(f0) & (f0 < ratio * med) & (db > ref - db_below)
    out, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and (low[j] or (j + 1 < len(low) and low[j + 1])): j += 1   # bridge 1-frame holes
            if (j - i) * hop_s >= min_s:
                out.append({"t": round(i * hop_s, 2), "dur": round((j - i) * hop_s, 2), "f0": round(float(np.nanmedian(f0[i:j])), 1),
                            "f0_voice": round(float(med), 1), "db_rel": round(float(db[i:j].max() - ref), 1)})
            i = j
        else: i += 1
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--prev"); ap.add_argument("ids", nargs="*", default=["01", "02", "03", "04", "05"])
    a = ap.parse_args()
    for id_ in a.ids:
        for t in [x for x in (a.prev, a.tag) if x]:
            y, sr = sf.read(f"{A}/{t}/{id_}/dubbed_24k.wav")
            r = detect(y, sr); print(f"{id_} {t}: {len(r)} creak runs", " ".join(f"{c['t']}s/{c['dur']}s f0 {c['f0']} ({c['db_rel']} dB)" for c in r), flush=True)
