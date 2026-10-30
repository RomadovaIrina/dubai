#!/usr/bin/env python3
"""r9: two faces on screen (spec 8.2: only the SPEAKER gets lips). Which face speaks, and which face did the pipeline re-lip?
  speaker = the face whose mouth opening moves with the source speech (Silero VAD on the source audio): mean opening while speaking minus
            while silent, and the correlation of the opening with the VAD mask;
  re-lipped = the face nearest to the face the pipeline tracked (the generator's single-face MediaPipe pass on the source, timing cache).
    python scripts/r9/twoface_check.py --src <video> --work <e2e work dir> --tracked <source landmarks npz> --out twoface.json [--sheet sheet.jpg]"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))


def norm(a, W, H):
    a = np.array(a, dtype=float)
    if np.nanmax(np.abs(a[..., 0])) <= 1.5: a[..., 0] *= W; a[..., 1] *= H
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--work", required=True); ap.add_argument("--tracked", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--sheet", default=""); a = ap.parse_args()
    import clean_face_stack as cfs
    w = pathlib.Path(a.work); npz = cfs.run_mediapipe(pathlib.Path(a.src), w / "twoface_mp2.npz", mode="image", num_faces=2, every=2)
    d = np.load(npz); W, H = [int(x) for x in d["wh"]]; la = norm(d["lm_all"], W, H)          # (T, 2, 478, 2), unordered
    T = len(la); fps = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "json", a.src],
                                                      capture_output=True, text=True).stdout)["streams"][0]["avg_frame_rate"].split("/")[0]) / float(
        json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "json", a.src], capture_output=True, text=True).stdout)["streams"][0]["avg_frame_rate"].split("/")[1])
    idx = np.arange(0, T, 2); la = la[idx]; t = idx / fps
    ok = np.isfinite(la[..., 0]).all(-1)
    cen = np.nanmean(la, axis=2)                                                                 # (N, 2, 2) face centres
    pts = cen[ok]; c = pts[np.argsort(pts[:, 0])][[0, -1]].copy()                                # 2-means on the centres, seeded left / right
    for _ in range(20):
        lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), 1); c = np.array([pts[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
    op = np.full((len(la), 2), np.nan); hh = np.full(2, np.nan); hs = [[], []]
    for i in range(len(la)):
        for f in range(2):
            if not ok[i, f]: continue
            k = int(np.argmin(((cen[i, f] - c) ** 2).sum(-1))); L = la[i, f]; h = L[:, 1].max() - L[:, 1].min()
            op[i, k] = np.linalg.norm(L[13] - L[14]) / h; hs[k].append(h)                         # inner-lip gap / face height
    import tempfile
    wav = tempfile.mktemp(suffix=".wav"); subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-ac", "1", "-ar", "16000", wav], check=True)
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    ts = get_speech_timestamps(read_audio(wav, sampling_rate=16000), load_silero_vad(), sampling_rate=16000, return_seconds=True)
    sp = np.zeros(len(t), bool)
    for s in ts: sp |= (t >= s["start"]) & (t <= s["end"])
    faces = []
    for k in range(2):
        o = op[:, k]; m = np.isfinite(o)
        r = float(np.corrcoef(o[m], sp[m].astype(float))[0, 1]) if m.sum() > 10 and sp[m].std() > 0 else None
        faces.append({"face": k, "centre_px": [round(float(x)) for x in c[k]], "visible_frac": round(float(m.mean()), 3), "height_px_median": round(float(np.median(hs[k])), 1) if hs[k] else None,
                      "mouth_open_speech": round(float(np.nanmean(o[m & sp])), 4) if (m & sp).any() else None, "mouth_open_silence": round(float(np.nanmean(o[m & ~sp])), 4) if (m & ~sp).any() else None,
                      "mouth_motion_std_speech": round(float(np.nanstd(np.diff(o[m & sp]))), 4) if (m & sp).sum() > 3 else None, "corr_open_vs_speech": round(r, 3) if r is not None else None})
    spk = max(range(2), key=lambda k: (faces[k]["mouth_motion_std_speech"] or 0) + 0.0)
    res = {"speech_s": round(sum(s["end"] - s["start"] for s in ts), 2), "faces": faces, "speaker_by_mouth_motion": spk}
    if a.tracked and pathlib.Path(a.tracked).exists():
        tr = np.load(a.tracked); key = "lm" if "lm" in tr.files else tr.files[0]; tl = norm(tr[key], W, H)
        if tl.ndim == 3 and len(tl) >= T:
            tc = np.nanmean(tl[idx], axis=1); g = np.isfinite(tc[:, 0])
            near = np.argmin(((tc[g][:, None] - c[None]) ** 2).sum(-1), 1)
            res["tracked_face_share"] = [round(float((near == k).mean()), 3) for k in range(2)]; res["relipped_face"] = int(np.bincount(near, minlength=2).argmax())
            res["relipped_is_speaker"] = res["relipped_face"] == spk
        else: res["tracked_note"] = f"tracked landmarks shape {tl.shape} vs {T} source frames"
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1)); print("== twoface", json.dumps(res))


if __name__ == "__main__":
    main()
