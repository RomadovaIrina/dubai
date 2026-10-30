#!/usr/bin/env python3
"""r9: per-window lip-sync diagnosis of one E2E output (the user: "lip-sync goes to hell" on m03).
Per --win s window (25 fps master):
  SyncNet best offset / conf on the speaker track (gate per-frame distances), dub speech share, source speech share (Silero on the source),
  dub-only / source-only / both seconds, LatentSync share, fallback weight, output mouth opening vs dub energy correlation, output vs SOURCE
  mouth opening correlation (does the new mouth still copy the Russian one?), and the mouth openness level out / source.
    python scripts/r9/sync_diag.py --id m03 [--win 3] --out diag.json"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
A = pathlib.Path("/workspace/dub/artifacts/r4a"); B = pathlib.Path("/workspace/dub/artifacts/r4m")


def vad(wav):
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    return get_speech_timestamps(read_audio(str(wav), sampling_rate=16000), load_silero_vad(), sampling_rate=16000, return_seconds=True)


def mask(ts, n, fps=25):
    m = np.zeros(n, bool)
    for s in ts: m[int(s["start"] * fps):int(np.ceil(s["end"] * fps))] = True
    return m


def opening(lm):   # inner-lip gap / face height per frame
    h = np.nanmax(lm[:, :, 1], 1) - np.nanmin(lm[:, :, 1], 1)
    return np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / h


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0: return None
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--win", type=float, default=3.0); ap.add_argument("--out", required=True); a = ap.parse_args()
    i = a.id; FT, GT, AT = f"r9f_{i}", f"r9g_{i}", f"r9_{i}"; W = B / GT / f"work_{i}_{GT}"
    import clean_face_stack as cfs
    tmp = pathlib.Path(tempfile.mkdtemp())
    out = B / FT / f"{i}_{FT}.mp4"; master = W / "video" / "master_25fps.mp4"
    o25 = tmp / "out25.mp4"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-r", "25", "-an", "-c:v", "libx264", "-crf", "12", str(o25)], check=True)
    lo = np.load(cfs.run_mediapipe(o25, tmp / "out25_mp.npz"))["lm"]; ls = np.load(W / "video" / "master_mediapipe.npz")["lm"]
    n = min(len(lo), len(ls)); lo, ls = lo[:n], ls[:n]
    sw = W / "video" / "own_audio_16k.wav"   # the source audio on the master timeline
    dub = A / AT / i / "dubbed_16k.wav"
    S = mask(vad(sw), n); D = mask(vad(dub), n)
    import soundfile as sf
    y, sr = sf.read(str(dub)); y = y if y.ndim == 1 else y.mean(1); hop = sr // 25
    env = np.array([np.sqrt(np.mean(y[k * hop:(k + 1) * hop] ** 2)) if k * hop < len(y) else 0 for k in range(n)])
    ys, _ = sf.read(str(sw)); envs = np.array([np.sqrt(np.mean(ys[k * 640:(k + 1) * 640] ** 2)) if k * 640 < len(ys) else 0 for k in range(n)])
    oo, osrc = opening(lo), opening(ls)
    sp = np.load(B / FT / f"{i}_{FT}.sync_perframe.npz")["d"]; fb = np.load(W / f"{i}_{GT}_25fps_fb_fb7.npz")
    lsm, fbw = fb["ls_mask"][:n].astype(bool), fb["fb"][:n]
    man = json.loads((B / FT / f"{i}_{FT}.manifest.json").read_text())
    rows = []; Wf = int(a.win * 25)
    for w0 in range(0, n, Wf):
        sl = slice(w0, min(n, w0 + Wf)); blk = sp[w0:w0 + Wf]; okb = np.isfinite(blk).all(1)
        if okb.sum() >= 20:
            m = blk[okb].mean(0); best = 15 - int(np.argmin(m)); conf = round(float(np.median(m) - m.min()), 2); c0 = round(float(np.median(m) - m[15]), 2)
        else: best = conf = c0 = None
        s, d = S[sl], D[sl]
        rows.append({"t": [round(w0 / 25, 1), round(min(n, w0 + Wf) / 25, 1)], "sync_off": best, "sync_conf": conf, "sync_conf0": c0,
                     "src_speech": round(float(s.mean()), 2), "dub_speech": round(float(d.mean()), 2),
                     "dub_only_s": round(float((d & ~s).sum()) / 25, 2), "src_only_s": round(float((s & ~d).sum()) / 25, 2), "both_s": round(float((s & d).sum()) / 25, 2),
                     "ls_share": round(float(lsm[sl].mean()), 2), "fallback_w": round(float(np.nanmean(fbw[sl])), 3),
                     "open_vs_dub": corr(oo[sl], env[sl]), "open_vs_src_mouth": corr(oo[sl], osrc[sl]), "src_mouth_vs_src_audio": corr(osrc[sl], envs[sl]),
                     "open_out_over_src": round(float(np.nanmean(oo[sl]) / max(1e-6, np.nanmean(osrc[sl]))), 2)})
    tot = {"open_vs_dub": corr(oo, env), "open_vs_src_mouth": corr(oo, osrc), "src_mouth_vs_src_audio": corr(osrc, envs),
           "dub_only_s": round(float((D & ~S).sum()) / 25, 2), "src_only_s": round(float((S & ~D).sum()) / 25, 2), "both_s": round(float((S & D).sum()) / 25, 2)}
    res = {"id": i, "win": a.win, "total": tot, "rows": rows, "pass_through": [(s["start_s"], s["end_s"]) for s in man["lipsync"]["segments"] if s["action"] != "LATENT_SYNC"]}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("total", tot); print("pass-through", res["pass_through"])
    for r in rows: print(r)


if __name__ == "__main__":
    main()
