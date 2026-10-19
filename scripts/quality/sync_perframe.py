#!/usr/bin/env python3
"""Per-frame / per-second lip-sync analysis (round 3, track A).

SyncNet part: the SAME detector / crop / model as LatentSync's eval/eval_sync_conf.py (S3FD tracks, 13-frame median crop, syncnet_v2),
but instead of one number per video it keeps the per-frame distance matrix d[frame, offset] (offsets -15..+15 video frames) and reports
  * global: conf / AV offset exactly as upstream (median(mean_d) - min(mean_d) per track, mean over tracks);
  * per second s (bins of --bin s, context +-(--ctx) s): conf_best = median - min of the window-mean distances, offset_best, and
    conf_at0 = median - d(offset 0) (sync quality at the global offset; the best offset of a 1-2 s window is noisy).
Lip part (--mp-npz = MediaPipe npz of the SAME video, mp_landmarks.py): inner-lip gap / IOD per frame vs the driving audio's log-RMS
envelope at 25 fps, per-second Pearson r at lag 0 and the best lag in +-3 frames; also per-second dub / source speech activity.

    python scripts/quality/sync_perframe.py --video OUT.mp4 [--audio dub_16k.wav] [--src-audio own_16k.wav] [--mp-npz OUT_mp.npz] --out X.json
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
FPS = 25; VSHIFT = 15


def _ls_root():
    from syncnet_common import LS
    return LS


def syncnet_dists(video: pathlib.Path, audio: pathlib.Path | None, device="cuda", keep=False):
    """-> list of tracks {f0, frame (global idx of the 5-frame clip centre), d (N, 31)}"""
    import cv2, torch
    LS = _ls_root(); sys.path.insert(0, str(LS))
    work = pathlib.Path(tempfile.mkdtemp(prefix="syncpf_")); (work / "checkpoints").symlink_to(LS / "checkpoints")
    old = os.getcwd(); os.chdir(work)
    try:
        v = video.resolve()
        if audio is not None:
            v = work / "in.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(video.resolve()), "-i", str(pathlib.Path(audio).resolve()), "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(v)], check=True)
        from eval.syncnet import SyncNetEval
        from eval.syncnet_detect import SyncNetDetector
        from python_speech_features import mfcc
        from scipy.io import wavfile
        det = SyncNetDetector(device=device, detect_results_dir="detect_results"); starts = {}
        orig_crop = det.crop_video

        def crop_video(track, cropfile, *a, **k):
            starts[pathlib.Path(cropfile).name] = int(track["frame"][0]); return orig_crop(track, cropfile, *a, **k)
        det.crop_video = crop_video
        det(video_path=str(v), min_track=50)
        net = SyncNetEval(device=device); net.loadParameters(str(LS / "checkpoints/auxiliary/syncnet_v2.model")); S = net._SyncNetEval__S__; S.eval()
        tracks = []
        for cf in sorted((work / "detect_results/crop").glob("*.mp4")):
            cap = cv2.VideoCapture(str(cf)); ims = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                ims.append(cv2.resize(fr, (224, 224)))
            cap.release()
            wav = work / "a.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(cf), "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", str(wav)], check=True)
            sr, au = wavfile.read(str(wav)); cc = np.stack([np.array(i) for i in zip(*mfcc(au, sr))])
            im = np.stack(ims, 3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32); cct = cc[None, None].astype(np.float32)
            n = min(len(ims), len(au) // 640) - 5
            if n < 10: continue
            imf, ccf = [], []
            with torch.no_grad():
                for i in range(0, n, 64):
                    j = range(i, min(n, i + 64))
                    imf.append(S.forward_lip(torch.from_numpy(np.concatenate([im[:, :, f:f + 5] for f in j])).to(device)).cpu())
                    ccf.append(S.forward_aud(torch.from_numpy(np.concatenate([cct[:, :, :, f * 4:f * 4 + 20] for f in j])).to(device)).cpu())
            imf, ccf = torch.cat(imf), torch.cat(ccf)
            ccp = torch.nn.functional.pad(ccf, (0, 0, VSHIFT, VSHIFT)); W = 2 * VSHIFT + 1
            d = torch.stack([torch.nn.functional.pairwise_distance(imf[[i]].repeat(W, 1), ccp[i:i + W]) for i in range(n)]).numpy()
            f0 = starts.get(cf.stem, 0)
            tracks.append({"crop": cf.name, "f0": f0, "frame": f0 + np.arange(n) + 2, "d": d})
        return tracks
    finally:
        os.chdir(old)
        if not keep: shutil.rmtree(work, ignore_errors=True)


def env25(wav: pathlib.Path, n: int):
    import soundfile as sf
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    hop = sr // FPS; out = np.zeros(n, np.float32)
    for i in range(n):
        x = a[i * hop:(i + 1) * hop]
        out[i] = np.sqrt(np.mean(x ** 2)) if len(x) else 0.0
    return out


def lip_gap(npz: pathlib.Path):
    z = np.load(npz); lm = z["lm"]
    iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1)            # outer eye corners
    gap = np.linalg.norm(lm[:, 13] - lm[:, 14], axis=1) / iod        # inner lips centre
    return gap


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or a[ok].std() < 1e-6 or b[ok].std() < 1e-6: return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--audio", default=None, help="driving audio to evaluate against (default: the video's own track)")
    ap.add_argument("--src-audio", default=None, help="source speech 16 kHz (per-second source activity column)")
    ap.add_argument("--mp-npz", default=None); ap.add_argument("--out", required=True); ap.add_argument("--bin", type=float, default=1.0); ap.add_argument("--ctx", type=float, default=0.5)
    ap.add_argument("--frames", default=None, help="a:b restrict the per-second table to these frames")
    a = ap.parse_args()
    tracks = syncnet_dists(pathlib.Path(a.video), pathlib.Path(a.audio) if a.audio else None)
    res = {"video": a.video, "audio": a.audio, "tracks": []}
    confs, offs = [], []
    for t in tracks:
        md = t["d"].mean(0); confs.append(float(np.median(md) - md.min())); offs.append(int(VSHIFT - md.argmin()))
        res["tracks"].append({"f0": t["f0"], "n": len(t["d"]), "conf": round(confs[-1], 3), "offset": offs[-1]})
    res["global"] = {"conf": round(float(np.mean(confs)), 3) if confs else None, "av_offset": int(np.mean(offs)) if offs else None}
    import cv2
    cap = cv2.VideoCapture(a.video); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    D = np.full((N, 2 * VSHIFT + 1), np.nan, np.float32)
    for t in tracks:
        ok = t["frame"] < N; D[t["frame"][ok]] = t["d"][ok]
    np.savez_compressed(pathlib.Path(a.out).with_suffix(".npz"), d=D)
    gap = lip_gap(pathlib.Path(a.mp_npz)) if a.mp_npz else None
    if gap is not None and len(gap) < N: gap = np.concatenate([gap, np.full(N - len(gap), np.nan)])
    drv = pathlib.Path(a.audio) if a.audio else None
    e = np.log(env25(drv, N) + 1e-4) if drv else None
    es = env25(pathlib.Path(a.src_audio), N) if a.src_audio else None
    ed = env25(drv, N) if drv else None
    thr_d = 0.1 * np.percentile(ed, 95) if ed is not None else None; thr_s = 0.1 * np.percentile(es, 95) if es is not None else None
    lo, hi = (0, N) if not a.frames else map(int, a.frames.split(":"))
    B = int(round(a.bin * FPS)); C = int(round(a.ctx * FPS)); rows = []
    for s0 in range(lo, hi, B):
        r = {"t": round(s0 / FPS, 2)}; w0, w1 = max(0, s0 - C), min(N, s0 + B + C)
        dd = D[w0:w1]; dd = dd[np.isfinite(dd[:, 0])]
        r["n"] = int(len(dd))
        if len(dd) >= 10:
            md = dd.mean(0); r["conf_best"] = round(float(np.median(md) - md.min()), 3); r["offset_best"] = int(VSHIFT - md.argmin()); r["conf_at0"] = round(float(np.median(md) - md[VSHIFT]), 3)
        if ed is not None: r["dub_act"] = round(float((ed[s0:s0 + B] > thr_d).mean()), 2)
        if es is not None: r["src_act"] = round(float((es[s0:s0 + B] > thr_s).mean()), 2)
        if gap is not None and e is not None:
            g, ee = gap[w0:w1], e[w0:w1]; r["lip_r0"] = None if np.isnan(c := corr(g, ee)) else round(c, 3)
            best = (np.nan, 0)
            for lag in range(-3, 4):   # lag > 0: lips follow the audio by `lag` frames
                gg = gap[max(0, w0 + lag):max(0, w1 + lag)]; m = min(len(gg), len(ee)); c = corr(gg[:m], ee[:m])
                if np.isfinite(c) and (not np.isfinite(best[0]) or c > best[0]): best = (c, lag)
            r["lip_rbest"] = None if not np.isfinite(best[0]) else round(best[0], 3); r["lip_lag"] = best[1]
            r["gap_mean"] = round(float(np.nanmean(gap[s0:s0 + B])), 4) if np.isfinite(gap[s0:s0 + B]).any() else None
        rows.append(r)
    res["per_second"] = rows
    if gap is not None and e is not None:
        res["lip_env_r_global"] = round(corr(gap, e), 3)
        lags = {}
        for lag in range(-4, 5):
            g = gap[max(0, lag):N + min(0, lag)]; ee = e[max(0, -lag):N - max(0, lag)]; lags[lag] = round(corr(g, ee), 3)
        res["lip_env_r_by_lag"] = lags
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    c0 = [r["conf_at0"] for r in rows if "conf_at0" in r]
    print(json.dumps({"global": res["global"], "seconds": len(rows), "conf_at0_mean": round(float(np.mean(c0)), 3) if c0 else None,
                      "lip_env_r": res.get("lip_env_r_global"), "by_lag": res.get("lip_env_r_by_lag")}))


if __name__ == "__main__":
    main()
