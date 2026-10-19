#!/usr/bin/env python3
"""Exact source time of every 25 fps master frame (round 2 finding: `ffmpeg -i SRC -r 25` keeps a frame-dropping schedule that lags the
nominal clock m/25 by ~60 ms on 30 fps sources). Writes <out>.json {"j": source frame index per master frame, "tau_s": its time (pts * tb,
the same convention as fps_composite.py's t_k)} for fps_composite.py --master-timing.

Method: a frame-index video with the SOURCE timestamps (PyAV, same time base / pts; each frame = its index as 14 black / white blocks), run
through make_master's exact ffmpeg command, blocks decoded back. Checked against pixel matching of the real master to the source
(min MSE over j-3..j+3 on every 5th master frame): the agreement is reported.
    python scripts/quality/r3_master_timing.py --src SRC.mp4 --master WORK/master_25fps.mp4 --out timing/<id>_master_timing.json
"""
import argparse, json, pathlib, subprocess, tempfile
import numpy as np

NB = 14; BS = 24


def main():
    import av, cv2
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--master", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = av.open(a.src); s = c.streams.video[0]; tb = s.time_base; rate = s.average_rate; pts = []; grays = []
    for fr in c.decode(s):
        pts.append(fr.pts); grays.append(cv2.resize(fr.to_ndarray(format="gray"), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32))
    W, H = s.codec_context.width, s.codec_context.height; c.close(); pts = np.array(pts, np.int64); n = len(pts)
    with tempfile.TemporaryDirectory() as td:
        idx = pathlib.Path(td) / "idx.mp4"; out = pathlib.Path(td) / "idx_25.mp4"
        o = av.open(str(idx), "w"); st = o.add_stream("libx264rgb", rate=rate); st.width, st.height = W, H; st.pix_fmt = "rgb24"; st.time_base = tb
        st.codec_context.time_base = tb; st.options = {"crf": "0", "preset": "ultrafast"}
        for k in range(n):
            img = np.zeros((H, W, 3), np.uint8)
            for b in range(NB):
                if (k >> b) & 1: img[8:8 + BS, 8 + b * (BS + 4):8 + b * (BS + 4) + BS] = 255
            f = av.VideoFrame.from_ndarray(img, format="rgb24"); f.pts = int(pts[k]); f.time_base = tb
            for p in st.encode(f): o.mux(p)
        for p in st.encode(): o.mux(p)
        o.close()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(idx), "-r", "25", "-crf", "18", "-an", str(out)], check=True)   # = make_master
        c = av.open(str(out)); j = []
        for fr in c.decode(video=0):
            g = fr.to_ndarray(format="gray")
            j.append(sum(1 << b for b in range(NB) if g[8 + BS // 2, 8 + b * (BS + 4) + BS // 2] > 127))
        c.close()
    j = np.array(j, np.int64)
    # independent check on the real master: pixel matching
    c = av.open(a.master); agree = tot = 0; nm = 0; bad = []
    for m, fr in enumerate(c.decode(video=0)):
        nm += 1
        if m % 5 or m >= len(j): continue
        g = cv2.resize(fr.to_ndarray(format="gray"), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
        cand = range(max(0, j[m] - 3), min(n, j[m] + 4)); e = [float(np.mean((grays[q] - g) ** 2)) for q in cand]; best = list(cand)[int(np.argmin(e))]
        srt = sorted(e); tot += 1
        if best == j[m] or (len(srt) > 1 and srt[1] - srt[0] < 0.05 * srt[0] + 0.2): agree += 1   # ambiguous (static) frames count as agreement
        else: bad.append((m, int(j[m]), int(best)))
    c.close()
    tau = (pts[np.clip(j, 0, n - 1)] * float(tb)).tolist(); nominal = np.arange(len(j)) / 25.0
    rep = {"src": a.src, "master": a.master, "source_frames": n, "master_frames_idx": len(j), "master_frames_real": nm, "j": j.tolist(), "tau_s": tau,
           "lag_ms": {"mean": round(float(np.mean(np.array(tau) - nominal)) * 1000, 1), "min": round(float(np.min(np.array(tau) - nominal)) * 1000, 1), "max": round(float(np.max(np.array(tau) - nominal)) * 1000, 1)},
           "src_start_s": float(pts[0] * tb), "pixel_check": {"checked": tot, "agree": agree, "disagree_examples": bad[:10]}}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(rep))
    print(json.dumps({k: rep[k] for k in ("source_frames", "master_frames_idx", "master_frames_real", "lag_ms", "src_start_s", "pixel_check")}))


if __name__ == "__main__":
    main()
