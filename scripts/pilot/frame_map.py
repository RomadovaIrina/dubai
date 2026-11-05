#!/usr/bin/env python3
"""Exact source-frame <-> 25 fps master-frame mapping (track A, 2026-10-16; spec 9.4: the final MP4 must have the SOURCE frame rate).

The video stage works on a 25 fps CFR master made by `ffmpeg -i SRC -r 25` (face_aware_latentsync.make_master). Which source frame each
master frame holds is decided by ffmpeg's vsync (drop / duplicate with a drift tolerance), NOT by nearest timestamps: on 03 (30 fps CFR) the
first 12 master frames copy source frames 0-11 one-to-one and the master then stays a constant 2 source frames (67 ms) BEHIND the source
clock, while the dubbed audio lives on the source clock. So the mapping is measured, not computed: every master frame is matched to a source
frame by grey-thumbnail difference, searched monotonically (step 0..4 source frames) from the previous match.

Returned dict (stored as manifest["frame_map"]):
  source_pts         per source frame: presentation time (s, first frame = 0), from ffprobe best_effort_timestamp_time
  master_to_source   per master frame k (time k / 25): index of the source frame it shows
  match_mad          per master frame: mean abs grey difference of the best match (0-255; re-encode noise ~1-2)
  lag_s              per master frame: source_pts[master_to_source[k]] - k / 25  (< 0 = master content is OLDER than its timestamp)
  source_master_pos  per source frame j: source_pts[j] * 25 = fractional master index on the AUDIO clock (what a 30 fps composite needs)
  summary            counts: dropped source frames, duplicated, ambiguous matches, lag median / min / max
"""
from __future__ import annotations
import os, pathlib, subprocess
import numpy as np
FPS = 25.0


def _pts(path: pathlib.Path) -> np.ndarray:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    v = np.array([float(x.strip(",")) for x in out if x.strip(",") not in ("", "N/A")])
    return v - v[0] if len(v) else v


def _thumbs(frames_or_path, size=(58, 106)) -> np.ndarray:
    import cv2
    out = []
    if isinstance(frames_or_path, (str, pathlib.Path)):
        cap = cv2.VideoCapture(str(frames_or_path))
        while True:
            ok, f = cap.read()
            if not ok: break
            out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA))
        cap.release()
    else:
        out = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA) for f in frames_or_path]
    return np.stack(out).astype(np.float32)


def exact_master_to_source(src) -> np.ndarray:
    """r15 audit (default; FRAME_MAP=thumb restores the greedy thumbnail match): which source frame each master frame holds, decided EXACTLY
    by replaying the master command (face_aware_latentsync.make_master: `-r 25`, or DUB_MASTER_TIMING=fps `-vf fps=25`) on a tiny index video
    with the source's own timestamps (each frame's index as 14 bit blocks), as scripts/quality/r3_master_timing.py does. The greedy match
    disagreed with that map on 249 of 2258 old03 master frames (up to 3 source frames: a landmark spike when FAST_MPREUSE seeds the master pass)."""
    import av, tempfile
    c = av.open(str(src)); s = c.streams.video[0]; tb = s.time_base; rate = s.average_rate
    pts = np.array(sorted(p.pts for p in c.demux(s) if p.pts is not None), np.int64); c.close()
    NB, BS = 14, 16; W, H = 8 + NB * (BS + 4), BS + 16
    tm = ["-vf", f"fps={FPS:g}"] if os.environ.get("DUB_MASTER_TIMING", "r25") == "fps" else ["-r", f"{FPS:g}"]
    with tempfile.TemporaryDirectory() as td:
        idx = pathlib.Path(td) / "idx.mp4"; out = pathlib.Path(td) / "idx_25.mp4"
        o = av.open(str(idx), "w"); st = o.add_stream("libx264rgb", rate=rate); st.width, st.height = W, H; st.pix_fmt = "rgb24"; st.time_base = tb
        st.codec_context.time_base = tb; st.options = {"crf": "0", "preset": "ultrafast"}
        for k, p in enumerate(pts):
            img = np.zeros((H, W, 3), np.uint8)
            for b in range(NB):
                if (k >> b) & 1: img[8:8 + BS, 8 + b * (BS + 4):8 + b * (BS + 4) + BS] = 255
            f = av.VideoFrame.from_ndarray(img, format="rgb24"); f.pts = int(p); f.time_base = tb
            for q in st.encode(f): o.mux(q)
        for q in st.encode(): o.mux(q)
        o.close()
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(idx)] + tm + ["-crf", "18", "-an", str(out)], check=True)
        c = av.open(str(out)); j = []
        for fr in c.decode(video=0):
            g = fr.to_ndarray(format="gray"); j.append(sum(1 << b for b in range(NB) if g[8 + BS // 2, 8 + b * (BS + 4) + BS // 2] > 127))
        c.close()
    return np.array(j, np.int64)


def frame_map(src: pathlib.Path, master_frames, max_step: int = 4) -> dict:
    """src: source video; master_frames: the 25 fps master (path or in-memory BGR frames)."""
    sp = _pts(pathlib.Path(src)); S = _thumbs(src); M = _thumbs(master_frames)
    sp = sp[:len(S)] if len(sp) >= len(S) else np.concatenate([sp, sp[-1] + np.arange(1, len(S) - len(sp) + 1) / FPS])
    jm = np.zeros(len(M), int); mad = np.zeros(len(M)); amb = 0; prev = 0
    for k in range(len(M)):
        lo, hi = (0, min(len(S), 9)) if k == 0 else (prev, min(len(S), prev + max_step + 1))
        d = np.abs(S[lo:hi] - M[k]).mean(axis=(1, 2)); o = np.argsort(d)
        jm[k] = lo + int(o[0]); mad[k] = float(d[o[0]]); prev = jm[k]
        if len(o) > 1 and d[o[1]] < 1.1 * d[o[0]] + 0.05: amb += 1
    jm_thumb = jm.copy(); method = "grey 58x106 thumbnail match, monotone search (step 0..%d source frames)" % max_step
    if os.environ.get("FRAME_MAP", "exact") != "thumb":   # r15 audit (default): the exact replayed map (exact_master_to_source)
        try:
            ex = exact_master_to_source(src)
            if len(ex) >= len(M) and ex[:len(M)].max() < len(S):
                jm = ex[:len(M)].astype(int); mad = np.abs(S[jm] - M).mean(axis=(1, 2)); method = "exact: master command replayed on an index video (thumbnail match kept as a check)"
            else: print(f"  [frame_map] exact map unusable ({len(ex)} frames, max {ex.max() if len(ex) else -1}); thumbnail match kept", flush=True)
        except Exception as e_:
            print(f"  [frame_map] exact map failed ({e_!r}); thumbnail match kept", flush=True)
    used = np.bincount(jm, minlength=len(S)); lag = sp[jm] - np.arange(len(M)) / FPS
    return {"method": method, "master_fps": FPS, "thumb_disagree": int((jm_thumb != jm).sum()),
            "source_frames": int(len(S)), "master_frames": int(len(M)), "source_pts": np.round(sp, 5).tolist(),
            "master_to_source": jm.tolist(), "match_mad": np.round(mad, 2).tolist(), "lag_s": np.round(lag, 4).tolist(),
            "source_master_pos": np.round(sp * FPS, 4).tolist(),
            "summary": {"dropped_source_frames": int((used == 0).sum()), "duplicated_source_frames": int((used > 1).sum()), "ambiguous_matches": int(amb),
                        "match_mad_p99": round(float(np.percentile(mad, 99)), 2), "lag_s_median": round(float(np.median(lag)), 4),
                        "lag_s_min": round(float(lag.min()), 4), "lag_s_max": round(float(lag.max()), 4)}}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source"); ap.add_argument("master"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); r = frame_map(pathlib.Path(a.source), pathlib.Path(a.master))
    print(json.dumps(r["summary"]))
    if a.out: pathlib.Path(a.out).write_text(json.dumps(r))
