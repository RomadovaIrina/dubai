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
import pathlib, subprocess
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
    used = np.bincount(jm, minlength=len(S)); lag = sp[jm] - np.arange(len(M)) / FPS
    return {"method": "grey 58x106 thumbnail match, monotone search (step 0..%d source frames)" % max_step, "master_fps": FPS,
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
