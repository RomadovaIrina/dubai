#!/usr/bin/env python3
"""Labelled side-by-side review clip: N panels (e.g. ORIGINAL | BASELINE | CANDIDATE), the SAME frames of each input (frame offsets
per input), H.264 crf 18, audio from one of the inputs. --zoom crops every panel to a fixed face box (2x) so mouth twitches are visible.

    python scripts/quality/make_sbs.py --out sbs/03_47.8s.mp4 --start 47.0 --dur 8 \
        --panel ORIGINAL=master_25fps.mp4@0 --panel BASELINE=03_L3.mp4@40 --panel CANDIDATE=03_S1.mp4@40 --audio-from 2 [--zoom x1,y1,x2,y2]
  (@T = time in the source timeline where that file starts, e.g. a window clip starting at 40 s)
"""
from __future__ import annotations
import argparse, pathlib, subprocess
import numpy as np
FPS = 25


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--start", type=float, required=True); ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--panel", action="append", required=True, help="LABEL=path@offset_s"); ap.add_argument("--audio-from", type=int, default=-1)
    ap.add_argument("--zoom", default=None, help="x1,y1,x2,y2 face box in source pixels (2x crop of every panel)"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    panels = []
    for p in a.panel:
        lab, rest = p.split("=", 1); path, off = (rest.rsplit("@", 1) + ["0"])[:2] if "@" in rest else (rest, "0")
        panels.append((lab, pathlib.Path(path), float(off)))
    n = int(round(a.dur * FPS)); caps = []
    for lab, path, off in panels:
        cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((a.start - off) * FPS))); caps.append(cap)
    zoom = [int(v) for v in a.zoom.split(",")] if a.zoom else None
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".tmp_" + out.stem + "_v.mp4")
    proc = None
    for i in range(n):
        tiles = []
        for (lab, _, _), cap in zip(panels, caps):
            ok, f = cap.read()
            if not ok: f = np.zeros_like(tiles[-1]) if tiles else np.zeros((848, 464, 3), np.uint8)
            if zoom:
                x1, y1, x2, y2 = zoom; f = cv2.resize(f[y1:y2, x1:x2], ((x2 - x1) * 2, (y2 - y1) * 2), interpolation=cv2.INTER_CUBIC)
            bar = np.zeros((30, f.shape[1], 3), np.uint8)
            cv2.putText(bar, f"{lab}  {a.start + i / FPS:6.2f}s", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, f]))
        sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
        frame = np.hstack(sum([[t, sep] for t in tiles], [])[:-1])
        if frame.shape[1] % 2: frame = np.hstack([frame, np.zeros((frame.shape[0], 1, 3), np.uint8)])
        if frame.shape[0] % 2: frame = np.vstack([frame, np.zeros((1, frame.shape[1], 3), np.uint8)])
        if proc is None:
            h, w = frame.shape[:2]
            proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                                     "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close(); proc.wait()
    lab, path, off = panels[a.audio_from]
    final_tmp = out.with_name(".tmp_" + out.name)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-ss", f"{a.start - off:.3f}", "-t", f"{a.dur:.3f}", "-i", str(path),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(final_tmp)], check=True)
    tmp.unlink(missing_ok=True); final_tmp.replace(out)
    print(out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
