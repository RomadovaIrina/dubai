#!/usr/bin/env python3
"""Labelled side-by-side review clip for panels with DIFFERENT frame rates (25 fps master-grid outputs vs source-fps composites):
every panel is cut by TIME and resampled to one display rate by its own timestamps (a 25 fps panel shows its real cadence: one frame
repeated in six at 30 fps), so judder differences stay visible. Audio from one panel (or none for slow motion).

    python scripts/quality/sbs_mixed_fps.py --out x.mp4 --start 22.5 --dur 5 --fps 30 \
        --panel ORIGINAL=03.mp4 --panel CAND5=03_final.mp4 --panel OPT1=03_OPT1.mp4 [--audio-from 1] [--zoom x1,y1,x2,y2] [--speed 0.5]
"""
from __future__ import annotations
import argparse, pathlib, subprocess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--start", type=float, required=True); ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--panel", action="append", required=True, help="LABEL=path"); ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--audio-from", type=int, default=-1); ap.add_argument("--zoom", default=None, help="x1,y1,x2,y2 crop (2x) in source pixels")
    ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--title", default="")
    a = ap.parse_args()
    ins, flt, labs = [], [], []
    for i, p in enumerate(a.panel):
        lab, path = p.split("=", 1); labs.append(f"[p{i}]")
        ins += ["-ss", f"{a.start:.3f}", "-t", f"{a.dur:.3f}", "-i", path]
        crop = ""
        if a.zoom:
            x1, y1, x2, y2 = [int(v) for v in a.zoom.split(",")]; crop = f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale={2 * (x2 - x1)}:{2 * (y2 - y1)}:flags=lanczos,"
        lab_e = lab.replace(":", "\\:").replace("'", "")
        flt.append(f"[{i}:v]fps={a.fps},{crop}setsar=1,drawtext=text='{lab_e}':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.55[p{i}]")
    title = (a.title + "  " if a.title else "") + f"%{{pts\\:hms\\:{a.start:.2f}}}" + (f"  x{a.speed:g} speed" if a.speed != 1 else "")
    st = f"{''.join(labs)}hstack=inputs={len(labs)}" + (f",setpts={1 / a.speed:.4f}*PTS" if a.speed != 1 else "") + \
         f",drawtext=text='{title}':x=8:y=h-34:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[v]"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex", ";".join(flt + [st]), "-map", "[v]"]
    if a.audio_from >= 0 and a.speed == 1: cmd += ["-map", f"{a.audio_from}:a:0", "-c:a", "aac", "-shortest"]
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(".tmp_" + out.name)
    cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", f"{a.fps * a.speed:g}" if a.speed != 1 else f"{a.fps:g}", str(tmp)]
    subprocess.run(cmd, check=True); tmp.rename(out); print(out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
