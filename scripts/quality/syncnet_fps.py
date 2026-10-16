#!/usr/bin/env python3
"""SyncNet (LatentSync evaluator) for outputs at ANY frame rate: a non-25 fps video is first rendered to 25 fps with the same conversion
as the pipeline's master (ffmpeg -vf fps=25, nearest frame), because the evaluator's own `-r 25` re-timing shifts 30 fps inputs (the
original 30 fps 03 scores AV offset 2 that way). Scores the muxed audio ("mix") and, with --dub, the dub voice only.
    python scripts/quality/syncnet_fps.py VIDEO [--dub dubbed_aac.m4a] [--json out.json]"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))


def fps_of(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()
    n, d = (o.split("/") + ["1"])[:2]; return float(n) / float(d)


def main() -> int:
    from syncnet_common import eval_syncnet
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--dub", default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args(); tmp = pathlib.Path(tempfile.mkdtemp()); res = {"fps": round(fps_of(a.video), 3)}
    v25 = pathlib.Path(a.video)
    if abs(res["fps"] - 25.0) > 0.01:
        v25 = tmp / "v25.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video, "-vf", "fps=25", "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "12",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", str(v25)], check=True); res["evaluated_as"] = "fps=25 rendition"
    o = eval_syncnet(v25); res["mix"] = [o["confidence"], o["av_offset_frames"]]
    if a.dub:
        t = tmp / "dub.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(v25), "-i", a.dub, "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(t)], check=True)
        o = eval_syncnet(t); res["dub_only"] = [o["confidence"], o["av_offset_frames"]]
    print(json.dumps(res))
    if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
