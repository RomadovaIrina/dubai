#!/usr/bin/env python3
"""SyncNet on the DUB-SPEECH parts only (2026-10-14).

With the thought pause policy the dub is silent for much longer while the face is still on screen (03: source speech without dub 11 -> 32 s),
and SyncNet averages its confidence over every window of the video: silent windows have no lip-audio correlation, so the whole-video number
drops even if the lips are as well synced as before wherever the dub speaks. This cuts the output video (with the dub voice only) to the dub
speech intervals (Silero VAD of dubbed_16k, >= --min-s, +-0.2 s), concatenates them and scores that with the same SyncNet evaluator.

    python scripts/quality/syncnet_speech_only.py RUN/final/03_final.manifest.json [more manifests]
"""
from __future__ import annotations
import json, pathlib, shutil, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))


def main() -> int:
    import argparse
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    from syncnet_common import eval_syncnet
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+"); ap.add_argument("--min-s", type=float, default=1.0)
    a = ap.parse_args(); model = load_silero_vad(); out = {}
    for mp in a.manifests:
        man = json.loads(pathlib.Path(mp).read_text()); dur = man["source"]["duration_s"]
        ts = get_speech_timestamps(read_audio(man["dubbed_track"]["wav16k"], sampling_rate=16000), model, sampling_rate=16000, return_seconds=True)
        iv = [(max(0.0, t["start"] - 0.2), min(dur, t["end"] + 0.2)) for t in ts if t["end"] - t["start"] >= a.min_s]
        tmp = pathlib.Path(tempfile.mkdtemp()); parts = []
        try:
            for k, (s, e) in enumerate(iv):
                p = tmp / f"p{k:03d}.mp4"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", man["output"], "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                                "-i", man["dubbed_track"]["wav16k"], "-map", "0:v:0", "-map", "1:a:0", "-r", "25", "-c:v", "libx264", "-crf", "16", "-c:a", "aac", str(p)], check=True)
                parts.append(p)
            lst = tmp / "list.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts)); cat = tmp / "speech_only.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(cat)], check=True)
            o = eval_syncnet(cat)
            out[mp] = {"segments": len(iv), "seconds": round(sum(e - s for s, e in iv), 1), "confidence": o["confidence"], "av_offset": o["av_offset_frames"]}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(mp, json.dumps(out[mp]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
