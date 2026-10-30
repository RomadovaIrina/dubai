#!/usr/bin/env python3
"""r8-sep: separate the raw test videos with one separator, model loaded once (background_bed.separate_background), into a stem cache
<out>/<id>/{bg_44k.wav, vocals_44k.wav, sep.json}. The first video is preceded by a 10-s warm-up so its separate_s is not the CUDA init.
    DUB_ALLOW_NC_BED=1 [DUB_SEP_BG=residual] flock /workspace/dub/gpu.lock python scripts/r8/sep_stems.py --sep htdemucs --out /workspace/dub/r8a/stems_htdemucs 01 02 ..."""
from __future__ import annotations
import os
import argparse, json, pathlib, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
RAW = pathlib.Path(os.environ.get("DUB_INPUTS", "/workspace/dub/inputs/test_videos_raw"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sep", required=True, choices=["spleeter", "hdemucs", "htdemucs", "htdemucs_ft"]); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("ids", nargs="+"); a = ap.parse_args()
    import background_bed as bb, subprocess
    O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:   # warm-up: 10 s of the first video (model load + kernels), not timed per video
        src = next(RAW.glob(f"{a.ids[0]}.*")); w = pathlib.Path(td) / "w.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-t", "10", "-vn", str(w)], check=True)
        t = time.perf_counter(); bb.separate_background(w, pathlib.Path(td), a.device, a.sep); warm = round(time.perf_counter() - t, 2)
    rows = []
    for id_ in a.ids:
        src = next(RAW.glob(f"{id_}.*")); d = O / id_; d.mkdir(exist_ok=True)
        r = bb.separate_background(src, d, a.device, a.sep); r["id"] = id_; r["warmup_s"] = warm
        r["s_per_video_min"] = round(r["separate_s"] / (r["seconds"] / 60), 3); (d / "sep.json").write_text(json.dumps(r, indent=1)); rows.append(r)
        print(id_, a.sep, bb.SEP_BG, f"sep {r['separate_s']} s / {r['seconds']} s = {r['s_per_video_min']} s per video-min (load {r['load_s']}, warm-up {warm})", flush=True)
    (O / "sep_all.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
