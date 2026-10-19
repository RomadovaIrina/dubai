#!/usr/bin/env python3
"""SyncNet (LatentSync's evaluator) on window stage videos with the run's DUB VOICE only (comparable to "SyncNet dub-only").
    python scripts/quality/window_syncnet.py --dir DECOMP_DIR [--stages master S0_ls_raw ...] --out syncnet.json
The window start / length come from DECOMP_DIR/window_manifest.json; the dub voice is <run work dir>/dubbed_aac.m4a."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--stages", nargs="*", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args(); d = pathlib.Path(a.dir); wm = json.loads((d / "window_manifest.json").read_text())
    man = json.loads(pathlib.Path(wm["source_manifest"]).read_text()); dub = pathlib.Path(man["work_dir"]) / "dubbed_aac.m4a"
    A, B = wm["window"]; fps = wm.get("fps", 25)
    from syncnet_common import eval_syncnet
    stages = a.stages or sorted(p.stem for p in d.glob("*.mp4") if not p.stem.startswith(("sbs", ".")))
    res = {}
    for s in stages:
        tmp = d / f".sync_{s}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(d / f"{s}.mp4"), "-ss", f"{A / fps:.3f}", "-t", f"{(B - A) / fps:.3f}", "-i", str(dub),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "13", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(tmp)], check=True)
        try:
            o = eval_syncnet(tmp); res[s] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}
        except Exception as e:
            res[s] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        tmp.unlink(missing_ok=True); print(s, res[s], flush=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
