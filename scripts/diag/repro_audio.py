#!/usr/bin/env python3
"""Diagnostic: re-run ONLY the audio stages of run_clean_pipeline_05.py (production `--alignment burst` path, unchanged code) and
keep every intermediate needed for the audio-lineage analysis. No pipeline file is modified; the video stage is replaced by a stop.

    source scripts/env.sh
    python scripts/diag/repro_audio.py --id 01 --out /workspace/dub/tts_diag/repro [-- extra runner args]

Outputs (<out>/<id>/):  work_<id>/ (audio16k, ref_*.wav, tts_u*_p*.wav = RAW Chatterbox, aligned_u*.wav, dubbed_24k.wav, dubbed_aac.m4a),
                        <id>.manifest.json (same schema as the production manifest, written by the runner on the stop exception),
                        atempo/<tag>_<burst>.wav (every chunk right after the atempo filter, exactly what place_groups produced).
"""
from __future__ import annotations
import argparse, pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality"))
import run_clean_pipeline_05 as rcp   # noqa: E402
import e1_burst_align as e1           # noqa: E402


class StopAfterAudio(RuntimeError):
    pass


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=str(ROOT / "test_videos")); ap.add_argument("--target-lang", default="en")
    a, extra = ap.parse_known_args()
    src = next(p for p in pathlib.Path(a.video_dir).iterdir() if p.stem == a.id and p.suffix.lower() in (".mp4", ".mov"))
    out = pathlib.Path(a.out) / a.id; out.mkdir(parents=True, exist_ok=True)
    trace = out / "atempo"; trace.mkdir(exist_ok=True)
    orig_ff = e1.ff

    def traced_ff(args):
        orig_ff(args)
        if "-af" in args and str(args[args.index("-af") + 1]).startswith("atempo=") and args[-1].endswith("_t.wav"):
            name = pathlib.Path(args[-1]).name[len("grp_"):-len("_t.wav")]      # e.g. u03_1 or u03_slot
            shutil.copy(args[-1], trace / f"{name}.wav")
    e1.ff = traced_ff

    def stop(*_a, **_k):
        raise StopAfterAudio("audio stages done (video stage intentionally skipped)")
    rcp.run_lipsync = stop
    argv = ["run_clean_pipeline_05.py", "--input", str(src), "--output", str(out / f"{a.id}.mp4"), "--target-lang", a.target_lang,
            "--alignment", "burst", "--no-codeformer", "--work-dir", str(out), *extra]
    sys.argv = argv
    try:
        rcp.main()
    except StopAfterAudio:
        print("OK: audio stages finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
