#!/usr/bin/env python3
"""Listening clips for the timing A/B (Agent C): ORIGINAL master frames (lips NOT re-synced) + a candidate dub track, one mp4 per
(window, policy), label burned in (policy, window, "lips NOT re-synced", gap marks). Dub voice only, no background bed (HDEMUCS = NO),
unless a track is given explicitly. A red bar at the top marks the residual "source speaks, dub silent" gaps > --mark-s of that policy
(from its dub_gaps.py JSON) so the listener can find them.

    python scripts/quality/listen_clips.py --master work/master_25fps.mp4 --vid 03 --out OUTBOX/audio-p1 \
        --track ref=AB/03_ref/dubbed_24k.wav:AB/gaps/03_ref.gaps.json --track L1=... --window 22.5-27.5 --window 32.5-39
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True); ap.add_argument("--vid", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--track", action="append", required=True, help="name=dub.wav[:gaps.json]"); ap.add_argument("--window", action="append", required=True, help="t0-t1 (s)")
    ap.add_argument("--mark-s", type=float, default=0.3); ap.add_argument("--suffix", default="dubvoice-only_nobed_lips-NOT-resynced")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); made = []
    for w in a.window:
        t0, t1 = (float(x) for x in w.split("-")); d = t1 - t0
        for i, tr in enumerate(a.track):
            name, rest = tr.split("=", 1); wav, _, gj = rest.partition(":")
            vf = [f"drawtext=fontfile={FONT}:text='{a.vid} {name}  {t0:.1f}-{t1:.1f}s  {'ORIGINAL AUDIO' if name == 'orig' else 'dub voice only, lips NOT re-synced'}':x=10:y=h-32:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6",
                  f"drawtext=fontfile={FONT}:text='%{{pts\\:hms\\:{t0:.3f}}}':x=w-170:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6"]
            if gj:
                for g in json.loads(pathlib.Path(gj).read_text())["gaps"]:
                    if g["seconds"] > a.mark_s and g["t1"] > t0 and g["t0"] < t1:
                        vf.append(f"drawbox=x=0:y=0:w=iw:h=14:color=red@0.9:t=fill:enable='between(t,{g['t0'] - t0:.3f},{g['t1'] - t0:.3f})'")
            f = out / f"{a.vid}_{int(t0):03d}s_{i}_{name}_{'original-audio_reference' if name == 'orig' else a.suffix}.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{d:.3f}", "-i", a.master, "-ss", f"{t0:.3f}", "-t", f"{d:.3f}", "-i", wav,
                            "-map", "0:v:0", "-map", "1:a:0", "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", str(f)], check=True)
            made.append(f.name)
    print("\n".join(made)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
