#!/usr/bin/env python3
"""Phase 5 human-review clips: baseline | candidate side by side (labels, timecode, green bar where LatentSync is active in the
candidate), candidate audio on the stacked clip, plus separate A (baseline) and B (candidate) cuts with their own audio.
    python scripts/quality/make_comparison.py --id 04 --baseline /tmp/dabai_quality/baseline/04_baseline.mp4 \
        --candidate /tmp/dabai_quality/candidates/04_e1.mp4 --tag e1 --ranges 8-20 33-48 [--out /tmp/dabai_quality/comparisons]
Without --ranges the worst scenes are picked from the candidate manifest: longest LatentSync segments (up to 3, 12 s each).
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess
FPS = 25


def ls_intervals(mp4: pathlib.Path):
    mp = mp4.with_name(mp4.stem + ".manifest.json")
    if not mp.exists(): return []
    segs = json.loads(mp.read_text())["lipsync"]["segments"]
    return [(s["start_frame"] / FPS, s["end_frame"] / FPS) for s in segs if s["action"] == "LATENT_SYNC"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--baseline", required=True); ap.add_argument("--candidate", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--ranges", nargs="*", default=None, help="start-end seconds"); ap.add_argument("--out", default="/tmp/dabai_quality/comparisons"); ap.add_argument("--clip-s", type=float, default=12.0)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); cand = pathlib.Path(a.candidate); base = pathlib.Path(a.baseline)
    ranges = [tuple(float(x) for x in r.split("-")) for r in a.ranges] if a.ranges else None
    if not ranges:
        ls = sorted(ls_intervals(cand), key=lambda x: -(x[1] - x[0]))[:3]
        ranges = [(max(0.0, s - 1.0), max(0.0, s - 1.0) + min(a.clip_s, e - s + 2.0)) for s, e in sorted(ls)] or [(0.0, a.clip_s)]
    made = []
    for k, (s, e) in enumerate(ranges, 1):
        d = e - s; stem = f"{a.id}_{a.tag}_scene{k}_{s:.0f}s"
        bar = "".join(f",drawbox=x=0:y=0:w=iw:h=8:color=lime@0.8:t=fill:enable='between(t,{max(ls_s - s, 0):.3f},{min(ls_e - s, d):.3f})'" for ls_s, ls_e in ls_intervals(cand) if ls_e > s and ls_s < e)
        lab = lambda txt: f"drawtext=text='{txt}':x=8:y=14:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55"
        tc = f"drawtext=text='%{{eif\\:t+{s:.2f}\\:d}}.%{{eif\\:mod(t*100\\,100)\\:d\\:2}}s':x=8:y=h-30:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55"
        fc = (f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,{lab('A baseline')},{tc}[l];"
              f"[1:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,{lab('B ' + a.tag)}{bar},{tc}[r];[l][r]hstack=inputs=2[v];"
              f"[1:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a]")
        p = out / f"{stem}_sidebyside_audioB.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(base), "-i", str(cand), "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "128k", str(p)], check=True)
        for tag, src in (("A_baseline", base), ("B_" + a.tag, cand)):
            q = out / f"{stem}_{tag}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", str(src), "-vf", f"{lab(tag.replace('_', ' '))},{tc}", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "128k", str(q)], check=True)
        made.append(str(p)); print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
