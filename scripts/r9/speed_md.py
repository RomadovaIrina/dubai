#!/usr/bin/env python3
"""r9: the speed baseline table (s/video-min, production only) from the speed.py json rows of both servers.
    python scripts/r9/speed_md.py --out reports/experiments/r9-e2e/speed.md speed_a.json speed_f.json"""
import argparse, json, pathlib

COLS = [("audio", "audio"), ("audio.base_pipeline", "a: base (ASR, diar, Qwen, TTS)"), ("audio.base.tts", "a: base TTS"), ("audio.base.loads", "a: model loads"),
        ("audio.oneproc(retake/clause/align)", "a: retake / clause / align"), ("audio.rest(place/hush/soften/remix)", "a: place / hush / remix"),
        ("sep", "sep (htdemucs)"), ("face_gen", "face_gen"), ("face_gen.latentsync", "fg: LatentSync + CF"), ("face_gen.own_composite(discarded)", "fg: own composite (discarded)"),
        ("face_comp", "face_comp"), ("production", "PRODUCTION")]
ORDER = ["m19", "m01", "m04", "m03", "m15", "m17", "m18", "m21", "m20"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("rows", nargs="+"); a = ap.parse_args()
    rows = [r for p in a.rows for r in json.loads(pathlib.Path(p).read_text())]; rows.sort(key=lambda r: ORDER.index(r["id"]) if r["id"] in ORDER else 99)
    fmt = lambda v: "" if v is None else f"{v:.0f}"
    L = ["| video | server | s | " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * (3 + len(COLS))]
    for r in rows: L.append(f"| {r['id']} | {r['server']} | {r['dur_s']:.0f} | " + " | ".join(fmt(r.get(c[0] + " /vmin")) for c in COLS) + " |")
    def agg(sel, name):
        vm = sum(r["dur_s"] for r in sel) / 60
        L.append(f"| **{name}** | | {sum(r['dur_s'] for r in sel):.0f} | " + " | ".join("**" + fmt(sum((r.get(c[0]) or 0) for r in sel) / vm) + "**" for c in COLS) + " |")
    agg(rows, "all 9 (time-weighted)"); agg([r for r in rows if r["id"] != "m21"], "the 8 with a face")
    agg([r for r in rows if r["id"] not in ("m21", "m01", "m04")], "the 6 with a face at <= 720x1280, 30 fps")
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
