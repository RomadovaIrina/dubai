#!/usr/bin/env python3
"""Old vs new at chosen timestamps: cut the audio of both final videos around each spot, run the independent ASRs, write old/new wav samples + a table.
    python scripts/diag/compare_spots.py --old first_res/01_e2e_final.mp4 --new result_videos/.../01_final.mp4 --times 35.3 37.8 --out <dir> [--win 1.6]"""
import argparse, json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot")); import tts_qa as q
ap = argparse.ArgumentParser(); ap.add_argument("--old"); ap.add_argument("--new"); ap.add_argument("--times", nargs="+", type=float); ap.add_argument("--out"); ap.add_argument("--win", type=float, default=1.6); ap.add_argument("--manifest")
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); SR = 24000
def audio(p):
    t = out / "_t.wav"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(t)], check=True); y, _ = sf.read(str(t), dtype="float32"); t.unlink(); return y
yo, yn = audio(a.old), audio(a.new); qa = q.TtsQA(); rows = []
for t in a.times:
    r = {"t": t}
    for tag, y in (("old", yo), ("new", yn)):
        seg = y[max(0, int((t - 0.3) * SR)):int((t + a.win) * SR)]; sf.write(str(out / f"t{t:06.1f}_{tag}.wav"), seg, SR, subtype="PCM_16")
        x = qa.assess("", seg) if False else None
        y16 = qa.to16(seg); r[tag] = {"whisper": qa.whisper(y16), "ctc": qa.ctc(y16)}
    rows.append(r); print(f"t={t:6.1f}s\n   OLD whisper: {r['old']['whisper']!r}\n   NEW whisper: {r['new']['whisper']!r}\n   OLD ctc: {r['old']['ctc'][:70]!r}\n   NEW ctc: {r['new']['ctc'][:70]!r}")
(out / "spots.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
