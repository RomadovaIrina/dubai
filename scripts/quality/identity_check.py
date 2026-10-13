#!/usr/bin/env python3
"""Is a speed lever really quality-neutral? Compare two finished runs of the same video: translations / TTS texts / placements, LatentSync
segments, dubbed track samples and every decoded output frame.
    python scripts/quality/identity_check.py --a RUN_A/final/05_final.manifest.json --b RUN_B/final/05_final.manifest.json
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib
import numpy as np


def frames_digest(p):
    import cv2
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(hashlib.md5(f.tobytes()).hexdigest())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    a = ap.parse_args(); A, B = json.loads(pathlib.Path(a.a).read_text()), json.loads(pathlib.Path(a.b).read_text())
    rep = {}
    rep["translations_identical"] = [u.get("translation_fitted") or u["translation"] for u in A["units"]] == [u.get("translation_fitted") or u["translation"] for u in B["units"]]
    rep["parts_identical"] = [[t["text"] for t in u["tts_parts"]] for u in A["units"]] == [[t["text"] for t in u["tts_parts"]] for u in B["units"]]
    pa = [pl["placed"] for u in A["units"] for pl in u["alignment"]["placements"] if pl.get("placed")]; pb = [pl["placed"] for u in B["units"] for pl in u["alignment"]["placements"] if pl.get("placed")]
    rep["placements_identical"] = pa == pb
    import soundfile as sf
    ya, _ = sf.read(A["dubbed_track"]["wav24k"]); yb, _ = sf.read(B["dubbed_track"]["wav24k"])
    rep["dub_track_max_abs_diff"] = float(np.abs(ya[:min(len(ya), len(yb))] - yb[:min(len(ya), len(yb))]).max()) if len(ya) and len(yb) else None
    sa = [(s["action"], s["start_frame"], s["end_frame"]) for s in A.get("lipsync", {}).get("segments", [])]
    sb = [(s["action"], s["start_frame"], s["end_frame"]) for s in B.get("lipsync", {}).get("segments", [])]
    rep["ls_segments_identical"] = sa == sb
    fa, fb = frames_digest(A["output"]), frames_digest(B["output"])
    rep["frames"] = [len(fa), len(fb)]; rep["frames_differing"] = sum(x != y for x, y in zip(fa, fb)) + abs(len(fa) - len(fb))
    rep["wall_s"] = [A["stage_seconds"]["total"], B["stage_seconds"]["total"]]
    print(json.dumps(rep, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
