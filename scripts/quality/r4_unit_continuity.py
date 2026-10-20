#!/usr/bin/env python3
"""Round 4 A: voice continuity BETWEEN consecutive sentences (units) of the same speaker on the placed dub track: voiced level (dB, frames within
20 dB of the unit's peak) and median F0 per unit; jumps = |difference| between consecutive same-speaker units (r3_joins.py covers the joins
INSIDE a sentence). Every unit is its own Chatterbox generation (own seed), so level / pitch can step between sentences.
    python scripts/quality/r4_unit_continuity.py MANIFEST     (dubbed_24k.wav next to it)"""
import json, pathlib, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r3_joins import f0_med


def unit_stats(mp, source=False):
    """source=True: the same per-unit numbers on the source vocal stem over the unit's source span (natural sentence-to-sentence variation)."""
    mp = pathlib.Path(mp); m = json.loads(mp.read_text())
    y, sr = sf.read(str(mp.parent / ("vocals_44k.wav" if source else "dubbed_24k.wav")), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    rows = []
    for u in m["units"]:
        pl = [{"placed": (u["start"], u["end"])}] if source else ((u.get("alignment") or {}).get("placements") or [])
        if not pl: continue
        seg = np.concatenate([y[int(p["placed"][0] * sr):int(p["placed"][1] * sr)] for p in pl])
        if len(seg) < sr // 5: continue
        h = int(0.02 * sr); fr = seg[: len(seg) // h * h].reshape(-1, h); d = 10 * np.log10((fr ** 2).mean(1) + 1e-12); v = d[d > d.max() - 20]
        rows.append({"unit": u["id"], "speaker": u.get("speaker"), "level_db": round(float(10 * np.log10(np.mean(10 ** (v / 10)))), 2), "f0": f0_med(seg, sr)})
    return rows


def analyse(mp, source=False):
    rows = unit_stats(mp, source); dl, df = [], []
    for a, b in zip(rows, rows[1:]):
        if a["speaker"] != b["speaker"]: continue
        dl.append(abs(b["level_db"] - a["level_db"]))
        if a["f0"] and b["f0"]: df.append(abs(12 * np.log2(b["f0"] / a["f0"])))
    p = lambda x, q: round(float(np.percentile(x, q)), 2) if x else None
    return {"pairs": len(dl), "level_jump_p50_db": p(dl, 50), "level_jump_p90_db": p(dl, 90), "f0_jump_p50_st": p(df, 50), "f0_jump_p90_st": p(df, 90), "rows": rows}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        for src in (False, True): r = analyse(f, src); print(f, "SOURCE" if src else "DUB", json.dumps({k: v for k, v in r.items() if k != "rows"}))
