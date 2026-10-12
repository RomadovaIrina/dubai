#!/usr/bin/env python3
"""Deterministic edge-cut check of the TTS spans of a run (no ASR): a span that starts or ends while the voice is still loud was cut
inside a sound ("swallowed ending", "voice breaks off", clipped onset), whereas a clean cut lands in a pause / on a decay.

For every tts_u<id>_p<k>.wav of the manifest: 10 ms RMS frames, reference = 90th percentile of the span; head = max RMS of the first
`edge_ms` (after the 5 ms fade), tail = max RMS of the last `edge_ms` (before the fade), both in dB relative to the reference.
hot = within `hot_db` of the reference. Sentence-final tails (last span of a unit) and sentence-initial heads are reported separately
from the inner cuts between bursts.

    python scripts/quality/edge_cuts.py --manifest RUN/03_final.manifest.json [--json out.json]
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np, soundfile as sf


def edges(y: np.ndarray, sr: int, edge_ms: float = 25.0, fade_ms: float = 5.0) -> tuple[float, float, float]:
    h = int(0.01 * sr); n = len(y) // h
    if n < 6:
        return 0.0, 0.0, -120.0
    rms = 20 * np.log10(np.sqrt(np.mean(y[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12))
    ref = float(np.percentile(rms, 90)); f = int(np.ceil(fade_ms / 10)); k = max(1, int(round(edge_ms / 10)))
    head = float(rms[f:f + k].max()) - ref; tail = float(rms[n - f - k:n - f].max()) - ref
    return round(head, 1), round(tail, 1), round(ref, 1)


def check(man: dict, hot_db: float = -15.0) -> dict:
    rows = []
    for u in man["units"]:
        tps = u.get("tts_parts") or []
        for k, tp in enumerate(tps):
            y, sr = sf.read(tp["file"], dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            head, tail, ref = edges(y, sr)
            rows.append({"unit": u["id"], "part": k, "text": tp["text"], "seconds": tp["seconds"], "head_db": head, "tail_db": tail, "ref_db": ref,
                         "initial": k == 0, "final": k == len(tps) - 1, "hot_head": head > hot_db, "hot_tail": tail > hot_db})
    fin = [r for r in rows if r["final"]]; ini = [r for r in rows if r["initial"]]; inner_t = [r for r in rows if not r["final"]]; inner_h = [r for r in rows if not r["initial"]]
    summ = {"spans": len(rows), "hot_db": hot_db,
            "final_tails_hot": sum(r["hot_tail"] for r in fin), "final_tails": len(fin), "final_tail_db_median": float(np.median([r["tail_db"] for r in fin])) if fin else None,
            "initial_heads_hot": sum(r["hot_head"] for r in ini), "initial_heads": len(ini),
            "inner_tails_hot": sum(r["hot_tail"] for r in inner_t), "inner_heads_hot": sum(r["hot_head"] for r in inner_h), "inner_cuts": len(inner_t)}
    return {"summary": summ, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--json", default=None); ap.add_argument("--hot-db", type=float, default=-15.0)
    a = ap.parse_args()
    rep = check(json.loads(pathlib.Path(a.manifest).read_text()), a.hot_db)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep["summary"]))
    for r in rep["rows"]:
        if (r["final"] and r["hot_tail"]) or (r["initial"] and r["hot_head"]) or r["hot_tail"] or r["hot_head"]:
            print(f"  u{r['unit']:02d}p{r['part']} {'F' if r['final'] else ' '}{'I' if r['initial'] else ' '} head {r['head_db']:+6.1f} tail {r['tail_db']:+6.1f}  {r['text'][-60:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
