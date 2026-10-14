#!/usr/bin/env python3
"""Rhythm of a dub run (2026-10-14 human feedback: "пауза НЕ ТАМ", "не «день ……… сегодня хороший», а «день сегодня хороший ………»").

From a run manifest (no models): every placed dub part on the timeline, then
  * pauses INSIDE a sentence (between parts of one unit) vs BETWEEN sentences, counts > 0.3 / 0.8 / 1.5 s and their seconds;
  * weak pauses = an inside-sentence pause > 0.3 s after a part that does not end on punctuation ("test how our | application");
  * tempo (atempo of every part: > 1.0 / max / mean, < 1.0 = slowed down), shift of a part behind its source anchor;
  * source speech without dub / dub outside source speech (Silero VAD of the source = manifest speech_intervals, the dub = placements).

    python scripts/quality/thought_metrics.py RUN/final/04_final.manifest.json [more manifests] [--json out.json]
"""
from __future__ import annotations
import argparse, json, pathlib, re
import numpy as np

PUNCT = re.compile(r"[,;:—–.!?…]['\"»)]*$")


def placed_parts(man: dict) -> list[dict]:
    out = []
    for u in man["units"]:
        for p in u["alignment"].get("placements", []):
            if p.get("placed"):
                out.append({"unit": u["id"], "a": p["placed"][0], "b": p["placed"][1], "text": p.get("text", ""), "atempo": p.get("atempo", 1.0),
                            "shift": p.get("shift_s", 0.0), "how": p.get("how", "")})
    return sorted(out, key=lambda x: x["a"])


def _cover(iv: list, n: int, fps: float = 100.0) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in iv:
        m[max(0, int(a * fps)):min(n, int(np.ceil(b * fps)))] = True
    return m


def metrics(man: dict) -> dict:
    pp = placed_parts(man); dur = man["source"]["duration_s"]; n = int(dur * 100) + 1
    inside, between = [], []
    for x, y in zip(pp, pp[1:]):
        g = y["a"] - x["b"]
        (inside if x["unit"] == y["unit"] else between).append({"gap": g, "after": x["text"][-40:], "before": y["text"][:30], "t": round(x["b"], 2)})
    weak = [d for d in inside if d["gap"] > 0.3 and not PUNCT.search(d["after"].strip())]
    src = _cover([(v["start"], v["end"]) for v in man["speech_intervals"]], n); dub = _cover([(x["a"], x["b"]) for x in pp], n)
    at = np.array([x["atempo"] for x in pp] or [1.0]); sh = np.array([x["shift"] for x in pp] or [0.0])
    def cnt(lst, t): return sum(1 for d in lst if d["gap"] > t)
    return {"parts": len(pp), "units": len(man["units"]),
            "inside_pauses_gt_0.3": cnt(inside, 0.3), "inside_pauses_gt_0.8": cnt(inside, 0.8), "inside_pauses_gt_1.5": cnt(inside, 1.5),
            "inside_pause_s": round(sum(d["gap"] for d in inside if d["gap"] > 0.3), 2), "weak_pauses_gt_0.3": len(weak),
            "between_pauses_gt_1.5": cnt(between, 1.5), "between_pause_max_s": round(max([d["gap"] for d in between] or [0.0]), 2),
            "atempo_max": round(float(at.max()), 3), "atempo_mean": round(float(at.mean()), 3), "parts_sped_up": int((at > 1.001).sum()), "parts_slowed": int((at < 0.999).sum()),
            "shift_max_s": round(float(sh.max()), 2), "parts_shifted_gt_0.1": int((sh > 0.1).sum()),
            "source_speech_s": round(src.sum() / 100, 1), "source_without_dub_s": round((src & ~dub).sum() / 100, 1), "dub_outside_source_s": round((dub & ~src).sum() / 100, 1),
            "weak_examples": [f"{d['t']}s '…{d['after']}' | {d['gap']:.1f}s | '{d['before']}…'" for d in weak][:8],
            "inside_examples": [f"{d['t']}s '…{d['after']}' | {d['gap']:.1f}s | '{d['before']}…'" for d in inside if d["gap"] > 0.8][:8]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+"); ap.add_argument("--json", default=None)
    a = ap.parse_args(); rows = {}
    for mp in a.manifests:
        man = json.loads(pathlib.Path(mp).read_text()); r = metrics(man); rows[mp] = r
        print(f"== {mp}"); print(json.dumps({k: v for k, v in r.items() if not k.endswith("examples")}))
        for e in r["inside_examples"]: print("   inside:", e)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
