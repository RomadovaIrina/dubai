#!/usr/bin/env python3
"""Residual "speaker talks, dub silent" gaps of one run as a list for the VIDEO side (Agent C -> Agent B, 2026-10-16).

Input: a dub_gaps.py JSON (+ the run's manifest for the fit-up history). Output: <out>/<vid>_<policy>.residual.json with, per gap,
master frames [f0, f1) at 25 fps, the LatentSync frames inside it, the class (natural < --natural-s / residual), where it sits in the source
burst, whether Whisper places source words there (asr_words_in_gap; False = breath / hesitation / a word tail outside its Whisper timestamp:
a still mouth may be acceptable there, check the frames), the sentence and the dub parts around it, the unit's fit-up history, and whether
layer 3 (a burst-targeted, more literal translation) could still remove it: residual, words in the gap, cause short_fill / inner_gap.

    python scripts/quality/residual_for_video.py --gaps GAPS/03_x.gaps.json --manifest RUN/recut.manifest.json --policy "L1+0.875" --flags "..." --out OUT
"""
from __future__ import annotations
import argparse, json, pathlib

FPS = 25


def fitup_state(u: dict | None) -> str:
    if not u: return "no_unit"
    fu = u.get("fit_up") or []
    if not fu: return "not_triggered"
    res = [r.get("result") or "" for r in fu]
    if any("accept" in r for r in res): return "accepted_still_short" if not all("accept" in r for r in res) else "accepted"
    return "rejected: " + ((fu[-1].get("reason") or res[-1])[:60])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaps", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--policy", required=True); ap.add_argument("--flags", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--natural-s", type=float, default=0.35)
    a = ap.parse_args()
    G = json.loads(pathlib.Path(a.gaps).read_text()); man = json.loads(pathlib.Path(a.manifest).read_text()); units = {u["id"]: u for u in man["units"]}
    vid = pathlib.Path(man["input"]).stem; rows = []
    asr_words = [w for seg in man["asr"]["segments"] for w in seg.get("words", [])]
    def words_in(t0, t1):   # Whisper word times are wide (04 "популярностью" 35.38-37.16 s across a pause): >= 0.15 s or half the word overlapping
        return " ".join(w["word"].strip() for w in asr_words if min(w["end"], t1) - max(w["start"], t0) >= min(0.15, 0.5 * max(1e-3, w["end"] - w["start"])))
    for i, g in enumerate(sorted(G["gaps"], key=lambda g: g["t0"])):
        u = units.get(g["unit"]); natural = g["seconds"] < a.natural_s; sw = words_in(g["t0"], g["t1"]); words = bool(sw)
        fs = fitup_state(u)
        rows.append({"id": f"{vid}-g{i:02d}", "t0": g["t0"], "t1": g["t1"], "seconds": g["seconds"], "frames": g["frames"], "ls_frames": g["ls_frames_fixed"],
                     "in_latentsync": g["ls_frames_fixed"] > 0, "class": "natural" if natural else "residual", "position_in_burst": g["position_in_burst"],
                     "source_burst": g["source_burst"], "asr_words_in_gap": words, "source_words": sw, "cause": g["cause"], "unit": g["unit"],
                     "sentence_src": g["sentence_src"], "sentence_dub": g["sentence_dub"], "prev_dub": g["prev_part"], "next_dub": g["next_part"], "unit_fit_up": fs,
                     "layer3_could_fill": (not natural) and words and g["cause"] in ("short_fill", "inner_gap")})
    res = [r for r in rows if r["class"] == "residual"]; vis = [r for r in res if r["in_latentsync"]]
    out = {"video": vid, "policy": a.policy, "flags": a.flags, "fps": FPS, "frames_are": "master_25fps.mp4 frame indices, [f0, f1)", "manifest": a.manifest, "dub_track": G["dub_track"],
           "definition": f"source Silero VAD speaking AND dub Silero VAD (min speech 60 ms) silent, 10 ms grid; natural = shorter than {a.natural_s} s",
           "summary": {"gaps": len(rows), "residual": len(res), "residual_s": round(sum(r["seconds"] for r in res), 2), "residual_in_latentsync": len(vis),
                       "residual_ls_frames": sum(r["ls_frames"] for r in vis), "residual_without_asr_words": sum(1 for r in res if not r["asr_words_in_gap"]),
                       "residual_layer3_candidates": sum(1 for r in res if r["layer3_could_fill"]), "longest_s": max([r["seconds"] for r in rows] + [0])},
           "gaps": rows}
    o = pathlib.Path(a.out); o.mkdir(parents=True, exist_ok=True); f = o / f"{vid}_{a.policy.replace(' ', '').replace('+', '_')}.residual.json"
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False)); s = out["summary"]
    print(f"{vid} {a.policy:12s} residual {s['residual']} ({s['residual_s']} s), in LS {s['residual_in_latentsync']} ({s['residual_ls_frames']} frames), no ASR words {s['residual_without_asr_words']}, "
          f"layer-3 candidates {s['residual_layer3_candidates']}, longest {s['longest_s']} s -> {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
