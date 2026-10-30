#!/usr/bin/env python3
"""r8 speed gate against the ENVELOPE of several sequential upstream draws (the same code, other seeds): quality swings by seed as much as
any lever, so a candidate build FAILS an axis only when it is worse than the worst draw by more than the tolerance.
    python scripts/r8/envelope_gate.py --new r8b5 --draws "01:r8e2e0b,r8s2b,r8s3 02:... 04:r8e2e0b,r8s2,r8s3" [--out gate.md]
Axes (worse = ... ; tolerance): faith mean (lower; 0.05), coverage = source speaks & dub silent in spans > 0.3 s (higher; 0.3 s),
longest dub gap beyond the source pause (higher; 0.3 s), pause W1 vs source (higher; 0.05 s), dub speaks & source silent (higher; 0.5 s),
WER of the final mix (higher; 0.01), cut-word ends / inserted mid-sentence gaps / dead air after words (higher; 0)."""
import argparse, json, pathlib, sys

A = pathlib.Path("/workspace/dub/artifacts/r4a")
AXES = [("faith", "faith mean (judge 1-5)", -1, 0.05), ("cov", "source speaks, dub silent, spans > 0.3 s (s)", 1, 0.3),
        ("gap", "longest dub gap beyond the source pause (s)", 1, 0.3), ("w1", "pause W1 vs source (s)", 1, 0.05),
        ("over", "dub speaks, source silent (s)", 1, 0.5), ("wer", "WER final mix", 1, 0.01),
        ("cut", "cut-word ends", 1, 0), ("ins", "inserted mid-sentence gaps", 1, 0), ("dead", "dead air after words", 1, 0)]


def rows(tag, id_):
    m = json.loads((A / tag / f"{id_}.metrics.json").read_text()); c = lambda n: json.loads((A / tag / "cov" / f"{n}_{id_}.json").read_text())["summary"]
    cov = c("cov"); cu = c("cuts"); mx = c("mix")
    return {"faith": (m.get("faith") or {}).get("mean"), "cov": cov["beyond_breaths_s"], "gap": m.get("gap_excess_max_s"), "w1": m.get("pauses_w1_s"),
            "over": cov["dub_speaks_src_silent_s"], "wer": (m.get("asr") or {}).get("wer_final_mix"), "cut": cu["cut_word_end"], "ins": cu["inserted_gap"], "dead": mx["dead"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--draws", required=True); ap.add_argument("--out")
    a = ap.parse_args(); md = [f"# envelope gate: `{a.new}` vs the min / max of the sequential draws", ""]; fails = 0
    for spec in a.draws.split():
        id_, tags = spec.split(":"); tags = tags.split(","); D = [rows(t, id_) for t in tags]; N = rows(a.new, id_)
        md += [f"## {id_} (draws: {', '.join(tags)})", "", "| axis | draws | envelope | new | verdict |", "|---|---|---|---:|---|"]
        for k, name, sign, tol in AXES:
            vals = [d[k] for d in D if d[k] is not None]; v = N[k]
            if not vals or v is None: md.append(f"| {name} | n/a | | {v} | |"); continue
            lo, hi = min(vals), max(vals)
            worst = hi if sign > 0 else lo
            bad = (v > worst + tol) if sign > 0 else (v < worst - tol)
            better = (v < lo) if sign > 0 else (v > hi)
            fails += bad
            md.append(f"| {name} | {' / '.join(str(x) for x in vals)} | {lo} .. {hi} | {v} | {'FAIL' if bad else 'better' if better else 'inside'} |")
        md.append("")
    md.insert(1, f"Result: {'PASS' if not fails else f'FAIL ({fails} axis rows)'}"); text = "\n".join(md) + "\n"
    if a.out: pathlib.Path(a.out).write_text(text)
    print(text); sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
