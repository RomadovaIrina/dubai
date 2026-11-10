#!/usr/bin/env python3
"""Collect scan_breaks moments over all videos -> BREAKS.md (all moments, per video) + worst10.txt (id|t|label) for moment_png.py.
A moment is marked 'also 08.11' when the 08.11 final has a moment within 0.6 s (same classes or not), else 'new in now'.
    python3 worst10.py --now out/now/scan [--old out/0811/scan] --out DIR [--n 10] [--per-video 3]"""
import argparse, json, pathlib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--now", required=True); ap.add_argument("--old", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--per-video", type=int, default=3); ap.add_argument("--min-rank", type=float, default=1.2)
    a = ap.parse_args(); O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    now = {p.name.split(".")[0]: json.load(open(p)) for p in sorted(pathlib.Path(a.now).glob("*.breaks.json"))}
    old = {p.name.split(".")[0]: json.load(open(p)) for p in sorted(pathlib.Path(a.old).glob("*.breaks.json"))} if a.old else {}
    allm = []
    for vid, r in now.items():
        for m in r["moments"]:
            if m["rank"] < a.min_rank: continue
            o = old.get(vid)
            if o is None: m["vs"] = "–"
            else:
                hit = [x for x in o["moments"] if x["rank"] >= a.min_rank and x["t0"] - 0.6 <= m["t"] <= x["t1"] + 0.6]
                m["vs"] = ("also 08.11 (r%.1f)" % max(x["rank"] for x in hit)) if hit else "new in now"
            m["vid"] = vid; allm.append(m)
    allm.sort(key=lambda m: -m["rank"])
    pick, per = [], {}
    for m in allm:
        if per.get(m["vid"], 0) >= a.per_video: continue
        pick.append(m); per[m["vid"]] = per.get(m["vid"], 0) + 1
        if len(pick) >= a.n: break
    def sigs(m): return ", ".join(f"{e['sig']} {e['val']} (was {e['base']})" for e in sorted(m["ev"], key=lambda e: -e["score"])[:4])
    L = ["# now_1011: tail-like breaks over ALL seconds (Q, day 4)\n",
         "Rule: a per-frame signal jumps vs the 2 s before it (4-frame mean / median before, the rule that catches the m03 / m18 / q01 tails).",
         "Signals: res_out / res_low (lower-face residual vs source), jaw_dev / lm_dev (ours' jaw / stable landmarks vs the source's, % IOD),",
         "mouth_sharp, lip_thick, seam (jaw-band wobble), ghost (second chin). rank = score x (1 + 0.5 per extra signal at the same moment).\n",
         "## Worst %d\n" % len(pick), "| # | video | t (s) | window | rank | classes | context | vs 08.11 | signals (value, was) | png |", "|---|---|---|---|---|---|---|---|---|---|"]
    W = []
    for i, m in enumerate(pick, 1):
        png = f"w{i:02d}_{m['vid']}_{m['t']:.2f}.jpg"
        L.append(f"| {i} | {m['vid']} | {m['t']:.2f} | {m['t0']:.2f}-{m['t1']:.2f} | {m['rank']} | {' / '.join(m['classes'])} | {m['ctx'] or '-'} | {m.get('vs', '–')} | {sigs(m)} | {png} |")
        W.append(f"{m['vid']}|{m['t']:.2f}|{png}|{m['vid']} {m['t']:.2f}s {'/'.join(m['classes'])} r{m['rank']}")
    L += ["\n## Per video (moments with rank >= %.1f)\n" % a.min_rank]
    for vid, r in now.items():
        ms = sorted([m for m in allm if m["vid"] == vid], key=lambda m: m["t"])
        L.append(f"\n### {vid} ({r['dur']} s, ghost median {r['ghost']['median']}, p95 {r['ghost']['p95']}; pauses {len(r['pauses'])}, cuts {len(r['cuts'])})\n")
        L += ["| t | window | rank | classes | context | vs 08.11 | signals |", "|---|---|---|---|---|---|---|"]
        L += [f"| {m['t']:.2f} | {m['t0']:.2f}-{m['t1']:.2f} | {m['rank']} | {' / '.join(m['classes'])} | {m['ctx'] or '-'} | {m.get('vs', '–')} | {sigs(m)} |" for m in ms] or ["| – | | | none | | | |"]
    (O / "BREAKS.md").write_text("\n".join(L) + "\n"); (O / "worst10.txt").write_text("\n".join(W) + "\n")
    print("\n".join(L[:len(pick) + 8]))


if __name__ == "__main__":
    main()
