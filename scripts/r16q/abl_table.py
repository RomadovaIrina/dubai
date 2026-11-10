#!/usr/bin/env python3
"""Q ablation table: one row per (video, arm) from r16q_defects JSONs (+ det.json black, + scan_breaks, + eval.json speech@0).
    python3 abl_table.py --q QDIR --abl ABLDIR --videos m03 m18 ... [--md OUT.md] [--json OUT.json]
Arms are read as QDIR/out/<set>/<video>.json for set in: d0811, abl_P0_oval ... abl_P6_pause, today (in ladder order)."""
import argparse, json, os

ARMS = [("d0811", "08.11 file"), ("abl_P0_oval", "P0 oval"), ("abl_P1_tight", "P1 +tight"), ("abl_P2_parse", "P2 +parse"), ("abl_P3_cmshadow", "P3 +cm-shadow (side)"),
        ("abl_P4_occl", "P4 P2+occl"), ("abl_P5_cap", "P5 +cap"), ("abl_P6_pause", "P6 +pause"), ("today", "today (+tail)")]


def row(q, abl, s, v):
    p = f"{q}/out/{s}/{v}"
    if not os.path.exists(p + ".json"): return None
    r = json.load(open(p + ".json")); d = {}
    try: bk = json.load(open(p + ".det.npz.det.json"))["det"]["black"]; d["black_fr"] = bk["flag_frames"]; d["black_share"] = bk.get("black_share_ours")
    except Exception: d["black_fr"] = d["black_share"] = None
    g = r.get("ghost", {}); d["ghost_s"] = round(sum(b - a for a, b, _ in g.get("runs", [])), 1); d["ghost_p95"] = g.get("excess_p95")
    d["seam"] = r["seam"]["excess_median"]; d["seam_p95"] = r["seam"]["excess_p95"]; d["tail"] = r["tail"]["local_max"]
    d["soft"] = len(r["mouth"]["soft_runs"]); d["sharp"] = r["mouth"]["sharp_median"]; d["thin"] = len(r["mouth"]["lip_thin_runs"]); d["fat"] = len(r["mouth"]["lip_fat_runs"])
    e = r.get("edge", {})
    for k, nm in (("left", "L"), ("right", "R")):
        x = e.get(k, {}); d[f"edge{nm}"] = (x.get("excess") or [None])[0]; d[f"edge{nm}_p95"] = (x.get("excess") or [None, None])[1]
        d[f"st{nm}"] = (x.get("static_abs") or [None])[0]; d[f"dL{nm}"] = (x.get("static_dL") or [None])[0]
    x = e.get("jaw", {}); d["stJ"] = (x.get("static_abs") or [None])[0]; d["dLJ"] = (x.get("static_dL") or [None])[0]; d["darkJ"] = x.get("dark_share")
    o = r.get("outside", {}); d["out_l"] = o.get("wob_l", {}).get("ratio_p95"); d["out_r"] = o.get("wob_r", {}).get("ratio_p95"); d["out_n"] = o.get("wob_n", {}).get("ratio_p95")
    try: b = json.load(open(f"{q}/out/{s}/scan/{v}.breaks.json")); d["moments"] = sum(m["rank"] >= 3 for m in b["moments"])
    except Exception: d["moments"] = None
    d["speech0"] = None
    if s.startswith("abl_"):
        ev = f"{abl}/work_{s[4:]}/eval_repost_{v}/eval.json"
        if os.path.exists(ev): d["speech0"] = json.load(open(ev)).get("speech_at0")
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--q", required=True); ap.add_argument("--abl", required=True); ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--md", default=None); ap.add_argument("--json", default=None); ap.add_argument("--speech", default=None, help="JSON {set: {video: speech@0}} for sets without an eval dir")
    a = ap.parse_args(); extra = json.load(open(a.speech)) if a.speech else {}
    cols = ["black_fr", "black_share", "ghost_s", "seam", "edgeL", "edgeR", "edgeR_p95", "stL", "stR", "stJ", "dLJ", "darkJ", "out_l", "out_r", "out_n", "tail", "soft", "thin", "fat", "moments", "speech0"]
    L = ["| video | arm | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]; allr = {}
    for v in a.videos:
        for s, nm in ARMS:
            d = row(a.q, a.abl, s, v)
            if d is None: continue
            if d["speech0"] is None: d["speech0"] = extra.get(s, {}).get(v)
            allr.setdefault(v, {})[s] = d
            L.append(f"| {v} | {nm} | " + " | ".join("" if d.get(c) is None else str(d[c]) for c in cols) + " |")
    out = "\n".join(L)
    if a.md: open(a.md, "w").write(out + "\n")
    if a.json: json.dump(allr, open(a.json, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
