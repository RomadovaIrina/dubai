#!/usr/bin/env python3
"""r4 merged: outbox/r4_merged_m2_WATCH.md for a NOT gate-passing watch candidate: per video the failing / better gate rows (r4_gate compare
NEW vs PREV), the shaky-window slide NEW vs PREV, and the timecodes of the worst LatentSync window joints of NEW (r3_joints *_worst), so the
viewer knows where to look.
    python scripts/r4/r4_watch_note.py --new DIR:LABEL --prev DIR:LABEL --gate GATE.md --ids 04 03 --out outbox/r4_merged_m2_WATCH.md"""
import argparse, json, pathlib, re


def tc(t): return f"{int(t) // 60}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--prev", required=True); ap.add_argument("--gate", required=True)
    ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    nd, nl = a.new.split(":"); pd, pl = a.prev.split(":"); g = pathlib.Path(a.gate).read_text().splitlines()
    hdr = next((l for l in g if l.startswith("| metric")), "")
    cols = [c.strip().split()[0] for c in hdr.strip("|").split("|")[1:]]
    L = ["# r4 merged m2: WATCH candidate (NOT gate-passing)", "",
         f"m2 = the video agent's r4_video_v1 stack (v4 generator + v4 composite minus seam-LF) on audio pb1 (the user's choice). PREV = v4-repro on this server",
         "(full v4 stack on the v4 audio reproduction). Files: `<id>_r4_merged_m2_WATCH.mp4`, `<id>_r4_merged_m2_WATCH_sbs.mp4` (V4 | M2).",
         "The failing video rows (slide, window joints) come from the GENERATOR on the new audio, not from the composite: the full v4 composite on",
         "m2's raw layer scores the same (04: slide 0.184, joints identical).", ""]
    for id_ in a.ids:
        if id_ not in cols: continue
        ci = cols.index(id_) + 1; rows = []
        for l in g:
            if not l.startswith("| ") or l.startswith("| metric"): continue
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) > ci and re.search(r"FAIL|better", c[ci]): rows.append(f"| {c[0]} | {c[ci]} |")
        ns, ps = (json.loads((pathlib.Path(d) / f"{id_}_{lb}.score.json").read_text()) for d, lb in ((nd, nl), (pd, pl)))
        jf = pathlib.Path(nd) / f"{id_}_{nl}.joints_full.json"; jw = json.loads(jf.read_text()).get("lowfreq_worst", []) if jf.exists() else []
        L += [f"## {id_}", "", "| gate row (FAIL / better) | NEW / PREV |", "|---|---|"] + (rows or ["| (none) | |"]) + [""]
        L += ["Slide (lower face vs the head, fixed shaky windows): " + "; ".join(f"{tc(w['window_s'][0])}-{tc(w['window_s'][1])}: {w['nosebase']['slide_px']['mean']:.3f} vs "
              f"{p['nosebase']['slide_px']['mean']:.3f} px" for w, p in zip(ns["slide"], ps["slide"]))]
        if jw: L += ["Worst LatentSync window joints in m2 (time, joint motion / inner motion; > 2 = a possible jump): " + ", ".join(f"{tc(t)} ({r:.1f}x)" for t, r in jw[:6])]
        L += [f"Lip flash candidates: {ns.get('flash_strong')} vs {ps.get('flash_strong')}.", ""]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(a.out)


if __name__ == "__main__":
    main()
