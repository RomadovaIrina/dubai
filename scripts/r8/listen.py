#!/usr/bin/env python3
"""r8 listen list for an AB WATCH: every PREV span where the source speaks and the dub is silent (coverage.py) and every pacing event
(pacing.py), with what NEW has at the same moment; timecodes in the AB file (A = PREV at t, B = NEW at t + offset). Markdown to stdout.
    python scripts/r8/listen.py --id 02 --ver r8v1 --prev cov_prev.json --new cov_new.json --pace-prev p.json --pace-new p2.json --offset 112.0"""
import argparse, json, pathlib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id"); ap.add_argument("--ver"); ap.add_argument("--prev"); ap.add_argument("--new")
    ap.add_argument("--pace-prev"); ap.add_argument("--pace-new"); ap.add_argument("--offset", type=float, default=0.0); ap.add_argument("--pver", default="v3a1"); a = ap.parse_args()
    ld = lambda p: json.loads(pathlib.Path(p).read_text()) if p and pathlib.Path(p).exists() else None
    P, N, PP, PN = ld(a.prev), ld(a.new), ld(a.pace_prev), ld(a.pace_new)
    mm = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
    o = [f"# {a.id}: {a.pver} → {a.ver} listen list", "", f"AB file: `{a.id}_audio_{a.ver}_WATCH_AB_prev-then-new.mp4`; A ({a.pver}) at t, B ({a.ver}) at t + {a.offset:.1f} s.", ""]
    if P and N:
        sp, sn = P["summary"], N["summary"]
        o += ["| | " + a.pver + " | " + a.ver + " |", "|---|---:|---:|"]
        for k, lab in (("src_speaks_dub_silent_s", "source speaks, dub silent (s, all)"), ("beyond_breaths_s", "  of it in spans > 0.3 s"), ("on_face_s", "  on the face"),
                       ("off_face_s", "  off screen"), ("dub_speaks_src_silent_s", "dub speaks, source silent (s)"), ("spans", "spans > 0.3 s")):
            o.append(f"| {lab} | {sp[k]} | {sn[k]} |")
        o += ["", f"## Where the dub disappeared in {a.pver} (source speaks, dub silent > 0.3 s)", "", f"| A at | B at | s {a.pver} | cause | RU | still silent in " + a.ver + " (s) |", "|---|---|---:|---|---|---:|"]
        for x in P["spans"]:
            left = sum(max(0.0, min(x["t1"], y["t1"]) - max(x["t0"], y["t0"])) for y in N["spans"])
            o.append(f"| {mm(x['t0'])}–{mm(x['t1'])} | {mm(x['t0'] + a.offset)} | {x['len_s']} | {x['cause']} | {x['ru']} | {round(left, 2)} |")
        new_only = [y for y in N["spans"] if not any(min(x["t1"], y["t1"]) > max(x["t0"], y["t0"]) for x in P["spans"])]
        if new_only:
            o += ["", f"## New silent spans in {a.ver}", "", "| B at | s | cause | RU |", "|---|---:|---|---|"]
            o += [f"| {mm(y['t0'] + a.offset)} ({mm(y['t0'])}) | {y['len_s']} | {y['cause']} | {y['ru']} |" for y in new_only]
    for lab, Q, off in ((a.pver, PP, 0.0), (a.ver, PN, a.offset)):
        if not Q: continue
        o += ["", f"## Pacing events, {lab}", "", "summary: " + ", ".join(f"{k} {v}" for k, v in Q["summary"].items()), "", "| at (AB file) | kind | detail |", "|---|---|---|"]
        o += [f"| {mm(e['t0'] + off)} | {e['kind']} | " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "t0", "t1")) + " |" for e in Q["events"]]
    print("\n".join(o))


if __name__ == "__main__":
    main()
