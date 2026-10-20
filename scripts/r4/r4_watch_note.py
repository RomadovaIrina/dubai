#!/usr/bin/env python3
"""r4 merged: outbox/r4_merged_m2_WATCH.md for a NOT gate-passing watch candidate: per video the failing / better gate rows (r4_gate compare
NEW vs PREV), the shaky-window slide NEW vs PREV, and the timecodes of the worst LatentSync window joints of NEW (r3_joints *_worst), so the
viewer knows where to look.
    python scripts/r4/r4_watch_note.py --new DIR:LABEL --prev DIR:LABEL --gate GATE.md --ids 04 03 --out outbox/r4_merged_m2_WATCH.md"""
import argparse, json, pathlib, re


def tc(t): return f"{int(t) // 60}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--prev", required=True); ap.add_argument("--gate", required=True)
    ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="m2"); ap.add_argument("--intro", default=None); a = ap.parse_args()
    nd, nl = a.new.split(":"); pd, pl = a.prev.split(":"); g = pathlib.Path(a.gate).read_text().splitlines()
    hdr = next((l for l in g if l.startswith("| metric")), "")
    cols = [c.strip().split()[0] for c in hdr.strip("|").split("|")[1:]]
    intro = a.intro or ("m2 = the video agent's r4_video_v1 stack (v4 generator + v4 composite minus seam-LF) on audio pb1 (the user's choice). PREV = v4-repro on this "
                        "server (full v4 stack on the v4 audio reproduction). The failing video rows (slide, window joints) come from the GENERATOR on the new audio, not from "
                        "the composite: the full v4 composite on m2's raw layer scores the same (04: slide 0.184, joints identical).")
    L = [f"# r4 merged {a.name}: WATCH candidate", "", intro, "",
         f"Files: `<id>_r4_merged_{a.name}_WATCH.mp4`, `<id>_r4_merged_{a.name}_WATCH_sbs.mp4` (V4 | {a.name.upper()}). Gate verdict per video below (rows FAIL / better).", ""]
    for id_ in a.ids:
        if id_ not in cols: continue
        ci = cols.index(id_) + 1; rows = []
        for l in g:
            if not l.startswith("| ") or l.startswith("| metric"): continue
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) > ci and re.search(r"FAIL|better", c[ci]): rows.append(f"| {c[0]} | {c[ci]} |")
        ns, ps = (json.loads((pathlib.Path(d) / f"{id_}_{lb}.score.json").read_text()) for d, lb in ((nd, nl), (pd, pl)))
        jf = pathlib.Path(nd) / f"{id_}_{nl}.joints_full.json"; jw = json.loads(jf.read_text()).get("lowfreq_worst", []) if jf.exists() else []
        fails = [r for r in rows if "FAIL" in r and "s / video-min" not in r]
        L += [f"## {id_}: gate {'FAIL (' + str(len(fails)) + ' row' + ('s' if len(fails) > 1 else '') + ')' if fails else 'PASS'}", "", "| gate row (FAIL / better) | NEW / PREV |", "|---|---|"] + (rows or ["| (none) | |"]) + [""]
        L += ["Slide (lower face vs the head, fixed shaky windows): " + "; ".join(f"{tc(w['window_s'][0])}-{tc(w['window_s'][1])}: {w['nosebase']['slide_px']['mean']:.3f} vs "
              f"{p['nosebase']['slide_px']['mean']:.3f} px" for w, p in zip(ns["slide"], ps["slide"]))]
        if jw: L += [f"Worst LatentSync window joints in {a.name} (time, joint motion / inner motion; > 2 = a possible jump): " + ", ".join(f"{tc(t)} ({r:.1f}x)" for t, r in jw[:6])]
        L += [f"Lip flash candidates: {ns.get('flash_strong')} vs {ps.get('flash_strong')}."]
        mn, mp_ = ns.get("mouth_audio") or {}, ps.get("mouth_audio") or {}
        if mn: L += [f"Mouth / dub-audio mismatch (talking mouth while the dub is silent + closed mouth while it speaks, face visible): {mp_.get('mouth_audio_mismatch_s')} s (v4-repro) -> "
                     f"{mn.get('mouth_audio_mismatch_s')} s; worst: " + ", ".join(f"{tc(t)} {k} ({d:.1f} s)" for k, t, d in (mn.get("worst") or [])[:4])]
        L += [""]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(a.out)


if __name__ == "__main__":
    main()
