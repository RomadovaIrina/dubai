#!/usr/bin/env python3
"""r16 Q 09.11: score r16q_defects.py outputs against Irina's verdicts (ФИДБЕК_0911). Expected = what she flagged (True) / called fine (False);
None = no verdict for that check. Prints a hit / miss table and writes CALIB.md.
    python3 defects_calib.py DIR   (DIR/<id>.json from r16q_defects.py)"""
import json, sys, pathlib

# check -> id -> (expected, where) ; times in s
EXP = {
    "tail": {"f0811_m03": (True, "40.1-40.4"), "full_m18": (True, "19-20.3"), "f0811_q01": (True, "last 1 s"), "f0811_q01_subs": (True, "last 1 s"),
             "f0811_q04": (False, ""), "full_old03": (False, ""), "long5": (False, ""), "q03_e2e_alpha_full": (False, ""), "q05_e2e_full": (None, ""),
             "f0811_q02": (None, ""), "f0811_q06": (False, "no KeySync")},
    "outside": {"q03_e2e_alpha_full": (True, "neck / collar, whole video"), "full_old03": (True, "right cheek ~64-66"), "long5": (True, "neck 108-150"),
                "q05_e2e_full": (True, "window on head turns"), "f0811_m03": (False, ""), "f0811_q01": (False, ""), "f0811_q01_subs": (False, ""),
                "f0811_q04": (False, "beard boil is inside the face"), "full_m18": (False, ""), "f0811_q06": (False, "no KeySync"), "f0811_q02": (None, "")},
    "chroma": {"long5": (True, "108-150, 116 blue mouth"), "f0811_m03": (None, "shadow on the cheek"), "f0811_q01": (False, ""), "f0811_q04": (False, ""),
               "full_old03": (False, ""), "full_m18": (False, ""), "q03_e2e_alpha_full": (False, ""), "q05_e2e_full": (None, ""), "f0811_q06": (False, ""),
               "f0811_q01_subs": (False, ""), "f0811_q02": (None, "")},
    "occlude": {"q05_e2e_full": (True, "cup ~12 s + turns"), "f0811_q02": (True, "hand 34.8-38.6"), "long5": (True, "hair over the cheek 108-150"),
                "f0811_m03": (False, ""), "f0811_q01": (False, ""), "full_m18": (False, ""), "full_old03": (False, ""), "f0811_q04": (False, "")},
}


OUT_SHARE = float(__import__("os").environ.get("OUT_SHARE", "0.05"))   # share of frames whose band excess > THR out_share


def flagged(r, check):
    if check == "tail":
        t = r["tail"]; loc = t.get("local", {})
        return t["flag"], "local " + ", ".join(f"{k} {v['ratio']} @{v['at_s']}" for k, v in loc.items()) + f"; global z {t['score']}"
    if check == "outside":
        o = r["outside"]; w = {k: v for k, v in o.items() if k.startswith("wob_")}
        return bool(o.get("flag")), "; ".join(f"{k[4:]} p95 {v.get('ratio_p95')} {[x[:2] for x in v.get('runs_1s', [])[:3]]}" for k, v in w.items())
    if check == "chroma": c = r["chroma"]; return c["share"] >= 0.02, f"{c['share']*100:.1f}% median {c['median_blue']} {[x[:2] for x in c['runs'][:3]]}"
    if check == "occlude":
        o = r["occlude"]; return bool(o["occl_runs"] or o["turn_runs"] or o.get("paint_runs")), f"paint {[x[:2] for x in o.get("paint_runs", [])[:3]]} occl {[x[:2] for x in o['occl_runs'][:4]]} turn {[x[:2] for x in o['turn_runs'][:4]]}"


D = pathlib.Path(sys.argv[1]); lines = ["# Defect checks vs Irina's verdicts (09.11)\n"]; score = {}
for check, exp in EXP.items():
    lines += [f"\n## {check}\n", "| video | Irina | check | detail | |", "|---|---|---|---|---|"]
    for vid, (e, where) in exp.items():
        p = D / f"{vid}.json"
        if not p.exists(): lines.append(f"| {vid} | {e} {where} | – | not scored | |"); continue
        f, det = flagged(json.load(open(p)), check)
        mark = "" if e is None else ("OK" if f == e else ("MISS" if e else "FALSE+"))
        if e is not None: score.setdefault(check, [0, 0]); score[check][0] += (f == e); score[check][1] += 1
        lines.append(f"| {vid} | {'flag' if e else ('fine' if e is False else '?')} {where} | {'FLAG' if f else '-'} | {det} | {mark} |")
lines.insert(1, "Agreement: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in score.items()) + "\n")
(D / "CALIB.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:3])); print(json.dumps(score))
