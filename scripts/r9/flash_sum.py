#!/usr/bin/env python3
"""r9: mouth flash table r9 vs r9fx from r3_flash.py json files (flash_<tag>.json).   python flash_sum.py <id> ..."""
import json, sys, pathlib
D = pathlib.Path("/workspace/dub/r9/diag")
rows = ["| video | arm | flash frames | per s | strong | texture | brightness | colour | worst moments (s) |", "|---|---|---|---|---|---|---|---|---|"]


def kind(e):
    return max((abs(e.get("z_hf", 0)), "texture"), (abs(e.get("z_lum", 0)), "brightness"), (abs(e.get("z_col", 0)), "colour"))[1]


for i in sys.argv[1:]:
    for arm, T in (("r9", f"r9f_{i}"), ("r9fx", f"r9f_{i}fx3" if i == "m03" else f"r9f_{i}fx")):
        p = D / f"flash_{T}.json"
        if not p.exists(): continue
        d = json.loads(p.read_text()); L = d["flash_list"]; dur = d["frames"] / 30.0
        strong = [e for e in L if (e.get("mc_ratio") or 0) >= 2 and (e.get("mc_vs_src") or 0) >= 2]
        k = {n: sum(1 for e in L if kind(e) == n) for n in ("texture", "brightness", "colour")}
        top = sorted(L, key=lambda e: -max(abs(e.get("z_hf", 0)), abs(e.get("z_lum", 0)), abs(e.get("z_col", 0))))[:6]
        worst = ", ".join(f"{e['t']:.2f}" for e in sorted(top, key=lambda e: e["t"]))
        rows.append(f"| {i} | {arm} | {len(L)} | {len(L) / dur:.1f} | {len(strong)} | {k['texture']} | {k['brightness']} | {k['colour']} | {worst} |")
print("\n".join(rows))
