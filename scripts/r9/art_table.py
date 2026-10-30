#!/usr/bin/env python3
"""r9: the mouth-artefact table (final output events per class, and where they first appear) from artefacts.py json files.
    python art_table.py <out.md> art_<id>[<arm>].json ..."""
import json, sys, pathlib
C = ("a_stick_unstick", "b_amplitude", "c_teeth", "d_slide")
rows = ["| video / arm | LS frames | a stick / unstick | b amplitude | c teeth | d slide | origin of the final events |", "|---|---|---|---|---|---|---|"]
tot = {}
for p in sys.argv[2:]:
    d = json.loads(pathlib.Path(p).read_text()); f = d["layers"]["L4 final"]; org = {}
    for c, o in d["final_events_by_origin"].items():
        for k, v in o.items(): org[k] = org.get(k, 0) + v; tot[k] = tot.get(k, 0) + v
    cells = " | ".join(f"{f[c]['events']} ({f[c]['frames']} fr)" for c in C)
    rows.append(f"| {d['id']} {d['arm']} | {d['ls_frames']} | {cells} | " + ", ".join(f"{k} {v}" for k, v in sorted(org.items(), key=lambda x: -x[1])) + " |")
rows.append("| **all** | | | | | | " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items(), key=lambda x: -x[1])) + " |")
pathlib.Path(sys.argv[1]).write_text("\n".join(rows) + "\n"); print("\n".join(rows))
