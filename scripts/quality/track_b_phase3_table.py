#!/usr/bin/env python3
"""Track B Phase 3 table for one video: cand5 / clean0 (25 fps master grid, ratios vs the 25 fps master) and the source-fps variants of
fps_composite.py (ratios vs the SOURCE at its own fps), common metric mouth_jitter_hr.py v4, SyncNet, audio md5, cost, feather.
    python scripts/quality/track_b_phase3_table.py --id 03 [--human 24 35] [--out table.md]"""
from __future__ import annotations
import argparse, json, pathlib

R = pathlib.Path("/workspace/dub/artifacts/restoration-temporal")


def jl(p):
    p = pathlib.Path(p); return json.loads(p.read_text()) if p.exists() else None


def row(name, mj, sync=None, comp=None, lic="YES", note=""):
    if not mj: return None
    c = mj.get("classes", {}).get("ls_core", {}); sw = mj.get("classes", {}).get("switch", {}); w = mj.get("latentsync_window_boundaries", {})
    g = lambda blk, k, reg="mouth": ((blk.get(k) or {}).get(reg) or {}).get("ratio")
    ps = {x["t"]: x for x in mj.get("per_second", [])}
    return {"variant": name, "license": lic, "resid": g(c, "resid"), "sharp": g(c, "sharp"), "lf_sharp": g(c, "sharp", "lowerface"), "d1": g(c, "d1"),
            "bnd": ((w.get("d1") or {}).get("video") or {}).get("phase0_over_inner"), "bnd15": ((w.get("boundary_d1_jump") or {}).get("video") or {}).get("share_gt_1_5"),
            "sw_resid": g(sw, "resid"), "sw_jerk": g(sw, "jerk"), "upper": g(c, "resid", "upper"), "shear95": g(c, "shear95", "lowerface"),
            "frozen": {t: (ps.get(t) or {}).get("frozen_mouth") for t in HUMAN}, "frozen_rank": {t: (ps.get(t) or {}).get("rank_frozen_mouth") for t in HUMAN},
            "sync": sync, "cost": (comp or {}).get("s_per_video_min"), "edge": ((comp or {}).get("edge_px_measured") or {}).get("median"),
            "fallback": (comp or {}).get("fallback_frames"), "note": note}


def main() -> int:
    global HUMAN
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--human", nargs="*", type=int, default=[24, 35]); ap.add_argument("--out", default=None)
    a = ap.parse_args(); HUMAN = a.human; i = a.id; rows = []
    base = (jl(f"/workspace/dub/artifacts/baseline/{i}/report_baseline_cand5.json") or [{}])[0]
    rows.append(row("cand5 (25 fps, CodeFormer)", jl(R / f"metric_validation/{i}_cand5_v4.json") or jl(R / f"full/final/{i}_cand5.mouth_jitter_hr.json"),
                    [(base.get("syncnet") or {}).get("confidence"), (base.get("syncnet_dub_only") or {}).get("confidence")] if isinstance(base.get("syncnet"), dict) else None,
                    lic="NO (CodeFormer, buffalo_l)", note="rendered reference"))
    ev = jl(R / f"eval/{i}_clean0.eval.json") or {}; cs = jl(R / f"fps/{i}/{i}_clean0.syncnet.json")
    rows.append(row("clean0 (25 fps)", jl(R / f"eval/{i}_clean0.mouth_jitter_hr.json"),
                    [cs["mix"][0], cs["dub_only"][0]] if cs else [(ev.get("syncnet_mix") or {}).get("confidence"), (ev.get("syncnet_dub_only") or {}).get("confidence")],
                    note="track A recipe on this server"))
    md5 = jl(R / f"fps/{i}/{i}_audio_md5.json") or {}
    for v, lab in (("F0_clean0fps", "F0 source fps (clean0 composite, 12 px edge)"), ("F1_C2", "F1 + C2 window-seam interpolation"), ("F2_C2T", "F2 + source skin texture"),
                   ("F3_C2TL", "F3 + lip texture"), ("OPT1_still", "OPT1 = F2 + still lower face in dub-silent stretches"), ("OPT2_orig", "OPT2 = F2 + original pass-through there")):
        s = jl(R / f"fps/{i}/{i}_{v}.syncnet.json") or {}
        r = row(lab, jl(R / f"fps/{i}/{i}_{v}.mouth_jitter_hr.json"), [(s.get("mix") or [None])[0], (s.get("dub_only") or [None])[0]] if s else None,
                jl(R / f"fps/{i}/{i}_{v}.fps_composite.json"), note={True: "audio md5 = baseline", False: "audio md5 DIFFERS"}.get(md5.get(v), ""))
        if r: rows.append(r)
    f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    hdr = ["variant", "licence", "mouth resid", "mouth sharp", "lower-face sharp", "LS-seam jump (d1)", "seams > 1.5x", "switch resid / jerk", "upper resid",
           "frozen-mouth " + " / ".join(f"{t}s" for t in HUMAN) + " (rank)", "SyncNet mix / dub-only", "post-proc s/min", "edge px", "fallback", "note"]
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in [x for x in rows if x]:
        fr = " / ".join(f"{f(r['frozen'][t])} ({f(r['frozen_rank'][t])})" for t in HUMAN)
        L.append("| " + " | ".join([r["variant"], r["license"], f(r["resid"]), f(r["sharp"]), f(r["lf_sharp"]), f(r["bnd"]), f(r["bnd15"]), f"{f(r['sw_resid'])} / {f(r['sw_jerk'])}",
                                    f(r["upper"]), fr, " / ".join(f(x) for x in (r["sync"] or [None, None])), f(r["cost"]), f(r["edge"]), f(r["fallback"]), r["note"]]) + " |")
    t = "\n".join(L) + "\n"; print(t)
    if a.out: pathlib.Path(a.out).write_text(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
