#!/usr/bin/env python3
"""Round 3 track B: one markdown table row per (video, variant) from the metrics dir (r3_eval.sh outputs, syncnet json, composite json).
    python scripts/quality/r3b_table.py 04:T3 04:F_Sdets 03:F_Sdets ... [--md out.md]"""
import argparse, glob, json, pathlib
import numpy as np
R = pathlib.Path("/workspace/dub/artifacts/r3-face")


def load(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return None


def row(id_, tag):
    M = R / "metrics"; out = {"video": id_, "variant": tag}
    sl = [load(p) for p in sorted(glob.glob(str(M / f"{id_}_{tag}_slide_*.json")))]; sl = [s for s in sl if s]
    if sl:
        out["slide"] = "%.3f / %.2f" % (np.mean([s["regions"]["nosebase"]["slide_px"]["mean"] for s in sl]), max(s["regions"]["nosebase"]["slide_px"]["p95"] for s in sl))
        out["offset"] = "%.2f" % np.mean([s["regions"]["nosebase"]["abs_px"]["mean"] for s in sl])
        out["cheeks"] = "%.3f" % np.mean([s["regions"]["cheeks"]["slide_px"]["mean"] for s in sl])
    f = load(M / f"{id_}_{tag}.flicker.json")
    if f:
        e = f["excess"]; out.update({"band_up": "%.2f" % e["band_up"]["mean"], "skin_up": "%.2f" % e["skin_up"]["mean"], "chin": "%.2f" % e["chin"]["mean"],
                                     "colour": "%.2f" % f["colour_flicker"]["mean"], "sharp_r3": "%.2f" % f["sharpness_ratio"]})
    v4 = [load(p) for p in sorted(glob.glob(str(M / f"{id_}_{tag}_v4_*.json")))]; v4 = [v for v in v4 if v]
    if v4:
        def avg(metric, reg):
            w = [v["all"][metric][reg]["frames"] for v in v4]; return np.average([v["all"][metric][reg]["ratio"] for v in v4], weights=w)
        out.update({"mouth vel": "%.2f" % avg("vel", "mouth"), "mouth jerk": "%.2f" % avg("jerk", "mouth"), "mouth resid": "%.2f" % avg("resid", "mouth"),
                    "mouth sharp": "%.2f" % avg("sharp", "mouth"), "lf jerk": "%.2f" % avg("jerk", "lowerface"), "lf shear": "%.2f" % avg("shear", "lowerface")})
        bj = [v["latentsync_window_boundaries"]["boundary_d1_jump"] for v in v4 if v.get("latentsync_window_boundaries")]
        if bj:
            nb = sum(b["video"]["boundaries"] for b in bj)
            out["seams >1.5x (src)"] = "%d %% (%d %%)" % (round(100 * sum(b["video"]["share_gt_1_5"] * b["video"]["boundaries"] for b in bj) / max(nb, 1)),
                                                           round(100 * sum(b["reference"]["share_gt_1_5"] * b["reference"]["boundaries"] for b in bj) / max(nb, 1)))
    s = load(M / f"{id_}_{tag}.syncnet.json")
    if s: out["SyncNet mix / dub"] = "%.2f / %.2f" % (s["mix"][0], s.get("dub_only", [float("nan")])[0])
    c = load(R / "comp" / f"{id_}_{tag}.fps_composite.json")
    if c: out["edge px"] = c["edge_px_measured"]["median"]; out["s/min"] = c["s_per_video_min"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rows", nargs="+"); ap.add_argument("--md", default=None); a = ap.parse_args()
    rows = [row(*r.split(":")) for r in a.rows]; cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    print(md)
    if a.md: pathlib.Path(a.md).write_text(md + "\n")


if __name__ == "__main__":
    main()
