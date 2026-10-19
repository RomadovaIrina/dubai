#!/usr/bin/env python3
"""Seam-fix A/B table (track A, 2026-10-16) for ls_window_ab.py arms: cost (LatentSync s vs the C0 arm), SyncNet, common metric v4
(mouth_jitter_hr.py) whole-window ratios, the window seams at the frames each arm RECORDED (ls_seams: window starts, or the first new frame
after a continuation overlap), scored on the common metric's own per-frame series, and the human-marked ranges.

Seam score per seam s: jump = d1[s] / median(d1[s-4 .. s+4] without s), for the video and (control) for the reference at the same frames;
reported: median jump, share of seams with a jump > 1.5, and d1 / resid at seams over inside, video vs reference.

    python scripts/quality/seam_table.py --dir artifacts/lipsync-temporal/p2c_seams --arms B_C0 B_W24 B_K24o2 ... [--md out.md]
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def seam_scores(series: dict, seams: list, n: int) -> dict:
    def arr(k):
        return np.array([np.nan if v is None else v for v in series.get(k, [None] * n)], float)
    out = {}
    for who in ("v", "r"):
        d1, rs = arr(f"d1_mouth_{who}"), arr(f"resid_mouth_{who}")
        sm = np.zeros(n, bool); sm[[s for s in seams if 0 < s < n]] = True
        jumps = []
        for s in seams:
            if not (4 <= s < n - 4): continue
            nb = [d1[k] for k in range(s - 4, s + 5) if k != s and not sm[k] and np.isfinite(d1[k])]
            if nb and np.isfinite(d1[s]): jumps.append(d1[s] / max(np.median(nb), 1e-6))
        inner = ~sm & np.isfinite(d1); inner[:1] = False
        out[who] = {"n": len(jumps), "jump_median": round(float(np.median(jumps)), 3) if jumps else None,
                    "share_gt_1_5": round(float(np.mean(np.array(jumps) > 1.5)), 3) if jumps else None,
                    "d1_seam_over_inner": round(float(np.nanmean(d1[sm]) / max(np.nanmean(d1[inner]), 1e-6)), 3) if sm.any() else None,
                    "resid_seam_over_inner": round(float(np.nanmean(rs[sm]) / max(np.nanmean(rs[inner & np.isfinite(rs)]), 1e-6)), 3) if sm.any() else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--md", default=None)
    a = ap.parse_args(); P = pathlib.Path(a.dir); rows = []
    base = {}
    for arm in a.arms:
        m = json.loads((P / f"03_{arm}.relmanifest.json").read_text()); cm = json.loads((P / f"cm_{arm}.json").read_text())
        sn = json.loads((P / f"03_{arm}.syncnet.json").read_text()) if (P / f"03_{arm}.syncnet.json").exists() else {}
        hum = json.loads((P / f"cm_{arm}_human.json").read_text()) if (P / f"cm_{arm}_human.json").exists() else None
        ls = m["lipsync"]; win = arm.split("_")[0]; n = len(cm["series"]["frames"])
        seams = [x for s in ls["segments"] if s["action"] == "LATENT_SYNC" for x in (s.get("ls_seams") or range(s["start_frame"] + 16, s["end_frame"], 16))]
        lss = ls["latentsync_inference_total_s"] or sum((c.get("seconds") or 0) for c in (ls.get("runner_calls") or []))
        if arm.endswith("_C0"): base[win] = lss
        al = cm["all"]; ps = [x["max_d1_jump"] for x in cm["per_second"] if x.get("max_d1_jump") is not None]
        sc = seam_scores(cm["series"], seams, n)
        def rr(k, reg="mouth"): v = al[k][reg]; return v.get("ratio") if isinstance(v, dict) else v
        row = {"arm": arm, "ls_s": round(lss, 1), "cost_vs_C0": None, "seams": len(seams), "syncnet_mix": sn.get("final_mix", {}).get("confidence"),
               "syncnet_dub": sn.get("dub_only", {}).get("confidence"), "av": sn.get("dub_only", {}).get("av_offset"),
               "resid": rr("resid"), "twitch": rr("twitch_index"), "acc": rr("acc"), "vel": rr("vel"), "sharp": rr("sharp"), "upper_resid": rr("resid", "upper"),
               "seam_jump_med_v": sc["v"]["jump_median"], "seam_jump_med_r": sc["r"]["jump_median"], "seam_share_gt1_5_v": sc["v"]["share_gt_1_5"],
               "seam_share_gt1_5_r": sc["r"]["share_gt_1_5"], "seam_d1_step_v": sc["v"]["d1_seam_over_inner"], "seam_d1_step_r": sc["r"]["d1_seam_over_inner"],
               "per_sec_max_d1_jump_mean": round(float(np.mean(ps)), 3) if ps else None, "per_sec_max_d1_jump_max": round(float(np.max(ps)), 3) if ps else None,
               "human_resid": None, "human_twitch": None}
        if hum:
            row["human_resid"] = hum["all"]["resid"]["mouth"].get("ratio"); row["human_twitch"] = hum["all"]["twitch_index"]["mouth"].get("ratio")
        rows.append(row)
    for r in rows:
        b = base.get(r["arm"].split("_")[0])
        if b: r["cost_vs_C0"] = f"{100 * (r['ls_s'] / b - 1):+.0f}%"
    cols = ["arm", "ls_s", "cost_vs_C0", "seams", "seam_jump_med_v", "seam_jump_med_r", "seam_share_gt1_5_v", "seam_share_gt1_5_r", "seam_d1_step_v", "seam_d1_step_r",
            "per_sec_max_d1_jump_mean", "per_sec_max_d1_jump_max", "resid", "twitch", "acc", "vel", "sharp", "upper_resid", "human_resid", "human_twitch", "syncnet_mix", "syncnet_dub", "av"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join("–" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    print("\n".join(md))
    if a.md: pathlib.Path(a.md).write_text("\n".join(md) + "\n")
    (P / "seam_table.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
