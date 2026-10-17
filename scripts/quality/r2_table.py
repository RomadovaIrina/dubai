#!/usr/bin/env python3
"""Round 2 (track A): one markdown table per window from the arm outputs in r2/win (slide_metric, SyncNet, common metric v4, manifest).
    python scripts/quality/r2_table.py [--win DIR] [--out reports/experiments/lipsync-temporal/r2_stage1_table.md] [--json ...]
"""
from __future__ import annotations
import argparse, json, pathlib, re

ARM_DOC = {"clean0": "clean0 (box ±2 anchors, mask ±3, rounded anchors, fp16 affine)", "raw": "raw MediaPipe per frame (rounded, fp16)",
           "rawnr32": "raw + sub-pixel anchors + fp32 affine", "c032": "clean0 + fp32 affine", "c0nr32": "clean0 + sub-pixel anchors + fp32 affine",
           "flow4": "flow-driven anchors (sigma 4)", "flow4nr": "flow + sub-pixel", "flow4nr32": "flow-driven + sub-pixel + fp32",
           "kal": "Kalman CV+RTS (q5, r0.5)", "kalnr32": "Kalman + sub-pixel + fp32", "sub3nr32": "MediaPipe every 3rd frame + flow track + sub-pixel + fp32",
           "wrap": "NO generation, clean0 geometry (crop -> paste only)", "wrapnr": "no generation, sub-pixel", "wrap32": "no generation, fp32",
           "wrapnr32": "NO generation, sub-pixel + fp32", "wrapflownr32": "no generation, flow + sub-pixel + fp32", "c0nr": "clean0 + sub-pixel"}


def g(d, *ks):
    for k in ks:
        if d is None: return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def f(x, n=3):
    return "–" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win", default="/workspace/dub/artifacts/lipsync-temporal/r2/win")
    ap.add_argument("--out", default="reports/experiments/lipsync-temporal/r2_stage1_table.md"); ap.add_argument("--json", default="reports/experiments/lipsync-temporal/r2_stage1_table.json")
    a = ap.parse_args(); W = pathlib.Path(a.win); rows = {}
    for sj in sorted(W.glob("*.slide.json")):
        m = re.match(r"(\d\d)_(w\d)_(.+)\.slide\.json", sj.name)
        if not m: continue
        vid, w, arm = m.groups(); s = json.loads(sj.read_text()); base = W / f"{vid}_{w}_{arm}"
        man = json.loads(base.with_suffix(".manifest.json").read_text()) if base.with_suffix(".manifest.json").exists() else {}
        sy = json.loads((W / f"{vid}_{w}_{arm}.sync.json").read_text()) if (W / f"{vid}_{w}_{arm}.sync.json").exists() else {}
        v4 = json.loads((W / f"{vid}_{w}_{arm}.v4.json").read_text()) if (W / f"{vid}_{w}_{arm}.v4.json").exists() else {}
        ls_frames = sum(x["end_frame"] - x["start_frame"] for x in g(man, "lipsync", "segments") or [] if x["action"] == "LATENT_SYNC")
        ls_s = g(man, "lipsync", "latentsync_inference_total_s")
        nb = s["regions"]["nosebase"]; ch = s["regions"]["cheeks"]
        rows.setdefault((vid, w), []).append({
            "arm": arm, "nose_slide_mean": g(nb, "slide_px", "mean"), "nose_slide_p95": g(nb, "slide_px", "p95"), "nose_slide_x_mean": g(nb, "slide_x_px", "mean"),
            "nose_abs_mean": g(nb, "abs_px", "mean"), "nose_abs_p95": g(nb, "abs_px", "p95"), "cheek_slide_mean": g(ch, "slide_px", "mean"),
            "corr_shake": nb.get("corr_shake_x"), "upper_abs": g(s, "regions", "upper", "abs_px", "mean"), "iod": s.get("iod_px"),
            "sync_dub": g(sy, "dub_only", "conf"), "av_dub": g(sy, "dub_only", "av_offset"), "sync_mix": g(sy, "mix", "conf"),
            "shear": g(v4, "all", "shear", "ratio"), "lf_jerk": g(v4, "all", "jerk", "lowerface", "ratio"), "mouth_resid": g(v4, "all", "resid", "mouth", "ratio"),
            "mouth_sharp": g(v4, "all", "sharp", "mouth", "ratio"),
            "ls_s": ls_s, "ls_s_per_ls_min": round(60 * ls_s / (ls_frames / 25), 1) if ls_s and ls_frames else None})
    out = ["# Round 2 stage 1 — lower-face position arms (25 fps windows)", "",
           "slide = |d_t − d_(t−1)| of the nose-base (alae / nostrils / subnasale / philtrum top) displacement output vs master on the same frame (DIS flow, "
           "inside the composite mask), px at master resolution; abs = |d_t|; corr = corr of d_x with the high-passed rigid head displacement; upper = sanity "
           "(source region, flow noise floor). shear / lf jerk / mouth resid / sharp = common metric v4 ratios vs the master window. LS s/min = LatentSync seconds per "
           "minute of LS frames (CPU shared with evals; for ranking only).", ""]
    for (vid, w), rs in sorted(rows.items()):
        order = list(ARM_DOC); rs.sort(key=lambda r: order.index(r["arm"]) if r["arm"] in order else 99)
        out += [f"## {vid} {w} (IOD {rs[0]['iod']} px)", "",
                "| arm | nose slide mean / p95 | slide x | nose abs mean / p95 | cheek slide | corr shake | upper | SyncNet dub / mix (AV) | shear | lf jerk | mouth resid | sharp | LS s/min |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            out.append(f"| {r['arm']} | {f(r['nose_slide_mean'])} / {f(r['nose_slide_p95'])} | {f(r['nose_slide_x_mean'])} | {f(r['nose_abs_mean'])} / {f(r['nose_abs_p95'])} | "
                       f"{f(r['cheek_slide_mean'])} | {f(r['corr_shake'],2)} | {f(r['upper_abs'])} | {f(r['sync_dub'],2)} / {f(r['sync_mix'],2)} ({f(r['av_dub'])}) | "
                       f"{f(r['shear'],2)} | {f(r['lf_jerk'],2)} | {f(r['mouth_resid'],2)} | {f(r['mouth_sharp'],2)} | {f(r['ls_s_per_ls_min'],0)} |")
        out.append("")
    out += ["Arms: " + "; ".join(f"`{k}` = {v}" for k, v in ARM_DOC.items()), ""]
    pathlib.Path(a.out).write_text("\n".join(out)); pathlib.Path(a.json).write_text(json.dumps({f"{k[0]}_{k[1]}": v for k, v in rows.items()}, indent=1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
